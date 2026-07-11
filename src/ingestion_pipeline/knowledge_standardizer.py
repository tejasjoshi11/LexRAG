"""Stage 6 of the LexRAG ingestion pipeline: knowledge standardization.

This module implements the knowledge standardizer, which transforms one
immutable :class:`~src.contracts.enriched_document.EnrichedDocument` into
one immutable :class:`~src.contracts.standardized_document.StandardizedDocument`.

Standardization is a purely deterministic rendering step: it never
discovers new semantic facts and it never hardcodes business knowledge.
The semantic metadata already extracted by Stage 5 (court, judges,
jurisdiction, statutes, legal entities, keywords, semantic relationships,
legal domain) is rendered into a fixed-order canonical metadata block --
this is the only mechanism by which that metadata survives into
downstream retrieval, since ``StandardizedDocument`` carries no semantic
fields of its own and Stage 7 onward never sees ``EnrichedDocument``
again. ``parsed_document.raw_text`` is then rewritten using an
externally supplied canonical-vocabulary table (e.g. expanding "I.R.C."
to "Internal Revenue Code"); this module defines no vocabulary of its
own, and an empty table leaves the body unchanged.

Every character of ``standardized_content`` is covered by exactly one
entry in ``standardized_offset_mappings``. Entries covering the
synthesized metadata header carry the sentinel source span ``(-1, -1)``,
since that header is rendered, not quoted, and has no natural character
span in ``ParsedDocument.raw_text``. Entries covering the body carry an
exact, non-sentinel source span, without ever recomputing page offsets.
``standardized_content`` is composed of a deterministic metadata header
followed by the standardized document body.

The stage never parses PDFs, cleans text, extracts semantic metadata,
generates embeddings, performs retrieval, chunks documents, or mutates
``ParsedDocument`` or ``EnrichedDocument``. Unknown or missing semantic
fields are rendered with a fixed placeholder rather than treated as a
failure (fail-open policy); only structurally inconsistent or malformed
input is ever treated as an error.
"""

from collections.abc import Callable
import logging
import re

from src.contracts.enriched_document import EnrichedDocument, SemanticRelationships
from src.contracts.parsed_document import ParsedDocument
from src.contracts.standardized_document import (
    StandardizedDocument,
    StandardizedOffsetMappings,
)
from src.shared.exceptions import StandardizationError
from src.shared.types import DocumentID

__all__ = [
    "standardize_document",
    "PermanentStandardizationError",
    "RetryableStandardizationError",
]

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# Error Hierarchy
# ============================================================================
# Both subclasses remain instances of the existing StandardizationError, so
# callers that only handle StandardizationError continue to work. Callers
# that need retry semantics can distinguish the two subclasses.


class PermanentStandardizationError(StandardizationError):
    """Raised when standardization fails in a way that will recur on retry.

    The supplied EnrichedDocument is internally inconsistent (e.g. its
    document_id disagrees with the document_id carried by its own
    ParsedDocument), has no raw text to standardize, or an assembled
    StandardizedDocument would carry invalid offset mappings. Rerunning
    the stage with the same input will fail again; the underlying data
    must be corrected upstream first.
    """


class RetryableStandardizationError(StandardizationError):
    """Raised when standardization fails in a way that may not recur on retry.

    The stage may be safely rerun using the same EnrichedDocument, without
    repeating Discovery, Parsing, Cleaning, or Semantic Extraction.
    """


# ============================================================================
# Canonical Vocabulary Contract
# ============================================================================
# This module defines no canonical vocabulary of its own. Abbreviation
# expansion, controlled-vocabulary substitution, and canonical wording are
# business knowledge; per "configuration over hardcoding", the ordered
# (pattern, canonical replacement) table must be supplied by the caller
# (sourced from project configuration) via the `canonical_vocabulary`
# parameter of `standardize_document`. An empty table is a valid,
# side-effect-free choice: the body is then carried through unchanged.

# Sentinel (source_start, source_end) pair used for standardized_content
# spans that are rendered rather than quoted from ParsedDocument.raw_text
# (currently: the metadata header). Never produced for the body, which
# always carries an exact, non-negative source span.
_NO_SOURCE_PROVENANCE: tuple[int, int] = (-1, -1)

# Fixed labels and placeholders for the canonical metadata block. Field
# order, labels, and placeholder text never change, so identical
# EnrichedDocument input always renders an identical block.
_UNKNOWN_PLACEHOLDER = "Unknown"
_EMPTY_SEQUENCE_PLACEHOLDER = "None identified"
_FIELD_SEPARATOR = "; "
_METADATA_BODY_SEPARATOR = "\n\n"
_MAX_METADATA_SEQUENCE_ITEMS = 20
_MAX_METADATA_RELATIONSHIPS = 20


# ============================================================================
# Public API
# ============================================================================


def standardize_document(
    enriched_document: EnrichedDocument,
    *,
    metadata_block_builder: Callable[[EnrichedDocument], str] | None = None,
    canonical_vocabulary: tuple[tuple[re.Pattern[str], str], ...] = (),
) -> StandardizedDocument:
    """Produce a deterministic standardized representation of one document.

    Combines the semantic metadata already extracted onto
    ``enriched_document`` with ``enriched_document.parsed_document.raw_text``
    into an immutable ``StandardizedDocument``: a fixed-order canonical
    metadata block followed by a canonically-worded body. No new semantic
    facts are discovered or inferred; every field rendered was already
    present on the input. ``ParsedDocument`` is carried through unchanged
    and is never mutated.

    Args:
        enriched_document: The immutable semantic metadata to standardize.
        metadata_block_builder: Renders the canonical metadata block from
            ``enriched_document``. Defaults to the built-in fixed-order
            renderer.
        canonical_vocabulary: Ordered (pattern, canonical replacement)
            table applied to the document body, sourced from project
            configuration -- this module hardcodes no vocabulary of its
            own. Defaults to an empty tuple, which leaves the body
            unchanged.

    Returns:
        StandardizedDocument: The immutable standardized representation.

    Raises:
        PermanentStandardizationError: If ``enriched_document`` is
            internally inconsistent, its ``ParsedDocument`` has no raw
            text to standardize, or the assembled result would carry
            invalid offset mappings. Retrying with the same input will
            fail again.
        RetryableStandardizationError: If standardization fails
            unexpectedly. The stage may be safely rerun using the same
            EnrichedDocument, without repeating any upstream stage.
    """
    document_id = _validate_input(enriched_document)

    metadata_block_builder = metadata_block_builder or _build_metadata_block

    try:
        metadata_block = metadata_block_builder(enriched_document)
        _LOGGER.debug(
            "Constructed metadata block of length %d for document_id %r",
            len(metadata_block),
            document_id,
        )
        body, body_mappings = _standardize_body(
            enriched_document.parsed_document.raw_text,
            canonical_vocabulary,
        )
    except StandardizationError:
        raise
    except Exception as exc:
        _LOGGER.exception(
            "Knowledge standardization failed for document_id %r",
            document_id,
        )
        raise RetryableStandardizationError(
            "Knowledge standardization failed unexpectedly for "
            f"document_id {document_id!r}; the stage may be safely "
            "rerun with the same EnrichedDocument."
        ) from exc

    standardized_content = f"{metadata_block}{_METADATA_BODY_SEPARATOR}{body}"
    header_length = len(metadata_block) + len(_METADATA_BODY_SEPARATOR)
    no_source_start, no_source_end = _NO_SOURCE_PROVENANCE

    standardized_offset_mappings: StandardizedOffsetMappings = (
        (0, header_length, no_source_start, no_source_end),
        *(
            (
                standardized_start + header_length,
                standardized_end + header_length,
                source_start,
                source_end,
            )
            for standardized_start, standardized_end, source_start, source_end in body_mappings
        ),
    )

    _validate_output(
        standardized_content=standardized_content,
        standardized_offset_mappings=standardized_offset_mappings,
        raw_text_length=len(enriched_document.parsed_document.raw_text),
        document_id=document_id,
    )

    return StandardizedDocument(
        parsed_document=enriched_document.parsed_document,
        document_id=document_id,
        standardized_content=standardized_content,
        standardized_offset_mappings=standardized_offset_mappings,
    )


# ============================================================================
# Input Validation
# ============================================================================


def _validate_input(enriched_document: EnrichedDocument) -> DocumentID:
    """Validate internal consistency of one EnrichedDocument.

    Returns:
        DocumentID: The validated canonical document identifier.

    Raises:
        PermanentStandardizationError: If ``enriched_document.document_id``
            disagrees with the canonical document_id carried by its own
            ParsedDocument, or if there is no raw text to standardize.
    """
    canonical_document_id = enriched_document.parsed_document.document_id

    if enriched_document.document_id != canonical_document_id:
        raise PermanentStandardizationError(
            "EnrichedDocument.document_id disagrees with the canonical "
            "document_id carried by its own ParsedDocument: "
            f"{enriched_document.document_id!r} != {canonical_document_id!r}"
        )

    if not enriched_document.parsed_document.raw_text.strip():
        raise PermanentStandardizationError(
            "ParsedDocument.raw_text is empty; there is no content to "
            f"standardize for document_id {canonical_document_id!r}"
        )

    return canonical_document_id


# ============================================================================
# Metadata Block Assembly
# ============================================================================


def _build_metadata_block(enriched_document: EnrichedDocument) -> str:
    """Render already-extracted semantic metadata as a canonical block.

    Field order, labels, and placeholder text are fixed, so identical
    EnrichedDocument input always renders an identical block. This block
    never introduces a semantic fact that was not already present on
    ``enriched_document``; missing fields render as a fixed placeholder
    rather than being inferred or omitted.
    """
    lines = (
        f"Document ID: {enriched_document.document_id}",
        f"Court: {_scalar_or_placeholder(enriched_document.court)}",
        f"Jurisdiction: {_scalar_or_placeholder(enriched_document.jurisdiction)}",
        f"Legal Domain: {_scalar_or_placeholder(enriched_document.legal_domain)}",
        f"Judges: {_sequence_or_placeholder(enriched_document.judges)}",
        f"Statutes: {_sequence_or_placeholder(enriched_document.statutes)}",
        f"Legal Entities: {_sequence_or_placeholder(enriched_document.legal_entities)}",
        f"Keywords: {_sequence_or_placeholder(enriched_document.keywords)}",
        f"Semantic Relationships: "
        f"{_render_relationships(enriched_document.semantic_relationships)}",
    )
    return "\n".join(lines)


def _scalar_or_placeholder(value: str | None) -> str:
    """Render an optional scalar field, or the fixed placeholder if absent."""
    return value if value is not None else _UNKNOWN_PLACEHOLDER


def _sequence_or_placeholder(values: tuple[str, ...]) -> str:
    """Render a tuple field in its given order, or the fixed placeholder."""

    if not values:
        return _EMPTY_SEQUENCE_PLACEHOLDER

    return _FIELD_SEPARATOR.join(
        values[:_MAX_METADATA_SEQUENCE_ITEMS]
    )


def _render_relationships(
    relationships: SemanticRelationships,
) -> str:
    """Render relationship triples as ``subject -predicate-> object``."""

    if not relationships:
        return _EMPTY_SEQUENCE_PLACEHOLDER

    return _FIELD_SEPARATOR.join(
        f"{subject} -{predicate}-> {obj}"
        for subject, predicate, obj in relationships[:_MAX_METADATA_RELATIONSHIPS]
    )


# ============================================================================
# Body Standardization
# ============================================================================


def _standardize_body(
    raw_text: str,
    canonical_vocabulary: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[str, tuple[tuple[int, int, int, int], ...]]:
    """Apply canonical wording to raw_text and record source offset spans.

    Returns the standardized body text together with a tuple of
    ``(standardized_start, standardized_end, source_start, source_end)``
    mappings that jointly and contiguously cover every character of the
    returned body, so any position in the standardized body can be traced
    back to an exact character span in ``ParsedDocument.raw_text``.
    """
    matches = _find_vocabulary_matches(raw_text, canonical_vocabulary)

    body_parts: list[str] = []
    mappings: list[tuple[int, int, int, int]] = []
    standardized_cursor = 0
    source_cursor = 0

    for source_start, source_end, replacement in matches:
        if source_start > source_cursor:
            unchanged = raw_text[source_cursor:source_start]
            body_parts.append(unchanged)
            mappings.append((
                standardized_cursor,
                standardized_cursor + len(unchanged),
                source_cursor,
                source_start,
            ))
            standardized_cursor += len(unchanged)

        body_parts.append(replacement)
        mappings.append((
            standardized_cursor,
            standardized_cursor + len(replacement),
            source_start,
            source_end,
        ))
        standardized_cursor += len(replacement)
        source_cursor = source_end

    if source_cursor < len(raw_text):
        trailing = raw_text[source_cursor:]
        body_parts.append(trailing)
        mappings.append((
            standardized_cursor,
            standardized_cursor + len(trailing),
            source_cursor,
            len(raw_text),
        ))

    return "".join(body_parts), tuple(mappings)


def _find_vocabulary_matches(
    raw_text: str,
    canonical_vocabulary: tuple[tuple[re.Pattern[str], str], ...],
) -> tuple[tuple[int, int, str], ...]:
    """Find non-overlapping canonical-vocabulary matches in source order.

    Patterns are applied in table order; a pattern never claims a span
    already claimed by an earlier pattern, so replacement is unambiguous
    and depends only on ``canonical_vocabulary`` and ``raw_text``.
    """
    claimed: list[tuple[int, int]] = []
    matches: list[tuple[int, int, str]] = []

    for pattern, replacement in canonical_vocabulary:
        for match in pattern.finditer(raw_text):
            start, end = match.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            matches.append((start, end, replacement))

    matches.sort(key=lambda item: item[0])
    return tuple(matches)


# ============================================================================
# Output Validation
# ============================================================================


def _validate_output(
    *,
    standardized_content: str,
    standardized_offset_mappings: StandardizedOffsetMappings,
    raw_text_length: int,
    document_id: DocumentID,
) -> None:
    """Check structural invariants of a freshly assembled StandardizedDocument.

    Every character of ``standardized_content`` must be covered by exactly
    one contiguous, in-order mapping entry, starting at position 0. An
    entry's source span must either be an exact, in-bounds span into
    ``ParsedDocument.raw_text``, or the ``_NO_SOURCE_PROVENANCE`` sentinel.

    Raises:
        PermanentStandardizationError: If any offset mapping is out of
            range, out of order, or leaves a gap, indicating a malformed
            standardized representation.
    """
    content_length = len(standardized_content)
    expected_next_start = 0

    for standardized_start, standardized_end, source_start, source_end in (
        standardized_offset_mappings
    ):
        is_sentinel = (source_start, source_end) == _NO_SOURCE_PROVENANCE
        in_bounds = (
            0 <= standardized_start <= standardized_end <= content_length
            and (
                is_sentinel
                or 0 <= source_start <= source_end <= raw_text_length
            )
        )
        if not in_bounds:
            raise PermanentStandardizationError(
                "Invalid standardized offset mapping "
                f"{(standardized_start, standardized_end, source_start, source_end)!r} "
                f"for document_id {document_id!r}"
            )
        if standardized_start != expected_next_start:
            raise PermanentStandardizationError(
                "Standardized offset mappings are not contiguous from "
                f"position 0 for document_id {document_id!r}: expected "
                f"start {expected_next_start}, got {standardized_start}"
            )
        expected_next_start = standardized_end

    if expected_next_start != content_length:
        raise PermanentStandardizationError(
            "Standardized offset mappings do not cover the full "
            f"standardized_content for document_id {document_id!r}: "
            f"covered up to {expected_next_start} of {content_length}"
        )