"""Stage 7 of the LexRAG ingestion pipeline: intelligent chunking.

This module implements the intelligent chunker, which transforms one
immutable :class:`~src.contracts.standardized_document.StandardizedDocument`
into an ordered, immutable ``tuple[RetrievalChunk, ...]``.

Chunking is a purely deterministic, rule-based segmentation step: it never
discovers new semantic information, never invents section structure that
is not already present in ``standardized_content``, and never recomputes
provenance that Stage 6 already established. Per the frozen Stage 7
architecture (ADR-010), heading and section detection are regex/rule
based only -- no LLMs, no NLP inference, no ML-based classification.

Metadata header handling follows ADR-009 exactly: the deterministic
metadata header Stage 6 renders at the start of ``standardized_content``
is prepended, verbatim, to the ``chunk_text`` of the document's first
chunk only. It never becomes a standalone chunk, is never repeated, and
never contributes to that chunk's ``page_start``, ``page_end``, or
``source_span`` -- those are derived solely from the chunk's real (body)
characters.

Every chunk's ``page_start``, ``page_end``, and ``source_span`` are
derived exclusively from ``standardized_offset_mappings`` (Stage 6's
provenance) and ``parsed_document.page_offsets`` -- this module never
scans or re-derives offsets against ``parsed_document.raw_text`` directly.

``parsed_document.page_offsets`` is consumed per its immutable contract:
an ordered, contiguous ``tuple[tuple[int, int], ...]`` where each
``(char_start, char_end)`` pair is the raw-text character span for one
page in document order. Page numbers are derived from positional index
at translation time (index + 1), isolated behind ``_resolve_page``.

Heading detection supports both newline-preserving bodies and single-line
bodies. When line breaks are present, line-anchored rules are applied.
When absent, fallback inline rules detect explicit legal heading markers
without depending on ``\n``.

The stage never parses PDFs, cleans text, extracts semantic metadata,
generates embeddings, performs retrieval, or mutates ``ParsedDocument``
or ``StandardizedDocument``.
"""

from collections.abc import Callable, Sequence
import hashlib
import logging
import re
from typing import NamedTuple, TypeAlias

from src.contracts.parsed_document import ParsedDocument
from src.contracts.retrieval_chunk import (
    RetrievalChunk,
    SectionHierarchy,
    SourceSpan,
)
from src.contracts.standardized_document import (
    StandardizedDocument,
    StandardizedOffsetMappings,
)
from src.shared.exceptions import ChunkingError
from src.shared.types import ChunkID, DocumentID, PageNumber

__all__ = [
    "chunk_document",
    "HeadingMatch",
    "HeadingBoundaries",
    "PermanentChunkingError",
    "RetryableChunkingError",
]

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# Error Hierarchy
# ============================================================================
# Both subclasses remain instances of the existing ChunkingError, so callers
# that only handle ChunkingError continue to work. Callers that need retry
# semantics can distinguish the two subclasses. Mirrors the Stage 6 pattern
# exactly (PermanentStandardizationError / RetryableStandardizationError).


class PermanentChunkingError(ChunkingError):
    """Raised when chunking fails in a way that will recur on retry.

    The supplied StandardizedDocument is internally inconsistent, malformed
    (e.g. its offset mappings do not begin with the expected sentinel
    header entry), or the requested configuration is invalid (e.g.
    ``chunk_overlap_tokens >= max_chunk_tokens``). Rerunning the stage with
    the same input will fail again; the underlying data or configuration
    must be corrected first.
    """


class RetryableChunkingError(ChunkingError):
    """Raised when chunking fails in a way that may not recur on retry.

    The stage may be safely rerun using the same StandardizedDocument,
    without repeating Discovery, Parsing, Cleaning, Semantic Extraction, or
    Knowledge Standardization.
    """


# ============================================================================
# Sentinel / Internal Types
# ============================================================================

# Sentinel (source_start, source_end) pair identifying a
# standardized_offset_mappings entry that covers synthesized metadata-
# header text rather than a real span of ParsedDocument.raw_text. Defined
# identically to Stage 6's own sentinel so the two stages agree on meaning
# without either importing the other's private constants.
_NO_SOURCE_PROVENANCE: tuple[int, int] = (-1, -1)

_METADATA_HEADER_BODY_SEPARATOR_HINT = (
    "the sentinel mapping entry's end position already includes any "
    "separator Stage 6 inserted between the header and the body"
)


class HeadingMatch(NamedTuple):
    """One detected heading boundary within the standardized body.

    Attributes:
        position: Character offset *relative to the body text* (i.e. to
            ``standardized_content`` with the metadata header already
            stripped off) where this heading's text begins.
        end_position: Character offset (body-relative, exclusive) where
            this heading's text ends.
        level: Nesting depth, 1 = outermost. A match at a shallower level
            closes all currently open deeper levels; a match at a deeper
            level nests under the most recent shallower match.
        text: The heading's literal text, preserved verbatim.
    """

    position: int
    end_position: int
    level: int
    text: str


HeadingBoundaries: TypeAlias = tuple[HeadingMatch, ...]


class _Section(NamedTuple):
    """One leaf section: its hierarchy, heading, and absolute char range.

    ``body_start``/``body_end`` are absolute offsets into
    ``standardized_content`` (not body-relative), and never include the
    section's own governing heading line.
    """

    section_hierarchy: SectionHierarchy
    heading: str | None
    body_start: int
    body_end: int


# ============================================================================
# Public API
# ============================================================================


def chunk_document(
    standardized_document: StandardizedDocument,
    *,
    heading_detector: Callable[[str], HeadingBoundaries] | None = None,
    token_counter: Callable[[str], int] | None = None,
    chunk_id_hasher: Callable[[str], str] | None = None,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
    chunker_version: str,
) -> tuple[RetrievalChunk, ...]:
    """Produce an ordered, immutable sequence of RetrievalChunks.

    Detects section/heading structure already present in
    ``standardized_document.standardized_content``, windows each section
    into token-bounded, boundary-aligned chunks, and translates every
    chunk's character range back into page-accurate provenance using
    ``standardized_offset_mappings`` and ``parsed_document.page_offsets``.
    No new semantic facts are discovered; only structure already rendered
    by Stage 6 is detected and used.

    Args:
        standardized_document: The immutable standardized representation
            to chunk.
        heading_detector: Detects heading boundaries in the body text
            (metadata header already excluded). Defaults to the built-in
            deterministic, regex-based detector.
        token_counter: Deterministic token-count approximation for a
            string. Defaults to a coarse whitespace-split count; should be
            overridden with a real tokenizer matching the configured
            embedding model in production.
        chunk_id_hasher: Deterministic digest function used by chunk-id
            construction. Receives the canonical identity payload and
            returns a hex-like digest string; defaults to SHA-256.
        max_chunk_tokens: Maximum tokens per chunk, including the metadata
            header for the document's first chunk.
        chunk_overlap_tokens: Tokens of trailing context repeated at the
            start of the next window within the same section.
        chunker_version: Deterministic descriptor of the chunking
            configuration in effect (rule table, tokenizer identity, size
            settings). Participates in ``chunk_id`` construction so a
            configuration change never collides with a prior run's ids.

    Returns:
        tuple[RetrievalChunk, ...]: Ordered (document order), non-empty,
        immutable sequence of chunks.

    Raises:
        PermanentChunkingError: If ``standardized_document`` is malformed,
            configuration is invalid, or an assembled chunk sequence would
            violate a structural invariant. Retrying with the same input
            will fail again.
        RetryableChunkingError: If chunking fails unexpectedly. The stage
            may be safely rerun using the same StandardizedDocument.
    """
    document_id = _validate_input(
        standardized_document,
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    heading_detector = heading_detector or _default_heading_detector
    token_counter = token_counter or _default_token_counter
    chunk_id_hasher = chunk_id_hasher or _default_chunk_id_hasher

    try:
        standardized_content = standardized_document.standardized_content
        mappings = standardized_document.standardized_offset_mappings

        body_start = _locate_header_body_split(mappings)
        header_region = standardized_content[:body_start]
        body_text = standardized_content[body_start:]

        headings = heading_detector(body_text)
        _LOGGER.debug(
            "Detected %d headings for document_id %r",
            len(headings),
            document_id,
        )
        sections = _build_section_hierarchy(body_text, body_start, headings)

        _LOGGER.debug(
            "Built %d sections for document_id %r",
            len(sections),
            document_id,
        )

        chunks = _build_retrieval_chunks(
            sections=sections,
            standardized_content=standardized_content,
            standardized_offset_mappings=mappings,
            header_region=header_region,
            document_id=document_id,
            chunker_version=chunker_version,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            token_counter=token_counter,
            chunk_id_hasher=chunk_id_hasher,
            page_offsets=standardized_document.parsed_document.page_offsets,
        )
        _LOGGER.debug(
            "Generated %d chunks for document_id %r",
            len(chunks),
            document_id,
        )
    except ChunkingError:
        raise
    except Exception as exc:
        _LOGGER.exception(
            "Intelligent chunking failed unexpectedly for document_id %r",
            document_id,
        )
        raise RetryableChunkingError(
            "Intelligent chunking failed unexpectedly for document_id "
            f"{document_id!r}; the stage may be safely rerun with the same "
            "StandardizedDocument."
        ) from exc

    _validate_output(chunks, header_region=header_region, document_id=document_id)

    return chunks


# ============================================================================
# Input Validation
# ============================================================================


def _validate_input(
    standardized_document: StandardizedDocument,
    *,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
) -> DocumentID:
    """Validate configuration and internal consistency of one input.

    Full offset-mapping contiguity validation is Stage 6's own
    responsibility (enforced before a StandardizedDocument can exist at
    all); Stage 7 only checks what it directly depends on here.

    Returns:
        DocumentID: The validated canonical document identifier.

    Raises:
        PermanentChunkingError: If the configuration is invalid, or
            ``standardized_document.document_id`` disagrees with the
            canonical document_id carried by its own ParsedDocument, or
            there is no content to chunk.
    """
    if max_chunk_tokens <= 0:
        raise PermanentChunkingError(
            f"max_chunk_tokens must be positive; got {max_chunk_tokens!r}"
        )
    if chunk_overlap_tokens < 0:
        raise PermanentChunkingError(
            f"chunk_overlap_tokens must be non-negative; got {chunk_overlap_tokens!r}"
        )
    if chunk_overlap_tokens >= max_chunk_tokens:
        raise PermanentChunkingError(
            "chunk_overlap_tokens must be smaller than max_chunk_tokens; "
            f"got chunk_overlap_tokens={chunk_overlap_tokens!r}, "
            f"max_chunk_tokens={max_chunk_tokens!r}"
        )

    canonical_document_id = standardized_document.parsed_document.document_id
    if standardized_document.document_id != canonical_document_id:
        raise PermanentChunkingError(
            "StandardizedDocument.document_id disagrees with the canonical "
            "document_id carried by its own ParsedDocument: "
            f"{standardized_document.document_id!r} != {canonical_document_id!r}"
        )

    if not standardized_document.standardized_content:
        raise PermanentChunkingError(
            f"standardized_content is empty for document_id {canonical_document_id!r}; "
            "there is no content to chunk"
        )

    return canonical_document_id



# ============================================================================
# Header / Body Split
# ============================================================================


def _locate_header_body_split(mappings: StandardizedOffsetMappings) -> int:
    """Return the absolute position where the standardized body begins.

    Found from Stage 6 provenance -- the sentinel-provenance mapping entry
    that always covers the metadata header -- never by searching
    ``standardized_content`` for header-like text. This keeps Stage 7
    dependent only on Stage 6's provenance contract, not on text
    formatting that could change independently of it.
    """
    if not mappings:
        raise PermanentChunkingError(
            "standardized_offset_mappings is empty; cannot locate the "
            "metadata header boundary"
        )

    first_std_start, first_std_end, first_source_start, first_source_end = mappings[0]

    if first_std_start != 0 or (first_source_start, first_source_end) != _NO_SOURCE_PROVENANCE:
        raise PermanentChunkingError(
            "Expected the first standardized_offset_mappings entry to "
            "start at 0 and carry sentinel provenance for the metadata "
            f"header; got {mappings[0]!r} ({_METADATA_HEADER_BODY_SEPARATOR_HINT})"
        )

    return first_std_end


# ============================================================================
# Heading Detection (default, rule-based, injectable)
# ============================================================================
# Ordered (pattern, level) rules. A later pattern never claims a span
# already claimed by an earlier one, so detection is unambiguous and
# depends only on body_text and this table -- never on model inference,
# per ADR-010. This table is the *default*; callers may inject their own
# heading_detector entirely, or wrap this one with a different table via
# functools.partial.

_DEFAULT_HEADING_RULES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"^[ \t]*(PART\s+[IVXLCDM]+\b[^\n]*)$", re.MULTILINE), 1),
    (
        re.compile(
            r"^[ \t]*((?:ARTICLE|SECTION)\s+[0-9]+(?:\.[0-9]+)*\b[^\n]*)$",
            re.MULTILINE | re.IGNORECASE,
        ),
        2,
    ),
    (re.compile(r"^[ \t]*(\xa7+\s?[0-9]+(?:\.[0-9]+)*\b[^\n]*)$", re.MULTILINE), 2),
    (
        re.compile(r"^[ \t]*([0-9]+(?:\.[0-9]+)*\.?\s+[A-Z][^\n]{0,120})$", re.MULTILINE),
        3,
    ),
    (re.compile(r"^[ \t]*([A-Z][A-Z0-9 ,'\-/&]{3,80})[ \t]*$", re.MULTILINE), 2),
)

_DEFAULT_INLINE_HEADING_RULES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(PART\s+[IVXLCDM]+\b[^.;:\n]{0,120})"), 1),
    (
        re.compile(r"\b((?:ARTICLE|SECTION)\s+[0-9]+(?:\.[0-9]+)*\b[^.;:\n]{0,120})", re.IGNORECASE),
        2,
    ),
    (re.compile(r"(\xa7+\s?[0-9]+(?:\.[0-9]+)*\b[^.;:\n]{0,120})"), 2),
    (re.compile(r"\b([0-9]+(?:\.[0-9]+)*\.?\s+[A-Z][^.;:\n]{0,120})"), 3),
)


def _collect_heading_matches(
    body_text: str,
    rules: tuple[tuple[re.Pattern[str], int], ...],
) -> list[tuple[int, int, int, str]]:
    """Collect non-overlapping heading spans using ordered regex rules."""
    claimed: list[tuple[int, int]] = []
    raw_matches: list[tuple[int, int, int, str]] = []

    for pattern, level in rules:
        for match in pattern.finditer(body_text):
            start, end = match.span(1)
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            raw_matches.append((start, end, level, match.group(1).strip()))

    raw_matches.sort(key=lambda item: item[0])
    return raw_matches


def _default_heading_detector(body_text: str) -> HeadingBoundaries:
    """Deterministic, rule-based heading detector (the injectable default).

    Applies ``_DEFAULT_HEADING_RULES`` in table order over ``body_text``
    only (the metadata header has already been excluded by the time this
    is called). No ML, no NLP inference, no LLMs, per ADR-010.
    """
    rules = _DEFAULT_HEADING_RULES if "\n" in body_text else _DEFAULT_INLINE_HEADING_RULES
    raw_matches = _collect_heading_matches(body_text, rules)

    return tuple(
        HeadingMatch(position=position, end_position=end_position, level=level, text=text)
        for position, end_position, level, text in raw_matches
    )


def _default_token_counter(text: str) -> int:
    """Coarse, deterministic token-count approximation (injectable default).

    Splits on whitespace. Exists only so the stage is runnable without a
    real tokenizer configured; production use should inject a tokenizer
    matching the configured embedding model.
    """
    return len(text.split())


# ============================================================================
# Section Hierarchy
# ============================================================================


def _build_section_hierarchy(
    body_text: str,
    body_start: int,
    headings: HeadingBoundaries,
) -> tuple[_Section, ...]:
    """Convert detected heading boundaries into contiguous leaf sections.

    ``headings`` positions are relative to ``body_text``; returned
    sections use absolute offsets into ``standardized_content``. A
    section's body range never includes its own governing heading line --
    the heading text is carried separately on ``_Section.heading``. If no
    headings are detected at all, the entire body becomes a single
    top-level section with an empty hierarchy and no heading: a
    deliberate fail-open default, never an error, mirroring Stage 6's own
    "missing structure renders as a fixed default, never a failure"
    policy.
    """
    body_end_absolute = body_start + len(body_text)

    if not headings:
        return (_Section((), None, body_start, body_end_absolute),)

    ordered_headings = tuple(sorted(headings, key=lambda heading: heading.position))

    sections: list[_Section] = []
    stack: list[str] = []

    if ordered_headings[0].position > 0:
        sections.append(
            _Section((), None, body_start, body_start + ordered_headings[0].position)
        )

    for index, heading in enumerate(ordered_headings):
        while len(stack) >= heading.level:
            stack.pop()
        stack.append(heading.text)

        section_start_relative = _derive_section_start(body_text, heading.end_position)
        section_start = body_start + section_start_relative

        next_position = (
            ordered_headings[index + 1].position
            if index + 1 < len(ordered_headings)
            else len(body_text)
        )
        section_end = body_start + next_position

        if section_start < section_end:
            sections.append(_Section(tuple(stack), heading.text, section_start, section_end))

    return tuple(sections) if sections else (_Section((), None, body_start, body_end_absolute),)


def _derive_section_start(body_text: str, heading_end_position: int) -> int:
    """Derive section body start from the exact heading match boundary.

    For line headings, consume one physical line break (CRLF/LF/CR) and
    leading indentation on the next line. For inline headings, consume
    only horizontal spacing and never skip non-whitespace body tokens.
    """
    if heading_end_position >= len(body_text):
        return heading_end_position

    cursor = heading_end_position

    if body_text[cursor] == "\r":
        cursor += 1
        if cursor < len(body_text) and body_text[cursor] == "\n":
            cursor += 1
        while cursor < len(body_text) and body_text[cursor] in " \t":
            cursor += 1
        return cursor

    if body_text[cursor] == "\n":
        cursor += 1
        while cursor < len(body_text) and body_text[cursor] in " \t":
            cursor += 1
        return cursor

    while cursor < len(body_text) and body_text[cursor] in " \t":
        cursor += 1
    return cursor


# ============================================================================
# Windowing (orchestration -> generation -> boundary selection -> overlap)
# ============================================================================


def _window_section(
    section: _Section,
    standardized_content: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
    first_window_max_tokens: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Produce ordered, boundary-aligned windows covering one section.

    Thin orchestrator: delegates the token-budget walk to
    ``_generate_windows``, which in turn uses ``_find_best_boundary`` for
    boundary selection and ``_apply_overlap`` for computing each window's
    successor start. ``first_window_max_tokens``, when given, applies only
    to this section's very first window (used to reserve room for the
    metadata header on the document's first chunk).
    """
    return _generate_windows(
        text=standardized_content,
        start=section.body_start,
        end=section.body_end,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        token_counter=token_counter,
        first_window_max_tokens=first_window_max_tokens,
    )


def _generate_windows(
    *,
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
    first_window_max_tokens: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Walk ``text[start:end]`` into token-bounded, boundary-aligned windows."""
    windows: list[tuple[int, int]] = []
    cursor = start
    is_first_iteration = True

    while cursor < end:
        budget = (
            first_window_max_tokens
            if is_first_iteration and first_window_max_tokens is not None
            else max_tokens
        )
        ideal_cut = _advance_by_token_budget(text, cursor, end, budget, token_counter)
        cut = _find_best_boundary(text, cursor, ideal_cut, end)
        windows.append((cursor, cut))

        if cut >= end:
            break

        cursor = _apply_overlap(text, cursor, cut, overlap_tokens, token_counter)
        is_first_iteration = False

    return tuple(windows)


def _advance_by_token_budget(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> int:
    """Find the furthest position <= end whose span from start fits max_tokens.

    Deterministic binary search over character position using
    ``token_counter``. Assumes a monotonic counter (more characters never
    yields fewer tokens), true of any standard tokenizer.
    """
    if token_counter(text[start:end]) <= max_tokens:
        return end

    low, high = start, end
    while low < high:
        mid = (low + high + 1) // 2
        if token_counter(text[start:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    return max(low, start + 1)


def _find_best_boundary(
    text: str,
    window_start: int,
    ideal_cut: int,
    section_end: int,
) -> int:
    """Choose the actual window end at or before ``ideal_cut``.

    Preference order: paragraph break, sentence break, whitespace, hard
    cut. Heading boundaries are not a separate tier here: a heading's own
    line is already excluded from every section's body range by
    ``_build_section_hierarchy``, so there is never heading text inside a
    single section's windowed span to protect against splitting -- that
    requirement is satisfied structurally, upstream of this function.
    Never returns a position before ``window_start + 1`` (guarantees
    forward progress) or after ``section_end``.
    """
    if ideal_cut >= section_end:
        return section_end

    search_region = text[window_start:ideal_cut]

    paragraph_break = search_region.rfind("\n\n")
    if paragraph_break > 0:
        return window_start + paragraph_break + 2

    sentence_break = max(search_region.rfind(". "), search_region.rfind(".\n"))
    if sentence_break > 0:
        return window_start + sentence_break + 2

    whitespace_break = search_region.rfind(" ")
    if whitespace_break > 0:
        return window_start + whitespace_break + 1

    return max(ideal_cut, window_start + 1)


def _apply_overlap(
    text: str,
    window_start: int,
    window_end: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
) -> int:
    """Compute the next window's start, stepping back by overlap_tokens.

    Snaps the overlap start forward to the nearest whitespace so the next
    window doesn't begin mid-word. Never returns a position at or before
    ``window_start`` (guarantees forward progress across windows).
    """
    if overlap_tokens <= 0:
        return window_end

    low, high = window_start, window_end
    while low < high:
        mid = (low + high) // 2
        if token_counter(text[mid:window_end]) <= overlap_tokens:
            high = mid
        else:
            low = mid + 1

    overlap_start = low
    next_space = text.find(" ", overlap_start, window_end)
    if next_space != -1:
        overlap_start = next_space + 1

    return max(overlap_start, window_start + 1)


# ============================================================================
# Metadata Header Budget & Chunk Text Composition
# ============================================================================


def _reserve_header_budget(
    max_chunk_tokens: int,
    header_region: str,
    token_counter: Callable[[str], int],
) -> int:
    """Return the token budget available for the document's first window.

    Ensures header + first chunk's body together satisfy
    ``max_chunk_tokens`` by construction (the budget is reduced before
    windowing runs), rather than by trimming after the fact.
    """

    header_tokens = token_counter(header_region)
    remaining = max_chunk_tokens - header_tokens

    if remaining <= 0:
        raise PermanentChunkingError(
            "The metadata header alone consumes the entire max_chunk_tokens "
            f"budget ({header_tokens} tokens >= {max_chunk_tokens}); increase "
            "max_chunk_tokens or shorten the configured metadata header."
        )

    return remaining


def _build_chunk_text(
    standardized_content: str,
    window_start: int,
    window_end: int,
    *,
    header_region: str | None,
) -> str:
    """Compose one chunk's text.

    Per ADR-009: an exact, contiguous body substring, optionally prefixed
    with the document's metadata header verbatim (``header_region`` is
    only ever non-None for the document's first chunk). Never
    paraphrased, summarized, or otherwise reworded.
    """
    body_text = standardized_content[window_start:window_end]
    if header_region is None:
        return body_text
    return f"{header_region}{body_text}"


# ============================================================================
# Offset & Page Translation
# ============================================================================


def _translate_to_source(
    window_start: int,
    window_end: int,
    standardized_offset_mappings: StandardizedOffsetMappings,
) -> SourceSpan:
    """Derive a chunk's (source_start, source_end) from Stage 6 provenance.

    Each chunk boundary is projected through the contiguous standardized
    offset table so a chunk that lands inside a single coarse body mapping
    still receives a distinct source span. Never scans ParsedDocument.raw_text
    directly and never recomputes provenance outside
    ``standardized_offset_mappings``.
    """
    source_start = _map_standardized_boundary_to_source(
        window_start,
        standardized_offset_mappings,
    )
    source_end = _map_standardized_boundary_to_source(
        window_end,
        standardized_offset_mappings,
    )

    if source_end < source_start:
        raise PermanentChunkingError(
            "Mapped source span is inverted for standardized content "
            f"range [{window_start}, {window_end})"
        )

    return (source_start, source_end)


def _map_standardized_boundary_to_source(
    standardized_position: int,
    standardized_offset_mappings: StandardizedOffsetMappings,
) -> int:
    """Project one standardized boundary position back into source space.

    The mapping table is contiguous by construction. For a boundary that
    falls inside a mapped span, we interpolate proportionally within that
    span; for a boundary that lands exactly on a span edge, we return the
    corresponding source edge.
    """
    for std_start, std_end, source_start, source_end in standardized_offset_mappings:
        if standardized_position < std_start:
            break

        if standardized_position > std_end:
            continue

        if (source_start, source_end) == _NO_SOURCE_PROVENANCE:
            if standardized_position == std_end:
                continue
            raise PermanentChunkingError(
                "Chunk boundary intersects the metadata header region, "
                "which has no source provenance"
            )

        if standardized_position == std_end:
            return source_end

        standardized_width = std_end - std_start
        source_width = source_end - source_start

        if standardized_width <= 0:
            raise PermanentChunkingError(
                "Encountered an empty standardized provenance span while "
                "mapping a chunk boundary"
            )

        if source_width <= 0:
            return source_start

        relative_position = standardized_position - std_start
        mapped_offset = round(relative_position * source_width / standardized_width)
        mapped_offset = max(0, min(mapped_offset, source_width))

        return source_start + mapped_offset

    raise PermanentChunkingError(
        "Standardized boundary is not covered by any provenance mapping"
    )


def _translate_to_pages(
    source_span: SourceSpan,
    page_offsets: Sequence[tuple[int, int]],
) -> tuple[PageNumber, PageNumber]:
    """Resolve a source_span into (page_start, page_end).

    ``parsed_document.page_offsets`` carries per-page source spans only;
    page number is derived as (offset entry index + 1).
    """
    source_start, source_end = source_span
    last_char = max(source_start, source_end - 1)
    return (_resolve_page(page_offsets, source_start), _resolve_page(page_offsets, last_char))


def _resolve_page(
    page_offsets: Sequence[tuple[int, int]],
    char_offset: int,
) -> PageNumber:
    """Binary search page_offsets for the page number containing char_offset."""
    # Stage 2 guarantees page_offsets are sorted and contiguous, so a binary
    # search over the page spans is valid here.
    low, high = 0, len(page_offsets) - 1

    while low <= high:
        mid = (low + high) // 2
        page_start, page_end = page_offsets[mid]
        if char_offset < page_start:
            high = mid - 1
        elif char_offset >= page_end:
            low = mid + 1
        else:
            return mid + 1

    raise PermanentChunkingError(
        f"char_offset {char_offset} is not covered by any entry in "
        "parsed_document.page_offsets"
    )


# ============================================================================
# Stable Chunk Identity
# ============================================================================


def _build_chunk_id(
    document_id: DocumentID,
    chunker_version: str,
    source_span: SourceSpan,
    *,
    hash_builder: Callable[[str], str],
) -> ChunkID:
    """Deterministic chunk_id from exactly document_id, chunker_version,
    and source_span -- no chunk text, no ordinal position alone, no
    timestamp, no randomness, per the frozen Stable Chunk IDs invariant.
    """
    source_start, source_end = source_span
    canonical = f"{document_id}\x00{chunker_version}\x00{source_start}\x00{source_end}"
    digest = hash_builder(canonical)[:16]
    return f"{document_id}:{digest}"


def _default_chunk_id_hasher(payload: str) -> str:
    """Default chunk-id digest implementation (SHA-256, injectable)."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Chunk Assembly
# ============================================================================


def _build_retrieval_chunks(
    *,
    sections: tuple[_Section, ...],
    standardized_content: str,
    standardized_offset_mappings: StandardizedOffsetMappings,
    header_region: str,
    document_id: DocumentID,
    chunker_version: str,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
    token_counter: Callable[[str], int],
    chunk_id_hasher: Callable[[str], str],
    page_offsets: Sequence[tuple[int, int]],
) -> tuple[RetrievalChunk, ...]:
    """Window every section in document order and assemble immutable chunks.

    Only the very first window of the very first section reserves budget
    for, and is prefixed with, the metadata header (ADR-009) -- every
    other window across every section uses the full ``max_chunk_tokens``
    and carries no header text.
    """
    header_budget = _reserve_header_budget(max_chunk_tokens, header_region, token_counter)

    chunks: list[RetrievalChunk] = []
    is_first_window_of_document = True

    for section_index, section in enumerate(sections):
        windows = _window_section(
            section,
            standardized_content,
            max_tokens=max_chunk_tokens,
            overlap_tokens=chunk_overlap_tokens,
            token_counter=token_counter,
            first_window_max_tokens=header_budget if section_index == 0 else None,
        )

        for window_start, window_end in windows:
            chunk_text = _build_chunk_text(
                standardized_content,
                window_start,
                window_end,
                header_region=header_region if is_first_window_of_document else None,
            )
            source_span = _translate_to_source(
                window_start, window_end, standardized_offset_mappings
            )
            page_start, page_end = _translate_to_pages(source_span, page_offsets)
            chunk_id = _build_chunk_id(
                document_id,
                chunker_version,
                source_span,
                hash_builder=chunk_id_hasher,
            )

            chunks.append(
                RetrievalChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    page_start=page_start,
                    page_end=page_end,
                    source_span=source_span,
                    section_hierarchy=section.section_hierarchy,
                    heading=section.heading,
                    chunk_text=chunk_text,
                )
            )
            is_first_window_of_document = False

    return tuple(chunks)


# ============================================================================
# Output Validation
# ============================================================================


def _validate_output(
    chunks: tuple[RetrievalChunk, ...],
    *,
    header_region: str,
    document_id: DocumentID,
) -> None:
    """Check structural invariants of a freshly assembled chunk sequence.

    Raises:
        PermanentChunkingError: If the sequence is empty, chunk_ids are
            not unique, ordering does not match document order, or the
            metadata header is missing from the first chunk or present in
            any other.
    """
    if not chunks:
        raise PermanentChunkingError(
            f"Chunking produced zero RetrievalChunks for document_id {document_id!r}"
        )

    seen_ids: set[str] = set()
    previous_source_start = -1

    for index, chunk in enumerate(chunks):
        if chunk.chunk_id in seen_ids:
            raise PermanentChunkingError(
                f"Duplicate chunk_id {chunk.chunk_id!r} for document_id {document_id!r}"
            )
        seen_ids.add(chunk.chunk_id)

        source_start, _ = chunk.source_span
        if source_start < previous_source_start:
            raise PermanentChunkingError(
                "RetrievalChunks are not ordered by document position for "
                f"document_id {document_id!r} at index {index}"
            )
        previous_source_start = source_start

        starts_with_header = chunk.chunk_text.startswith(header_region)
        if index == 0 and not starts_with_header:
            raise PermanentChunkingError(
                "The first RetrievalChunk does not carry the metadata "
                f"header for document_id {document_id!r}"
            )
        if index != 0 and starts_with_header:
            raise PermanentChunkingError(
                "The metadata header is repeated outside the first "
                f"RetrievalChunk for document_id {document_id!r}"
            )