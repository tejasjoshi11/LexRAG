"""Stage 5 of the LexRAG ingestion pipeline: semantic metadata extraction.

This module implements the semantic metadata extractor, which combines one
immutable :class:`~src.contracts.cleaned_document.CleanedDocument` and its
matching :class:`~src.contracts.catalog_metadata.CatalogMetadata` into one
immutable :class:`~src.contracts.enriched_document.EnrichedDocument`.

Extraction is purely rule-based and deterministic: court, judges,
jurisdiction, statutes, legal entities, keywords, semantic relationships,
and legal domain are all derived from ``cleaned_document.cleaned_text`` and
``catalog_metadata.category`` using fixed patterns and lexicons. No
randomness, network access, or external model is involved, so identical
inputs always produce an identical ``EnrichedDocument``.

The stage never rewrites, cleans, normalizes, or chunks text, never
generates embeddings, never answers questions, never performs retrieval,
and never mutates ``ParsedDocument`` or any other upstream contract.
"""

from collections.abc import Callable, Iterable
import logging
import re

from src.contracts.catalog_metadata import CatalogMetadata
from src.contracts.cleaned_document import CleanedDocument
from src.contracts.enriched_document import (
    EnrichedDocument,
    RelationshipTriple,
    SemanticRelationships,
)
from src.shared.exceptions import SemanticExtractionError
from src.shared.types import Category, DocumentID
from src.shared.utils import normalize_whitespace
from dataclasses import dataclass

__all__ = [
    "extract_semantic_metadata",
    "PermanentSemanticExtractionError",
    "RetryableSemanticExtractionError",
]

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# Error Hierarchy
# ============================================================================
# Both subclasses remain instances of the existing SemanticExtractionError,
# so callers that only handle SemanticExtractionError continue to work.
# Callers that need retry semantics can distinguish the two subclasses.


class PermanentSemanticExtractionError(SemanticExtractionError):
    """Raised when extraction fails in a way that will recur on retry.

    The supplied CleanedDocument and CatalogMetadata are inconsistent, or
    the cleaned text contains no extractable content. Rerunning the stage
    with the same inputs will fail again; the underlying data must be
    corrected upstream first.
    """


class RetryableSemanticExtractionError(SemanticExtractionError):
    """Raised when extraction fails in a way that may not recur on retry.

    The stage may be safely rerun using the same CleanedDocument and
    CatalogMetadata, without repeating Stage 3 or Stage 4.
    """


# ============================================================================
# Deterministic Extraction Patterns and Lexicons
# ============================================================================
# cleaned_text has already had all whitespace (including newlines) collapsed
# by the cleaner, so it is one continuous, single-spaced string. Every
# capture group below is bounded to avoid unbounded or catastrophic matches.

_CIRCUIT_COURT_PATTERN = re.compile(
    r"UNITED STATES COURT OF APPEALS FOR THE\s+"
    r"([A-Z]+(?:\s+[A-Z]+){0,3})\s+CIRCUIT",
    re.IGNORECASE,
)

_DISTRICT_COURT_PATTERN = re.compile(
    r"UNITED STATES DISTRICT COURT FOR THE\s+"
    r"([A-Z][A-Z\s]+?)(?=\s+(?:DIVISION|CIVIL|CASE|NO\.|$))",
    re.IGNORECASE,
)

_FIXED_COURT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"UNITED STATES TAX COURT", re.IGNORECASE), "United States Tax Court"),
    (
        re.compile(r"SUPREME COURT OF THE UNITED STATES", re.IGNORECASE),
        "Supreme Court of the United States",
    ),
    (re.compile(r"BOARD OF TAX APPEALS", re.IGNORECASE), "Board of Tax Appeals"),
    (
        re.compile(r"UNITED STATES COURT OF FEDERAL CLAIMS", re.IGNORECASE),
        "United States Court of Federal Claims",
    ),
    (
        re.compile(r"UNITED STATES BANKRUPTCY COURT", re.IGNORECASE),
        "United States Bankruptcy Court",
    ),
)

_STATE_JURISDICTION_PATTERN = re.compile(
    r"STATE OF\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)",
)

_JUDGE_TITLE_PATTERN = re.compile(
    r"\b(?:Before\s+)?Judge\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?)\b",
)

_JUDGE_SIGNOFF_PATTERN = re.compile(
    r"\b([A-Z][A-Z]+),\s?(?:C\.?J\.|J\.)",
)

# A citation's number, e.g. "501(c)(3)" or "162(a)" or "1.61-1". Deliberately
# excludes bare trailing periods so a citation never swallows the sentence
# punctuation that follows it.
_CITATION_TAIL = r"\d+(?:\.\d+)*(?:\(\w+\))*(?:-\d+)*"

_STATUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\b\d{{1,4}}\s+U\.S\.C\.(?:A\.)?\s*§{{1,2}}\s*{_CITATION_TAIL}"),
    re.compile(
        rf"\b(?:I\.?R\.?C\.?|Internal Revenue Code)\s*§{{1,2}}\s*{_CITATION_TAIL}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bTreas\.?\s*Reg\.?\s*§{{1,2}}\s*{_CITATION_TAIL}",
        re.IGNORECASE,
    ),
)

_LEGAL_ENTITY_SUFFIX_PATTERN = re.compile(
    r"\b(?:[A-Z][\w&.,'-]*(?:\s+[A-Z][\w&.,'-]*){0,5})\s+"
    r"(?:Inc\.|Incorporated|L\.?L\.?C\.?|Corp\.|Corporation|Co\.|Company|"
    r"L\.?P\.|L\.?L\.?P\.|N\.A\.)"
)

_LEGAL_ENTITY_EXACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bInternal Revenue Service\b", re.IGNORECASE), "Internal Revenue Service"),
    (re.compile(r"\bIRS\b", re.IGNORECASE), "Internal Revenue Service"),
    (
        re.compile(r"\bCommissioner of Internal Revenue\b", re.IGNORECASE),
        "Commissioner of Internal Revenue",
    ),
    (re.compile(r"\bCommissioner\b", re.IGNORECASE), "Commissioner"),
    (
        re.compile(r"\bSecretary of the Treasury\b", re.IGNORECASE),
        "Secretary of the Treasury",
    ),
    (re.compile(r"\bTreasury(?: Department)?\b", re.IGNORECASE), "Treasury"),
    (re.compile(r"\bUnited States\b", re.IGNORECASE), "United States"),
    (re.compile(r"\bTax Court\b", re.IGNORECASE), "Tax Court"),
    (re.compile(r"\bPetitioners?\b", re.IGNORECASE), "Petitioner"),
    (re.compile(r"\bRespondents?\b", re.IGNORECASE), "Respondent"),
    (re.compile(r"\bTaxpayers?\b", re.IGNORECASE), "Taxpayer"),
)

_KEYWORD_LEXICON: tuple[str, ...] = (
    "income tax",
    "capital gains",
    "estate tax",
    "gift tax",
    "deduction",
    "exemption",
    "tax credit",
    "withholding",
    "deficiency",
    "penalty",
    "statute of limitations",
    "audit",
    "taxpayer",
    "petitioner",
    "respondent",
    "internal revenue service",
    "tax court",
    "appeal",
    "jurisdiction",
    "liability",
)

_CATEGORY_LEGAL_DOMAINS: dict[Category, str] = {
    "tax": "Tax Law",
    "judgments": "Case Law",
    "acts": "Statutory Law",
    "pov": "Points of View",
}


@dataclass(frozen=True, slots=True)
class ExtractionPolicy:
    extract_court: bool
    extract_judges: bool
    extract_jurisdiction: bool
    extract_statutes: bool
    extract_legal_entities: bool
    extract_keywords: bool
    derive_relationships: bool


_EXTRACTION_POLICIES: dict[Category, ExtractionPolicy] = {
    "acts": ExtractionPolicy(
        extract_court=False,
        extract_judges=False,
        extract_jurisdiction=False,
        extract_statutes=True,
        extract_legal_entities=True,
        extract_keywords=True,
        derive_relationships=True,
    ),
    "judgments": ExtractionPolicy(
        extract_court=True,
        extract_judges=True,
        extract_jurisdiction=True,
        extract_statutes=True,
        extract_legal_entities=True,
        extract_keywords=True,
        derive_relationships=True,
    ),
    "pov": ExtractionPolicy(
        extract_court=False,
        extract_judges=False,
        extract_jurisdiction=False,
        extract_statutes=True,
        extract_legal_entities=True,
        extract_keywords=True,
        derive_relationships=True,
    ),
    "tax": ExtractionPolicy(
        extract_court=False,
        extract_judges=False,
        extract_jurisdiction=False,
        extract_statutes=True,
        extract_legal_entities=True,
        extract_keywords=True,
        derive_relationships=True,
    )
}

# ============================================================================
# Public API
# ============================================================================


def extract_semantic_metadata(
    cleaned_document: CleanedDocument,
    catalog_metadata: CatalogMetadata,
    *,
    court_extractor: Callable[[str], str | None] | None = None,
    judge_extractor: Callable[[str], tuple[str, ...]] | None = None,
    jurisdiction_extractor: Callable[[str, str | None], str | None] | None = None,
    statute_extractor: Callable[[str], tuple[str, ...]] | None = None,
    legal_entity_extractor: Callable[[str], tuple[str, ...]] | None = None,
    keyword_extractor: Callable[[str], tuple[str, ...]] | None = None,
    legal_domain_classifier: Callable[[Category], str | None] | None = None,
    relationship_deriver: Callable[..., SemanticRelationships] | None = None,
) -> EnrichedDocument:
    """Extract deterministic semantic metadata for one cleaned document.

    Combines ``cleaned_document`` and its matching ``catalog_metadata`` into
    an immutable ``EnrichedDocument`` describing court, judges, jurisdiction,
    statutes, legal entities, keywords, semantic relationships, and legal
    domain. Extraction is purely rule-based; identical inputs always
    produce an identical result. Neither input is mutated.

    Each extraction step may be overridden by injecting an alternate
    callable, allowing extraction logic to be composed or replaced without
    modifying this module.

    Args:
        cleaned_document: The mechanically cleaned document to analyze.
        catalog_metadata: The catalog metadata describing the same document.
        court_extractor: Extracts the court name from cleaned text.
            Defaults to the built-in pattern-based extractor.
        judge_extractor: Extracts judge names from cleaned text. Defaults
            to the built-in pattern-based extractor.
        jurisdiction_extractor: Infers jurisdiction from cleaned text and
            the extracted court. Defaults to the built-in extractor.
        statute_extractor: Extracts statute and regulation citations.
            Defaults to the built-in pattern-based extractor.
        legal_entity_extractor: Extracts named legal entities. Defaults to
            the built-in pattern-based extractor.
        keyword_extractor: Extracts recognized domain keywords. Defaults to
            the built-in lexicon-based extractor.
        legal_domain_classifier: Maps a catalog category to a legal domain.
            Defaults to the built-in category mapping.
        relationship_deriver: Derives semantic relationship triples from
            the extracted fields. Defaults to the built-in deriver.

    Returns:
        EnrichedDocument: The immutable generated semantic metadata.

    Raises:
        PermanentSemanticExtractionError: If the supplied inputs are
            inconsistent or contain no extractable content. Retrying with
            the same inputs will fail again.
        RetryableSemanticExtractionError: If extraction fails unexpectedly.
            The stage may be safely rerun using the same CleanedDocument
            and CatalogMetadata, without repeating Stage 3 or Stage 4.
    """
    document_id = _validate_inputs(cleaned_document, catalog_metadata)

    court_extractor = court_extractor or _extract_court
    judge_extractor = judge_extractor or _extract_judges
    jurisdiction_extractor = jurisdiction_extractor or _extract_jurisdiction
    statute_extractor = statute_extractor or _extract_statutes
    legal_entity_extractor = legal_entity_extractor or _extract_legal_entities
    keyword_extractor = keyword_extractor or _extract_keywords
    legal_domain_classifier = legal_domain_classifier or _classify_legal_domain
    relationship_deriver = relationship_deriver or _derive_relationships

    text = cleaned_document.cleaned_text
    policy = _EXTRACTION_POLICIES[catalog_metadata.category]

    try:
        court = (
            court_extractor(text)
            if policy.extract_court
            else None
        )
        judges = (
            judge_extractor(text)
            if policy.extract_judges
            else tuple()
        )
        jurisdiction = (
            jurisdiction_extractor(text, court)
            if policy.extract_jurisdiction
            else None
        )
        statutes = (
            statute_extractor(text)
            if policy.extract_statutes
            else tuple()
        )
        legal_entities = (
            legal_entity_extractor(text)
            if policy.extract_legal_entities
            else tuple()
        )
        keywords = (
            keyword_extractor(text)
            if policy.extract_keywords
            else tuple()
        )
        legal_domain = legal_domain_classifier(
            catalog_metadata.category
        )
        semantic_relationships = (
            relationship_deriver(
                document_id=document_id,
                court=court,
                judges=judges,
                statutes=statutes,
                legal_entities=legal_entities,
            )
            if policy.derive_relationships
            else tuple()
        )
    except SemanticExtractionError:
        raise
    except Exception as exc:
        _LOGGER.exception(
            "Semantic metadata extraction failed for document_id %r",
            document_id,
        )
        raise RetryableSemanticExtractionError(
            "Semantic metadata extraction failed unexpectedly for "
            f"document_id {document_id!r}; the stage may "
            "be safely rerun with the same inputs."
        ) from exc

    return EnrichedDocument(
        parsed_document=cleaned_document.parsed_document,
        document_id=document_id,
        court=court,
        judges=judges,
        jurisdiction=jurisdiction,
        statutes=statutes,
        legal_entities=legal_entities,
        keywords=keywords,
        semantic_relationships=semantic_relationships,
        legal_domain=legal_domain,
    )


# ============================================================================
# Input Validation
# ============================================================================


def _validate_inputs(
    cleaned_document: CleanedDocument,
    catalog_metadata: CatalogMetadata,
) -> DocumentID:
    """Validate that both inputs describe the same, non-empty document.

    Returns:
        DocumentID: The validated canonical document identifier.

    Raises:
        PermanentSemanticExtractionError: If the inputs reference different
            documents or the cleaned text has no extractable content.
    """
    task_document_id = cleaned_document.parsed_document.document_id

    if task_document_id != catalog_metadata.document_id:
        raise PermanentSemanticExtractionError(
            "CleanedDocument and CatalogMetadata reference different "
            f"document_id values: {task_document_id!r} != "
            f"{catalog_metadata.document_id!r}"
        )

    if not cleaned_document.cleaned_text.strip():
        raise PermanentSemanticExtractionError(
            "Cleaned text is empty; semantic metadata cannot be extracted "
            f"for document_id {catalog_metadata.document_id!r}"
        )

    return task_document_id



# ============================================================================
# Field Extractors
# ============================================================================


def _extract_court(text: str) -> str | None:
    """Detect the first recognized court name mentioned in the text."""

    for pattern, canonical_name in _FIXED_COURT_PATTERNS:
        if pattern.search(text) is not None:
            return canonical_name

    circuit_match = _CIRCUIT_COURT_PATTERN.search(text)
    if circuit_match is not None:
        circuit_name = normalize_whitespace(circuit_match.group(1)).title()
        return f"United States Court of Appeals for the {circuit_name} Circuit"

    district_match = _DISTRICT_COURT_PATTERN.search(text)
    if district_match is not None:
        district_name = normalize_whitespace(district_match.group(1)).title()
        return f"United States District Court for the {district_name}"

    return None


def _extract_judges(text: str) -> tuple[str, ...]:
    """Extract unique judge names in first-encountered order."""

    matches: list[tuple[int, str]] = []

    for pattern in (_JUDGE_TITLE_PATTERN, _JUDGE_SIGNOFF_PATTERN):
        for match in pattern.finditer(text):
            name = normalize_whitespace(match.group(1)).title()
            matches.append((match.start(), name))

    matches.sort(key=lambda item: item[0])

    names = _unique_in_order(name for _, name in matches)

    return tuple(
        name
        for name in names
        if not any(
            name != other and name in other.split() for other in names
        )
    )

def _extract_jurisdiction(text: str, court: str | None) -> str | None:
    """Infer jurisdiction from the extracted court or explicit text mentions."""

    if court is not None and "united states" in court.lower():
        return "United States (Federal)"

    state_match = _STATE_JURISDICTION_PATTERN.search(text)
    if state_match is not None:
        state_name = normalize_whitespace(state_match.group(1)).title()
        return f"State of {state_name}"

    return None


def _extract_statutes(text: str) -> tuple[str, ...]:
    """Extract unique statute and regulation citations in first-seen order."""

    matches: list[tuple[int, str]] = []

    for pattern in _STATUTE_PATTERNS:
        for match in pattern.finditer(text):
            citation = normalize_whitespace(match.group(0))
            matches.append((match.start(), citation))

    matches.sort(key=lambda item: item[0])

    return _unique_in_order(citation for _, citation in matches)


def _extract_legal_entities(text: str) -> tuple[str, ...]:
    """Extract unique named legal entities in first-seen order."""

    matches: list[tuple[int, str]] = []

    for pattern, canonical_name in _LEGAL_ENTITY_EXACT_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), canonical_name))

    for match in _LEGAL_ENTITY_SUFFIX_PATTERN.finditer(text):
        matches.append((match.start(), normalize_whitespace(match.group(0))))

    matches.sort(key=lambda item: item[0])

    return _unique_in_order(entity for _, entity in matches)


def _extract_keywords(text: str) -> tuple[str, ...]:
    """Return lexicon keywords present in the text, in lexicon order."""

    lowered_text = text.lower()

    return tuple(
        keyword for keyword in _KEYWORD_LEXICON if keyword in lowered_text
    )


def _classify_legal_domain(category: Category) -> str | None:
    """Map a configured catalog category to a canonical legal domain."""

    return _CATEGORY_LEGAL_DOMAINS.get(category)


def _derive_relationships(
    *,
    document_id: DocumentID,
    court: str | None,
    judges: tuple[str, ...],
    statutes: tuple[str, ...],
    legal_entities: tuple[str, ...],
) -> SemanticRelationships:
    """Derive semantic relationship triples from already-extracted fields."""

    relationships: list[RelationshipTriple] = []

    if court is not None:
        relationships.append((document_id, "heard_in", court))

    relationships.extend(
        (document_id, "presided_by", judge) for judge in judges
    )
    relationships.extend(
        (document_id, "cites", statute) for statute in statutes
    )
    relationships.extend(
        (document_id, "mentions", entity) for entity in legal_entities
    )

    return tuple(relationships)


# ============================================================================
# Shared Helpers
# ============================================================================


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    """Return values in their given order with duplicates removed."""

    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)

    return tuple(ordered)