"""Immutable enriched document contract."""

from dataclasses import dataclass
from typing import TypeAlias

from src.contracts.parsed_document import ParsedDocument
from src.shared.types import DocumentID

# ============================================================================
# Contract-specific Type Aliases
# ============================================================================

RelationshipTriple: TypeAlias = tuple[
    str,
    str,
    str,
]

SemanticRelationships: TypeAlias = tuple[
    RelationshipTriple,
    ...,
]

# ============================================================================
# Enriched Document
# ============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class EnrichedDocument:
    """Immutable generated semantic metadata for a document."""
    
    parsed_document: ParsedDocument
    document_id: DocumentID
    court: str | None
    judges: tuple[str, ...]
    jurisdiction: str | None
    statutes: tuple[str, ...]
    legal_entities: tuple[str, ...]
    keywords: tuple[str, ...]
    semantic_relationships: SemanticRelationships
    legal_domain: str | None