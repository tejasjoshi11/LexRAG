"""Immutable standardized document contract."""

from dataclasses import dataclass
from typing import TypeAlias

from src.contracts.parsed_document import ParsedDocument
from src.shared.types import DocumentID

# ============================================================================
# Contract-specific Type Aliases
# ============================================================================

StandardizedOffsetMappings: TypeAlias = tuple[
    tuple[int, int, int, int],
    ...,
]

# ============================================================================
# Standardized Document
# ============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class StandardizedDocument:
    """Immutable standardized non-canonical document representation."""
    parsed_document: ParsedDocument
    document_id: DocumentID
    standardized_content: str
    standardized_offset_mappings: StandardizedOffsetMappings