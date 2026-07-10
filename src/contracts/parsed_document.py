"""Immutable parsed document contract."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

from src.shared.types import DocumentID
from src.contracts.document_task import DocumentTask

# ============================================================================
# Contract-specific Type Aliases
# ============================================================================

MetadataValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | datetime
    | None
)

PageOffsets: TypeAlias = tuple[
    tuple[int, int],
    ...
]

IntrinsicMetadata: TypeAlias = tuple[
    tuple[str, MetadataValue],
    ...
]

# ============================================================================
# Parsed Document
# ============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class ParsedDocument:
    """Immutable canonical representation of one parsed PDF."""

    document_task: DocumentTask

    raw_text: str

    page_count: int

    page_offsets: PageOffsets

    intrinsic_metadata: IntrinsicMetadata

    @property
    def document_id(self) -> DocumentID:
        """Return the canonical document identifier."""
        return self.document_task.catalog_metadata.document_id