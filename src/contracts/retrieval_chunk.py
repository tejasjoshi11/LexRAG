"""Immutable retrieval chunk contract."""

from dataclasses import dataclass
from typing import TypeAlias

from src.shared.types import (
    ChunkID,
    DocumentID,
    PageNumber,
)

# ============================================================================
# Contract-specific Type Aliases
# ============================================================================

SourceSpan: TypeAlias = tuple[
    int,
    int,
]

SectionHierarchy: TypeAlias = tuple[
    str,
    ...,
]

# ============================================================================
# Retrieval Chunk
# ============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RetrievalChunk:
    """Immutable structurally contextualized retrieval unit."""

    chunk_id: ChunkID
    document_id: DocumentID
    page_start: PageNumber
    page_end: PageNumber
    source_span: SourceSpan
    section_hierarchy: SectionHierarchy
    heading: str | None
    chunk_text: str