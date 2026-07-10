"""Immutable embedded chunk contract."""

from dataclasses import dataclass
from typing import TypeAlias

from src.shared.types import (
    ChunkID,
    DocumentID,
)

# ============================================================================
# Contract-specific Type Aliases
# ============================================================================

EmbeddingVector: TypeAlias = tuple[
    float,
    ...,
]

# ============================================================================
# Embedded Chunk
# ============================================================================

@dataclass(
    frozen=True,
    slots=True,
)
class EmbeddedChunk:

    chunk_id: ChunkID

    document_id: DocumentID

    embedding: EmbeddingVector

    embedding_model: str

    embedding_model_version: str

    embedding_dimension: int

    normalized: bool
