"""Search result payload contract."""

from dataclasses import dataclass

from src.shared.types import (
    ChunkID,
    DocumentID,
)


@dataclass(
    frozen=True,
    slots=True,
)
class SearchResultPayload:
    """Parsed backend payload from a vector search result."""

    chunk_id: ChunkID

    document_id: DocumentID
