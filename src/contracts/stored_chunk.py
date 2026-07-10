"""Stored chunk contract."""

from dataclasses import dataclass

from src.shared.types import (
    ChunkID,
    DocumentID,
    PageNumber,
)


@dataclass(
    frozen=True,
    slots=True,
)
class StoredChunk:
    """A chunk retrieved from the storage backend."""

    chunk_id: ChunkID

    document_id: DocumentID

    chunk_text: str

    heading: str | None

    page_start: PageNumber

    page_end: PageNumber
