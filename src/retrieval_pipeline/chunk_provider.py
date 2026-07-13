"""Chunk provider abstraction."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.contracts.stored_chunk import StoredChunk
from src.shared.types import ChunkID


class ChunkProvider(ABC):
    """Abstract base class for storage backends providing chunk retrieval."""

    @abstractmethod
    def get_chunk(
        self,
        chunk_id: ChunkID,
    ) -> StoredChunk | None:
        """Return a chunk by its identifier.

        Args:
            chunk_id:
                Retrieval chunk identifier.

        Returns:
            The stored chunk if found, otherwise None.
        """
