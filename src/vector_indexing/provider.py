"""Index provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.contracts.embedded_chunk import EmbeddedChunk


class VectorIndexProvider(ABC):
    """Abstract interface for vector index providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the index provider name."""

    @abstractmethod
    def index_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """Index embedded chunks."""