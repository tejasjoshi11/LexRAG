"""Keyword index provider abstraction.

This module defines the abstract interface implemented by every keyword
index backend. Implementations are responsible for persisting retrieval
chunks into a keyword search engine such as Elasticsearch.

The interface intentionally exposes only the operations required by the
LexRAG ingestion pipeline.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.contracts.retrieval_chunk import RetrievalChunk
from src.contracts.stored_chunk import StoredChunk
from src.retrieval_pipeline.chunk_provider import ChunkProvider
from src.shared.types import ChunkID


class KeywordIndexProvider(ChunkProvider, ABC):
    """Abstract base class for keyword index providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def index_chunks(
        self,
        chunks: list[RetrievalChunk],
    ) -> None:
        """Persist retrieval chunks into the keyword index.

        Args:
            chunks: Retrieval chunks to persist.
        """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        """Search the keyword index.

        Args:
            query:
                User search query.

            top_k:
                Maximum number of results.

        Returns:
            Raw keyword search results.
        """

    @abstractmethod
    def get_chunk(
        self,
        chunk_id: ChunkID,
    ) -> StoredChunk | None:
        """Return a retrieval chunk by its identifier.

        Args:
            chunk_id:
                Retrieval chunk identifier.

        Returns:
            Stored chunk if found, otherwise None.
        """