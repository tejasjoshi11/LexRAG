"""Retrieval provider abstraction.

This module defines the abstract interface implemented by every retrieval
backend used by the LexRAG query pipeline.

Implementations may perform semantic retrieval, keyword retrieval, or any
future retrieval strategy, but all expose the same interface and return
retrieval results through immutable contracts.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.contracts.query import Query
from src.contracts.retrieved_chunk import RetrievedChunk


class RetrievalProvider(ABC):
    """Abstract base class for retrieval providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the retrieval provider name."""

    @abstractmethod
    def retrieve(
        self,
        query: Query,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query:
                Processed user query.

            top_k:
                Maximum number of retrieved chunks.

        Returns:
            Ranked retrieval results.
        """

        