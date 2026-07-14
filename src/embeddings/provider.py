"""Embedding provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for document embedding providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the embedding provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the immutable embedding model identifier."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the immutable embedding model version."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate one embedding for each input document."""

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single search query."""