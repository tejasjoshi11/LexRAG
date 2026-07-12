"""Sentence Transformer embedding provider."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from src.embeddings.provider import EmbeddingProvider
from src.shared.config import embedding_model_name
from src.shared.exceptions import EmbeddingError

_LOGGER = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider backed by Sentence Transformers."""

    def __init__(self) -> None:
        """Load the embedding model."""
        self._model_name = embedding_model_name()

        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            _LOGGER.exception("Failed to load the embedding model.")
            raise EmbeddingError(
                "The embedding model could not be initialized."
            ) from exc

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        return self._model_name
    
    @property
    def model_version(self) -> str:
        """Return the embedding model version."""
        return self.model_name

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._model.get_embedding_dimension()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate one embedding per input document."""

        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=True,
            )

            return embeddings.tolist()
        except Exception as exc:
            _LOGGER.exception("Document embedding failed.")
            raise EmbeddingError(
                "Failed to generate document embeddings."
            ) from exc
    
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Generate an embedding for a single search query."""

        try:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

            return embedding.tolist()
        except Exception as exc:
            _LOGGER.exception("Query embedding failed.")
            raise EmbeddingError(
                "Failed to generate the query embedding."
            ) from exc