"""Qdrant Cloud index provider.

This module implements the concrete :class:`VectorIndexProvider` that writes
embedded chunks as vectors into a Qdrant Cloud collection.

Responsibilities are intentionally narrow:

- connect to Qdrant Cloud
- create the target collection if it does not exist
- convert :class:`EmbeddedChunk` instances into Qdrant point structures
- upsert all points in a single request

No retrieval, search, filtering, or reranking is performed here.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
import logging

from src.contracts.embedded_chunk import EmbeddedChunk
from src.vector_indexing.provider import VectorIndexProvider
from src.shared.config import qdrant_api_key, qdrant_url
from src.shared.constants import DATASTORE_NAMESPACE
from src.shared.exceptions import IndexingError, RetrievalError
from uuid import NAMESPACE_URL, uuid5

_LOGGER = logging.getLogger(__name__)

class QdrantProvider(VectorIndexProvider):
    """Index provider that writes vectors into Qdrant Cloud."""

    _COLLECTION_NAME: str = DATASTORE_NAMESPACE

    def __init__(self) -> None:
        """Initialize the Qdrant Cloud client."""
        try:
            self._client: QdrantClient = QdrantClient(
                url=qdrant_url(),
                api_key=qdrant_api_key(),
                timeout=120,
            )
        except Exception as exc:
            _LOGGER.exception("Failed to initialize the Qdrant client.")
            raise IndexingError(
                "The vector search service could not be initialized."
            ) from exc

        _LOGGER.info("Initialized Qdrant Cloud client.")

    @property
    def provider_name(self) -> str:
        """Return the index provider name."""
        return "qdrant"

    def index_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """Index embedded chunks into Qdrant Cloud.

        Args:
            chunks: The embedded chunks to index.

        Raises:
            ValueError: If any chunk is missing an embedding, if the
                embeddings do not all share the same dimension, or if an
                embedding's length disagrees with its declared dimension.
        """
        if not chunks:
            return

        self._validate_chunks(chunks)
        self._ensure_collection(len(chunks[0].embedding))

        points = self._build_points(chunks)

        try:
            self._client.upsert(
                collection_name=self._COLLECTION_NAME,
                points=points,
            )
        except Exception as exc:
            _LOGGER.exception("Failed to index chunks into Qdrant.")
            raise IndexingError(
                "Failed to index documents into the vector search service."
            ) from exc

        _LOGGER.info(f"Indexed {len(points)} chunks into Qdrant.")

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ):
        """Return the top-k semantic search results."""

        try:
            return self._client.query_points(
                collection_name=self._COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            ).points
        except Exception as exc:
            _LOGGER.exception("Qdrant semantic search failed.")
            raise RetrievalError(
                "Unable to retrieve relevant documents."
            ) from exc

    def _validate_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """Validate that every chunk has a consistent embedding.

        Each embedding must be present, all embeddings must share the same
        length, and each embedding's length must agree with the chunk's
        declared ``embedding_dimension``.

        Args:
            chunks: The embedded chunks to validate.

        Raises:
            ValueError: If a chunk has no embedding, if embeddings differ in
                length, or if an embedding length disagrees with its declared
                dimension.
        """
        expected_dimension: int | None = None

        for chunk in chunks:
            embedding = chunk.embedding

            if embedding is None or len(embedding) == 0:
                raise ValueError(
                    f"Chunk {chunk.chunk_id!r} has no embedding."
                )

            dimension = len(embedding)

            if dimension != chunk.embedding_dimension:
                raise ValueError(
                    f"Chunk {chunk.chunk_id!r} embedding length {dimension} "
                    f"disagrees with declared dimension "
                    f"{chunk.embedding_dimension}."
                )

            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise ValueError(
                    "All embeddings must have the same dimension. "
                    f"Expected {expected_dimension}, got {dimension} "
                    f"for chunk {chunk.chunk_id!r}."
                )

    def _ensure_collection(
        self,
        vector_size: int,
    ) -> None:
        """Create the Qdrant collection if it does not already exist.

        Collection creation is idempotent.

        Args:
            vector_size: The dimension of the vectors to store.
        """
        try:
            if self._client.collection_exists(self._COLLECTION_NAME):
                return

            self._client.create_collection(
                collection_name=self._COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as exc:
            _LOGGER.exception("Failed to ensure Qdrant collection.")
            raise IndexingError(
                "The vector search service collection could not be created."
            ) from exc

    def _build_points(
        self,
        chunks: list[EmbeddedChunk],
    ) -> list[PointStruct]:
        """Convert embedded chunks into Qdrant point structures.

        Args:
            chunks: The embedded chunks to convert.

        Returns:
            The list of point structures ready for upsert.
        """
        return [self._to_point(chunk) for chunk in chunks]

    def _to_point(
        self,
        chunk: EmbeddedChunk,
    ) -> PointStruct:
        """Convert a single embedded chunk into a Qdrant point structure.

        The chunk id is used as the point id, the embedding is stored as the
        vector, and the payload contains only non-vector metadata.

        Args:
            chunk: The embedded chunk to convert.

        Returns:
            The Qdrant point structure.
        """
        return PointStruct(
            id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
            vector=list(chunk.embedding),
            payload={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "embedding_model": chunk.embedding_model,
                "embedding_model_version": chunk.embedding_model_version,
                "normalized": chunk.normalized,
            },
        )