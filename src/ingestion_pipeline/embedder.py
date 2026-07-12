"""Stage 8 of the LexRAG ingestion pipeline: embedding.

This module implements the embedder, which transforms an ordered
``list[RetrievalChunk]`` into an ordered ``list[EmbeddedChunk]`` by
generating one embedding vector per chunk via an
:class:`~src.embeddings.provider.EmbeddingProvider`, validating every
vector, and L2-normalizing it.

The stage is a pure deterministic transformation. It owns normalization
but never loads models, reads configuration, performs batching/retries,
persists vectors, or talks to any datastore. It depends only on the
:class:`~src.embeddings.provider.EmbeddingProvider` interface -- never on
any concrete implementation.
"""

from __future__ import annotations

import math

from src.contracts.embedded_chunk import EmbeddedChunk
from src.contracts.retrieval_chunk import RetrievalChunk
from src.embeddings.provider import EmbeddingProvider
import logging

__all__ = [
    "embed_chunks",
]


# ============================================================================
# Public API
# ============================================================================

_LOGGER = logging.getLogger(__name__)


def embed_chunks(
    retrieval_chunks: list[RetrievalChunk],
    provider: EmbeddingProvider,
) -> list[EmbeddedChunk]:
    """Embed retrieval chunks into L2-normalized embedded chunks.

    Extracts ``chunk_text`` from every input chunk, requests embeddings
    from ``provider`` exactly once, validates each returned vector, L2-
    normalizes it, and assembles one immutable
    :class:`~src.contracts.embedded_chunk.EmbeddedChunk` per input chunk.
    Input order is preserved exactly.

    Args:
        retrieval_chunks: Ordered retrieval chunks to embed.
        provider: The embedding provider interface used to generate
            vectors. Only the interface is consumed; no concrete
            implementation is imported.

    Returns:
        list[EmbeddedChunk]: One embedded chunk per retrieval chunk, in
        the same order as the input. Every embedding is L2-normalized and
        stored as ``tuple[float, ...]``.

    Raises:
        ValueError: If the provider returns the wrong number of
            embeddings, or any embedding is empty, has the wrong
            dimension, contains non-finite values, or has zero norm.
    """
    if not retrieval_chunks:
        return []

    _LOGGER.info(f"Embedding {len(retrieval_chunks)} chunks via {provider.provider_name}...")

    texts = [chunk.chunk_text for chunk in retrieval_chunks]

    embeddings = provider.embed_documents(texts)

    _validate_embedding_count(embeddings, expected=len(retrieval_chunks))

    model_name = provider.model_name
    model_version = provider.model_version
    embedding_dimension = provider.embedding_dimension

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk, raw_embedding in zip(retrieval_chunks, embeddings, strict=True):
        _validate_embedding(raw_embedding, expected_dimension=embedding_dimension)
        normalized_embedding = _l2_normalize(raw_embedding)

        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                embedding=normalized_embedding,
                embedding_model=model_name,
                embedding_model_version=model_version,
                embedding_dimension=embedding_dimension,
                normalized=True,
            )
        )

    _LOGGER.info(f"Successfully embedded {len(embedded_chunks)} chunks.")
    return embedded_chunks


# ============================================================================
# Validation
# ============================================================================


def _validate_embedding_count(
    embeddings: list[list[float]],
    *,
    expected: int,
) -> None:
    """Ensure the provider returned exactly one embedding per input chunk.

    Raises:
        ValueError: If the counts differ.
    """
    if len(embeddings) != expected:
        raise ValueError(
            "Embedding provider returned an unexpected number of embeddings: "
            f"expected {expected}, got {len(embeddings)}"
        )


def _validate_embedding(
    embedding: list[float],
    *,
    expected_dimension: int,
) -> None:
    """Validate a single raw embedding vector.

    Checks, in order: non-empty, correct dimension, all values finite,
    non-zero norm.

    Raises:
        ValueError: If any check fails.
    """
    if not embedding:
        raise ValueError("Embedding provider returned an empty embedding vector")

    if len(embedding) != expected_dimension:
        raise ValueError(
            "Embedding dimension mismatch: expected "
            f"{expected_dimension}, got {len(embedding)}"
        )

    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("Embedding vector contains non-finite values")

    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0.0:
        raise ValueError("Embedding vector has zero norm")


# ============================================================================
# Normalization
# ============================================================================


def _l2_normalize(embedding: list[float]) -> tuple[float, ...]:
    """Return the L2-normalized embedding as an immutable tuple.

    The caller guarantees ``embedding`` is non-empty, finite, and has a
    non-zero norm, so division here is always safe.
    """
    norm = math.sqrt(sum(value * value for value in embedding))
    return tuple(value / norm for value in embedding)