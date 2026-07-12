"""Stage 9 of the LexRAG ingestion pipeline: indexing.

This module implements the indexer, the final stage of the ingestion
pipeline. It receives an ordered ``list[EmbeddedChunk]`` and forwards them
to an :class:`~src.vector_indexing.provider.VectorIndexProvider` for persistence into a
vector datastore (e.g., Qdrant).

The indexer is oblivious to the underlying datastore, credentials, network,
logging, or datastore logic, and it never imports or references any
concrete :class:`~src.vector_indexing.provider.VectorIndexProvider` implementation. It
depends only on the :class:`~src.vector_indexing.provider.VectorIndexProvider`
abstraction. Every :class:`~src.contracts.embedded_chunk.EmbeddedChunk` is
treated as immutable and is forwarded to the provider unchanged.
"""

from __future__ import annotations

from src.contracts.embedded_chunk import EmbeddedChunk
from src.vector_indexing.provider import VectorIndexProvider
import logging

__all__ = [
    "index_chunks",
]


# ============================================================================
# Public API
# ============================================================================

_LOGGER = logging.getLogger(__name__)


def index_chunks(
    embedded_chunks: list[EmbeddedChunk],
    provider: VectorIndexProvider,
) -> None:
    """Forward embedded chunks to an index provider for persistence.

    This is the final ingestion stage. It delegates all indexing work to
    ``provider.index_chunks`` exactly once, passing the input chunks through
    unchanged. No chunk is mutated.

    Args:
        embedded_chunks: The embedded chunks to index.
        provider: The index provider interface used to persist vectors.
            Only the interface is consumed; no concrete implementation is
            imported.
    """
    if not embedded_chunks:
        _LOGGER.info("No embedded chunks to index.")
        return

    _LOGGER.info(f"Indexing {len(embedded_chunks)} chunks via {provider.provider_name}...")
    provider.index_chunks(embedded_chunks)
    _LOGGER.info(f"Successfully indexed {len(embedded_chunks)} chunks.")