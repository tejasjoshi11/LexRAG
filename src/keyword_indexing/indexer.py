"""Stage 9 of the LexRAG ingestion pipeline: keyword indexing.

This module implements the keyword indexer, the final stage of the keyword
indexing pipeline. It receives an ordered ``list[RetrievalChunk]`` and
forwards them to a
:class:`~src.keyword_indexing.provider.KeywordIndexProvider` for persistence
into a keyword search index.

The stage is a thin orchestrator. It owns no validation, batching, retries,
logging, or datastore logic, and it never imports or references any
concrete :class:`~src.keyword_indexing.provider.KeywordIndexProvider`
implementation. It depends only on the
:class:`~src.keyword_indexing.provider.KeywordIndexProvider` abstraction.
Every :class:`~src.contracts.retrieval_chunk.RetrievalChunk` is treated as
immutable and is forwarded to the provider unchanged.
"""

from __future__ import annotations

from src.contracts.retrieval_chunk import RetrievalChunk
from src.keyword_indexing.provider import KeywordIndexProvider

__all__ = [
    "index_chunks",
]


# ============================================================================
# Public API
# ============================================================================


def index_chunks(
    retrieval_chunks: list[RetrievalChunk],
    provider: KeywordIndexProvider,
) -> None:
    """Forward retrieval chunks to a keyword index provider for persistence.

    This is the final keyword indexing stage. It delegates all indexing work
    to ``provider.index_chunks`` exactly once, passing the input chunks
    through unchanged. No chunk is mutated.

    Args:
        retrieval_chunks: The retrieval chunks to index.
        provider: The keyword index provider interface used to persist
            chunks. Only the interface is consumed; no concrete
            implementation is imported.
    """
    if not retrieval_chunks:
        return

    provider.index_chunks(retrieval_chunks)