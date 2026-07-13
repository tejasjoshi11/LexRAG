"""Reciprocal Rank Fusion.

This module implements Reciprocal Rank Fusion (RRF), a rank aggregation
algorithm used to combine multiple ranked retrieval result lists into a
single ranking.

The implementation is deterministic and independent of the underlying
retrieval backends.
"""

from __future__ import annotations

from src.contracts.retrieved_chunk import RetrievedChunk
from src.shared.types import ChunkID

_RRF_K: int = 60


def fuse(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked retrieval result lists.

    Args:
        ranked_lists:
            Ranked retrieval result lists.

        top_k:
            Maximum number of fused results.

    Returns:
        Final fused ranking.
    """
    if top_k <= 0:
        return []

    scores: dict[ChunkID, float] = {}
    chunks: dict[ChunkID, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            scores[chunk.chunk_id] = (
                scores.get(chunk.chunk_id, 0.0)
                + 1.0 / (_RRF_K + rank)
            )
            chunks[chunk.chunk_id] = chunk

    ranked_chunk_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        chunks[chunk_id]
        for chunk_id in ranked_chunk_ids[:top_k]
    ]