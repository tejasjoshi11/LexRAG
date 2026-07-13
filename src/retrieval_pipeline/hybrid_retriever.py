"""Hybrid retriever orchestration.

This module orchestrates the LexRAG hybrid retrieval pipeline.

It coordinates semantic retrieval, keyword retrieval, and reciprocal rank
fusion while remaining independent of any concrete retrieval backend.

Responsibilities are intentionally narrow:

- invoke semantic retrieval
- invoke keyword retrieval
- fuse retrieval results
- return the final ranked retrieval results

No prompt construction, LLM invocation, registry lookup, or response
generation is performed here.
"""

from __future__ import annotations
import logging

from src.contracts.query import Query
from src.contracts.retrieved_chunk import RetrievedChunk
from src.retrieval_pipeline.keyword_retriever import KeywordRetriever
from src.retrieval_pipeline.semantic_retriever import SemanticRetriever
from src.retrieval_pipeline.reciprocal_rank_fusion import fuse
from src.retrieval_pipeline.retrieval_provider import RetrievalProvider

_LOGGER = logging.getLogger(__name__)


class HybridRetriever(RetrievalProvider):
    """Hybrid retrieval pipeline orchestrator."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_retriever: KeywordRetriever,
    ) -> None:
        """Initialize the hybrid retriever."""
        self._semantic_retriever = semantic_retriever
        self._keyword_retriever = keyword_retriever

    @property
    def provider_name(self) -> str:
        """Return the retrieval provider name."""
        return "hybrid"

    def retrieve(
        self,
        query: Query,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks using hybrid retrieval."""

        keyword_results = self._keyword_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        semantic_results = self._semantic_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        fused_results = fuse(
            ranked_lists=[
                keyword_results,
                semantic_results,
            ],
            top_k=top_k,
        )
        _LOGGER.info(f"Hybrid retrieval fused into {len(fused_results)} results.")

        return fused_results