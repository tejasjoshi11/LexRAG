"""Keyword retriever backed by Elasticsearch BM25."""

from __future__ import annotations

from src.contracts.query import Query
from src.contracts.retrieved_chunk import RetrievalMethod, RetrievedChunk
from src.keyword_indexing.provider import KeywordIndexProvider
from src.metadata import DocumentRegistry
from src.retrieval_pipeline.chunk_provider import ChunkProvider
from src.retrieval_pipeline.retrieval_provider import RetrievalProvider
import logging

_LOGGER = logging.getLogger(__name__)


class KeywordRetriever(RetrievalProvider):
    """Keyword retriever backed by Elasticsearch."""

    def __init__(
        self,
        provider: KeywordIndexProvider,
        chunk_provider: ChunkProvider,
        document_registry: DocumentRegistry,
    ) -> None:
        """Initialize the keyword retriever."""
        self._provider = provider
        self._chunk_provider = chunk_provider
        self._document_registry = document_registry

    @property
    def provider_name(self) -> str:
        """Return the retrieval provider name."""
        return "keyword"

    def retrieve(
        self,
        query: Query,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant keyword search results."""

        hits = self._provider.search(
            query=query.query_text,
            top_k=top_k,
        )
        _LOGGER.info(f"Retrieved {len(hits)} hits from keyword provider.")

        retrieved_chunks: list[RetrievedChunk] = []

        for retrieval_rank, hit in enumerate(hits, start=1):
            chunk_id = hit["_id"]

            chunk = self._chunk_provider.get_chunk(
                chunk_id,
            )

            if chunk is None:
                continue

            registry_record = self._document_registry.get_registry_record(
                chunk.document_id,
            )

            if registry_record is None:
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=registry_record.title,
                    heading=chunk.heading,
                    source_url=registry_record.source_url,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_text=chunk.chunk_text,
                    retrieval_score=float(hit["_score"]),
                    retrieval_method=RetrievalMethod.KEYWORD,
                    retrieval_rank=retrieval_rank,
                )
            )

        return retrieved_chunks