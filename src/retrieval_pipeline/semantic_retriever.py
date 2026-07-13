"""Semantic retriever backed by Qdrant."""

from __future__ import annotations

from src.contracts.query import Query
from src.contracts.retrieved_chunk import RetrievalMethod, RetrievedChunk
from src.contracts.search_result_payload import SearchResultPayload
from src.embeddings.provider import EmbeddingProvider
from src.vector_indexing.qdrant_provider import QdrantProvider
import logging

_LOGGER = logging.getLogger(__name__)
from src.metadata import DocumentRegistry
from src.retrieval_pipeline.chunk_provider import ChunkProvider
from src.retrieval_pipeline.retrieval_provider import RetrievalProvider


class SemanticRetriever(RetrievalProvider):
    """Semantic retriever backed by Qdrant."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        qdrant_provider: QdrantProvider,
        chunk_provider: ChunkProvider,
        document_registry: DocumentRegistry,
    ) -> None:
        """Initialize the semantic retriever."""
        self._embedding_provider = embedding_provider
        self._qdrant_provider = qdrant_provider
        self._chunk_provider = chunk_provider
        self._document_registry = document_registry

    @property
    def provider_name(self) -> str:
        """Return the retrieval provider name."""
        return "semantic"

    def retrieve(
        self,
        query: Query,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant semantic search results."""

        query_embedding = self._embedding_provider.embed_query(
            query.query_text,
        )

        points = self._qdrant_provider.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        _LOGGER.info(f"Retrieved {len(points)} points from semantic provider.")

        retrieved_chunks: list[RetrievedChunk] = []

        for retrieval_rank, point in enumerate(points, start=1):
            payload = SearchResultPayload(
                chunk_id=point.payload["chunk_id"],
                document_id=point.payload["document_id"],
            )

            chunk = self._chunk_provider.get_chunk(
                payload.chunk_id,
            )

            if chunk is None:
                continue

            registry_record = self._document_registry.get_registry_record(
                payload.document_id,
            )

            if registry_record is None:
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=payload.chunk_id,
                    document_id=payload.document_id,
                    title=registry_record.title,
                    heading=chunk.heading,
                    source_url=registry_record.source_url,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_text=chunk.chunk_text,
                    retrieval_score=float(point.score),
                    retrieval_method=RetrievalMethod.SEMANTIC,
                    retrieval_rank=retrieval_rank,
                )
            )

        return retrieved_chunks