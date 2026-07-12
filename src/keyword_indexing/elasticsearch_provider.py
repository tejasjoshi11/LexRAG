"""Elasticsearch Serverless keyword index provider.

This module implements the concrete :class:`KeywordIndexProvider` that
writes retrieval chunks as keyword-searchable documents into an
Elasticsearch Serverless index.

Responsibilities are intentionally narrow:

- connect to Elasticsearch Serverless
- create the target index if it does not exist
- create the index mapping exactly once, at index creation time
- convert :class:`RetrievalChunk` instances into Elasticsearch bulk actions
- bulk index all documents in a single request

No retrieval, search, filtering, or reranking is performed here.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import TypeAlias

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

from src.contracts.retrieval_chunk import RetrievalChunk
from src.contracts.stored_chunk import StoredChunk
from src.keyword_indexing.provider import KeywordIndexProvider
from src.shared.config import (
    elasticsearch_api_key,
    elasticsearch_url,
)
from src.shared.constants import DATASTORE_NAMESPACE
from src.shared.exceptions import IndexingError, RetrievalError

from src.shared.types import ChunkID

BulkAction: TypeAlias = dict[str, Any]

_LOGGER = logging.getLogger(__name__)

class ElasticsearchProvider(KeywordIndexProvider):
    """Keyword index provider that writes documents into Elasticsearch."""

    _INDEX_NAME: str = DATASTORE_NAMESPACE

    def __init__(self) -> None:
        """Initialize the Elasticsearch client."""
        try:
            self._client = Elasticsearch(
                elasticsearch_url(),
                api_key=elasticsearch_api_key(),
                request_timeout=60,
            )
        except Exception as exc:
            _LOGGER.exception("Failed to initialize the Elasticsearch client.")
            raise IndexingError(
                "The search service could not be initialized."
            ) from exc

        _LOGGER.info("Initialized Elasticsearch client.")

    @property
    def provider_name(self) -> str:
        """Return the keyword index provider name."""
        return "elasticsearch"

    def index_chunks(
        self,
        chunks: list[RetrievalChunk],
    ) -> None:
        """Index retrieval chunks into Elasticsearch."""

        if not chunks:
            return

        self._ensure_index()

        actions = self._build_actions(chunks)

        try:
            success_count, errors = bulk(
                self._client,
                actions,
                raise_on_error=False,
            )

            if errors:
                raise IndexingError(
                    f"Failed to index {len(errors)} Elasticsearch documents."
                )

            self._client.indices.refresh(
                index=self._INDEX_NAME,
            )
        except IndexingError:
            raise
        except Exception as exc:
            _LOGGER.exception("Elasticsearch bulk indexing failed.")
            raise IndexingError(
                "Failed to index documents into the search service."
            ) from exc

        _LOGGER.info(f"Indexed {success_count} chunks into Elasticsearch.")
        
    def _ensure_index(self) -> None:
        """Create the Elasticsearch index if it does not already exist."""
        try:
            if self._client.indices.exists(index=self._INDEX_NAME):
                return

            self._create_mapping()
        except Exception as exc:
            _LOGGER.exception("Failed to ensure Elasticsearch index.")
            raise IndexingError(
                "The search service index could not be created."
            ) from exc

    def _create_mapping(self) -> None:
        """Create the Elasticsearch index and mapping."""

        self._client.indices.create(
            index=self._INDEX_NAME,
            mappings={
                "dynamic": "strict",
                "properties": {
                    "chunk_id": {
                        "type": "keyword",
                    },
                    "document_id": {
                        "type": "keyword",
                    },
                    "heading": {
                        "type": "text",
                    },
                    "section_hierarchy": {
                        "type": "text",
                    },
                    "chunk_text": {
                        "type": "text",
                    },
                    "page_start": {
                        "type": "integer",
                    },
                    "page_end": {
                        "type": "integer",
                    },
                },
            },
        )

    def _build_actions(
        self,
        chunks: list[RetrievalChunk],
    ) -> list[BulkAction]:
        """Convert retrieval chunks into Elasticsearch bulk actions.

        Args:
            chunks: Retrieval chunks to convert.

        Returns:
            Bulk indexing actions.
        """
        return [
            self._to_document(chunk)
            for chunk in chunks
        ]

    def _to_document(
        self,
        chunk: RetrievalChunk,
    ) -> BulkAction:
        """Convert a retrieval chunk into an Elasticsearch bulk action.

        Args:
            chunk: Retrieval chunk to convert.

        Returns:
            Elasticsearch bulk indexing action.
        """
        return {
            "_index": self._INDEX_NAME,
            "_id": chunk.chunk_id,
            "_source": {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "heading": chunk.heading,
                "section_hierarchy": " > ".join(
                    chunk.section_hierarchy
                ),
                "chunk_text": chunk.chunk_text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            },
        }
    
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, object]]:
        """Search the keyword index using BM25.

        Args:
            query:
                User search query.

            top_k:
                Maximum number of results.

        Returns:
            Raw Elasticsearch search hits.
        """
        if top_k <= 0:
            return []

        try:
            response = self._client.search(
                index=self._INDEX_NAME,
                size=top_k,
                track_total_hits=False,
                
                query={
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "heading^3",
                            "section_hierarchy^2",
                            "chunk_text",
                        ],
                    },
                }
            )
            return response["hits"]["hits"]
        except Exception as exc:
            _LOGGER.exception("Elasticsearch keyword search failed.")
            raise RetrievalError(
                "The search service is currently unavailable."
            ) from exc
    
    def get_chunk(
        self,
        chunk_id: ChunkID,
    ) -> StoredChunk | None:
        """Return a retrieval chunk by its identifier.

        Args:
            chunk_id:
                Retrieval chunk identifier.

        Returns:
            Stored chunk if found, otherwise None.
        """
        try:
            response = self._client.get(
                index=self._INDEX_NAME,
                id=chunk_id,
            )
        except NotFoundError:
            return None
        except Exception as exc:
            _LOGGER.exception("Elasticsearch chunk retrieval failed.")
            raise RetrievalError(
                "The search service is currently unavailable."
            ) from exc

        source = response["_source"]
        return StoredChunk(
            chunk_id=source["chunk_id"],
            document_id=source["document_id"],
            chunk_text=source["chunk_text"],
            heading=source.get("heading"),
            page_start=source["page_start"],
            page_end=source["page_end"],
        )