"""Manual integration test for ElasticsearchProvider."""

from __future__ import annotations

from src.contracts.retrieval_chunk import RetrievalChunk
from src.keyword_indexing.elasticsearch_provider import (
    ElasticsearchProvider,
)
from src.keyword_indexing.indexer import index_chunks


def make_chunk(
    chunk_id: str,
    document_id: str,
) -> RetrievalChunk:
    """Create one test RetrievalChunk."""

    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page_start=1,
        page_end=1,
        source_span=(0, 100),
        section_hierarchy=("Test Section",),
        heading="Test Heading",
        chunk_text=(
            "This is a sample retrieval chunk used for "
            "testing Elasticsearch keyword indexing."
        ),
    )


def main() -> None:
    """Run the Elasticsearch integration test."""

    print("=" * 70)
    print("Elasticsearch Integration Test")
    print("=" * 70)

    provider = ElasticsearchProvider()

    chunks = [
        make_chunk("chunk-1", "doc-1"),
        make_chunk("chunk-2", "doc-1"),
        make_chunk("chunk-3", "doc-2"),
    ]

    index_chunks(
        retrieval_chunks=chunks,
        provider=provider,
    )

    print(f"Successfully indexed {len(chunks)} chunks.")
    print("Index: lexrag")
    print("Elasticsearch integration successful.")
    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()