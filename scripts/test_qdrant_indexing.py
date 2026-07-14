"""Manual integration test for QdrantProvider."""

from __future__ import annotations

from src.contracts.embedded_chunk import EmbeddedChunk
from src.vector_indexing.qdrant_provider import QdrantProvider


def make_chunk(
    chunk_id: str,
    document_id: str,
) -> EmbeddedChunk:
    """Create one test EmbeddedChunk."""

    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        embedding=(0.1, 0.2, 0.3),
        embedding_model="test-model",
        embedding_model_version="v1",
        embedding_dimension=3,
        normalized=False,
    )


def main() -> None:
    print("=" * 70)
    print("Qdrant Cloud Integration Test")
    print("=" * 70)

    provider = QdrantProvider()

    chunks = [
        make_chunk("chunk-1", "doc-1"),
        make_chunk("chunk-2", "doc-1"),
        make_chunk("chunk-3", "doc-2"),
    ]

    provider.index_chunks(chunks)

    print(f"Successfully indexed {len(chunks)} chunks.")
    print("Collection: lexrag")
    print("Qdrant Cloud integration successful.")
    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()