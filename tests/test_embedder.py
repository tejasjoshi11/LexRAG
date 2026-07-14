"""Manual test script for the Stage 8 embedder."""

from __future__ import annotations

import math

from src.contracts.retrieval_chunk import RetrievalChunk
from src.embeddings.provider import EmbeddingProvider
from src.ingestion_pipeline.embedder import embed_chunks


class FakeProvider(EmbeddingProvider):
    """Simple fake embedding provider for testing."""

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def model_version(self) -> str:
        return "v1"

    @property
    def embedding_dimension(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ]


def make_chunk(chunk_id: str) -> RetrievalChunk:
    """Create one test RetrievalChunk."""

    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        page_start=1,
        page_end=1,
        source_span=(0, 10),
        section_hierarchy=(),
        heading=None,
        chunk_text=f"Example text for {chunk_id}",
    )


def main() -> None:
    print("=" * 70)
    print("Stage 8 Embedder Test")
    print("=" * 70)

    provider = FakeProvider()

    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2"),
        make_chunk("chunk-3"),
    ]

    embedded_chunks = embed_chunks(chunks, provider)

    print(f"Input chunks      : {len(chunks)}")
    print(f"Output chunks     : {len(embedded_chunks)}")
    print(f"Embedding model   : {provider.model_name}")
    print(f"Model version     : {provider.model_version}")
    print(f"Embedding dim     : {provider.embedding_dimension}")
    print()

    assert len(embedded_chunks) == len(chunks)

    for retrieval_chunk, embedded_chunk in zip(
        chunks,
        embedded_chunks,
        strict=True,
    ):
        assert retrieval_chunk.chunk_id == embedded_chunk.chunk_id
        assert retrieval_chunk.document_id == embedded_chunk.document_id

        norm = math.sqrt(
            sum(value * value for value in embedded_chunk.embedding)
        )

        assert abs(norm - 1.0) < 1e-6
        assert embedded_chunk.normalized is True

        print(f"{embedded_chunk.chunk_id}")
        print(f"  Dimension : {len(embedded_chunk.embedding)}")
        print(f"  Norm      : {norm:.6f}")
        print(f"  Normalized: {embedded_chunk.normalized}")
        print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()