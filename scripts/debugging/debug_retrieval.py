"""Manual integration test for the retrieval pipeline."""

from __future__ import annotations

from src.contracts.query import Query
from src.embeddings.sentence_transformer_provider import (
    SentenceTransformerProvider,
)
from src.vector_indexing.qdrant_provider import QdrantProvider
from src.keyword_indexing.elasticsearch_provider import (
    ElasticsearchProvider,
)
from src.metadata.document_registry import DocumentRegistry
from src.retrieval_pipeline.hybrid_retriever import HybridRetriever
from src.retrieval_pipeline.keyword_retriever import KeywordRetriever
from src.retrieval_pipeline.semantic_retriever import SemanticRetriever
from src.shared.constants import DEFAULT_TOP_K


from src.shared.config import (
    elasticsearch_url,
    elasticsearch_api_key,
)

print("Elasticsearch URL:", elasticsearch_url())
print("API Key Loaded:", bool(elasticsearch_api_key()))

def print_results(
    title: str,
    results,
) -> None:
    """Pretty-print retrieval results."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results.")
        return

    for index, chunk in enumerate(results, start=1):
        print(f"\n[{index}]")
        print(f"Title              : {chunk.title}")
        print(f"Source URL         : {chunk.source_url}")
        print(f"Document ID        : {chunk.document_id}")
        print(f"Chunk ID           : {chunk.chunk_id}")
        print(f"Pages              : {chunk.page_start}-{chunk.page_end}")
        print(f"Retrieval Method   : {chunk.retrieval_method}")
        print(f"Score              : {chunk.retrieval_score:.4f}")

        preview = chunk.chunk_text.replace("\n", " ")
        preview = preview[:250]

        print("Chunk Preview")
        print("-" * 80)
        print(preview)


def main() -> None:
    """Run retrieval integration tests."""

    embedding_provider = SentenceTransformerProvider()

    qdrant_provider = QdrantProvider()

    elasticsearch_provider = ElasticsearchProvider()
    document_registry = DocumentRegistry()
    print(f"Index Name: {elasticsearch_provider._INDEX_NAME}")
    count = elasticsearch_provider._client.count(
        index="lexrag",
    )

    print(f"Elasticsearch Document Count: {count['count']}")

    response = elasticsearch_provider._client.search(
        index="lexrag",
        size=10,
        query={
            "match_all": {}
        },
    )

    print("\n=== Elasticsearch Documents ===")

    for hit in response["hits"]["hits"]:
        print(f"_id: {hit['_id']}")
        print(hit["_source"])
        print("-" * 80)

    hits = elasticsearch_provider.search(
        query="judicial review",
        top_k=DEFAULT_TOP_K,
    )

    print(f"Elasticsearch Hits: {len(hits)}")

    keyword_retriever = KeywordRetriever(
        provider=elasticsearch_provider,
        chunk_provider=elasticsearch_provider,
        document_registry=document_registry,
    )

    query_embedding = embedding_provider.embed_query(
        "judicial review",
    )

    points = qdrant_provider.search(
        query_embedding=query_embedding,
        top_k=DEFAULT_TOP_K,
    )

    print(f"Qdrant Points: {len(points)}")

    semantic_retriever = SemanticRetriever(
        embedding_provider=embedding_provider,
        qdrant_provider=qdrant_provider,
        chunk_provider=elasticsearch_provider,
        document_registry=document_registry,
    )

    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
    )

    query = Query(
        query_text="What is judicial review?",
    )

    keyword_results = keyword_retriever.retrieve(
        query=query,
        top_k=DEFAULT_TOP_K,
    )

    semantic_results = semantic_retriever.retrieve(
        query=query,
        top_k=DEFAULT_TOP_K,
    )

    hybrid_results = hybrid_retriever.retrieve(
        query=query,
        top_k=DEFAULT_TOP_K,
    )

    print_results(
        "KEYWORD RETRIEVAL",
        keyword_results,
    )

    print_results(
        "SEMANTIC RETRIEVAL",
        semantic_results,
    )

    print_results(
        "HYBRID RETRIEVAL",
        hybrid_results,
    )

    print("\n")
    print("=" * 80)
    print("Retrieval pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()