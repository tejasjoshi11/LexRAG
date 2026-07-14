"""Application composition root.

Constructs and wires all backend dependencies for the LexRAG pipeline.
Contains no business logic, no retrieval, no routing, and no Streamlit
code — only dependency composition.
"""

from __future__ import annotations

from src.citations.default_citation_formatter import (
    DefaultCitationFormatter,
)
from src.embeddings.sentence_transformer_provider import (
    SentenceTransformerProvider,
)
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_generator import ResponseGenerator
from src.keyword_indexing.elasticsearch_provider import (
    ElasticsearchProvider,
)
from src.llm.models import Model, SUPPORTED_MODELS
from src.llm.registry import ProviderRegistry
from src.metadata import DocumentRegistry
from src.rag_pipeline.pipeline import RAGPipeline
from src.retrieval_pipeline.hybrid_retriever import HybridRetriever
from src.retrieval_pipeline.keyword_retriever import KeywordRetriever
from src.retrieval_pipeline.semantic_retriever import SemanticRetriever
from src.routing.router import Router
from src.vector_indexing.qdrant_provider import QdrantProvider


def create_pipeline() -> RAGPipeline:
    """Construct and return a fully wired RAGPipeline."""

    document_registry = DocumentRegistry()
    elasticsearch_provider = ElasticsearchProvider()

    semantic_retriever = SemanticRetriever(
        embedding_provider=SentenceTransformerProvider(),
        qdrant_provider=QdrantProvider(),
        chunk_provider=elasticsearch_provider,
        document_registry=document_registry,
    )

    keyword_retriever = KeywordRetriever(
        provider=elasticsearch_provider,
        chunk_provider=elasticsearch_provider,
        document_registry=document_registry,
    )

    return RAGPipeline(
        router=Router(),
        retrieval_provider=HybridRetriever(
            semantic_retriever=semantic_retriever,
            keyword_retriever=keyword_retriever,
        ),
        prompt_builder=PromptBuilder(),
        response_generator=ResponseGenerator(),
        citation_formatter=DefaultCitationFormatter(),
    )


def get_implemented_models() -> tuple[Model, ...]:
    """Return models whose providers are implemented.

    Filters SUPPORTED_MODELS using ProviderRegistry.supports().
    Models whose providers are not yet implemented are excluded.
    """

    registry = ProviderRegistry()

    return tuple(
        model
        for model in SUPPORTED_MODELS
        if registry.supports(model)
    )
