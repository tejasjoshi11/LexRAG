"""Shared exception hierarchy for the LexRAG project.

Defines the common exception hierarchy used across the ingestion pipeline,
query pipeline, API layer, evaluation, and future modules.

This module is dependency-free and safe to import from anywhere.
"""

__all__ = [
    "LexRAGError",
    "ConfigurationError",
    "ValidationError",
    "MetadataValidationError",
    "IngestionError",
    "DiscoveryError",
    "ParsingError",
    "CleaningError",
    "CatalogError",
    "StandardizationError",
    "SemanticExtractionError",
    "KnowledgeStandardizationError",
    "ChunkingError",
    "EmbeddingError",
    "IndexingError",
    "RegistryError",
    "QueryError",
    "RetrievalError",
    "RerankingError",
    "PromptBuilderError",
    "LLMError",
    "CitationError",
    "EvaluationError",
]


# ============================================================================
# Base Exceptions
# ============================================================================


class LexRAGError(Exception):
    """Base exception for all LexRAG-specific errors."""


# ============================================================================
# Configuration & Validation
# ============================================================================


class ConfigurationError(LexRAGError):
    """Raised when application configuration is invalid or missing."""


class ValidationError(LexRAGError):
    """Raised when data fails validation."""


class MetadataValidationError(ValidationError):
    """Raised when catalog metadata fails validation."""


# ============================================================================
# Ingestion Pipeline Exceptions
# ============================================================================


class IngestionError(LexRAGError):
    """Base exception for ingestion pipeline failures."""


class DiscoveryError(IngestionError):
    """Raised when document discovery or registry validation fails."""


class ParsingError(IngestionError):
    """Raised when PDF parsing fails."""


class CleaningError(IngestionError):
    """Raised when document cleaning fails."""


class CatalogError(IngestionError):
    """Raised when catalog metadata loading fails."""


class StandardizationError(IngestionError):
    """Raised when knowledge standardization fails."""


class SemanticExtractionError(IngestionError):
    """Raised when semantic metadata extraction fails."""


class KnowledgeStandardizationError(StandardizationError):
    """Raised when knowledge standardization fails."""


class ChunkingError(IngestionError):
    """Raised when structure-aware chunking fails."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


class IndexingError(IngestionError):
    """Raised when indexing into search backends fails."""


class RegistryError(IngestionError):
    """Raised when document registry operations fail."""


# ============================================================================
# Query Pipeline Exceptions
# ============================================================================


class QueryError(LexRAGError):
    """Base exception for query pipeline failures."""


class RetrievalError(QueryError):
    """Raised when document retrieval fails."""


class RerankingError(QueryError):
    """Raised when reranking retrieved results fails."""


class PromptBuilderError(QueryError):
    """Raised when prompt construction fails."""


class LLMError(QueryError):
    """Raised when an LLM provider operation fails."""


class CitationError(QueryError):
    """Raised when citation generation fails."""


# ============================================================================
# Evaluation
# ============================================================================


class EvaluationError(LexRAGError):
    """Raised when evaluation pipeline execution fails."""