"""Project-wide immutable constants for LexRAG.

This module provides globally reusable constant values referenced across
the ingestion pipeline, query pipeline, and shared utilities. It is
dependency-free and safe to import from any layer of the application.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

PROJECT_NAME: Final[str] = "LexRAG"

# ---------------------------------------------------------------------------
# Default encodings and algorithms
# ---------------------------------------------------------------------------

DEFAULT_ENCODING: Final[str] = "utf-8"
DEFAULT_HASH_ALGORITHM: Final[str] = "sha256"

# ---------------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------------

PDF_EXTENSION: Final[str] = ".pdf"
CSV_EXTENSION: Final[str] = ".csv"
JSON_EXTENSION: Final[str] = ".json"

# ---------------------------------------------------------------------------
# Pipeline stage identifiers
# ---------------------------------------------------------------------------
# Concise, stable identifiers corresponding to ingestion module names.
# Used for logging, metrics, tracing, and internal pipeline references.

STAGE_DISCOVERY: Final[str] = "discovery"
STAGE_PARSER: Final[str] = "parser"
STAGE_CLEANER: Final[str] = "cleaner"
STAGE_CATALOG_METADATA_LOADER: Final[str] = "catalog_loader"
STAGE_SEMANTIC_METADATA_EXTRACTION: Final[str] = "semantic_metadata"
STAGE_KNOWLEDGE_STANDARDIZER: Final[str] = "knowledge_standardizer"
STAGE_CHUNKING: Final[str] = "chunker"
STAGE_EMBEDDING: Final[str] = "embedder"
STAGE_INDEXING: Final[str] = "indexer"
STAGE_REGISTRY: Final[str] = "registry"

# ---------------------------------------------------------------------------
# Processing status values
# ---------------------------------------------------------------------------

PROCESSING_STATUS_PENDING: Final[str] = "pending"
PROCESSING_STATUS_IN_PROGRESS: Final[str] = "in_progress"
PROCESSING_STATUS_COMPLETED: Final[str] = "completed"
PROCESSING_STATUS_FAILED: Final[str] = "failed"
PROCESSING_STATUS_SKIPPED: Final[str] = "skipped"

# ---------------------------------------------------------------------------
# Catalog metadata field names
# ---------------------------------------------------------------------------

DOCUMENT_ID_FIELD: Final[str] = "document_id"
TITLE_FIELD: Final[str] = "title"
YEAR_FIELD: Final[str] = "year"
CATEGORY_FIELD: Final[str] = "category"
SOURCE_URL_FIELD: Final[str] = "source_url"
FILENAME_FIELD: Final[str] = "filename"
SUMMARY_FIELD: Final[str] = "summary"

# ---------------------------------------------------------------------------
# Configuration paths
# ---------------------------------------------------------------------------

CONFIGS_DIRECTORY: Final[str] = "configs"

CONFIG_FILE: Final[str] = "config.yaml"
SUPPORTED_CATEGORIES_FILE: Final[str] = "supported_categories.yaml"
MODELS_CONFIG_FILE: Final[str] = "models.yaml"

# ---------------------------------------------------------------------------
# Timestamp format
# ---------------------------------------------------------------------------

ISO8601_UTC_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DATASTORE_NAMESPACE: Final[str] = "lexrag"
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0
DEFAULT_TOP_K: Final[int] = 5

__all__ = [
    "PROJECT_NAME",
    "DEFAULT_ENCODING",
    "DEFAULT_HASH_ALGORITHM",
    "PDF_EXTENSION",
    "CSV_EXTENSION",
    "JSON_EXTENSION",
    "STAGE_DISCOVERY",
    "STAGE_PARSER",
    "STAGE_CLEANER",
    "STAGE_CATALOG_METADATA_LOADER",
    "STAGE_SEMANTIC_METADATA_EXTRACTION",
    "STAGE_KNOWLEDGE_STANDARDIZER",
    "STAGE_CHUNKING",
    "STAGE_EMBEDDING",
    "STAGE_INDEXING",
    "STAGE_REGISTRY",
    "PROCESSING_STATUS_PENDING",
    "PROCESSING_STATUS_IN_PROGRESS",
    "PROCESSING_STATUS_COMPLETED",
    "PROCESSING_STATUS_FAILED",
    "PROCESSING_STATUS_SKIPPED",
    "DOCUMENT_ID_FIELD",
    "TITLE_FIELD",
    "CATEGORY_FIELD",
    "YEAR_FIELD",
    "FILENAME_FIELD",
    "SUMMARY_FIELD",
    "SOURCE_URL_FIELD",
    "CONFIGS_DIRECTORY",
    "CONFIG_FILE",
    "SUPPORTED_CATEGORIES_FILE",
    "MODELS_CONFIG_FILE",
    "ISO8601_UTC_TIMESTAMP_FORMAT",
    "DATASTORE_NAMESPACE",
    "DEFAULT_LLM_TEMPERATURE",
    "DEFAULT_TOP_K",
]