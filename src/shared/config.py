"""Centralized configuration loading for the LexRAG project.

This module provides a dependency-light, centralized interface for loading
application configuration from YAML files. Configuration is loaded lazily,
cached after the first read, and exposed through small helper functions.

This module contains no business logic and is safe to import throughout the
application.
"""

import os
import re

from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

from yaml.resolver import BaseResolver

from src.shared.constants import (
    CONFIGS_DIRECTORY,
    CONFIG_FILE,
    MODELS_CONFIG_FILE,
    SUPPORTED_CATEGORIES_FILE,
)
from src.shared.exceptions import ConfigurationError
from src.shared.types import Category, PathLike

load_dotenv()

_QDRANT_URL_ENV = "QDRANT_URL"
_QDRANT_API_KEY_ENV = "QDRANT_API_KEY"

_ELASTICSEARCH_URL_ENV = "ELASTICSEARCH_URL"
_ELASTICSEARCH_API_KEY_ENV = "ELASTICSEARCH_API_KEY"

_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

_GROQ_API_KEY_ENV = "GROQ_API_KEY"

_NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"

_CATEGORY_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[a-z]+$")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct a mapping node while rejecting duplicate keys."""

    mapping: dict[Any, Any] = {}

    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)

        try:
            if key in mapping:
                raise yaml.YAMLError(f"Duplicate YAML mapping key: {key!r}")
        except TypeError as exc:
            raise yaml.YAMLError("YAML mapping keys must be hashable") from exc

        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@cache
def project_root() -> Path:
    """Return the LexRAG project root directory."""
    return Path(__file__).resolve().parents[2]


@cache
def configs_directory() -> Path:
    """Return the validated configuration directory."""

    directory = project_root() / CONFIGS_DIRECTORY

    if not directory.exists():
        raise ConfigurationError(
            f"Configuration directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise ConfigurationError(
            f"Configuration path is not a directory: {directory}"
        )

    return directory


@cache
def load_yaml(path: PathLike) -> Mapping[str, Any]:
    """Load and validate a YAML configuration file."""

    file_path = Path(path)

    if not file_path.is_absolute():
        file_path = configs_directory() / file_path

    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = yaml.load(file, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"Failed to load configuration file: {file_path}"
        ) from exc

    if data is None:
        data = {}

    if not isinstance(data, Mapping):
        raise ConfigurationError(
            f"Configuration file must contain a YAML mapping: {file_path}"
        )

    return data


@cache
def load_config(filename: PathLike) -> Mapping[str, Any]:
    """Load a configuration file from the configs directory."""
    return load_yaml(filename)


@cache
def config() -> Mapping[str, Any]:
    """Return the main application configuration."""
    return load_config(CONFIG_FILE)


@cache
def supported_categories() -> Mapping[str, Any]:
    """Return the supported document categories configuration."""
    return load_config(SUPPORTED_CATEGORIES_FILE)


@cache
def models_config() -> Mapping[str, Any]:
    """Return the model configuration."""
    return load_config(MODELS_CONFIG_FILE)


@cache
def catalog_metadata_path() -> Path:
    """Return the configured catalog metadata path."""

    application_config = config()

    if "catalog_metadata" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'catalog_metadata'"
        )

    return _resolve_configured_path(application_config["catalog_metadata"])


@cache
def registry_storage_path() -> Path:
    """Return the configured Registry storage path."""

    application_config = config()

    if "registry_storage" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'registry_storage'"
        )

    return _resolve_configured_path(application_config["registry_storage"])


@cache
def chunk_size() -> int:
    """Return the configured maximum chunk size in tokens."""

    application_config = config()

    if "chunking" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'chunking'"
        )

    chunking = application_config["chunking"]

    if not isinstance(chunking, Mapping):
        raise ConfigurationError(
            "Application configuration field 'chunking' must be a mapping"
        )

    if "max_chunk_tokens" not in chunking:
        raise ConfigurationError(
            "Chunking configuration is missing 'max_chunk_tokens'"
        )

    value = chunking["max_chunk_tokens"]

    if not isinstance(value, int):
        raise ConfigurationError(
            "Chunking configuration field 'max_chunk_tokens' must be an integer"
        )

    if value <= 0:
        raise ConfigurationError(
            "Chunking configuration field 'max_chunk_tokens' must be greater than zero"
        )

    return value


@cache
def chunk_overlap() -> int:
    """Return the configured chunk overlap in tokens."""

    application_config = config()

    if "chunking" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'chunking'"
        )

    chunking = application_config["chunking"]

    if not isinstance(chunking, Mapping):
        raise ConfigurationError(
            "Application configuration field 'chunking' must be a mapping"
        )

    if "chunk_overlap_tokens" not in chunking:
        raise ConfigurationError(
            "Chunking configuration is missing 'chunk_overlap_tokens'"
        )

    value = chunking["chunk_overlap_tokens"]

    if not isinstance(value, int):
        raise ConfigurationError(
            "Chunking configuration field 'chunk_overlap_tokens' must be an integer"
        )

    if value < 0:
        raise ConfigurationError(
            "Chunking configuration field 'chunk_overlap_tokens' must not be negative"
        )

    if value >= chunk_size():
        raise ConfigurationError(
            "Chunk overlap must be smaller than the configured chunk size"
        )

    return value


@cache
def chunker_version() -> str:
    """Return the configured chunker version."""

    application_config = config()

    if "chunking" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'chunking'"
        )

    chunking = application_config["chunking"]

    if not isinstance(chunking, Mapping):
        raise ConfigurationError(
            "Application configuration field 'chunking' must be a mapping"
        )

    if "chunker_version" not in chunking:
        raise ConfigurationError(
            "Chunking configuration is missing 'chunker_version'"
        )

    value = chunking["chunker_version"]

    if not isinstance(value, str):
        raise ConfigurationError(
            "Chunking configuration field 'chunker_version' must be a string"
        )

    if not value or value.isspace() or value.strip() != value:
        raise ConfigurationError(
            "Chunking configuration field 'chunker_version' is invalid"
        )

    return value


def _pipeline_version(
    key: str,
) -> str:
    """Return one configured pipeline version."""

    application_config = config()

    if "pipeline" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'pipeline'"
        )

    pipeline = application_config["pipeline"]

    if not isinstance(pipeline, Mapping):
        raise ConfigurationError(
            "Application configuration field 'pipeline' must be a mapping"
        )

    if key not in pipeline:
        raise ConfigurationError(
            f"Pipeline configuration is missing '{key}'"
        )

    value = pipeline[key]

    if not isinstance(value, str):
        raise ConfigurationError(
            f"Pipeline configuration field '{key}' must be a string"
        )

    if not value or value.isspace() or value.strip() != value:
        raise ConfigurationError(
            f"Pipeline configuration field '{key}' is invalid"
        )

    return value


@cache
def parser_version() -> str:
    """Return the configured parser version."""
    return _pipeline_version("parser_version")


@cache
def cleaner_version() -> str:
    """Return the configured cleaner version."""
    return _pipeline_version("cleaner_version")


@cache
def semantic_version() -> str:
    """Return the configured semantic metadata version."""
    return _pipeline_version("semantic_version")


@cache
def embedding_version() -> str:
    """Return the configured embedding version."""
    return _pipeline_version("embedding_version")


@cache
def pipeline_version() -> str:
    """Return the configured pipeline version."""
    return _pipeline_version("pipeline_version")


@cache
def embedding_model_name() -> str:
    """Return the configured embedding model name."""

    models = models_config()

    if "embedding" not in models:
        raise ConfigurationError(
            "Models configuration is missing 'embedding'"
        )

    embedding = models["embedding"]

    if not isinstance(embedding, Mapping):
        raise ConfigurationError(
            "Models configuration field 'embedding' must be a mapping"
        )

    if "model" not in embedding:
        raise ConfigurationError(
            "Embedding configuration is missing 'model'"
        )

    value = embedding["model"]

    if not isinstance(value, str):
        raise ConfigurationError(
            "Embedding configuration field 'model' must be a string"
        )

    if not value or value.isspace() or value.strip() != value:
        raise ConfigurationError(
            "Embedding configuration field 'model' is invalid"
        )

    return value


@cache
def qdrant_url() -> str:
    """Return the configured Qdrant Cloud URL.

    Returns:
        str: The configured Qdrant Cloud endpoint.

    Raises:
        ConfigurationError: If the environment variable is missing or empty.
    """
    value = os.getenv(_QDRANT_URL_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_QDRANT_URL_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_QDRANT_URL_ENV}' must not be empty."
        )

    return value


@cache
def qdrant_api_key() -> str:
    """Return the configured Qdrant Cloud API key.

    Returns:
        str: The configured Qdrant Cloud API key.

    Raises:
        ConfigurationError: If the environment variable is missing or empty.
    """
    value = os.getenv(_QDRANT_API_KEY_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_QDRANT_API_KEY_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_QDRANT_API_KEY_ENV}' must not be empty."
        )

    return value


@cache
def elasticsearch_url() -> str:
    """Return the configured Elasticsearch Serverless URL.

    Returns:
        str: The configured Elasticsearch endpoint.

    Raises:
        ConfigurationError: If the environment variable is missing or empty.
    """
    value = os.getenv(_ELASTICSEARCH_URL_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_ELASTICSEARCH_URL_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_ELASTICSEARCH_URL_ENV}' must not be empty."
        )

    return value


@cache
def elasticsearch_api_key() -> str:
    """Return the configured Elasticsearch Serverless API key.

    Returns:
        str: The configured Elasticsearch API key.

    Raises:
        ConfigurationError: If the environment variable is missing or empty.
    """
    value = os.getenv(_ELASTICSEARCH_API_KEY_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_ELASTICSEARCH_API_KEY_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_ELASTICSEARCH_API_KEY_ENV}' must not be empty."
        )

    return value


@cache
def gemini_api_key() -> str:
    """Return the configured Gemini API key.

    Returns:
        str: The configured Gemini API key.

    Raises:
        ConfigurationError:
            If the environment variable is missing or empty.
    """
    value = os.getenv(_GEMINI_API_KEY_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_GEMINI_API_KEY_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_GEMINI_API_KEY_ENV}' must not be empty."
        )

    return value


@cache
def groq_api_key() -> str:
    """Return the configured Groq API key.

    Returns:
        str: The configured Groq API key.

    Raises:
        ConfigurationError:
            If the environment variable is missing or empty.
    """
    value = os.getenv(_GROQ_API_KEY_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_GROQ_API_KEY_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_GROQ_API_KEY_ENV}' must not be empty."
        )

    return value


@cache
def nvidia_api_key() -> str:
    """Return the configured NVIDIA API key.

    Returns:
        str: The configured NVIDIA API key.

    Raises:
        ConfigurationError:
            If the environment variable is missing or empty.
    """
    value = os.getenv(_NVIDIA_API_KEY_ENV)

    if value is None:
        raise ConfigurationError(
            f"Environment variable '{_NVIDIA_API_KEY_ENV}' is not configured."
        )

    value = value.strip()

    if not value:
        raise ConfigurationError(
            f"Environment variable '{_NVIDIA_API_KEY_ENV}' must not be empty."
        )

    return value


@cache
def configured_categories() -> frozenset[Category]:
    """Return the immutable set of configured category identifiers."""

    category_config = supported_categories()

    if tuple(category_config.keys()) != ("categories",):
        raise ConfigurationError(
            "Supported categories configuration must contain only 'categories'"
        )

    raw_categories = category_config["categories"]

    if isinstance(raw_categories, (str, bytes, bytearray)) or not isinstance(
        raw_categories,
        Sequence,
    ):
        raise ConfigurationError(
            "Supported categories configuration field 'categories' must be a sequence"
        )

    if not raw_categories:
        raise ConfigurationError(
            "Supported categories configuration field 'categories' must not be empty"
        )

    categories: set[Category] = set()

    for raw_category in raw_categories:
        category = _validate_category_identifier(raw_category)

        if category in categories:
            raise ConfigurationError(
                f"Duplicate configured category: {category}"
            )

        categories.add(category)

    return frozenset(categories)


@cache
def configured_category_directories() -> tuple[tuple[Category, Path], ...]:
    """Return immutable configured category-directory pairs."""

    application_config = config()

    if "document_directories" not in application_config:
        raise ConfigurationError(
            "Application configuration is missing 'document_directories'"
        )

    raw_directories = application_config["document_directories"]

    if not isinstance(raw_directories, Mapping):
        raise ConfigurationError(
            "Application configuration field 'document_directories' must be a mapping"
        )

    if not raw_directories:
        raise ConfigurationError(
            "Application configuration field 'document_directories' must not be empty"
        )

    categories = configured_categories()
    category_directories: list[tuple[Category, Path]] = []
    seen_categories: set[Category] = set()

    for raw_category, raw_directory in raw_directories.items():
        category = _validate_category_identifier(raw_category)

        if category in seen_categories:
            raise ConfigurationError(
                f"Duplicate configured category directory: {category}"
            )

        if category not in categories:
            raise ConfigurationError(
                f"Unsupported configured category directory: {category}"
            )

        directory = _resolve_configured_path(raw_directory)
        seen_categories.add(category)
        category_directories.append((category, directory))

    if seen_categories != categories:
        missing_categories = tuple(sorted(categories - seen_categories))
        raise ConfigurationError(
            f"Configured category directories are missing categories: {missing_categories}"
        )

    return tuple(sorted(category_directories, key=lambda pair: pair[0]))


def _validate_category_identifier(value: Any) -> Category:
    """Validate and return one configured category identifier."""

    if not isinstance(value, str):
        raise ConfigurationError("Configured category identifier must be a string")

    if not value or value.isspace() or value.strip() != value:
        raise ConfigurationError("Configured category identifier is missing")

    if _CATEGORY_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ConfigurationError(
            f"Invalid configured category identifier: {value!r}"
        )

    return value


def _resolve_configured_path(value: Any) -> Path:
    """Resolve one configured path without checking its existence."""

    if not isinstance(value, str):
        raise ConfigurationError("Configured path must be a string")

    if not value or value.isspace() or value.strip() != value:
        raise ConfigurationError("Configured path is missing")

    if "\x00" in value:
        raise ConfigurationError(
            f"Invalid configured path: {value!r}"
        )

    directory = Path(value)

    if directory.is_absolute():
        return directory

    return project_root() / directory


__all__ = [
    "project_root",
    "configs_directory",
    "load_yaml",
    "load_config",
    "config",
    "supported_categories",
    "models_config",
    "catalog_metadata_path",
    "registry_storage_path",
    "chunk_size",
    "chunk_overlap",
    "chunker_version",
    "parser_version",
    "cleaner_version",
    "semantic_version",
    "embedding_version",
    "pipeline_version",
    "embedding_model_name",
    "qdrant_url",
    "qdrant_api_key",
    "elasticsearch_url",
    "elasticsearch_api_key",
    "gemini_api_key",
    "groq_api_key",
    "nvidia_api_key",
    "configured_categories",
    "configured_category_directories",
]
