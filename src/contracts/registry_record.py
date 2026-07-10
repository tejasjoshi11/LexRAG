"""Immutable registry record contract."""

from dataclasses import dataclass

from src.shared.types import (
    Category,
    ContentHash,
    DocumentID,
    ProcessingStatus,
    SourceURL,
)


# Registry Record

@dataclass(
    frozen=True,
    slots=True,
)
class RegistryRecord:
    """Represent document lifecycle and processing history."""

    document_id: DocumentID
    content_hash: ContentHash
    title: str
    category: Category
    source_url: SourceURL
    parser_version: str
    cleaner_version: str
    semantic_version: str
    chunker_version: str
    embedding_model: str
    embedding_version: str
    pipeline_version: str
    processing_timestamp: str
    processing_status: ProcessingStatus
    chunk_count: int
    summary: str | None = None