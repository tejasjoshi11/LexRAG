"""Immutable catalog metadata contract."""

from dataclasses import dataclass

from src.shared.types import (
    Category,
    DocumentID,
    FileName,
    SourceURL,
)

@dataclass(
    frozen=True,
    slots=True,
)
class CatalogMetadata:
    """Immutable catalog metadata loaded from a single metadata.csv row."""

    document_id: DocumentID
    title: str
    year: int
    category: Category
    source_url: SourceURL
    filename: FileName
    summary: str | None = None