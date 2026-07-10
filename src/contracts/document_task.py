"""Immutable document task contract."""

from dataclasses import dataclass
from pathlib import Path

from src.contracts.catalog_metadata import CatalogMetadata
from src.shared.types import ContentHash


@dataclass(
    frozen=True,
    slots=True,
)
class DocumentTask:
    """Immutable ingestion work item."""

    pdf_path: Path

    catalog_metadata: CatalogMetadata

    content_hash: ContentHash