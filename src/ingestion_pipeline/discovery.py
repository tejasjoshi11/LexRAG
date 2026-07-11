"""Deterministic document discovery and Registry-aware task orchestration."""

from collections.abc import Iterator
import logging
from pathlib import Path

from src.contracts.document_task import DocumentTask
from src.ingestion_pipeline.catalog_loader import find_catalog_metadata
from src.metadata import DocumentRegistry
from src.shared.config import configured_category_directories
from src.shared.exceptions import (
    ConfigurationError,
    DiscoveryError,
    MetadataValidationError,
    RegistryError,
)
from src.shared.types import Category
from src.shared.utils import compute_file_hash, is_pdf


_LOGGER = logging.getLogger(__name__)


def discover_documents() -> tuple[DocumentTask, ...]:
    """Discover changed PDF documents and return immutable ingestion tasks."""

    candidate_count = 0
    skipped_count = 0
    emitted_count = 0
    tasks: list[DocumentTask] = []

    _LOGGER.info("Discovery started")

    document_registry = DocumentRegistry()

    try:
        for category, pdf_path in _iter_candidate_pdfs():
            candidate_count += 1
            task = _build_document_task(category, pdf_path, document_registry)

            if task is None:
                skipped_count += 1
                continue

            tasks.append(task)
            emitted_count += 1
    except (
        ConfigurationError,
        DiscoveryError,
        MetadataValidationError,
        RegistryError,
    ):
        _LOGGER.exception("Discovery failed")
        raise

    result = tuple(tasks)

    _LOGGER.info(
        "Discovery completed: candidates=%d skipped=%d emitted=%d",
        candidate_count,
        skipped_count,
        emitted_count,
    )

    return result


def _iter_candidate_pdfs() -> Iterator[tuple[Category, Path]]:
    """Yield eligible direct-child PDF files in deterministic order."""

    for category, directory in configured_category_directories():
        try:
            if directory.is_symlink():
                raise DiscoveryError(
                    f"Configured document directory must not be a symlink: {directory}"
                )

            if not directory.exists():
                raise DiscoveryError(
                    f"Configured document directory does not exist: {directory}"
                )

            if not directory.is_dir():
                raise DiscoveryError(
                    f"Configured document directory is not a directory: {directory}"
                )
        except OSError as exc:
            raise DiscoveryError(
                f"Failed to access configured document directory: {directory}"
            ) from exc

        try:
            paths = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise DiscoveryError(
                f"Failed to enumerate configured document directory: {directory}"
            ) from exc

        for pdf_path in paths:
            try:
                if pdf_path.is_symlink() or not pdf_path.is_file():
                    continue
            except OSError as exc:
                raise DiscoveryError(
                    f"Failed to inspect discovered path: {pdf_path}"
                ) from exc

            if not is_pdf(pdf_path):
                continue

            yield category, pdf_path


def _build_document_task(
    category: Category,
    pdf_path: Path,
    document_registry: DocumentRegistry,
) -> DocumentTask | None:
    """Build one task or return None when Registry state is unchanged."""

    filename = pdf_path.name
    catalog_metadata = find_catalog_metadata(category, filename)
    content_hash = compute_file_hash(pdf_path)

    if document_registry.is_document_unchanged(catalog_metadata.document_id, content_hash):
        return None

    return DocumentTask(
        pdf_path=pdf_path,
        catalog_metadata=catalog_metadata,
        content_hash=content_hash,
    )


__all__ = [
    "discover_documents",
]
