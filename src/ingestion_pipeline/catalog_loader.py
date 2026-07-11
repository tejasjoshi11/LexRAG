"""Validated runtime access to immutable catalog metadata."""

import csv
from pathlib import Path
from threading import Lock

from src.contracts.catalog_metadata import CatalogMetadata
from src.shared.config import catalog_metadata_path, configured_categories
from src.shared.exceptions import ConfigurationError, MetadataValidationError
from src.shared.types import Category, FileName


_CATALOG_HEADERS = frozenset(
    {
        "document_id",
        "title",
        "year",
        "category",
        "source_url",
        "filename",
        "summary",
    }
)

_catalog_initialization_lock = Lock()
_catalog_metadata: tuple[CatalogMetadata, ...] | None = None
_catalog_index: dict[tuple[Category, FileName], CatalogMetadata] | None = None


def load_catalog_metadata() -> tuple[CatalogMetadata, ...]:
    """Load, validate, and return the immutable catalog metadata snapshot."""

    global _catalog_index, _catalog_metadata

    snapshot = _catalog_metadata
    if snapshot is not None:
        return snapshot

    with _catalog_initialization_lock:
        snapshot = _catalog_metadata

        if snapshot is None:
            snapshot, index = _load_catalog_snapshot()
            _catalog_index = index
            _catalog_metadata = snapshot

        return snapshot


def find_catalog_metadata(
    category: Category,
    filename: FileName,
) -> CatalogMetadata:
    """Return metadata matching one exact category and filename identity."""

    load_catalog_metadata()
    index = _catalog_index

    if index is None:
        raise MetadataValidationError("Catalog metadata index is unavailable")

    try:
        return index[(category, filename)]
    except (KeyError, TypeError) as exc:
        raise MetadataValidationError(
            "Catalog metadata record was not found for the supplied identity"
        ) from exc


def _load_catalog_snapshot() -> tuple[
    tuple[CatalogMetadata, ...],
    dict[tuple[Category, FileName], CatalogMetadata],
]:
    """Read, validate, sort, and index one complete catalog snapshot."""

    path = catalog_metadata_path()

    try:
        if not path.is_file():
            raise ConfigurationError(
                f"Configured catalog metadata path is not a readable file: {path}"
            )

        with path.open("r", encoding="utf-8-sig", newline="") as catalog_file:
            reader = csv.reader(catalog_file)
            headers = _read_headers(reader)
            _validate_headers(headers)
            records = _read_records(reader, headers)
    except OSError as exc:
        raise ConfigurationError(
            f"Unable to read configured catalog metadata file: {path}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise MetadataValidationError(
            f"Catalog metadata file is not valid UTF-8: {path}"
        ) from exc
    except csv.Error as exc:
        raise MetadataValidationError(
            f"Catalog metadata CSV parsing failed: {path}"
        ) from exc

    records.sort(key=lambda record: (record.category, record.filename))
    snapshot = tuple(records)
    index = {
        (record.category, record.filename): record
        for record in snapshot
    }

    return snapshot, index


def _read_headers(reader: csv.reader) -> list[str]:
    """Return the catalog CSV header row or raise for an empty source."""

    try:
        return next(reader)
    except StopIteration as exc:
        raise MetadataValidationError(
            "Catalog metadata CSV is missing its header row"
        ) from exc


def _validate_headers(headers: list[str]) -> None:
    """Validate the exact, unique catalog CSV header set."""

    if len(headers) != len(_CATALOG_HEADERS):
        raise MetadataValidationError(
            "Catalog metadata CSV must contain exactly seven headers"
        )

    if len(set(headers)) != len(headers):
        raise MetadataValidationError(
            "Catalog metadata CSV contains duplicate headers"
        )

    if set(headers) != _CATALOG_HEADERS:
        raise MetadataValidationError(
            "Catalog metadata CSV headers do not match the required schema"
        )


def _read_records(
    reader: csv.reader,
    headers: list[str],
) -> list[CatalogMetadata]:
    """Validate all CSV rows and construct immutable catalog metadata."""

    categories = configured_categories()
    records: list[CatalogMetadata] = []
    document_ids: set[str] = set()
    identities: set[tuple[Category, FileName]] = set()

    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(headers):
            raise MetadataValidationError(
                f"Catalog metadata row {row_number} has an invalid column count"
            )

        row = dict(zip(headers, values))
        metadata = _build_catalog_metadata(row, row_number, categories)
        identity = (metadata.category, metadata.filename)

        if metadata.document_id in document_ids:
            raise MetadataValidationError(
                f"Duplicate catalog document_id at row {row_number}: "
                f"{metadata.document_id!r}"
            )

        if identity in identities:
            raise MetadataValidationError(
                f"Duplicate catalog category and filename at row {row_number}: "
                f"{identity!r}"
            )

        document_ids.add(metadata.document_id)
        identities.add(identity)
        records.append(metadata)

    if not records:
        raise MetadataValidationError(
            "Catalog metadata CSV must contain at least one metadata record"
        )

    return records


def _build_catalog_metadata(
    row: dict[str, str],
    row_number: int,
    categories: frozenset[Category],
) -> CatalogMetadata:
    """Validate one CSV row and construct its immutable metadata contract."""

    document_id = _required_text(row["document_id"], "document_id", row_number)
    title = _required_text(row["title"], "title", row_number)
    year = _positive_integer(row["year"], "year", row_number)
    category = _category(row["category"], row_number, categories)
    source_url = _required_text(row["source_url"], "source_url", row_number)
    filename = _filename(row["filename"], row_number)
    summary = _optional_summary(row["summary"], row_number)

    return CatalogMetadata(
        document_id=document_id,
        title=title,
        year=year,
        category=category,
        source_url=source_url,
        filename=filename,
        summary=summary,
    )


def _required_text(value: str, field: str, row_number: int) -> str:
    """Validate one required, unmodified text field."""

    if not isinstance(value, str):
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} field {field!r} must be a string"
        )

    if not value or value.isspace() or value.strip() != value:
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} field {field!r} is invalid"
        )

    return value


def _optional_summary(value: str, row_number: int) -> str | None:
    """Validate one optional summary while preserving valid content exactly."""

    if not isinstance(value, str):
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} field 'summary' must be a string"
        )

    if value == "":
        return None

    if value.isspace() or value.strip() != value:
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} field 'summary' is invalid"
        )

    return value


def _positive_integer(value: str, field: str, row_number: int) -> int:
    """Validate and return one positive decimal integer field."""

    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} field {field!r} must be a positive integer"
        )

    integer_value = int(value)

    if integer_value <= 0:
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} field {field!r} must be a positive integer"
        )

    return integer_value


def _category(
    value: str,
    row_number: int,
    categories: frozenset[Category],
) -> Category:
    """Validate one exact configured category value."""

    if not isinstance(value, str) or value not in categories:
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} has an unsupported category: {value!r}"
        )

    return value


def _filename(value: str, row_number: int) -> FileName:
    """Validate one unmodified filename-only value with an extension."""

    filename = _required_text(value, "filename", row_number)

    if "/" in filename or "\\" in filename:
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} filename must not contain a path"
        )

    if not Path(filename).suffix:
        raise MetadataValidationError(
            f"Catalog metadata row {row_number} filename must include an extension"
        )

    return filename


__all__ = [
    "load_catalog_metadata",
    "find_catalog_metadata",
]
