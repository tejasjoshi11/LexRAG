"""Generic reusable utility helpers for the LexRAG project.

This module provides dependency-light, reusable helper functions shared
across the application. Utilities in this module are deterministic,
independent of business logic, and safe to import from any layer.
"""

import hashlib
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.shared.constants import (
    ISO8601_UTC_TIMESTAMP_FORMAT,
    PDF_EXTENSION,
)
from src.shared.exceptions import DiscoveryError
from src.shared.types import ContentHash, PathLike


def file_exists(path: PathLike) -> bool:
    """Return True if the path exists and is a regular file."""
    return Path(path).is_file()


def directory_exists(path: PathLike) -> bool:
    """Return True if the path exists and is a directory."""
    return Path(path).is_dir()


_HASH_CHUNK_SIZE = 64 * 1024


def compute_file_hash(path: Path) -> ContentHash:
    """Return the lowercase hexadecimal SHA-256 digest of a file."""

    digest = hashlib.sha256()

    try:
        with path.open("rb") as file:
            while chunk := file.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        raise DiscoveryError(
            f"Failed to compute SHA-256 hash for file: {path}"
        ) from exc

    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def utc_now_iso() -> str:
    """Return the current UTC timestamp using the project format."""
    return datetime.now(UTC).strftime(ISO8601_UTC_TIMESTAMP_FORMAT)


def generate_uuid() -> str:
    """Generate a random UUID4 string."""
    return str(uuid.uuid4())


def list_files(
    directory: PathLike,
    extension: str | None = None,
) -> list[Path]:
    """Return a sorted list of files in a directory.

    Parameters
    ----------
    directory:
        Directory to scan.

    extension:
        Optional file extension filter (e.g. ".pdf").

    Returns
    -------
    list[Path]
        Sorted list of matching files.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.

    NotADirectoryError
        If the supplied path is not a directory.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory_path}"
        )

    files = (
        path
        for path in directory_path.iterdir()
        if path.is_file()
    )

    if extension is not None:
        normalized_extension = extension.casefold()

        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        files = (
            path
            for path in files
            if path.suffix.casefold() == normalized_extension
        )

    return sorted(files)


def is_pdf(path: PathLike) -> bool:
    """Return True if the path has a PDF extension."""
    return Path(path).suffix.casefold() == PDF_EXTENSION.casefold()


__all__ = [
    "compute_file_hash",
    "directory_exists",
    "file_exists",
    "generate_uuid",
    "is_pdf",
    "list_files",
    "normalize_whitespace",
    "utc_now_iso",
]
