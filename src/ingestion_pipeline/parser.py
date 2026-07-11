"""Stateless parsing of one PDF document into its canonical contract."""

from pathlib import Path

import logging

import fitz

from src.contracts.document_task import DocumentTask
from src.contracts.parsed_document import ParsedDocument
from src.shared.exceptions import ParsingError


def parse_document(task: DocumentTask) -> ParsedDocument:
    """Parse one immutable document task into an immutable parsed document."""
    _LOGGER = logging.getLogger(__name__)

    try:
        with _open_document(task.pdf_path) as document:
            if document.needs_pass:
                raise ParsingError("PDF is encrypted or password-protected")

            page_count = document.page_count
            raw_text, page_offsets = _extract_raw_text(document, page_count)
            intrinsic_metadata = _extract_intrinsic_metadata(document)
    except ParsingError:
        raise
    
    except (
        fitz.FileDataError,
        fitz.EmptyFileError,
        fitz.FileNotFoundError,
        OSError,
    ) as exc:
        raise ParsingError(
            "Unable to parse PDF document."
        ) from exc

    return ParsedDocument(
        document_task=task,
        raw_text=raw_text,
        page_count=page_count,
        page_offsets=page_offsets,
        intrinsic_metadata=intrinsic_metadata,
    )


def _open_document(path: Path) -> fitz.Document:
    """Open one PDF document through the sole supported backend."""

    return fitz.open(path)


def _extract_raw_text(
    document: fitz.Document,
    page_count: int,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Extract ordered page text and its canonical character offsets."""

    page_texts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    current_offset = 0

    for page_number in range(page_count):
        page = document.load_page(page_number)
        page_text = page.get_text("text")

        if not isinstance(page_text, str):
            raise RuntimeError("PyMuPDF returned non-text page content")

        end_offset = current_offset + len(page_text)
        page_texts.append(page_text)
        page_offsets.append((current_offset, end_offset))
        current_offset = end_offset

    return "".join(page_texts), tuple(page_offsets)


def _extract_intrinsic_metadata(
    document: fitz.Document,
) -> tuple[tuple[str, str], ...]:
    """Return available intrinsic PDF metadata in canonical field order."""

    metadata = document.metadata
    intrinsic_metadata: list[tuple[str, str]] = []

    for backend_key, canonical_key in (
        ("title", "title"),
        ("author", "author"),
        ("subject", "subject"),
        ("keywords", "keywords"),
        ("creator", "creator"),
        ("producer", "producer"),
        ("creationDate", "creation_date"),
        ("modDate", "modification_date"),
        ("trapped", "trapped"),
    ):
        value = metadata.get(backend_key)

        if value is None or value == "":
            continue

        if not isinstance(value, str):
            raise RuntimeError("PyMuPDF returned an invalid metadata value")

        intrinsic_metadata.append((canonical_key, value))

    return tuple(intrinsic_metadata)


def _translate_backend_exception(error: Exception) -> ParsingError:
    """Translate one filesystem or PyMuPDF failure at the parser boundary."""

    return ParsingError("Unable to parse PDF document")


__all__ = ["parse_document"]
