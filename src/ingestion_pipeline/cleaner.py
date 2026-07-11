"""Stage 3 of the LexRAG ingestion pipeline: deterministic text cleaning.

This module implements the cleaner, which transforms one immutable
:class:`~src.contracts.parsed_document.ParsedDocument` into one immutable
:class:`~src.contracts.cleaned_document.CleanedDocument`.

The cleaner performs purely mechanical, deterministic cleaning of
``parsed_document.raw_text``: Unicode NFC normalization, line ending
normalization, removal of unsupported control characters, and whitespace
normalization. It never rewrites, summarizes, paraphrases, or infers
structure, and it never mutates the supplied ``ParsedDocument``.
"""

import re
import unicodedata
import logging

from src.contracts.cleaned_document import CleanedDocument
from src.contracts.parsed_document import ParsedDocument
from src.shared.utils import normalize_whitespace

__all__ = ["clean_document"]

_LOGGER = logging.getLogger(__name__)

# Unicode category "Cc" control characters (C0 controls, DEL, C1 controls),
# excluding the tab and newline characters the pipeline preserves.
_NON_PRINTABLE_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean_document(parsed_document: ParsedDocument) -> CleanedDocument:
    """Mechanically normalize a parsed document's raw text.

    Applies deterministic, mechanical cleaning to ``parsed_document.raw_text``
    and pairs the result with the original ``parsed_document`` in a new
    ``CleanedDocument``. The supplied ``parsed_document`` is preserved
    exactly as received and is never modified.

    Args:
        parsed_document: The immutable canonical parsed document to clean.

    Returns:
        CleanedDocument: The original parsed document paired with its
        mechanically cleaned text.
    """
    cleaned_text = _clean_text(parsed_document.raw_text)

    cleaned_document = CleanedDocument(
        parsed_document=parsed_document,
        cleaned_text=cleaned_text,
    )

    _LOGGER.info(
        f"Cleaned document {parsed_document.filename}."
    )

    return cleaned_document


def _clean_text(raw_text: str) -> str:
    """Run the deterministic cleaning pipeline over raw text.

    Pipeline: Unicode NFC normalization, line ending normalization, control
    character removal, then shared whitespace normalization.

    Args:
        raw_text: The canonical raw text extracted by the parser.

    Returns:
        str: The mechanically cleaned text.
    """
    text = unicodedata.normalize("NFC", raw_text)
    text = _normalize_line_endings(text)
    text = _remove_control_characters(text)
    text = normalize_whitespace(text)

    return text


def _normalize_line_endings(text: str) -> str:
    """Normalize CRLF and CR line endings to a single LF character.

    Args:
        text: Text that may contain CRLF or CR line endings.

    Returns:
        str: Text using only LF (``\\n``) line endings.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_control_characters(text: str) -> str:
    """Remove non-printable control characters, keeping tabs and newlines.

    Args:
        text: Text that may contain non-printable control characters.

    Returns:
        str: Text with all Unicode "Cc"-category control characters
        removed, except for ``\\t`` and ``\\n``.
    """
    return _NON_PRINTABLE_CONTROL_PATTERN.sub("", text)