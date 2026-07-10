"""Immutable cleaned document contract."""

from dataclasses import dataclass

from src.contracts.parsed_document import ParsedDocument


@dataclass(
    frozen=True,
    slots=True,
)
class CleanedDocument:
    """Immutable mechanically cleaned representation of a parsed document."""

    parsed_document: ParsedDocument

    cleaned_text: str