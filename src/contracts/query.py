"""Immutable processed user query."""

from dataclasses import dataclass


@dataclass(
    frozen=True,
    slots=True,
)
class Query:
    """Canonical user query."""

    query_text: str