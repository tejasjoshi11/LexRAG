"""Abstract citation formatter."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.contracts.citation import Citation
from src.contracts.retrieved_chunk import RetrievedChunk


class CitationFormatter(ABC):
    """Abstract base class for citation formatters."""

    @abstractmethod
    def format(
        self,
        *,
        retrieved_chunks: tuple[
            RetrievedChunk,
            ...,
        ],
    ) -> tuple[
        Citation,
        ...,
    ]:
        """Convert retrieved chunks into citations."""