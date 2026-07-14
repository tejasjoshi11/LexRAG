"""Default citation formatter."""

from __future__ import annotations

from src.citations.citation_formatter import (
    CitationFormatter,
)
from src.contracts.citation import Citation
from src.contracts.retrieved_chunk import RetrievedChunk


class DefaultCitationFormatter(
    CitationFormatter,
):
    """Create citations from retrieved chunks."""

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

        citations: list[Citation] = []

        seen: set[
            tuple[
                str,
                int,
                int,
            ]
        ] = set()

        for chunk in retrieved_chunks:

            key = (
                chunk.document_id,
                chunk.page_start,
                chunk.page_end,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                Citation(
                    title=chunk.title,
                    heading=chunk.heading,
                    source_url=chunk.source_url,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )

        return tuple(citations)