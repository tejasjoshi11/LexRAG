"""Final response returned by the LexRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.contracts.citation import Citation
from src.contracts.llm_response import LLMResponse
from src.contracts.retrieved_chunk import RetrievedChunk
from src.contracts.route import RouteDecision


@dataclass(
    frozen=True,
    slots=True,
)
class RAGResponse:
    """Final response returned by the RAG pipeline."""

    answer: str

    llm_response: LLMResponse

    route_decision: RouteDecision

    retrieved_chunks: tuple[
        RetrievedChunk,
        ...,
    ]

    citations: tuple[
        Citation,
        ...,
    ]