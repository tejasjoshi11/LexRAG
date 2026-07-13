"""RAG pipeline orchestrator."""

from __future__ import annotations

import logging

from src.citations.citation_formatter import CitationFormatter
from src.contracts.llm_response import FinishReason, LLMResponse
from src.contracts.query import Query
from src.contracts.rag_response import RAGResponse
from src.contracts.route import Route, RouteDecision
from src.generation.prompt_builder import PromptBuilder
from src.generation.prompts.general_chat_prompt import (
    GENERAL_CHAT_SYSTEM_PROMPT,
)
from src.generation.response_generator import ResponseGenerator
from src.llm.models import Model
from src.retrieval_pipeline.retrieval_provider import RetrievalProvider
from src.routing.router import Router
from src.routing.routing_responses import (
    CLARIFY_ANSWER,
    REJECT_ANSWER,
)
from src.shared.exceptions import QueryError

_LOGGER = logging.getLogger(__name__)


class RAGPipeline:
    """Top-level orchestrator for the LexRAG query pipeline."""

    def __init__(
        self,
        router: Router,
        retrieval_provider: RetrievalProvider,
        prompt_builder: PromptBuilder,
        response_generator: ResponseGenerator,
        citation_formatter: CitationFormatter,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            router:
                Query router.

            retrieval_provider:
                Retrieval backend.

            prompt_builder:
                Prompt builder.

            response_generator:
                Language model response generator.

            citation_formatter:
                Citation formatter.
        """

        self._router = router
        self._retrieval_provider = retrieval_provider
        self._prompt_builder = prompt_builder
        self._response_generator = response_generator
        self._citation_formatter = citation_formatter

    def run(
        self,
        *,
        query: str,
        top_k: int,
        model: Model,
    ) -> RAGResponse:
        """Execute the RAG pipeline.

        Args:
            query:
                User query.

            top_k:
                Maximum number of retrieved chunks.

            model:
                Language model to use for generation.

        Returns:
            Complete RAG pipeline response.
        """

        _LOGGER.info("Pipeline started.")

        route_decision = self._router.route(
            query=query,
        )

        _LOGGER.info(
            "Route selected: %s (confidence=%.2f).",
            route_decision.route.value,
            route_decision.confidence,
        )

        match route_decision.route:

            case Route.REJECT:
                response = self._build_early_response(
                    answer=REJECT_ANSWER,
                    route_decision=route_decision,
                )

            case Route.CLARIFY:
                response = self._build_early_response(
                    answer=CLARIFY_ANSWER,
                    route_decision=route_decision,
                )

            case Route.GENERAL_CHAT:
                response = self._handle_general_chat(
                    query=query,
                    model=model,
                    route_decision=route_decision,
                )

            case Route.LEGAL_RAG:
                response = self._handle_legal_rag(
                    query=query,
                    top_k=top_k,
                    model=model,
                    route_decision=route_decision,
                )

            case _:
                raise QueryError(
                    f"Unsupported route: {route_decision.route.value}"
                )

        _LOGGER.info("Pipeline completed.")

        return response

    def _handle_general_chat(
        self,
        *,
        query: str,
        model: Model,
        route_decision: RouteDecision,
    ) -> RAGResponse:
        """Handle a general chat query."""

        llm_response = self._response_generator.generate(
            model=model,
            system_prompt=GENERAL_CHAT_SYSTEM_PROMPT,
            user_prompt=query,
        )

        _LOGGER.info("Generation completed.")

        return RAGResponse(
            answer=llm_response.text,
            llm_response=llm_response,
            route_decision=route_decision,
            retrieved_chunks=(),
            citations=(),
        )

    def _handle_legal_rag(
        self,
        *,
        query: str,
        top_k: int,
        model: Model,
        route_decision: RouteDecision,
    ) -> RAGResponse:
        """Handle a legal RAG query."""

        query_contract = Query(query_text=query)

        retrieved_chunks = self._retrieval_provider.retrieve(
            query=query_contract,
            top_k=top_k,
        )

        _LOGGER.info(
            "Retrieval completed: %d chunks retrieved.",
            len(retrieved_chunks),
        )

        retrieved_chunks_tuple = tuple(retrieved_chunks)

        citations = self._citation_formatter.format(
            retrieved_chunks=retrieved_chunks_tuple,
        )

        prompt = self._prompt_builder.build(
            user_query=query,
            retrieved_chunks=retrieved_chunks,
        )

        _LOGGER.info("Prompt built.")

        llm_response = self._response_generator.generate(
            model=model,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )

        _LOGGER.info("Generation completed.")

        return RAGResponse(
            answer=llm_response.text,
            llm_response=llm_response,
            route_decision=route_decision,
            retrieved_chunks=retrieved_chunks_tuple,
            citations=citations,
        )

    def _build_early_response(
        self,
        *,
        answer: str,
        route_decision: RouteDecision,
    ) -> RAGResponse:
        """Build a RAGResponse for non-retrieval routes."""

        return RAGResponse(
            answer=answer,
            llm_response=LLMResponse(
                text=answer,
                provider="none",
                model="none",
                prompt_tokens=0,
                completion_tokens=0,
                reasoning_tokens=None,
                total_tokens=0,
                latency_ms=0.0,
                finish_reason=FinishReason.STOP,
            ),
            route_decision=route_decision,
            retrieved_chunks=(),
            citations=(),
        )
