"""Rendering helpers for the LexRAG Streamlit application.

Each function encapsulates one visual section of the assistant response.
Contains no business logic — only presentation over backend contracts.
"""

from __future__ import annotations

import streamlit as st

from src.contracts.citation import Citation
from src.contracts.rag_response import RAGResponse
from src.contracts.retrieved_chunk import RetrievedChunk
from src.contracts.route import Route


# ─── Example Queries ──────────────────────────────────────────────────────────

_EXAMPLE_QUERIES: tuple[str, ...] = (
    "What is judicial review?",
    "Summarize Brown v. Board of Education.",
    "Explain Article 21 of the Constitution.",
    "What is habeas corpus?",
)


# ─── Welcome Section ─────────────────────────────────────────────────────────


def render_welcome() -> None:
    """Render the welcome section shown when no conversation exists."""

    st.markdown("### Welcome to LexRAG")

    st.markdown(
        "Ask legal questions. Summarize judgments. "
        "Retrieve cited legal documents."
    )

    st.markdown(
        "Every legal response includes references to official sources "
        "whenever available."
    )

    st.divider()

    st.markdown("**Try an example:**")

    cols = st.columns(2)

    for index, query in enumerate(_EXAMPLE_QUERIES):
        with cols[index % 2]:
            if st.button(
                query,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state._example_query = query
                st.rerun()


# ─── Response Orchestrator ────────────────────────────────────────────────────


def render_response(
    response: RAGResponse,
    *,
    show_pipeline_details: bool,
    show_retrieved_context: bool,
) -> None:
    """Render a complete assistant response based on route.

    Dispatches to the appropriate rendering helpers depending on the
    route decision attached to the response.
    """

    route = response.route_decision.route

    _render_answer(response)

    if route == Route.LEGAL_RAG and response.citations:
        _render_sources(response.citations)

    if (
        show_retrieved_context
        and route == Route.LEGAL_RAG
        and response.retrieved_chunks
    ):
        _render_retrieved_context(response.retrieved_chunks)

    if show_pipeline_details and route in (
        Route.LEGAL_RAG,
        Route.GENERAL_CHAT,
    ):
        _render_pipeline_details(response)


# ─── Answer ───────────────────────────────────────────────────────────────────


def _render_answer(response: RAGResponse) -> None:
    """Render the assistant answer as Markdown."""

    st.markdown(response.answer)


# ─── Sources ──────────────────────────────────────────────────────────────────


def _render_sources(citations: tuple[Citation, ...]) -> None:
    """Render the Sources section for LEGAL_RAG responses.

    Each citation is displayed as a clean card-like block with title,
    page information, section heading, and an official source link.
    Empty metadata fields are hidden.
    """

    st.markdown("#### Sources")

    for index, citation in enumerate(citations):

        st.markdown(f"**{citation.title}**")

        details: list[str] = []

        if citation.page_start and citation.page_end:
            if citation.page_start == citation.page_end:
                details.append(f"Page {citation.page_start}")
            else:
                details.append(
                    f"Pages {citation.page_start}"
                    f"\u2013{citation.page_end}"
                )
        elif citation.page_start:
            details.append(f"Page {citation.page_start}")

        if citation.heading:
            details.append(f"Section: {citation.heading}")

        if details:
            st.markdown(" \u00b7 ".join(details))

        if citation.source_url:
            st.link_button(
                "\U0001f4cc Open Official Source",
                citation.source_url,
            )

        if index < len(citations) - 1:
            st.divider()


# ─── Retrieved Context ────────────────────────────────────────────────────────


def _render_retrieved_context(
    chunks: tuple[RetrievedChunk, ...],
) -> None:
    """Render the Retrieved Context section.

    Each chunk appears inside its own expander with metadata and the
    full chunk text rendered as Markdown.
    """

    st.markdown("#### Retrieved Context")

    for chunk in chunks:

        label = (
            f"{chunk.title} \u2014 "
            f"Score: {chunk.retrieval_score:.4f}"
        )

        with st.expander(label, expanded=False):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Document:** {chunk.title}")
                st.markdown(
                    f"**Retrieval Method:** "
                    f"{chunk.retrieval_method.value.title()}"
                )

            with col2:
                st.markdown(
                    f"**Similarity Score:** "
                    f"{chunk.retrieval_score:.4f}"
                )

                if chunk.page_start and chunk.page_end:
                    if chunk.page_start == chunk.page_end:
                        st.markdown(f"**Page:** {chunk.page_start}")
                    else:
                        st.markdown(
                            f"**Pages:** "
                            f"{chunk.page_start}\u2013{chunk.page_end}"
                        )
                elif chunk.page_start:
                    st.markdown(f"**Page:** {chunk.page_start}")

            if chunk.heading:
                st.markdown(f"**Section:** {chunk.heading}")

            st.divider()

            st.markdown(chunk.chunk_text)


# ─── Pipeline Details ─────────────────────────────────────────────────────────


def _render_pipeline_details(response: RAGResponse) -> None:
    """Render Pipeline Details inside a collapsed expander.

    Displays only data returned by the backend. Nothing is calculated
    or fabricated by the frontend.
    """

    with st.expander("Pipeline Details", expanded=False):

        route_decision = response.route_decision
        llm = response.llm_response

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"**Selected Route:** {route_decision.route.value}"
            )
            st.markdown(
                f"**Router Confidence:** "
                f"{route_decision.confidence:.2f}"
            )
            st.markdown(f"**Generation Model:** {llm.model}")
            st.markdown(
                f"**Retrieved Chunks:** "
                f"{len(response.retrieved_chunks)}"
            )

        with col2:
            st.markdown(
                f"**Generation Latency:** {llm.latency_ms:.0f} ms"
            )
            st.markdown(f"**Prompt Tokens:** {llm.prompt_tokens:,}")
            st.markdown(
                f"**Completion Tokens:** {llm.completion_tokens:,}"
            )
            st.markdown(f"**Total Tokens:** {llm.total_tokens:,}")

        st.markdown(
            f"**Finish Reason:** {llm.finish_reason.value}"
        )
