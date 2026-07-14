"""Page-level rendering for the LexRAG Streamlit application.

Composes atomic components from ``ui_components`` into complete page
sections.  Contains no business logic — only presentation over
backend contracts.
"""

from __future__ import annotations

import streamlit as st

from src.contracts.citation import Citation
from src.contracts.rag_response import RAGResponse
from src.contracts.retrieved_chunk import RetrievedChunk
from src.contracts.route import Route

from ui.components import (
    render_empty_state,
    render_pipeline_details,
    render_source_card,
)


# ─── Constants ────────────────────────────────────────────────────────────

_EXAMPLE_QUERIES: tuple[str, ...] = (
    "What is judicial review?",
    "Summarize Brown v. Board of Education.",
    "Explain habeas corpus.",
    "What is the standard deduction for 2024?",
)

_CAPABILITIES: tuple[str, ...] = (
    "Natural language legal Q&A",
    "Legal document summarization",
    "Hybrid semantic + keyword retrieval",
    "Source-grounded citations",
    "Official document links",
    "Page-level references",
)


# ─── Welcome Page ─────────────────────────────────────────────────────────


def render_welcome() -> None:
    """Render the full welcome page with capabilities and examples.

    Note: the footer is NOT rendered here — it is rendered once by
    ``app.py`` after all content to avoid duplication.
    """

    # ── Capabilities grid (custom HTML) ──────────────────────────────
    caps_html = "\n".join(
        '<div class="capability-item">'
        '  <span class="capability-dot"></span>'
        f"  <span>{cap}</span>"
        "</div>"
        for cap in _CAPABILITIES
    )

    # All HTML uses <div> instead of <h1>/<p> to avoid Streamlit's
    # internal markdown processing interfering with rendering.
    st.markdown(
        '<div class="welcome-container">'
        '  <div class="welcome-title">LexRAG</div>'
        '  <div class="welcome-subtitle">'
        "    US Tax &amp; Legal Q&amp;A System"
        "  </div>"
        '  <div class="welcome-description">'
        "    Source-grounded legal research powered by"
        "    hybrid retrieval and LLMs."
        "  </div>"
        '  <div class="welcome-section-label">Capabilities</div>'
        '  <div class="capabilities-grid">'
        f"    {caps_html}"
        "  </div>"
        '  <div class="welcome-divider"></div>'
        '  <div class="welcome-section-label">Try an example</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Example buttons (Streamlit for interactivity) ────────────────
    spacer_l, col_a, col_b, spacer_r = st.columns([1.2, 2, 2, 1.2])

    for index, query in enumerate(_EXAMPLE_QUERIES):
        with (col_a if index % 2 == 0 else col_b):
            if st.button(
                query,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state._example_query = query
                st.rerun()


# ─── Response Orchestrator ────────────────────────────────────────────────


def render_response(
    response: RAGResponse,
    *,
    show_pipeline_details: bool,
    show_retrieved_context: bool,
) -> None:
    """Render a complete assistant response.

    Dispatches to the appropriate rendering helpers depending on
    the route decision attached to the response.
    """

    route = response.route_decision.route

    # ── Answer ───────────────────────────────────────────────────────
    st.markdown(response.answer)

    # ── Sources ──────────────────────────────────────────────────────
    if route == Route.LEGAL_RAG and response.citations:
        _render_sources(
            response.citations,
            response.retrieved_chunks,
            show_context=show_retrieved_context,
        )
    elif route == Route.LEGAL_RAG and not response.citations:
        render_empty_state(
            "\u2014",
            "No citations available",
            "The model did not produce source references for "
            "this answer.",
        )

    # ── Pipeline details ─────────────────────────────────────────────
    if show_pipeline_details and route in (
        Route.LEGAL_RAG,
        Route.GENERAL_CHAT,
    ):
        render_pipeline_details(response)


# ─── Sources ──────────────────────────────────────────────────────────────


def _render_sources(
    citations: tuple[Citation, ...],
    chunks: tuple[RetrievedChunk, ...],
    *,
    show_context: bool,
) -> None:
    """Render unified source cards inside a collapsed expander."""

    count = len(citations)
    total_chunks = len(chunks)

    paired: list[tuple[Citation, RetrievedChunk | None]] = []
    for citation in citations:
        matched = _match_citation_to_chunk(citation, chunks)
        paired.append((citation, matched))

    # Preserve backend retrieval order. If citation matching disrupts the ordering,
    # restore the backend-defined ranking before rendering. Do not recompute or 
    # invent ranking in the presentation layer.
    def get_rank(chunk: RetrievedChunk | None) -> int:
        if chunk is None:
            return 99999
        try:
            return chunks.index(chunk)
        except ValueError:
            return 99999

    paired.sort(key=lambda pair: get_rank(pair[1]))

    with st.expander(f"Sources ({count})", expanded=False):
        for citation, chunk in paired:
            chunk_rank = get_rank(chunk) + 1 if chunk else None
            render_source_card(
                citation,
                chunk,
                chunk_rank=chunk_rank if chunk_rank and chunk_rank < 99999 else None,
                total_chunks=total_chunks,
                show_context=show_context,
            )


def _match_citation_to_chunk(
    citation: Citation,
    chunks: tuple[RetrievedChunk, ...],
) -> RetrievedChunk | None:
    """Find the best matching chunk for a citation.

    Matches first by title + page_start, then title + heading,
    and falls back to title alone.
    """

    # 1. Prefer title + page_start
    for chunk in chunks:
        if (
            chunk.title == citation.title
            and citation.page_start
            and chunk.page_start
            and citation.page_start == chunk.page_start
        ):
            return chunk
            
    # 2. Prefer title + heading (Section)
    for chunk in chunks:
        if (
            chunk.title == citation.title
            and citation.heading
            and chunk.heading
            and citation.heading == chunk.heading
        ):
            return chunk

    # 3. Fallback — title only
    for chunk in chunks:
        if chunk.title == citation.title:
            return chunk

    return None
