"""Reusable UI components for the LexRAG Streamlit application.

Atomic, presentation-only components.  Each function renders a
self-contained visual element using the CSS design system defined
in ``.streamlit/styles.css``.  Contains no business logic.

HTML rendering note — Streamlit's ``st.markdown(unsafe_allow_html=True)``
passes HTML through, but certain elements like ``<h1>``–``<h6>`` may
collide with Streamlit's internal markdown processing.  We therefore
avoid bare heading tags in custom HTML and use styled ``<div>`` elements
with explicit CSS classes instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.contracts.citation import Citation
from src.contracts.rag_response import RAGResponse
from src.contracts.retrieved_chunk import RetrievedChunk, RetrievalMethod

_logger = logging.getLogger(__name__)

# ─── Style Injection ──────────────────────────────────────────────────────

_CSS_PATH = Path(__file__).parent / "styles.css"


def inject_styles() -> None:
    """Load the Google Font link and inject the CSS design system."""

    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" '
        'crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@300;400;500;600;700&display=swap" '
        'rel="stylesheet">',
        unsafe_allow_html=True,
    )

    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ─── Sticky Header ───────────────────────────────────────────────────────


def render_sticky_header() -> None:
    """Render the branded header bar."""

    st.markdown(
        '<div class="lexrag-header">'
        '  <span class="lexrag-header-title">LexRAG</span>'
        '  <div class="lexrag-header-sep"></div>'
        '  <span class="lexrag-header-subtitle">'
        "    US Tax &amp; Legal Q&amp;A System"
        "  </span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Initialization Card ─────────────────────────────────────────────────

_INIT_STEPS: tuple[str, ...] = (
    "Loading AI embedding model",
    "Connecting to vector database",
    "Connecting to keyword index",
    "Preparing retrieval pipeline",
    "Finalizing startup",
)


def render_initialization_card() -> None:
    """Render the centred initialization progress card.

    Displays the pipeline startup stages with a single animated
    spinner at the top and step labels below.  Since
    ``create_pipeline()`` is a single blocking call we cannot
    track per-step progress, so the spinner reflects overall
    activity while the step list communicates what happens during
    initialization.
    """

    rows: list[str] = []

    for index, label in enumerate(_INIT_STEPS):
        if index == 0:
            icon = '<span class="spinner-sm"></span>'
            row_cls = "init-step active"
        else:
            icon = '<span class="init-step-icon pending">\u25CB</span>'
            row_cls = "init-step"

        rows.append(
            f'<div class="{row_cls}">'
            f"  {icon}"
            f"  <span>{label}</span>"
            f"</div>"
        )

    st.markdown(
        '<div class="init-overlay">'
        '  <div class="init-card">'
        '    <div class="init-card-title">LexRAG</div>'
        '    <div class="init-card-subtitle">Initializing</div>'
        '    <div class="init-steps">'
        + "\n".join(rows)
        + "    </div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_ready_card() -> None:
    """Render the brief 'LexRAG Ready' confirmation."""

    st.markdown(
        '<div class="ready-overlay">'
        '  <div class="ready-content">'
        '    <div class="ready-icon">\u2713</div>'
        '    <div class="ready-text">LexRAG Ready</div>'
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Processing Indicator ────────────────────────────────────────────────

_PROCESSING_HTML: str = (
    '<div class="processing-indicator">'
    '  <div class="processing-spinner"></div>'
    '  <span class="processing-text">Generating response\u2026</span>'
    "</div>"
)


def render_processing_indicator() -> None:
    """Render the in-chat processing spinner (ChatGPT-style)."""

    st.markdown(_PROCESSING_HTML, unsafe_allow_html=True)


# ─── Source Card ──────────────────────────────────────────────────────────


def render_source_card(
    citation: Citation,
    chunk: RetrievedChunk | None = None,
    chunk_rank: int | None = None,
    total_chunks: int = 0,
    *,
    show_context: bool = False,
) -> None:
    """Render a single source card with optional retrieved context.

    Combines citation metadata and (when enabled) the matched
    retrieved chunk into one cohesive card.
    """

    # ── metadata tags ────────────────────────────────────────────────
    tags: list[str] = []

    if chunk and chunk_rank is not None:
        tags.append(f"Rank #{chunk_rank}")
        if total_chunks > 0:
            tags.append(f"Chunk {chunk_rank} of {total_chunks}")
            
    if chunk:
        method = chunk.retrieval_method
        method_name = method.value.title()
        tags.append(f"{method_name} Retrieval")

        score = chunk.retrieval_score
        if method == RetrievalMethod.SEMANTIC:
            tags.append(f"Similarity {score * 100:.2f}%")
        elif method == RetrievalMethod.KEYWORD:
            tags.append(f"BM25 Score {score:.2f}")
        elif method == RetrievalMethod.HYBRID:
            tags.append(f"Hybrid Score {score:.4f}")
        else:
            tags.append(f"Score {score:.2f}")

    if citation.page_start and citation.page_end:
        if citation.page_start == citation.page_end:
            tags.append(f"Page {citation.page_start}")
        else:
            tags.append(
                f"Pages {citation.page_start}\u2013{citation.page_end}"
            )
    elif citation.page_start:
        tags.append(f"Page {citation.page_start}")

    meta_parts: list[str] = []
    for idx, tag in enumerate(tags):
        is_score = chunk and idx == len(tags) - 1
        cls = "source-card-tag score" if is_score else "source-card-tag"
        meta_parts.append(f'<span class="{cls}">{_esc(tag)}</span>')
        if idx < len(tags) - 1:
            meta_parts.append('<span class="source-card-dot"></span>')

    meta_html = (
        '<div class="source-card-meta">'
        + "".join(meta_parts)
        + "</div>"
    )

    # ── official source link ─────────────────────────────────────────
    link_html = ""
    if citation.source_url:
        url = _esc(citation.source_url)
        link_html = (
            f'<a class="source-card-link" href="{url}" '
            f'target="_blank" rel="noopener">'
            f"\U0001f4cc Official Source</a>"
        )

    # ── retrieved context ────────────────────────────────────────────
    context_html = ""
    if show_context and chunk:
        context_html = (
            '<details class="source-card-details">'
            "  <summary>Retrieved Context</summary>"
            '  <div class="source-card-context">'
            f'    <p class="source-card-context-text">'
            f"      {_esc(chunk.chunk_text)}</p>"
            "  </div>"
            "</details>"
        )

    st.markdown(
        f'<div class="source-card">'
        f'  <div class="source-card-title">{_esc(citation.title)}</div>'
        f"  {meta_html}"
        f"  {link_html}"
        f"  {context_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─── Empty State ──────────────────────────────────────────────────────────


def render_empty_state(
    icon: str,
    title: str,
    subtitle: str = "",
) -> None:
    """Render a professional empty-state card."""

    sub = (
        f'<div class="empty-state-subtitle">{_esc(subtitle)}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f'<div class="empty-state">'
        f'  <div class="empty-state-icon">{_esc(icon)}</div>'
        f'  <div class="empty-state-title">{_esc(title)}</div>'
        f"  {sub}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─── Error Messages ──────────────────────────────────────────────────────

_ERROR_PATTERNS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("connection", "timeout", "unreachable", "refused"),
        "Service temporarily unavailable.",
        "Check your network connection or try again in a moment.",
    ),
    (
        ("quota", "rate limit", "rate_limit", "429"),
        "API quota exceeded.",
        "Please wait a moment and try again.",
    ),
    (
        ("model", "unavailable", "not found", "does not exist"),
        "Selected language model is unavailable.",
        "Choose a different model from the sidebar.",
    ),
    (
        ("elasticsearch", "elastic"),
        "Keyword search service could not be reached.",
        "The search backend may be temporarily down.",
    ),
    (
        ("qdrant",),
        "Vector database could not be reached.",
        "The vector search backend may be temporarily down.",
    ),
    (
        ("api key", "api_key", "authentication", "unauthorized",
         "401", "403"),
        "Authentication error.",
        "Check your API key configuration.",
    ),
    (
        ("embedding",),
        "Embedding model encountered an error.",
        "The AI embedding service may be temporarily unavailable.",
    ),
)


def map_error_to_message(
    exc: Exception,
) -> tuple[str, str, str]:
    """Map an exception to ``(title, reason, suggestion)``."""

    combined = f"{type(exc).__name__} {exc}".lower()

    for keywords, reason, suggestion in _ERROR_PATTERNS:
        if any(kw in combined for kw in keywords):
            return "Unable to generate a response.", reason, suggestion

    return (
        "Unable to generate a response.",
        "An unexpected error occurred.",
        "Please try again or choose a different query.",
    )


def render_error_message(exc: Exception) -> None:
    """Render a human-friendly error card from a live exception."""

    title, reason, suggestion = map_error_to_message(exc)
    _render_error_parts(title, reason, suggestion)


def render_error_from_parts(
    title: str,
    reason: str,
    suggestion: str,
) -> None:
    """Render a human-friendly error card from stored parts."""

    _render_error_parts(title, reason, suggestion)


def _render_error_parts(
    title: str,
    reason: str,
    suggestion: str,
) -> None:
    """Shared rendering for error cards."""

    st.markdown(
        f'<div class="error-card">'
        f'  <div class="error-card-title">{_esc(title)}</div>'
        f'  <div class="error-card-row">'
        f'    <span class="error-card-label">Reason:</span>'
        f'    <span class="error-card-value">{_esc(reason)}</span>'
        f"  </div>"
        f'  <div class="error-card-row">'
        f'    <span class="error-card-label">Suggestion:</span>'
        f'    <span class="error-card-value">{_esc(suggestion)}</span>'
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─── Pipeline Details ────────────────────────────────────────────────────


def render_pipeline_details(response: RAGResponse) -> None:
    """Render pipeline metrics in a collapsible section.

    Displays only data returned by the backend — nothing is
    calculated or fabricated by the frontend.
    """

    rd = response.route_decision
    llm = response.llm_response

    with st.expander("Pipeline Details", expanded=False):
        st.markdown(
            f'<div class="pipeline-grid">'
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">Route</span>'
            f'    <span class="pipeline-metric-value">'
            f"      {rd.route.value}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">Confidence</span>'
            f'    <span class="pipeline-metric-value">'
            f"      {rd.confidence:.2f}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">Model</span>'
            f'    <span class="pipeline-metric-value">'
            f"      {_esc(llm.model)}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">'
            f"      Generation Latency</span>"
            f'    <span class="pipeline-metric-value">'
            f"      {llm.latency_ms:.0f} ms</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">'
            f"      Prompt Tokens</span>"
            f'    <span class="pipeline-metric-value">'
            f"      {llm.prompt_tokens:,}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">'
            f"      Completion Tokens</span>"
            f'    <span class="pipeline-metric-value">'
            f"      {llm.completion_tokens:,}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">'
            f"      Total Tokens</span>"
            f'    <span class="pipeline-metric-value">'
            f"      {llm.total_tokens:,}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">'
            f"      Retrieved Chunks</span>"
            f'    <span class="pipeline-metric-value">'
            f"      {len(response.retrieved_chunks)}</span>"
            f"  </div>"
            f'  <div class="pipeline-metric">'
            f'    <span class="pipeline-metric-label">'
            f"      Finish Reason</span>"
            f'    <span class="pipeline-metric-value">'
            f"      {llm.finish_reason.value}</span>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─── Footer ──────────────────────────────────────────────────────────────


def render_footer() -> None:
    """Render the professional footer."""

    st.markdown(
        '<div class="lexrag-footer">'
        '  <div class="lexrag-footer-brand">LexRAG</div>'
        '  <div class="lexrag-footer-tags">'
        '    <span class="lexrag-footer-tag">Hybrid Retrieval</span>'
        '    <span class="lexrag-footer-tag">Semantic Search</span>'
        '    <span class="lexrag-footer-tag">Keyword Search</span>'
        '    <span class="lexrag-footer-tag">Citation Grounding</span>'
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Utilities ────────────────────────────────────────────────────────────


def _esc(text: str) -> str:
    """Escape HTML special characters in user-facing text."""

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
