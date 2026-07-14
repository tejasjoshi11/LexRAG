"""LexRAG — Streamlit Application.

Production-quality Legal Retrieval-Augmented Generation frontend.
This module is a thin presentation layer over the existing RAGPipeline.
It handles page setup, sidebar controls, session state, the chat loop,
and delegates all rendering to ui_helpers and ui_components.

Architecture note — Streamlit re-executes this script top-to-bottom on
every interaction.  The browser receives the rendered page only AFTER
the script finishes (or calls ``st.stop()``).  Long-running work like
model loading must therefore be split across reruns using session state
so the browser can render intermediate UI before the heavy work begins.

State machine:

    INIT_SHOW  →  Render init card + st.rerun()
                  (browser receives init card immediately)
    INIT_LOAD  →  Backend loads while browser shows init card
                  On success → st.rerun() to RUNNING
    RUNNING    →  Normal operation (welcome / chat / etc.)
"""

from __future__ import annotations

import logging

import streamlit as st

from ui_components import (
    inject_styles,
    map_error_to_message,
    render_error_from_parts,
    render_error_message,
    render_footer,
    render_initialization_card,
    render_sticky_header,
)

_logger = logging.getLogger(__name__)

# ─── Application States ──────────────────────────────────────────────────

_STATE_INIT_SHOW = "init_show"    # Render init card, then rerun
_STATE_INIT_LOAD = "init_load"    # Load pipeline (browser shows init card)
_STATE_RUNNING   = "running"      # Normal operation


# ─── Page Configuration ──────────────────────────────────────────────────

st.set_page_config(
    page_title="LexRAG \u00b7 US Tax & Legal Q&A",
    page_icon="\u2696",
    layout="centered",
)


# ─── Instant Shell ────────────────────────────────────────────────────────
# CSS + header render instantly — no heavy imports triggered yet.

inject_styles()
render_sticky_header()


# ─── State Machine Bootstrap ─────────────────────────────────────────────

if "_app_state" not in st.session_state:
    st.session_state._app_state = _STATE_INIT_SHOW


# ═══════════════════════════════════════════════════════════════════════════
# STATE: INIT_SHOW
#
# Purpose: render the initialization card and immediately rerun so the
# browser actually displays it.  Without this split, the browser would
# show nothing until the heavy INIT_LOAD finishes.
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state._app_state == _STATE_INIT_SHOW:
    render_initialization_card()
    st.session_state._app_state = _STATE_INIT_LOAD
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STATE: INIT_LOAD
#
# The browser is now showing the init card from INIT_SHOW.  Streamlit
# keeps the previous page visible while a new run is in progress, so the
# user sees the init card throughout this entire block — even though it
# takes minutes for imports + model loading.
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state._app_state == _STATE_INIT_LOAD:

    # Re-render the init card so if Streamlit briefly shows THIS run's
    # output, the card is still there (not a blank page).
    render_initialization_card()

    # ── Deferred imports ─────────────────────────────────────────────
    # src.main transitively imports torch, sentence-transformers, etc.
    # This is the heaviest part of startup.
    try:
        from src.main import create_pipeline, get_implemented_models
    except Exception as exc:
        _logger.exception("Failed to import backend modules.")
        render_error_message(exc)
        st.stop()

    # ── Pipeline construction (cached) ───────────────────────────────
    @st.cache_resource
    def _load_pipeline():
        """Construct and cache the RAGPipeline."""
        return create_pipeline()

    @st.cache_resource
    def _load_available_models():
        """Discover and cache available language models."""
        return get_implemented_models()

    try:
        _load_pipeline()
        _load_available_models()
    except Exception as exc:
        _logger.exception("Pipeline initialization failed.")
        render_error_message(exc)
        st.stop()

    # ── Transition ───────────────────────────────────────────────────
    st.session_state._app_state = _STATE_RUNNING
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STATE: RUNNING — Normal application.
# ═══════════════════════════════════════════════════════════════════════════

# ── Deferred imports (instant now — Python modules already cached) ────────

from src.main import create_pipeline, get_implemented_models  # noqa: E402
from src.shared.constants import DEFAULT_TOP_K  # noqa: E402
from ui_helpers import render_response, render_welcome  # noqa: E402


# ── Cached resources (instant — already populated during INIT_LOAD) ──────

@st.cache_resource
def _load_pipeline():
    """Construct and cache the RAGPipeline."""
    return create_pipeline()


@st.cache_resource
def _load_available_models():
    """Discover and cache available language models."""
    return get_implemented_models()


try:
    pipeline = _load_pipeline()
    available_models = _load_available_models()
except Exception as exc:
    _logger.exception("Pipeline unavailable.")
    render_error_message(exc)
    st.stop()


# ── Show a brief "ready" toast on first transition to RUNNING ────────────

if "_showed_ready" not in st.session_state:
    st.session_state._showed_ready = True
    st.toast("\u2713 LexRAG Ready", icon="\u2696")


# ─── Session State ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(
        '<div class="sidebar-section-label">Generation Model</div>',
        unsafe_allow_html=True,
    )

    selected_model = st.selectbox(
        "Model",
        options=available_models,
        format_func=lambda m: m.display_name,
        key="model_selection",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-section-label">Retrieval</div>',
        unsafe_allow_html=True,
    )

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        key="top_k",
    )

    with st.expander("Advanced Settings"):

        show_pipeline_details = st.checkbox(
            "Show Pipeline Details",
            value=False,
            key="show_pipeline_details",
        )

        show_retrieved_context = st.checkbox(
            "Show Retrieved Context",
            value=False,
            key="show_retrieved_context",
        )

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── Chat Input ──────────────────────────────────────────────────────────

prompt = st.chat_input("Ask a legal question\u2026")

# Pick up any pending example query from the welcome section.
if "_example_query" in st.session_state:
    prompt = st.session_state.pop("_example_query")


# ─── Content ─────────────────────────────────────────────────────────────

if not st.session_state.messages and not prompt:

    render_welcome()

else:

    # ── Replay chat history ──────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message.get("error"):
                render_error_from_parts(
                    message["_error_title"],
                    message["_error_reason"],
                    message["_error_suggestion"],
                )
            else:
                render_response(
                    message["response"],
                    show_pipeline_details=show_pipeline_details,
                    show_retrieved_context=show_retrieved_context,
                )

    # ── Process new query ────────────────────────────────────────────
    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            _status = st.empty()
            _status.markdown(
                '<div class="processing-indicator">'
                '  <div class="processing-spinner"></div>'
                '  <span class="processing-text">'
                "    Generating response\u2026"
                "  </span>"
                "</div>",
                unsafe_allow_html=True,
            )

            try:
                response = pipeline.run(
                    query=prompt,
                    top_k=top_k,
                    model=selected_model,
                )

                _status.empty()

                render_response(
                    response,
                    show_pipeline_details=show_pipeline_details,
                    show_retrieved_context=show_retrieved_context,
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "response": response,
                })

            except Exception as exc:
                _status.empty()

                # Full traceback in terminal for debugging.
                _logger.exception(
                    "Query processing failed: %s", prompt
                )

                # User-friendly message in the UI.
                render_error_message(exc)

                # Store pre-mapped parts for session replay.
                title, reason, suggestion = map_error_to_message(exc)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(exc),
                    "error": True,
                    "_error_title": title,
                    "_error_reason": reason,
                    "_error_suggestion": suggestion,
                })

# ─── Footer (rendered exactly once, after all content) ────────────────────

render_footer()
