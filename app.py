"""LexRAG — Streamlit Application.

Production-inspired Legal Retrieval-Augmented Generation frontend.
This module is a thin presentation layer over the existing RAGPipeline.
It handles page setup, sidebar controls, session state, the chat loop,
and delegates all rendering to ui_helpers.
"""

import streamlit as st

from src.main import create_pipeline, get_implemented_models
from src.shared.constants import DEFAULT_TOP_K
from ui_helpers import render_response, render_welcome


# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="LexRAG",
    layout="centered",
)


# ─── Cached Resources ────────────────────────────────────────────────────────


@st.cache_resource
def _load_pipeline():
    """Construct and cache the RAGPipeline (runs once per process)."""
    return create_pipeline()


@st.cache_resource
def _load_available_models():
    """Discover and cache the list of available language models."""
    return get_implemented_models()


# ─── Pipeline Initialization ─────────────────────────────────────────────────

try:
    pipeline = _load_pipeline()
except Exception as exc:
    st.error(f"Failed to initialize the LexRAG pipeline: {exc}")
    st.stop()

available_models = _load_available_models()


# ─── Session State ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:

    st.header("Model Selection")

    selected_model = st.selectbox(
        "Model",
        options=available_models,
        format_func=lambda m: m.display_name,
        key="model_selection",
        label_visibility="collapsed",
    )

    with st.expander("Advanced Settings"):

        top_k = st.slider(
            "Number of Retrieved Chunks",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            key="top_k",
        )

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


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("LexRAG")
st.caption(
    "Production-Inspired Legal Retrieval-Augmented Generation System"
)


# ─── Chat Input (pinned to page bottom by Streamlit) ─────────────────────────

prompt = st.chat_input("Ask a legal question...")

# Pick up any pending example query from the welcome section.
if "_example_query" in st.session_state:
    prompt = st.session_state.pop("_example_query")


# ─── Content ─────────────────────────────────────────────────────────────────

if not st.session_state.messages and not prompt:

    # ── Welcome section (first visit, no pending query) ──────────────────
    render_welcome()

else:

    # ── Replay chat history ──────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message.get("error"):
                st.error(message["content"])
            else:
                render_response(
                    message["response"],
                    show_pipeline_details=show_pipeline_details,
                    show_retrieved_context=show_retrieved_context,
                )

    # ── Process new query ────────────────────────────────────────────────
    if prompt:

        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.status(
                    "Processing your request...",
                ) as status:
                    response = pipeline.run(
                        query=prompt,
                        top_k=top_k,
                        model=selected_model,
                    )
                    status.update(
                        label="Completed",
                        state="complete",
                    )

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
                st.error(
                    "An error occurred while processing your "
                    f"request: {exc}"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(exc),
                    "error": True,
                })
