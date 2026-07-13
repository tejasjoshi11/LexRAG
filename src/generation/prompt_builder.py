"""Build prompts for language model generation."""

from __future__ import annotations

from src.contracts.prompt import Prompt
from src.contracts.retrieved_chunk import RetrievedChunk
from src.generation.prompts.legal_rag_prompt import (
    LEGAL_RAG_SYSTEM_PROMPT,
)


class PromptBuilder:
    """Build prompts for language model generation."""

    def build(
        self,
        *,
        user_query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> Prompt:
        """Build the prompt for the language model.

        Args:
            user_query:
                User's legal question.

            retrieved_chunks:
                Retrieved legal chunks.

        Returns:
            Complete prompt.
        """

        system_prompt = self._build_system_prompt()

        context = self._build_context(
            retrieved_chunks,
        )

        user_prompt = self._build_user_prompt(
            user_query=user_query,
            context=context,
        )

        return Prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_system_prompt(
        self,
    ) -> str:
        """Build the system prompt."""

        return LEGAL_RAG_SYSTEM_PROMPT

    def _build_context(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """Build the retrieved context section."""

        if not retrieved_chunks:
            return "No legal documents were retrieved."

        separator = "\n" + ("-" * 80) + "\n"

        return separator.join(
            self._format_chunk(
                index=index,
                chunk=chunk,
            )
            for index, chunk in enumerate(
                retrieved_chunks,
                start=1,
            )
        )

    def _format_chunk(
        self,
        *,
        index: int,
        chunk: RetrievedChunk,
    ) -> str:
        """Format a retrieved chunk for the prompt."""

        return f"""Retrieved Document {index}

Title:
{chunk.title}

Document ID:
{chunk.document_id}

Heading:
{chunk.heading}

Pages:
{chunk.page_start}-{chunk.page_end}

Source:
{chunk.source_url}

Content:
{chunk.chunk_text}
"""

    def _build_user_prompt(
        self,
        *,
        user_query: str,
        context: str,
    ) -> str:
        """Build the user prompt."""

        return f"""# User Question

{user_query}

# Retrieved Legal Documents

{context}

# Instructions

Answer the user's question using ONLY the retrieved legal documents above.

If the retrieved documents do not contain sufficient information to answer the question, explicitly state that the available documents are insufficient.

Do not rely on outside knowledge.

Provide a clear, well-structured, and concise answer.
"""