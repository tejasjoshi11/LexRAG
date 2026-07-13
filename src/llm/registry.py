"""Language model provider registry."""

from __future__ import annotations

from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.models import (
    Model,
    Provider,
)
from src.llm.provider import LLMProvider


class ProviderRegistry:
    """Factory for creating language model providers."""

    def supports(self, model: Model) -> bool:
        """Return True if the model's provider is implemented."""
        return model.provider in {Provider.GEMINI, Provider.GROQ}

    def create(
        self,
        model: Model,
    ) -> LLMProvider:

        match model.provider:

            case Provider.GEMINI:
                return GeminiProvider(
                    model=model,
                )

            case Provider.GROQ:
                return GroqProvider(
                    model=model,
                )

            case Provider.NVIDIA:
                raise NotImplementedError(
                    "NVIDIA provider not implemented."
                )

            case _:
                raise ValueError(
                    f"Unsupported provider: {model.provider}"
                )