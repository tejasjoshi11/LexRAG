"""Abstract interface for Large Language Model providers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.contracts.llm_response import LLMResponse
from src.shared.constants import DEFAULT_LLM_TEMPERATURE


class LLMProvider(ABC):
    """Abstract base class for language model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response from the language model.

        Args:
            system_prompt:
                System instructions.

            user_prompt:
                User prompt.

            temperature:
                Sampling temperature.

            max_output_tokens:
                Maximum number of output tokens.

        Returns:
            Structured language model response.
        """