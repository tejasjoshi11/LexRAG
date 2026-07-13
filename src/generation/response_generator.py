"""High-level language model response generator."""

from __future__ import annotations

from src.contracts.llm_response import LLMResponse
from src.llm.models import Model
from src.llm.registry import ProviderRegistry
from src.shared.constants import DEFAULT_LLM_TEMPERATURE
import logging

_LOGGER = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses using a selected language model."""

    def __init__(self) -> None:
        """Initialize the response generator."""

        self._registry = ProviderRegistry()

    def generate(
        self,
        *,
        model: Model,
        system_prompt: str,
        user_prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response.

        Args:
            model:
                Language model to use.

            system_prompt:
                System instructions.

            user_prompt:
                User prompt.

            temperature:
                Sampling temperature.

            max_output_tokens:
                Maximum response length.

        Returns:
            Structured language model response.
        """

        provider = self._registry.create(model)

        _LOGGER.info(f"Generating response using {provider.provider_name}...")

        return provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )