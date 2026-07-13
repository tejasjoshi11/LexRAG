"""Groq language model provider."""

from __future__ import annotations

import re
import time
import logging

from groq import Groq

from src.contracts.llm_response import (
    FinishReason,
    LLMResponse,
)
from src.llm.models import Model
from src.llm.provider import LLMProvider
from src.shared.config import groq_api_key
from src.shared.constants import DEFAULT_LLM_TEMPERATURE
from src.shared.exceptions import LLMError

_LOGGER = logging.getLogger(__name__)

class GroqProvider(LLMProvider):
    """Groq implementation."""

    def __init__(
        self,
        model: Model,
    ) -> None:
        """Initialize the Groq provider."""

        self._model = model

        try:
            self._client = Groq(
                api_key=groq_api_key(),
            )
        except Exception as exc:
            _LOGGER.exception("Failed to initialize the Groq client.")
            raise LLMError(
                "The language model client could not be initialized."
            ) from exc

        _LOGGER.info(f"Initialized Groq provider for {model.id}.")

    @property
    def provider_name(self) -> str:
        """Return the provider name."""

        return "groq"

    @property
    def model_name(self) -> str:
        """Return the model identifier."""

        return self._model.id

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response from Groq."""

        start_time = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
            )
        except Exception as exc:
            _LOGGER.exception("Groq generation failed.")
            raise LLMError(
                "The language model is temporarily unavailable."
            ) from exc

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        _LOGGER.info(
            f"Generated response from Groq in {latency_ms:.0f}ms."
        )

        try:
            text = ""

            if response.choices:
                text = (
                    response.choices[0].message.content
                    or ""
                )

            text = self._remove_reasoning(
                text,
            )

            usage = response.usage

            prompt_tokens = (
                usage.prompt_tokens
                if usage is not None
                else 0
            )

            completion_tokens = (
                usage.completion_tokens
                if usage is not None
                else 0
            )

            total_tokens = (
                usage.total_tokens
                if usage is not None
                else prompt_tokens + completion_tokens
            )

            reasoning_tokens = getattr(
                usage,
                "reasoning_tokens",
                0,
            )

            finish_reason = FinishReason.OTHER

            if response.choices:
                finish_reason = self._map_finish_reason(
                    response.choices[0].finish_reason,
                )

            return LLMResponse(
                text=text,
                provider=self.provider_name,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )
        except Exception as exc:
            _LOGGER.exception("Failed to parse Groq response.")
            raise LLMError(
                "The language model returned an invalid response."
            ) from exc


    def _remove_reasoning(
        self,
        text: str,
    ) -> str:
        """Remove reasoning blocks from model output."""

        return re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

    def _map_finish_reason(
        self,
        reason: str | None,
    ) -> FinishReason:
        """Map Groq finish reason to the common contract."""

        match reason:
            case "stop":
                return FinishReason.STOP

            case "length":
                return FinishReason.MAX_TOKENS

            case "content_filter":
                return FinishReason.SAFETY

            case _:
                return FinishReason.OTHER