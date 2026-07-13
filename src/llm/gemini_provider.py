"""Google Gemini language model provider."""

from __future__ import annotations

import time
import logging

from google import genai
from google.genai.types import GenerateContentConfig

from src.contracts.llm_response import (
    FinishReason,
    LLMResponse,
)
from src.llm.models import Model
from src.llm.provider import LLMProvider
from src.shared.config import gemini_api_key
from src.shared.constants import DEFAULT_LLM_TEMPERATURE
from src.shared.exceptions import LLMError

_LOGGER = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):

    def __init__(
        self,
        model: Model,
    ) -> None:
        """Initialize the Gemini provider."""

        self._model = model

        try:
            self._client = genai.Client(
                api_key=gemini_api_key(),
            )
        except Exception as exc:
            _LOGGER.exception("Failed to initialize the Gemini client.")
            raise LLMError(
                "The language model client could not be initialized."
            ) from exc

        _LOGGER.info(f"Initialized Gemini provider for {model.id}.")

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"

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
        """Generate a response from Gemini."""

        start_time = time.perf_counter()

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
        except Exception as exc:
            _LOGGER.exception("Gemini generation failed.")
            raise LLMError(
                "The language model is temporarily unavailable."
            ) from exc

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        _LOGGER.info(
            f"Generated response from Gemini in {latency_ms:.0f}ms."
        )

        try:
            return self._build_response(
                response=response,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            _LOGGER.exception("Failed to parse Gemini response.")
            raise LLMError(
                "The language model returned an invalid response."
            ) from exc

    def _build_response(
        self,
        *,
        response,
        latency_ms: float,
    ) -> LLMResponse:
        """Convert a Gemini response into an LLMResponse."""

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        prompt_tokens = (
            usage.prompt_token_count
            if usage is not None
            else 0
        )

        completion_tokens = (
            usage.candidates_token_count
            if usage is not None
            else 0
        )

        reasoning_tokens = (
            getattr(
                usage,
                "thoughts_token_count",
                None,
            )
            if usage is not None
            else None
        )

        total_tokens = (
            usage.total_token_count
            if usage is not None
            else prompt_tokens + completion_tokens
        )

        finish_reason = FinishReason.OTHER

        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if candidates:
            raw_finish_reason = getattr(
                candidates[0],
                "finish_reason",
                None,
            )

            if raw_finish_reason is not None:
                try:
                    finish_reason = FinishReason(
                        raw_finish_reason.value,
                    )
                except ValueError:
                    finish_reason = FinishReason.OTHER

        return LLMResponse(
            text=response.text or "",
            provider=self.provider_name,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )