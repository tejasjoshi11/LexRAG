"""Language model response contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FinishReason(StrEnum):
    """Reason why language model generation finished."""

    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response returned by every language model."""

    text: str

    provider: str
    model: str

    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    total_tokens: int

    latency_ms: float

    finish_reason: FinishReason