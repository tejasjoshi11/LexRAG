"""Supported language models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    GROQ = "groq"
    NVIDIA = "nvidia"


@dataclass(frozen=True, slots=True)
class Model:
    """Language model metadata."""

    id: str
    display_name: str

    provider: Provider


GEMINI_FLASH = Model(
    id="gemini-3.1-flash",
    display_name="Gemini 3.1 Flash",
    provider=Provider.GEMINI,
)

GEMINI_FLASH_LITE = Model(
    id="gemini-3.1-flash-lite",
    display_name="Gemini 3.1 Flash Lite",
    provider=Provider.GEMINI,
)

GROQ_QWEN_32B = Model(
    id="qwen/qwen3-32b",
    display_name="Qwen 3 32B",
    provider=Provider.GROQ,
)

NVIDIA_NEMOTRON = Model(
    id="nvidia/llama-3.3-nemotron-super-49b-v1",
    display_name="Llama 3.3 Nemotron Super 49B",
    provider=Provider.NVIDIA,
)

SUPPORTED_MODELS: tuple[Model, ...] = (
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GROQ_QWEN_32B,
    NVIDIA_NEMOTRON,
)