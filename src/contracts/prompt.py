"""Language model prompt contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    """Complete prompt sent to a language model."""

    system_prompt: str

    user_prompt: str