"""Routing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Route(StrEnum):
    """Supported query routing destinations."""

    LEGAL_RAG = "legal_rag"

    GENERAL_CHAT = "general_chat"

    REJECT = "reject"

    CLARIFY = "clarify"


@dataclass(
    frozen=True,
    slots=True,
)
class RouteDecision:
    """Decision produced by the query router."""

    route: Route

    confidence: float

    reason: str