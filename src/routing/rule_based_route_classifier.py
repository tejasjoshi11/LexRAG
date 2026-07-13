"""Rule-based query route classifier."""

from __future__ import annotations

from src.contracts.route import (
    Route,
    RouteDecision,
)
from src.routing.route_classifier import RouteClassifier


class RuleBasedRouteClassifier(RouteClassifier):
    """Classify obvious conversational queries without using an LLM."""

    _GENERAL_CHAT_QUERIES = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "see you",
        "how are you",
        "how are you?",
        "who are you",
        "who are you?",
        "what can you do",
        "what can you do?",
    }

    def classify(
        self,
        *,
        query: str,
    ) -> RouteDecision | None:
        """Classify a query using simple rules.

        Args:
            query:
                User query.

        Returns:
            Route decision if matched, otherwise None.
        """

        normalized_query = self._normalize(
            query,
        )

        if normalized_query in self._GENERAL_CHAT_QUERIES:
            return RouteDecision(
                route=Route.GENERAL_CHAT,
                confidence=1.0,
                reason="Matched predefined conversational query.",
            )

        return None

    def _normalize(
        self,
        query: str,
    ) -> str:
        """Normalize a query for rule matching."""

        return " ".join(
            query.casefold().split(),
        )