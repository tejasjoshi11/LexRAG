"""Query router."""

from __future__ import annotations

from src.contracts.route import RouteDecision
from src.routing.llm_route_classifier import (
    LLMRouteClassifier,
)
from src.routing.rule_based_route_classifier import (
    RuleBasedRouteClassifier,
)


class Router:
    """Routes user queries."""

    def __init__(
        self,
    ) -> None:
        """Initialize the router."""

        self._rule_based_classifier = (
            RuleBasedRouteClassifier()
        )

        self._llm_classifier = (
            LLMRouteClassifier()
        )

    def route(
        self,
        *,
        query: str,
    ) -> RouteDecision:
        """Route a user query."""

        decision = (
            self._rule_based_classifier.classify(
                query=query,
            )
        )

        if decision is not None:
            return decision

        return self._llm_classifier.classify(
            query=query,
        )