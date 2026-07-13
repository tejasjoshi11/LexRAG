"""LLM-based query route classifier."""

from __future__ import annotations

import json

from src.contracts.route import (
    Route,
    RouteDecision,
)
from src.routing.routing_prompt import (
    ROUTING_SYSTEM_PROMPT,
)
from src.generation.response_generator import ResponseGenerator
from src.llm.models import GEMINI_FLASH_LITE
from src.routing.route_classifier import RouteClassifier
from src.shared.constants import DEFAULT_LLM_TEMPERATURE


class LLMRouteClassifier(RouteClassifier):
    """Route user queries using a lightweight language model."""

    def __init__(
        self,
    ) -> None:
        """Initialize the classifier."""

        self._generator = ResponseGenerator()

    def classify(
        self,
        *,
        query: str,
    ) -> RouteDecision:
        """Classify a query using an LLM."""

        response = self._generator.generate(
            model=GEMINI_FLASH_LITE,
            system_prompt=ROUTING_SYSTEM_PROMPT,
            user_prompt=query,
            temperature=DEFAULT_LLM_TEMPERATURE,
            max_output_tokens=128,
        )

        return self._parse_response(
            response.text,
        )

    def _parse_response(
        self,
        response: str,
    ) -> RouteDecision:
        """Parse the JSON returned by the routing model."""

        try:
            data = json.loads(response)

            route = Route(
                data["route"].lower(),
            )

            confidence = float(
                data.get(
                    "confidence",
                    0.0,
                )
            )

            reason = data.get(
                "reason",
                "",
            )

            return RouteDecision(
                route=route,
                confidence=confidence,
                reason=reason,
            )

        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ):
            return RouteDecision(
                route=Route.GENERAL_CHAT,
                confidence=0.0,
                reason="Invalid routing response.",
            )