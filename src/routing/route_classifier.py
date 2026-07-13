"""Abstract interface for query route classifiers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from src.contracts.route import RouteDecision


class RouteClassifier(ABC):
    """Abstract base class for query route classifiers."""

    @abstractmethod
    def classify(
        self,
        *,
        query: str,
    ) -> RouteDecision:
        """Classify a user query.

        Args:
            query:
                User query.

        Returns:
            Routing decision.
        """