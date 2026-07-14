from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.contracts.evaluation_result import EvaluationResult


class SummaryGenerator:
    """Generates a JSON summary from evaluation results."""

    def generate(
        self,
        results: list[EvaluationResult],
        output_path: Path,
    ) -> None:
        
        total = len(results)
        if total == 0:
            summary = {"message": "No results to summarize."}
            self._write_json(output_path, summary)
            return

        successes = [r for r in results if r.status == "SUCCESS"]
        failed = [r for r in results if r.status == "FAILED"]

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "questions_total": total,
            "questions_success": len(successes),
            "questions_failed": len(failed),
            "metrics": {
                "retrieval_accuracy": self._compute_stats([r.retrieval_accuracy for r in successes]),
                "faithfulness": self._compute_stats([r.faithfulness_score for r in successes]),
                "latency_ms": self._compute_stats([r.latency for r in successes]),
            }
        }

        self._write_json(output_path, summary)

    def _compute_stats(self, values: list[float]) -> dict:
        if not values:
            return {
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "std_deviation": 0.0,
            }

        return {
            "average": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "std_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
