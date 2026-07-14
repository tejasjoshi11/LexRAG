from __future__ import annotations

from pathlib import Path

from src.contracts.evaluation_pair import EvaluationPair
from src.contracts.evaluation_result import EvaluationResult
from src.contracts.runner_config import RunnerConfig
from src.evaluation.deepeval_evaluator import DeepEvalEvaluator
from src.evaluation.evaluation_writer import EvaluationWriter


class EvaluationRunner:
    """Evaluates EvaluationPairs using DeepEval evaluator."""

    def __init__(
        self,
        evaluator: DeepEvalEvaluator,
        writer: EvaluationWriter,
        output_path: Path,
        config: RunnerConfig,
    ) -> None:
        self._evaluator = evaluator
        self._writer = writer
        self._output_path = output_path
        self._config = config

    def run(
        self,
        evaluation_pairs: list[EvaluationPair],
        existing_results: list[EvaluationResult] | None = None,
    ) -> list[EvaluationResult]:

        evaluation_results: dict[str, EvaluationResult] = {
            res.question_id: res for res in (existing_results or [])
        }
        completed_ids = {
            q_id
            for q_id, res in evaluation_results.items()
            if res.status == "SUCCESS"
        }
        consecutive_failures = 0

        print("Evaluating...")

        for pair in evaluation_pairs:
            if pair.question_id in completed_ids:
                continue

            print(f"Question {pair.question_id}")
            result = self._evaluator.evaluate(pair)

            evaluation_results[result.question_id] = result
            if result.status == "SUCCESS":
                completed_ids.add(result.question_id)
                print("SUCCESS")

            if len(evaluation_results) % self._config.checkpoint_interval == 0:
                try:
                    self._writer.write(
                        self._output_path,
                        list(evaluation_results.values()),
                    )
                    print("Checkpoint saved.")
                except Exception as exc:
                    print(f"Checkpoint save failed: {exc}")

            if result.status == "FAILED":
                consecutive_failures += 1

                print(f"FAILED")

                if consecutive_failures >= self._config.max_consecutive_failures:
                    try:
                        self._writer.write(
                            self._output_path,
                            list(evaluation_results.values()),
                        )
                    except Exception as exc:
                        print(f"Final checkpoint save failed: {exc}")

                    print(f"Stopping evaluation after {self._config.max_consecutive_failures} consecutive failures.")
                    break
            else:
                consecutive_failures = 0

        return list(evaluation_results.values())
