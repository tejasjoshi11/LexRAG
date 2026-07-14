from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.contracts.evaluation_result import EvaluationResult


class EvaluationWriterError(Exception):
    """Raised when evaluation results cannot be written or read."""


class EvaluationWriter:
    """Reads and writes evaluation results to an Excel file."""

    def write(
        self,
        path: Path,
        results: list[EvaluationResult],
    ) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_results(results)

            dataframe = pd.DataFrame(
                [self._to_dict(res) for res in results]
            )

            dataframe.to_excel(path, index=False)
        except Exception as exc:
            raise EvaluationWriterError(
                f"Failed to write evaluation results: {exc}"
            ) from exc

    def read(
        self,
        path: Path,
    ) -> list[EvaluationResult]:
        if not path.exists():
            return []

        try:
            dataframe = pd.read_excel(path)
            results = []

            for _, row in dataframe.iterrows():
                f_score_raw = row.get("Faithfulness_Score")
                f_score = None if pd.isna(f_score_raw) or f_score_raw == "" else float(f_score_raw)

                results.append(
                    EvaluationResult(
                        question_id=str(row.get("Question_ID", "")),
                        retrieval_accuracy=float(row.get("Retrieval_Accuracy", 0.0)),
                        faithfulness_score=f_score,
                        faithfulness_reason=str(row.get("Faithfulness_Reason", "")),
                        faithfulness_passed=bool(row.get("Faithfulness_Passed", False)),
                        latency=float(row.get("Latency_ms", 0.0)),
                        provider=str(row.get("Provider", "")),
                        model=str(row.get("Model", "")),
                        status=str(row.get("Status", "")),
                        error_message=str(row.get("Error_Message", "")),
                    )
                )

            return results
        except Exception as exc:
            raise EvaluationWriterError(
                f"Failed to read evaluation results: {exc}"
            ) from exc

    @staticmethod
    def _validate_results(
        results: list[EvaluationResult],
    ) -> None:
        seen: set[str] = set()
        for res in results:
            if res.question_id in seen:
                raise EvaluationWriterError(
                    f"Duplicate Question_ID: {res.question_id}"
                )
            seen.add(res.question_id)

    @staticmethod
    def _to_dict(
        res: EvaluationResult,
    ) -> dict:
        return {
            "Question_ID": res.question_id,
            "Retrieval_Accuracy": res.retrieval_accuracy,
            "Faithfulness_Score": res.faithfulness_score,
            "Faithfulness_Reason": res.faithfulness_reason,
            "Faithfulness_Passed": res.faithfulness_passed,
            "Latency_ms": res.latency,
            "Provider": res.provider,
            "Model": res.model,
            "Status": res.status,
            "Error_Message": res.error_message,
        }
