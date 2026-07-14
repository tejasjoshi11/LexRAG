from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.contracts.generated_answer import GeneratedAnswerRow


class GeneratedAnswerWriterError(Exception):
    """Raised when generated answers cannot be written."""


class GeneratedAnswerWriter:
    """Writes generated answers to an Excel file."""

    def write(
        self,
        path: Path,
        answers: list[GeneratedAnswerRow],
    ) -> None:
        """
        Write generated answers to an Excel file.

        Args:
            path:
                Output Excel path.

            answers:
                Generated answers.

        Raises:
            GeneratedAnswerWriterError
        """

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            self._validate_answers(answers)

            dataframe = pd.DataFrame(
                [
                    self._to_dict(answer)
                    for answer in answers
                ]
            )

            dataframe.to_excel(path, index=False)

        except Exception as exc:
            raise GeneratedAnswerWriterError(
                f"Failed to write generated answers: {exc}"
            ) from exc

    def read(
        self,
        path: Path,
    ) -> list[GeneratedAnswerRow]:
        """
        Read generated answers from an Excel file.
        """
        if not path.exists():
            return []

        try:
            dataframe = pd.read_excel(path)
            answers = []

            for _, row in dataframe.iterrows():
                answers.append(
                    GeneratedAnswerRow(
                        question_id=str(row.get("Question_ID", "")),
                        generated_answer=str(row.get("Generated_Answer", "")),
                        retrieved_context=json.loads(str(row.get("Retrieved_Context", "[]"))),
                        retrieved_documents=json.loads(str(row.get("Retrieved_Documents", "[]"))),
                        retrieved_pages=json.loads(str(row.get("Retrieved_Pages", "[]"))),
                        provider=str(row.get("Provider", "")),
                        model=str(row.get("Model", "")),
                        latency=float(row.get("Latency_ms", 0.0)),
                        status=str(row.get("Status", "")),
                    )
                )

            return answers

        except Exception as exc:
            raise GeneratedAnswerWriterError(
                f"Failed to read generated answers: {exc}"
            ) from exc

    @staticmethod
    def _validate_answers(
        answers: list[GeneratedAnswerRow],
    ) -> None:

        seen: set[str] = set()

        for answer in answers:

            if answer.question_id in seen:
                raise GeneratedAnswerWriterError(
                    f"Duplicate Question_ID: {answer.question_id}"
                )

            seen.add(answer.question_id)

    @staticmethod
    def _to_dict(
        answer: GeneratedAnswerRow,
    ) -> dict:

        return {
            "Question_ID": answer.question_id,
            "Generated_Answer": answer.generated_answer,
            "Retrieved_Context": json.dumps(
                answer.retrieved_context,
                ensure_ascii=False,
            ),
            "Retrieved_Documents": json.dumps(
                answer.retrieved_documents,
                ensure_ascii=False,
            ),
            "Retrieved_Pages": json.dumps(
                answer.retrieved_pages,
            ),
            "Provider": answer.provider,
            "Model": answer.model,
            "Latency_ms": answer.latency,
            "Status": answer.status,
        }