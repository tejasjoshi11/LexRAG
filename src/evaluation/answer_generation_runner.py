from __future__ import annotations

from pathlib import Path

from src.contracts.rag_response import RAGResponse
from src.contracts.generated_answer import GeneratedAnswerRow
from src.contracts.golden_set import GoldenSetRow
from src.contracts.runner_config import RunnerConfig
from src.evaluation.generated_answer_writer import GeneratedAnswerWriter

from src.llm.models import Model
from src.rag_pipeline.pipeline import RAGPipeline


class AnswerGenerationRunner:
    """Generates answers for a Golden Set using the production RAG pipeline."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        model: Model,
        top_k: int,
        writer: GeneratedAnswerWriter,
        output_path: Path,
        config: RunnerConfig,
    ) -> None:
        self._pipeline = pipeline
        self._model = model
        self._top_k = top_k
        self._writer = writer
        self._output_path = output_path
        self._config = config

    def _run_question(
        self,
        question: GoldenSetRow,
    ) -> GeneratedAnswerRow:

        try:
            response = self._pipeline.run(
                query=question.question,
                top_k=self._top_k,
                model=self._model,
            )

            return self._to_generated_answer(
                question=question,
                response=response,
            )

        except Exception as exc:
            print(f"Question {question.question_id} failed: {exc}")
            return self._failed_answer(question)
        
    def run(
        self,
        golden_set: list[GoldenSetRow],
        existing_answers: list[GeneratedAnswerRow] | None = None,
    ) -> list[GeneratedAnswerRow]:

        answers_dict: dict[str, GeneratedAnswerRow] = {
            ans.question_id: ans for ans in (existing_answers or [])
        }
        completed_ids = {
            q_id
            for q_id, ans in answers_dict.items()
            if ans.status == "SUCCESS"
        }
        consecutive_failures = 0

        print(f"Processing {len(golden_set)} questions.")

        for row in golden_set:
            if row.question_id in completed_ids:
                continue

            answer = self._run_question(row)

            answers_dict[answer.question_id] = answer
            if answer.status == "SUCCESS":
                completed_ids.add(answer.question_id)

            if len(answers_dict) % self._config.checkpoint_interval == 0:
                try:
                    self._writer.write(
                        self._output_path,
                        list(answers_dict.values()),
                    )
                except Exception as exc:
                    print(f"Checkpoint save failed: {exc}")

            if answer.status == "FAILED":
                consecutive_failures += 1

                print(
                    f"Question {row.question_id} failed "
                    f"({consecutive_failures}/{self._config.max_consecutive_failures})"
                )

                if consecutive_failures >= self._config.max_consecutive_failures:

                    try:
                        self._writer.write(
                            self._output_path,
                            list(answers_dict.values()),
                        )
                    except Exception as exc:
                        print(f"Final checkpoint save failed: {exc}")

                    print(f"Stopping evaluation after {self._config.max_consecutive_failures} consecutive failures.")
                    break

            else:
                consecutive_failures = 0

        return list(answers_dict.values())


    def _to_generated_answer(
        self,
        question: GoldenSetRow,
        response: RAGResponse,
    ) -> GeneratedAnswerRow:

        return GeneratedAnswerRow(
            question_id=question.question_id,
            generated_answer=response.answer,
            retrieved_context=[
                chunk.chunk_text
                for chunk in response.retrieved_chunks
            ],
            retrieved_documents=[
                chunk.document_id
                for chunk in response.retrieved_chunks
            ],
            retrieved_pages=[
                chunk.page_start
                for chunk in response.retrieved_chunks
            ],
            provider=response.llm_response.provider,
            model=response.llm_response.model,
            latency=response.llm_response.latency_ms,
            status="SUCCESS",
        )

    def _failed_answer(
        self,
        question: GoldenSetRow,
    ) -> GeneratedAnswerRow:

        return GeneratedAnswerRow(
            question_id=question.question_id,
            generated_answer="",
            retrieved_context=[],
            retrieved_documents=[],
            retrieved_pages=[],
            provider=self._model.provider.value,
            model=self._model.id,
            latency=0.0,
            status="FAILED",
        )