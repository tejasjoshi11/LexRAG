from __future__ import annotations

import time
import os
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel

from src.contracts.evaluation_pair import EvaluationPair
from src.contracts.evaluation_result import EvaluationResult


class DeepEvalEvaluatorError(Exception):
    """Raised when DeepEval evaluation fails."""


class DeepEvalEvaluator:
    """Evaluates EvaluationPairs using DeepEval metrics."""

    def __init__(
        self,
        judge_model: str,
        threshold: float,
    ) -> None:
        self._judge_model = judge_model
        self._threshold = threshold

    def evaluate(self, pair: EvaluationPair) -> EvaluationResult:
        start_time = time.time()
        
        # 1. Retrieval Accuracy (Mean Reciprocal Rank)
        expected_doc = pair.source_document.strip().lower()
        expected_doc_id = expected_doc.split("_")[0]

        # Deduplicate retrieved documents while preserving first occurrence order
        seen_docs = set()
        unique_retrieved_docs = []
        for d in pair.retrieved_documents:
            doc_str = str(d).strip().lower()
            if doc_str not in seen_docs:
                seen_docs.add(doc_str)
                unique_retrieved_docs.append(doc_str)
        
        retrieval_accuracy = 0.0
        
        # Calculate MRR (1 / rank), rank starts from 1
        for i, doc in enumerate(unique_retrieved_docs):
            if doc == expected_doc_id:
                retrieval_accuracy = 1.0 / (i + 1)
                break

        max_retries = 3
        retry_delay = 20

        for attempt in range(max_retries):
            try:
                # 2. Faithfulness
                if "gemini" in self._judge_model.lower():
                    api_key = os.environ.get("GEMINI_API_KEY")
                    if not api_key:
                        raise DeepEvalEvaluatorError("GEMINI_API_KEY environment variable is not set.")
                    judge_model = GeminiModel(model=self._judge_model, api_key=api_key)
                else:
                    judge_model = self._judge_model

                test_case = LLMTestCase(
                    input=pair.question,
                    actual_output=pair.generated_answer,
                    retrieval_context=pair.retrieved_context,
                )
                
                metric = FaithfulnessMetric(
                    threshold=self._threshold,
                    model=judge_model,
                    include_reason=True,
                )
                
                metric.measure(test_case)
                
                latency = time.time() - start_time
                
                return EvaluationResult(
                    question_id=pair.question_id,
                    retrieval_accuracy=retrieval_accuracy,
                    faithfulness_score=metric.score,
                    faithfulness_reason=metric.reason,
                    faithfulness_passed=metric.is_successful,
                    latency=latency,
                    provider=pair.provider,
                    model=pair.model,
                    status="SUCCESS",
                    error_message="",
                )

            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries - 1:
                    print(f"429 Resource Exhausted. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

                latency = time.time() - start_time
                return EvaluationResult(
                    question_id=pair.question_id,
                    retrieval_accuracy=retrieval_accuracy,
                    faithfulness_score=None,
                    faithfulness_reason="Evaluation failed",
                    faithfulness_passed=False,
                    latency=latency,
                    provider=pair.provider,
                    model=pair.model,
                    status="FAILED",
                    error_message=str(exc),
                )
