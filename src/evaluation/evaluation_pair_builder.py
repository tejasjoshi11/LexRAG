from __future__ import annotations

from src.contracts.golden_set import GoldenSetRow
from src.contracts.generated_answer import GeneratedAnswerRow
from src.contracts.evaluation_pair import EvaluationPair


class EvaluationPairBuilderError(Exception):
    """Raised when evaluation pairs cannot be built."""


class EvaluationPairBuilder:
    """Builds EvaluationPair objects from Golden Set and Generated Answers."""

    def build(
        self,
        golden_set: list[GoldenSetRow],
        generated_answers: list[GeneratedAnswerRow],
    ) -> list[EvaluationPair]:
        
        golden_map = {row.question_id: row for row in golden_set}
        
        pairs: list[EvaluationPair] = []
        seen: set[str] = set()

        for answer in generated_answers:
            q_id = answer.question_id
            
            if q_id in seen:
                raise EvaluationPairBuilderError(
                    f"Duplicate Question_ID found in generated answers: {q_id}"
                )
            seen.add(q_id)
            
            golden_row = golden_map.get(q_id)
            if not golden_row:
                raise EvaluationPairBuilderError(
                    f"Question_ID {q_id} not found in Golden Set."
                )
                
            pairs.append(
                EvaluationPair(
                    question_id=q_id,
                    question=golden_row.question,
                    ground_truth_answer=golden_row.ground_truth_answer,
                    source_document=golden_row.source_document,
                    source_reference=golden_row.source_reference,
                    generated_answer=answer.generated_answer,
                    retrieved_context=answer.retrieved_context,
                    retrieved_documents=answer.retrieved_documents,
                    retrieved_pages=answer.retrieved_pages,
                    provider=answer.provider,
                    model=answer.model,
                    latency=answer.latency,
                    status=answer.status,
                )
            )
            
        missing_golden = set(golden_map.keys()) - seen
        if missing_golden:
            raise EvaluationPairBuilderError(
                f"Missing generated answers for Golden Set questions: {sorted(missing_golden)}"
            )

        return pairs
