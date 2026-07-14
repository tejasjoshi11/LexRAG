from dataclasses import dataclass
from typing import List

@dataclass(frozen=True, slots=True)
class EvaluationPair:
    """Represents one merged evaluation input."""
    question_id: str
    question: str
    ground_truth_answer: str
    source_document: str
    source_reference: str
    generated_answer: str
    retrieved_context: List[str]
    retrieved_documents: List[str]
    retrieved_pages: List[str]
    provider: str
    model: str
    latency: float
    status: str
