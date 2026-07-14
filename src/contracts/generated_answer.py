from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class GeneratedAnswerRow:
    """
    Represents one generated answer produced by LexRAG.
    """

    question_id: str

    generated_answer: str

    retrieved_context: List[str]

    retrieved_documents: List[str]

    retrieved_pages: List[str]

    provider: str

    model: str

    latency: float

    status: str