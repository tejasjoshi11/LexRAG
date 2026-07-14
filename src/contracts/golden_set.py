from dataclasses import dataclass
from typing import List

@dataclass(frozen=True, slots=True)
class GoldenSetRow:
    question_id: str
    question: str
    ground_truth_answer: str
    source_document: str
    source_reference: str