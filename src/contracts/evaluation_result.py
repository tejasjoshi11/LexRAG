from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Represents one evaluated answer."""
    question_id: str
    
    # Retrieval
    retrieval_accuracy: float
    
    # Faithfulness
    faithfulness_score: float | None
    faithfulness_reason: str
    faithfulness_passed: bool
    
    # Metadata
    latency: float
    provider: str
    model: str
    
    # Execution
    status: str
    error_message: str
