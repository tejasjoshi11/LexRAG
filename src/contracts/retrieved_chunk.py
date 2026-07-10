from dataclasses import dataclass
from enum import StrEnum

from src.shared.types import (
    ChunkID,
    DocumentID,
    PageNumber,
)


class RetrievalMethod(StrEnum):
    """Retrieval strategy used."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(
    frozen=True,
    slots=True,
)
class RetrievedChunk:
    """Retrieval result enriched with retrieval metadata."""

    chunk_id: ChunkID

    document_id: DocumentID

    title: str

    heading: str | None

    source_url: str

    page_start: PageNumber

    page_end: PageNumber

    chunk_text: str

    retrieval_score: float

    retrieval_method: RetrievalMethod

    retrieval_rank: int