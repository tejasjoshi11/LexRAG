from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ingestion_pipeline.discovery import discover_documents
from src.ingestion_pipeline.parser import parse_document
from src.ingestion_pipeline.cleaner import clean_document
from src.ingestion_pipeline.semantic_metadata import (
    extract_semantic_metadata,
)
from src.ingestion_pipeline.knowledge_standardizer import (
    standardize_document,
)
from src.ingestion_pipeline.chunker import (
    chunk_document,
)

from src.ingestion_pipeline.embedder import embed_chunks
from src.embeddings.provider import EmbeddingProvider
from src.embeddings.sentence_transformer_provider import (
    SentenceTransformerProvider,
)

from src.ingestion_pipeline.indexer import index_chunks
from src.vector_indexing.provider import VectorIndexProvider
from src.vector_indexing.qdrant_provider import QdrantProvider

from src.keyword_indexing.elasticsearch_provider import (
    ElasticsearchProvider,
)

from src.keyword_indexing.indexer import (
    index_chunks as keyword_index_chunks,
)

from src.keyword_indexing.provider import KeywordIndexProvider

from src.shared.config import (
    chunk_size,
    chunk_overlap,
    chunker_version,
)

from src.shared.exceptions import (
    CleaningError,
    ChunkingError,
    DiscoveryError,
    ParsingError,
    SemanticExtractionError,
    StandardizationError,
)

from src.contracts.document_task import DocumentTask
from src.contracts.parsed_document import ParsedDocument

from src.shared.types import (
    Category,
    ChunkID,
    DocumentID,
)

from src.metadata.document_registry import DocumentRegistry

from src.contracts.registry_record import RegistryRecord

from src.shared.constants import (
    PROCESSING_STATUS_COMPLETED,
    PROCESSING_STATUS_FAILED,
)

from src.shared.utils import (
    compute_file_hash,
    utc_now_iso,
)

from src.shared.config import (
    parser_version,
    cleaner_version,
    semantic_version,
    embedding_model_name,
    embedding_version,
    pipeline_version,
)

from src.shared.exceptions import RegistryError

CHUNK_PREVIEW_LIMIT = 10
SEPARATOR = "=" * 80


@dataclass(frozen=True, slots=True)
class ChunkPreview:
    chunk_id: ChunkID
    page_start: int
    page_end: int

@dataclass(slots=True)
class DocumentProcessingResult:
    document_id: DocumentID
    category: Category
    pdf_path: Path
    page_count: int
    cleaned_characters: int
    standardized_characters: int
    chunk_count: int
    chunk_previews: tuple[ChunkPreview, ...]
    court: str | None
    legal_domain: str | None
    embedded_chunk_count: int
    indexed: bool

@dataclass(slots=True)
class IngestionStatistics:
    discovered: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    total_embedded_chunks: int = 0
    total_indexed_chunks: int = 0

def print_separator(
    character: str = "=",
    width: int = 80,
) -> None:
    print(character * width)

def print_document_summary(
    summary: DocumentProcessingResult,
) -> None:
    """Print a processing summary for one document."""

    print()
    print_separator()
    print(f"Document ID          : {summary.document_id}")
    print(f"Category             : {summary.category}")
    print(f"PDF Path             : {summary.pdf_path}")
    print()

    print(f"Pages                : {summary.page_count}")
    print(f"Cleaned Characters   : {summary.cleaned_characters}")
    print(f"Standardized Chars   : {summary.standardized_characters}")
    print()

    if summary.court is not None:
        print(f"Court                : {summary.court}")

    if summary.legal_domain is not None:
        print(f"Legal Domain         : {summary.legal_domain}")

    print(f"Chunks Generated     : {summary.chunk_count}")
    print(f"Embedded Chunks      : {summary.embedded_chunk_count}")
    print(f"Indexed              : {'Yes' if summary.indexed else 'No'}")

    print()

    for index, chunk_preview in enumerate(
        summary.chunk_previews,
        start=1,
    ):
        print(
            f"Chunk {index:03d} : "
            f"{chunk_preview.chunk_id} "
            f"Pages {chunk_preview.page_start}-{chunk_preview.page_end}"
        )

    remaining = summary.chunk_count - len(summary.chunk_previews)

    if remaining > 0:
        print(f"... {remaining} more chunks")

    print_separator()

def print_final_summary(
    statistics: IngestionStatistics,
) -> None:
    """Print ingestion summary."""

    print()
    print_separator()
    print("LEXRAG INGESTION SUMMARY")
    print_separator()

    print(f"Documents Discovered : {statistics.discovered}")
    print(f"Successful           : {statistics.successful}")
    print(f"Failed               : {statistics.failed}")
    print(f"Total Chunks         : {statistics.total_chunks}")
    print(f"Total Embedded Chunks: {statistics.total_embedded_chunks}")
    print(f"Total Indexed Chunks : {statistics.total_indexed_chunks}")

    print_separator()

def run_ingestion() -> None:
    """Run the full ingestion pipeline over all discovered documents."""

    print_separator()
    print("LEXRAG INGESTION PIPELINE")
    print_separator()

    
    statistics = IngestionStatistics()
    document_registry = DocumentRegistry()
    try:
        document_tasks = discover_documents()
    except DiscoveryError as exc:
        statistics.failed = 1

        print()
        print_separator()
        print("FAILED")
        print_separator()
        print(type(exc).__name__)
        print(exc)
        print()

        print_final_summary(statistics)
        return

    statistics.discovered = len(document_tasks)

    if not document_tasks:
        print("No documents discovered.")
        print_final_summary(statistics)
        return
    
    print("Loading Sentence Transformer model...")
    embedding_provider = SentenceTransformerProvider()
    print("Sentence Transformer loaded.")

    print("Connecting to Qdrant...")
    index_provider = QdrantProvider()
    print("Connected to Qdrant.")

    print("Connecting to Elasticsearch...")
    keyword_provider = ElasticsearchProvider()
    print("Connected to Elasticsearch.")

    for index, task in enumerate(document_tasks, start=1):

        if task.catalog_metadata.document_id != "J01":
            continue

        content_hash = compute_file_hash(task.pdf_path)

        if document_registry.is_document_unchanged(
            document_id=task.catalog_metadata.document_id,
            content_hash=content_hash,
        ):
            print(
                f"Skipping {task.catalog_metadata.document_id}: already processed."
            )
            continue

        try:
            summary = process_document(
                task,
                embedding_provider,
                index_provider,
                keyword_provider,
            )

            statistics.successful += 1
            statistics.total_chunks += summary.chunk_count
            statistics.total_embedded_chunks += summary.embedded_chunk_count
            statistics.total_indexed_chunks += summary.embedded_chunk_count

            print_document_summary(summary)

            record = RegistryRecord(
                document_id=task.catalog_metadata.document_id,
                content_hash=content_hash,
                title=task.catalog_metadata.title,
                category=task.catalog_metadata.category,
                source_url=task.catalog_metadata.source_url,
                summary=task.catalog_metadata.summary,
                parser_version=parser_version(),
                cleaner_version=cleaner_version(),
                semantic_version=semantic_version(),
                chunker_version=chunker_version(),
                embedding_model=embedding_model_name(),
                embedding_version=embedding_version(),
                pipeline_version=pipeline_version(),
                chunk_count=summary.chunk_count,
                processing_timestamp=utc_now_iso(),
                processing_status=PROCESSING_STATUS_COMPLETED,
            )

            document_registry.save_registry_record(record)

        except (
            CleaningError,
            ParsingError,
            SemanticExtractionError,
            StandardizationError,
            ChunkingError,
            RegistryError,
        ) as exc:
            statistics.failed += 1

            record = RegistryRecord(
                document_id=task.catalog_metadata.document_id,
                content_hash=content_hash,
                title=task.catalog_metadata.title,
                category=task.catalog_metadata.category,
                source_url=task.catalog_metadata.source_url,
                summary=task.catalog_metadata.summary,
                parser_version=parser_version(),
                cleaner_version=cleaner_version(),
                semantic_version=semantic_version(),
                chunker_version=chunker_version(),
                embedding_model=embedding_model_name(),
                embedding_version=embedding_version(),
                pipeline_version=pipeline_version(),
                chunk_count=0,
                processing_timestamp=utc_now_iso(),
                processing_status=PROCESSING_STATUS_FAILED,
            )

            document_registry.save_registry_record(record)

            print()
            print_separator()
            print("FAILED")
            print_separator()
            print(task.catalog_metadata.document_id)
            print(task.pdf_path)
            print(type(exc).__name__)
            print(exc)
            print()

    print_final_summary(statistics)

def process_document(
    task: DocumentTask,
    embedding_provider: EmbeddingProvider,
    index_provider: VectorIndexProvider,
    keyword_provider: KeywordIndexProvider,
) -> DocumentProcessingResult:
    """Process one document through Stages 2–9."""

    parsed_document = parse_document(task)
    _validate_parsed_document(parsed_document)

    cleaned_document = clean_document(parsed_document)

    catalog_metadata = task.catalog_metadata

    enriched_document = extract_semantic_metadata(
        cleaned_document,
        catalog_metadata,
    )

    standardized_document = standardize_document(
        enriched_document,
    )

    retrieval_chunks = chunk_document(
        standardized_document,
        max_chunk_tokens=chunk_size(),
        chunk_overlap_tokens=chunk_overlap(),
        chunker_version=chunker_version(),
    )
    print(f"Chunks generated: {len(retrieval_chunks)}")
    print(f"Average chunk length: {sum(len(c.chunk_text) for c in retrieval_chunks) // len(retrieval_chunks)} characters")

    embedded_chunks = embed_chunks(
        retrieval_chunks,
        embedding_provider,
    )
    print(f"Embeddings generated: {len(embedded_chunks)}")
    index_chunks(
        embedded_chunks,
        index_provider,
    )

    keyword_index_chunks(
        retrieval_chunks=retrieval_chunks,
        provider=keyword_provider,
    )
    count = keyword_provider._client.count(
        index="lexrag",
    )

    print(
        f"Elasticsearch now contains "
        f"{count['count']} documents."
    )
    print(
        f"Keyword index updated with "
        f"{len(retrieval_chunks)} retrieval chunks."
    )
    chunk_previews = tuple(
        ChunkPreview(
            chunk_id=chunk.chunk_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        for chunk in retrieval_chunks[:CHUNK_PREVIEW_LIMIT]
    )

    return DocumentProcessingResult(
        document_id=catalog_metadata.document_id,
        category=catalog_metadata.category,
        pdf_path=task.pdf_path,
        page_count=parsed_document.page_count,
        cleaned_characters=len(cleaned_document.cleaned_text),
        standardized_characters=len(
            standardized_document.standardized_content
        ),
        chunk_count=len(retrieval_chunks),
        chunk_previews=chunk_previews,
        court=enriched_document.court,
        legal_domain=enriched_document.legal_domain,
        embedded_chunk_count=len(embedded_chunks),
        indexed=True,
    )

def _validate_parsed_document(parsed_document: ParsedDocument) -> None:
    """Reject empty or structurally invalid parser output early."""

    if parsed_document.page_count <= 0:
        raise ParsingError("Parsed document must contain at least one page")

    if len(parsed_document.page_offsets) != parsed_document.page_count:
        raise ParsingError("Parsed document page offsets do not match page count")

    if not parsed_document.raw_text.strip():
        raise ParsingError("Parsed document must contain non-empty text")

def main() -> None:
    """CLI entry point for the ingestion pipeline."""

    run_ingestion()

if __name__ == "__main__":
    main()