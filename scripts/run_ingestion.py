"""Dedicated offline ingestion entry point for the LexRAG pipeline.

Orchestrates the complete document ingestion pipeline from discovery
through indexing. This script composes and sequences existing pipeline
stages — it contains no business logic of its own.

Usage:
    python -m scripts.run_ingestion
    python -m scripts.run_ingestion --limit 5
    python -m scripts.run_ingestion --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.contracts.registry_record import RegistryRecord
from src.embeddings.sentence_transformer_provider import (
    SentenceTransformerProvider,
)
from src.ingestion_pipeline.chunker import chunk_document
from src.ingestion_pipeline.cleaner import clean_document
from src.ingestion_pipeline.discovery import discover_documents
from src.ingestion_pipeline.embedder import embed_chunks
from src.ingestion_pipeline.indexer import index_chunks as vector_index_chunks
from src.ingestion_pipeline.knowledge_standardizer import (
    standardize_document,
)
from src.ingestion_pipeline.parser import parse_document
from src.ingestion_pipeline.semantic_metadata import (
    extract_semantic_metadata,
)
from src.keyword_indexing.elasticsearch_provider import (
    ElasticsearchProvider,
)
from src.keyword_indexing.indexer import (
    index_chunks as keyword_index_chunks,
)
from src.metadata import DocumentRegistry
from src.shared.config import (
    chunk_overlap,
    chunk_size,
    chunker_version,
    cleaner_version,
    embedding_model_name,
    embedding_version,
    parser_version,
    pipeline_version,
    semantic_version,
)
from src.shared.constants import (
    PROCESSING_STATUS_COMPLETED,
    PROCESSING_STATUS_FAILED,
)
from src.shared.exceptions import LexRAGError
from src.shared.utils import compute_file_hash, utc_now_iso
from src.vector_indexing.qdrant_provider import QdrantProvider

_LOGGER = logging.getLogger(__name__)

_SEPARATOR = "=" * 60


# ============================================================================
# CLI
# ============================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the LexRAG offline ingestion pipeline.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all).",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )

    return parser.parse_args()


# ============================================================================
# Logging Configuration
# ============================================================================


def _configure_logging(level: str) -> None:
    """Configure the root logger for the ingestion pipeline."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    # Silence noisy third-party loggers that flood the terminal
    # especially during model initialization or HTTP requests
    for logger_name in [
        "httpx",
        "httpcore",
        "sentence_transformers",
        "elastic_transport",
        "qdrant_client",
        "urllib3",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# ============================================================================
# Document Processing
# ============================================================================


def _process_document(
    task,
    *,
    embedding_provider: SentenceTransformerProvider,
    vector_provider: QdrantProvider,
    keyword_provider: ElasticsearchProvider,
    document_registry: DocumentRegistry,
) -> tuple[int, int]:
    """Process one document through all ingestion stages.

    Returns:
        A tuple of (chunk_count, embedded_chunk_count).
    """

    document_id = task.catalog_metadata.document_id
    content_hash = compute_file_hash(task.pdf_path)

    # ── Stage 2: Parse ───────────────────────────────────────────────────
    stage_start = time.perf_counter()
    parsed_document = parse_document(task)
    _LOGGER.info(
        "Parsed %s (%d pages) in %.2f s",
        document_id,
        parsed_document.page_count,
        time.perf_counter() - stage_start,
    )

    # ── Stage 3: Clean ───────────────────────────────────────────────────
    stage_start = time.perf_counter()
    cleaned_document = clean_document(parsed_document)
    _LOGGER.info(
        "Cleaned %s (%d chars) in %.2f s",
        document_id,
        len(cleaned_document.cleaned_text),
        time.perf_counter() - stage_start,
    )

    # ── Stage 4: Catalog Metadata ────────────────────────────────────────
    # Already loaded during discovery via find_catalog_metadata().
    catalog_metadata = task.catalog_metadata

    # ── Stage 5: Semantic Metadata ───────────────────────────────────────
    stage_start = time.perf_counter()
    enriched_document = extract_semantic_metadata(
        cleaned_document,
        catalog_metadata,
    )
    _LOGGER.info(
        "Extracted semantic metadata for %s in %.2f s",
        document_id,
        time.perf_counter() - stage_start,
    )

    # ── Stage 6: Knowledge Standardization ───────────────────────────────
    stage_start = time.perf_counter()
    standardized_document = standardize_document(enriched_document)
    _LOGGER.info(
        "Standardized %s (%d chars) in %.2f s",
        document_id,
        len(standardized_document.standardized_content),
        time.perf_counter() - stage_start,
    )

    # ── Stage 7: Chunking ────────────────────────────────────────────────
    stage_start = time.perf_counter()
    retrieval_chunks = chunk_document(
        standardized_document,
        max_chunk_tokens=chunk_size(),
        chunk_overlap_tokens=chunk_overlap(),
        chunker_version=chunker_version(),
    )
    chunk_count = len(retrieval_chunks)
    _LOGGER.info(
        "Chunked %s into %d chunks in %.2f s",
        document_id,
        chunk_count,
        time.perf_counter() - stage_start,
    )

    # ── Stage 8: Embedding ───────────────────────────────────────────────
    stage_start = time.perf_counter()
    embedded_chunks = embed_chunks(
        list(retrieval_chunks),
        embedding_provider,
    )
    embedded_chunk_count = len(embedded_chunks)
    _LOGGER.info(
        "Embedded %d chunks for %s in %.2f s",
        embedded_chunk_count,
        document_id,
        time.perf_counter() - stage_start,
    )

    # ── Stage 9a: Vector Indexing (Qdrant) ───────────────────────────────
    stage_start = time.perf_counter()
    vector_index_chunks(embedded_chunks, vector_provider)
    _LOGGER.info(
        "Qdrant indexing completed for %s (%d vectors) in %.2f s",
        document_id,
        embedded_chunk_count,
        time.perf_counter() - stage_start,
    )

    # ── Stage 9b: Keyword Indexing (Elasticsearch) ───────────────────────
    stage_start = time.perf_counter()
    keyword_index_chunks(
        retrieval_chunks=list(retrieval_chunks),
        provider=keyword_provider,
    )
    _LOGGER.info(
        "Elasticsearch indexing completed for %s (%d docs) in %.2f s",
        document_id,
        chunk_count,
        time.perf_counter() - stage_start,
    )

    # ── Registry: Record Success ─────────────────────────────────────────
    record = RegistryRecord(
        document_id=document_id,
        content_hash=content_hash,
        title=catalog_metadata.title,
        category=catalog_metadata.category,
        source_url=catalog_metadata.source_url,
        summary=catalog_metadata.summary,
        parser_version=parser_version(),
        cleaner_version=cleaner_version(),
        semantic_version=semantic_version(),
        chunker_version=chunker_version(),
        embedding_model=embedding_model_name(),
        embedding_version=embedding_version(),
        pipeline_version=pipeline_version(),
        chunk_count=chunk_count,
        processing_timestamp=utc_now_iso(),
        processing_status=PROCESSING_STATUS_COMPLETED,
    )
    document_registry.save_registry_record(record)

    return chunk_count, embedded_chunk_count


def _record_failure(
    task,
    *,
    document_registry: DocumentRegistry,
) -> None:
    """Persist a FAILED registry record for one document."""

    content_hash = compute_file_hash(task.pdf_path)
    catalog_metadata = task.catalog_metadata

    record = RegistryRecord(
        document_id=catalog_metadata.document_id,
        content_hash=content_hash,
        title=catalog_metadata.title,
        category=catalog_metadata.category,
        source_url=catalog_metadata.source_url,
        summary=catalog_metadata.summary,
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


# ============================================================================
# Summary
# ============================================================================


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _print_summary(
    *,
    discovered: int,
    processed: int,
    failed: int,
    total_chunks: int,
    total_vectors: int,
    total_keyword_docs: int,
    elapsed: float,
) -> None:
    """Print a concise execution summary to stdout."""

    print()
    print(_SEPARATOR)
    print("LexRAG Ingestion Summary")
    print(_SEPARATOR)
    print(f"Documents discovered  : {discovered:,}")
    print(f"Processed             : {processed:,}")
    print(f"Failed                : {failed:,}")
    print(f"Chunks generated      : {total_chunks:,}")
    print(f"Vectors indexed       : {total_vectors:,}")
    print(f"Keyword docs indexed  : {total_keyword_docs:,}")
    print(f"Elapsed time          : {_format_elapsed(elapsed)}")
    print(_SEPARATOR)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Run the complete ingestion pipeline. Returns exit code."""

    args = _parse_args()
    _configure_logging(args.log_level)

    pipeline_start = time.perf_counter()

    _LOGGER.info("Ingestion pipeline started")

    # ── Stage 1: Discovery ───────────────────────────────────────────────

    _LOGGER.info("Discovering documents...")
    stage_start = time.perf_counter()

    try:
        document_tasks = discover_documents()
    except LexRAGError:
        _LOGGER.exception("Document discovery failed")
        return 1

    discovered = len(document_tasks)
    _LOGGER.info(
        "Discovered %d documents in %.2f s",
        discovered,
        time.perf_counter() - stage_start,
    )

    if not document_tasks:
        _LOGGER.info("No documents to process")
        _print_summary(
            discovered=0,
            processed=0,
            failed=0,
            total_chunks=0,
            total_vectors=0,
            total_keyword_docs=0,
            elapsed=time.perf_counter() - pipeline_start,
        )
        return 0

    # ── Apply --limit ────────────────────────────────────────────────────

    if args.limit is not None:
        document_tasks = document_tasks[: args.limit]
        _LOGGER.info(
            "Limited to %d document(s) (--limit %d)",
            len(document_tasks),
            args.limit,
        )

    # ── Initialize Providers (once) ──────────────────────────────────────

    _LOGGER.info("Initializing embedding provider...")
    try:
        embedding_provider = SentenceTransformerProvider()
    except LexRAGError:
        _LOGGER.exception("Failed to initialize embedding provider")
        return 1

    _LOGGER.info("Initializing Qdrant provider...")
    try:
        vector_provider = QdrantProvider()
    except LexRAGError:
        _LOGGER.exception("Failed to initialize Qdrant provider")
        return 1

    _LOGGER.info("Initializing Elasticsearch provider...")
    try:
        keyword_provider = ElasticsearchProvider()
    except LexRAGError:
        _LOGGER.exception("Failed to initialize Elasticsearch provider")
        return 1

    document_registry = DocumentRegistry()

    _LOGGER.info("All providers initialized")

    # ── Process Documents ────────────────────────────────────────────────

    processed = 0
    failed = 0
    total_chunks = 0
    total_vectors = 0
    total_keyword_docs = 0

    for index, task in enumerate(document_tasks, start=1):
        document_id = task.catalog_metadata.document_id

        _LOGGER.info(
            "Processing document %d/%d: %s",
            index,
            len(document_tasks),
            document_id,
        )

        doc_start = time.perf_counter()

        content_hash = compute_file_hash(task.pdf_path)

        if document_registry.is_document_unchanged(
            document_id=document_id,
            content_hash=content_hash,
        ):
            _LOGGER.info(
                "Skipping %s: already processed and unchanged",
                document_id,
            )
            continue

        try:
            chunk_count, embedded_count = _process_document(
                task,
                embedding_provider=embedding_provider,
                vector_provider=vector_provider,
                keyword_provider=keyword_provider,
                document_registry=document_registry,
            )

            processed += 1
            total_chunks += chunk_count
            total_vectors += embedded_count
            total_keyword_docs += chunk_count

            _LOGGER.info(
                "Completed %s in %.2f s (%d chunks)",
                document_id,
                time.perf_counter() - doc_start,
                chunk_count,
            )

        except Exception:
            failed += 1
            _LOGGER.exception(
                "Failed to process %s after %.2f s",
                document_id,
                time.perf_counter() - doc_start,
            )

            try:
                _record_failure(
                    task,
                    document_registry=document_registry,
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to record failure for %s",
                    document_id,
                )

    # ── Summary ──────────────────────────────────────────────────────────

    elapsed = time.perf_counter() - pipeline_start

    _LOGGER.info(
        "Ingestion pipeline finished: %d processed, %d failed in %s",
        processed,
        failed,
        _format_elapsed(elapsed),
    )

    _print_summary(
        discovered=discovered,
        processed=processed,
        failed=failed,
        total_chunks=total_chunks,
        total_vectors=total_vectors,
        total_keyword_docs=total_keyword_docs,
        elapsed=elapsed,
    )

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
