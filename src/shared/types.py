"""
Shared reusable type definitions for LexRAG.

This module defines dependency-free type aliases that are shared across
the ingestion pipeline, retrieval pipeline, API layer, evaluation, and
utilities.

The module intentionally contains no business logic, validation, or runtime
behavior and is safe to import from any part of the project.
"""

from pathlib import Path
from typing import TypeAlias, Literal

# ============================================================================
# Generic Type Aliases
# ============================================================================

PathLike: TypeAlias = str | Path

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONPrimitive
    | dict[str, "JSONValue"]
    | list["JSONValue"]
)

JSONDict: TypeAlias = dict[str, JSONValue]
JSONList: TypeAlias = list[JSONValue]

MetadataDict: TypeAlias = dict[str, JSONValue]

Headers: TypeAlias = dict[str, str]

FloatVector: TypeAlias = list[float]

DocumentID: TypeAlias = str
ChunkID: TypeAlias = str
Category: TypeAlias = str
ContentHash: TypeAlias = str
SourceURL: TypeAlias = str
FileName: TypeAlias = str
PageNumber: TypeAlias = int

ProcessingStatus: TypeAlias = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "skipped",
]

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "PathLike",
    "JSONPrimitive",
    "JSONValue",
    "JSONDict",
    "JSONList",
    "MetadataDict",
    "Headers",
    "FloatVector",
    "DocumentID",
    "ChunkID",
    "Category",
    "ContentHash",
    "SourceURL",
    "FileName",
    "PageNumber",
    "ProcessingStatus",
]