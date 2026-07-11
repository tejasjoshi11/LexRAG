"""Persistent, thread-safe DocumentRegistry state for processed documents."""

from collections.abc import Iterable, Mapping
from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from types import MappingProxyType
from typing import Any

from src.contracts.registry_record import RegistryRecord
from src.shared.config import registry_storage_path
from src.shared.constants import (
    PROCESSING_STATUS_COMPLETED,
    PROCESSING_STATUS_FAILED,
    PROCESSING_STATUS_IN_PROGRESS,
    PROCESSING_STATUS_PENDING,
    PROCESSING_STATUS_SKIPPED,
)
from src.shared.exceptions import RegistryError
from src.shared.types import ContentHash, DocumentID


_RECORD_FIELD_NAMES = tuple(field.name for field in fields(RegistryRecord))
_RECORD_FIELD_NAME_SET = frozenset(_RECORD_FIELD_NAMES)
_RECORDS_KEY = "records"
_VALID_PROCESSING_STATUSES = frozenset(
    {
        PROCESSING_STATUS_PENDING,
        PROCESSING_STATUS_IN_PROGRESS,
        PROCESSING_STATUS_COMPLETED,
        PROCESSING_STATUS_FAILED,
        PROCESSING_STATUS_SKIPPED,
    }
)
_UNCHANGED_PROCESSING_STATUS = PROCESSING_STATUS_COMPLETED


class DocumentRegistry:
    """A thread-safe registry to manage document metadata."""

    def __init__(self) -> None:
        """Initialize the document registry."""
        self._lock = RLock()
        self._snapshot: tuple[RegistryRecord, ...] | None = None
        self._index: Mapping[DocumentID, RegistryRecord] | None = None

    def is_document_unchanged(
        self,
        document_id: DocumentID,
        content_hash: ContentHash,
    ) -> bool:
        """Return whether Registry lifecycle state considers a document unchanged."""
        self._validate_identity_value(document_id, "document_id")
        self._validate_identity_value(content_hash, "content_hash")

        _, index = self._state()
        record = index.get(document_id)

        return (
            record is not None
            and record.content_hash == content_hash
            and record.processing_status == _UNCHANGED_PROCESSING_STATUS
        )

    def get_registry_record(
        self,
        document_id: DocumentID,
    ) -> RegistryRecord | None:
        """Return the Registry record for a document."""
        self._validate_identity_value(document_id, "document_id")
        _, index = self._state()
        return index.get(document_id)

    def save_registry_record(self, record: RegistryRecord) -> None:
        """Validate and atomically persist one current Registry record."""
        self._validate_registry_record(record)

        with self._lock:
            snapshot, index = self._state()

            if index.get(record.document_id) == record:
                return

            next_records = [
                current_record
                for current_record in snapshot
                if current_record.document_id != record.document_id
            ]
            next_records.append(record)
            next_snapshot, next_index = self._build_state(next_records)

            self._persist_snapshot(next_snapshot)
            self._index = next_index
            self._snapshot = next_snapshot

    def _state(self) -> tuple[
        tuple[RegistryRecord, ...],
        Mapping[DocumentID, RegistryRecord],
    ]:
        """Return the initialized immutable Registry snapshot and lookup index."""
        snapshot = self._snapshot
        index = self._index

        if snapshot is not None and index is not None:
            return snapshot, index

        with self._lock:
            snapshot = self._snapshot
            index = self._index

            if snapshot is None or index is None:
                snapshot, index = self._load_state()
                self._index = index
                self._snapshot = snapshot

            return snapshot, index

    def _load_state(self) -> tuple[
        tuple[RegistryRecord, ...],
        Mapping[DocumentID, RegistryRecord],
    ]:
        """Load and validate the complete Registry snapshot from persistent storage."""
        path = registry_storage_path()

        try:
            if not path.exists():
                return self._build_state(())

            if not path.is_file():
                raise RegistryError(
                    f"Registry storage path is not a file: {path}"
                )

            with path.open("r", encoding="utf-8") as registry_file:
                payload = json.load(
                    registry_file,
                    object_pairs_hook=self._unique_json_object,
                )
        except RegistryError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RegistryError(
                f"Failed to load Registry storage: {path}"
            ) from exc

        return self._deserialize_snapshot(payload)

    def _deserialize_snapshot(
        self,
        payload: Any,
    ) -> tuple[tuple[RegistryRecord, ...], Mapping[DocumentID, RegistryRecord]]:
        """Deserialize and validate a complete internal Registry snapshot."""
        if not isinstance(payload, Mapping) or set(payload) != {_RECORDS_KEY}:
            raise RegistryError("Registry storage has an invalid root structure")

        raw_records = payload[_RECORDS_KEY]

        if not isinstance(raw_records, list):
            raise RegistryError("Registry storage records must be a list")

        records: list[RegistryRecord] = []

        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise RegistryError("Registry storage contains an invalid record")

            if set(raw_record) != _RECORD_FIELD_NAME_SET:
                raise RegistryError("Registry storage record fields are invalid")

            try:
                record = RegistryRecord(**dict(raw_record))
            except (TypeError, ValueError) as exc:
                raise RegistryError("Registry storage record is malformed") from exc

            records.append(record)

        return self._build_state(records)

    def _build_state(
        self,
        records: Iterable[RegistryRecord],
    ) -> tuple[tuple[RegistryRecord, ...], Mapping[DocumentID, RegistryRecord]]:
        """Validate records and build immutable deterministic Registry state."""
        index: dict[DocumentID, RegistryRecord] = {}

        for record in records:
            self._validate_registry_record(record)

            if record.document_id in index:
                raise RegistryError(
                    f"Registry contains duplicate document_id: {record.document_id!r}"
                )

            index[record.document_id] = record

        snapshot = tuple(index[document_id] for document_id in sorted(index))
        return snapshot, MappingProxyType(dict(index))

    def _validate_registry_record(self, record: RegistryRecord) -> None:
        """Validate one immutable RegistryRecord before persistence or publication."""
        if not isinstance(record, RegistryRecord):
            raise RegistryError("Registry record must be a RegistryRecord instance")

        self._validate_required_text(record.document_id, "document_id")
        self._validate_required_text(record.content_hash, "content_hash")
        self._validate_required_text(record.title, "title")
        self._validate_required_text(record.category, "category")
        self._validate_required_text(record.source_url, "source_url")
        self._validate_required_text(record.parser_version, "parser_version")
        self._validate_required_text(record.cleaner_version, "cleaner_version")
        self._validate_required_text(record.semantic_version, "semantic_version")
        self._validate_required_text(record.chunker_version, "chunker_version")
        self._validate_required_text(record.embedding_model, "embedding_model")
        self._validate_required_text(record.embedding_version, "embedding_version")
        self._validate_required_text(record.pipeline_version, "pipeline_version")
        self._validate_required_text(record.processing_timestamp, "processing_timestamp")
        self._validate_required_text(record.processing_status, "processing_status")

        if record.processing_status not in _VALID_PROCESSING_STATUSES:
            raise RegistryError("Registry record has an invalid lifecycle state")

        if isinstance(record.chunk_count, bool) or not isinstance(record.chunk_count, int):
            raise RegistryError("Registry record chunk_count must be an integer")

        if record.chunk_count < 0:
            raise RegistryError("Registry record chunk_count must not be negative")

        if record.summary is not None:
            if (
                not isinstance(record.summary, str)
                or not record.summary
                or record.summary.isspace()
                or record.summary.strip() != record.summary
            ):
                raise RegistryError(
                    "Registry record summary must be a non-whitespace string or None"
                )

    def _validate_required_text(self, value: str, field_name: str) -> None:
        """Validate one required Registry text field."""
        if (
            not isinstance(value, str)
            or not value
            or value.isspace()
            or value.strip() != value
        ):
            raise RegistryError(
                f"Registry record field {field_name!r} must be a non-empty string"
            )

    def _validate_identity_value(self, value: str, name: str) -> None:
        """Validate one exact Registry lookup identity component."""
        try:
            self._validate_required_text(value, name)
        except RegistryError as exc:
            raise RegistryError(
                f"Registry lookup {name!r} must be a non-empty string"
            ) from exc

    def _persist_snapshot(self, snapshot: tuple[RegistryRecord, ...]) -> None:
        """Atomically persist one complete Registry snapshot."""
        path = registry_storage_path()
        temporary_path: Path | None = None

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                _RECORDS_KEY: [
                    {
                        field_name: getattr(record, field_name)
                        for field_name in _RECORD_FIELD_NAMES
                    }
                    for record in snapshot
                ]
            }

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    payload,
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, path)
            temporary_path = None
        except (OSError, TypeError, ValueError) as exc:
            raise RegistryError(
                f"Failed to persist Registry storage: {path}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _unique_json_object(
        self,
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        """Construct one JSON object while rejecting duplicate keys."""
        result: dict[str, Any] = {}

        for key, value in pairs:
            if key in result:
                raise RegistryError(
                    f"Registry storage contains a duplicate key: {key!r}"
                )

            result[key] = value

        return result
