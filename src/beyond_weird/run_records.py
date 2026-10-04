"""Resumable JSONL record helpers for API experiments."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULT_KEY_FIELDS = ("id", "variety", "condition", "provider", "model")


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(*parts: str) -> str:
    payload = "\n\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def result_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in RESULT_KEY_FIELDS)


def successful(record: dict[str, Any], valid_labels: set[str]) -> bool:
    prediction = str(record.get("predicted_category", "")).strip().lower()
    return (
        not record.get("error")
        and not bool(record.get("parse_error"))
        and prediction in valid_labels
    )


def read_latest_records(path: str | Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return latest
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected object at {source}:{line_number}")
            latest[result_key(record)] = record
    return latest


def append_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def compact_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    ordered = sorted(
        records,
        key=lambda record: tuple(str(record.get(field, "")) for field in RESULT_KEY_FIELDS),
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for record in ordered:
            append_record(handle, record)
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
