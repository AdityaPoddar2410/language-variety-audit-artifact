#!/usr/bin/env python3
"""Validate and checksum one canonical evaluation JSONL file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from language_variety_audit.prompts import CONTENT_CATEGORIES
from language_variety_audit.run_records import portable_source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-model", required=True)
    args = parser.parse_args()

    rows = []
    with args.path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise SystemExit(f"Non-object JSON at line {line_number}")
            rows.append(record)

    keys = [
        (row.get("id"), row.get("variety"), row.get("condition"), row.get("provider"), row.get("model"))
        for row in rows
    ]
    valid_labels = set(CONTENT_CATEGORIES)
    errors = [row for row in rows if row.get("error") or row.get("parse_error")]
    invalid_labels = [
        row for row in rows if str(row.get("predicted_category", "")).strip().lower() not in valid_labels
    ]
    wrong_models = [row for row in rows if row.get("model") != args.expected_model]
    required = {
        "run_id",
        "timestamp_utc",
        "benchmark_sha256",
        "prompt_sha256",
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
        "attempt_count",
        "raw_response",
    }
    missing_metadata = [row for row in rows if required - row.keys()]

    failures = []
    if len(rows) != args.expected_rows:
        failures.append(f"rows={len(rows)} expected={args.expected_rows}")
    if len(set(keys)) != len(keys):
        failures.append(f"duplicate_keys={len(keys) - len(set(keys))}")
    if errors:
        failures.append(f"error_rows={len(errors)}")
    if invalid_labels:
        failures.append(f"invalid_label_rows={len(invalid_labels)}")
    if wrong_models:
        failures.append(f"wrong_model_rows={len(wrong_models)}")
    if missing_metadata:
        failures.append(f"missing_metadata_rows={len(missing_metadata)}")
    if failures:
        raise SystemExit("Validation failed: " + ", ".join(failures))

    report = {
        "path": portable_source(args.path),
        "sha256": sha256(args.path),
        "rows": len(rows),
        "unique_keys": len(set(keys)),
        "model": args.expected_model,
        "run_ids": sorted({str(row["run_id"]) for row in rows}),
        "benchmark_sha256": sorted({str(row["benchmark_sha256"]) for row in rows}),
        "prompt_versions": sorted({str(row["prompt_version"]) for row in rows}),
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "total_latency_seconds": sum(float(row.get("latency_seconds") or 0) for row in rows),
    }
    report_path = args.path.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
