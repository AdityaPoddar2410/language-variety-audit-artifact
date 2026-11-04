#!/usr/bin/env python3
"""Audit benchmark and result artifacts before analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from language_variety_audit.run_records import portable_source

CURRENT_RESULT_SPECS = {
    "gpt41-evaluations.jsonl": ("gpt-4.1-mini-2025-04-14", 1200),
    "gemini25-evaluations.jsonl": ("gemini-2.5-flash-lite", 1200),
    "gemma3-local-evaluations.jsonl": ("gemma3:12b", 1200),
    "llama31-local-evaluations.jsonl": ("llama3.1:8b", 1200),
    "mistral7b-local-evaluations.jsonl": ("mistral:7b", 1200),
    "gpt41-probabilities-v2.jsonl": ("gpt-4.1-mini-2025-04-14", 600),
    "gpt4o-probabilities-v2.jsonl": ("gpt-4o-mini-2024-07-18", 600),
}

EXPECTED_BENCHMARK_COLUMNS = [
    "id",
    "word_or_phrase",
    "hinglish_example",
    "indian_english_translation",
    "indian_meaning",
    "category",
    "sae_example",
    "dataset_type",
]
EXPECTED_LABELS = {
    "normal",
    "offensive",
    "sexually_suggestive",
    "harassment",
    "violence",
    "self_harm",
}
VARIETY_COLUMNS = {
    "sae": "sae_example",
    "indian_english": "indian_english_translation",
    "hinglish": "hinglish_example",
}
RESULT_KEY = ["id", "variety", "condition", "provider", "model"]
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "url": re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)"),
    "handle": re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def numeric_ids(series: pd.Series) -> tuple[pd.Series, list[str]]:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series[converted.isna() & series.notna()].astype(str).tolist()
    return converted.dropna().astype(int), invalid


def audit_benchmark(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    missing_columns = [c for c in EXPECTED_BENCHMARK_COLUMNS if c not in frame.columns]
    present_required = [c for c in EXPECTED_BENCHMARK_COLUMNS if c in frame.columns]
    ids, invalid_ids = numeric_ids(frame["id"]) if "id" in frame else (pd.Series(dtype=int), [])

    missing_values = {
        column: int(frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum())
        for column in present_required
    }
    duplicate_ids = sorted(ids[ids.duplicated(keep=False)].unique().tolist())
    expected_ids = set(range(1, 201))
    observed_ids = set(ids.tolist())

    label_counts = (
        frame["category"].map(clean_value).value_counts(dropna=False).sort_index().to_dict()
        if "category" in frame
        else {}
    )
    labels = set(label_counts)
    dataset_type_counts = (
        frame["dataset_type"].map(clean_value).value_counts(dropna=False).sort_index().to_dict()
        if "dataset_type" in frame
        else {}
    )

    text_audits: dict[str, Any] = {}
    pii_candidates: list[dict[str, Any]] = []
    for variety, column in VARIETY_COLUMNS.items():
        if column not in frame:
            continue
        text = frame[column].map(clean_value)
        lengths = text.map(lambda value: len(value.split()))
        duplicated = frame.loc[text.duplicated(keep=False) & text.ne(""), ["id", column]]
        text_audits[variety] = {
            "column": column,
            "duplicate_text_rows": duplicated.to_dict(orient="records"),
            "leading_or_trailing_whitespace_rows": frame.loc[
                frame[column].notna() & frame[column].astype(str).ne(text), "id"
            ].astype(str).tolist(),
            "word_count": {
                "mean": round(float(lengths.mean()), 4),
                "median": float(lengths.median()),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "in_6_to_10_band": int(lengths.between(6, 10).sum()),
                "shorter_than_4": frame.loc[lengths.lt(4), "id"].astype(int).tolist(),
                "longer_than_15": frame.loc[lengths.gt(15), "id"].astype(int).tolist(),
            },
        }
        for row_id, value in zip(frame["id"], text, strict=True):
            for kind, pattern in PII_PATTERNS.items():
                matches = pattern.findall(value)
                if matches:
                    pii_candidates.append(
                        {"id": int(row_id), "variety": variety, "kind": kind, "matches": matches}
                    )

    triplet_duplicates = 0
    if all(column in frame for column in VARIETY_COLUMNS.values()):
        normalized = frame[list(VARIETY_COLUMNS.values())].map(clean_value)
        triplet_duplicates = int(normalized.duplicated(keep=False).sum())

    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "missing_required_columns": missing_columns,
        "unexpected_columns": [c for c in frame.columns if c not in EXPECTED_BENCHMARK_COLUMNS],
        "missing_or_blank_values": missing_values,
        "invalid_ids": invalid_ids,
        "duplicate_ids": duplicate_ids,
        "missing_expected_ids": sorted(expected_ids - observed_ids),
        "unexpected_ids": sorted(observed_ids - expected_ids),
        "label_counts": label_counts,
        "unknown_labels": sorted(labels - EXPECTED_LABELS),
        "missing_labels": sorted(EXPECTED_LABELS - labels),
        "dataset_type_counts": dataset_type_counts,
        "duplicate_triplet_rows": triplet_duplicates,
        "text": text_audits,
        "pii_candidates": pii_candidates,
    }


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    invalid_lines: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            if not isinstance(value, dict):
                invalid_lines.append(line_number)
                continue
            rows.append(value)
    return rows, invalid_lines


def audit_jsonl_results(path: Path, benchmark: pd.DataFrame) -> dict[str, Any]:
    records, invalid_lines = read_jsonl(path)
    frame = pd.DataFrame(records)
    required_columns = RESULT_KEY + ["predicted_category", "parse_error", "error"]
    missing_columns = [c for c in required_columns if c not in frame.columns]

    coverage: list[dict[str, Any]] = []
    duplicate_keys: list[dict[str, Any]] = []
    invalid_predictions: dict[str, int] = {}
    text_mismatches: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    unexpected_ids: list[int] = []

    if not frame.empty and not any(c not in frame.columns for c in RESULT_KEY):
        duplicate_mask = frame.duplicated(RESULT_KEY, keep=False)
        if duplicate_mask.any():
            duplicate_keys = frame.loc[duplicate_mask, RESULT_KEY].to_dict(orient="records")

        grouped = frame.groupby(["provider", "model", "condition", "variety"], dropna=False)
        for key, group in grouped:
            errors = group["error"].notna() & group["error"].astype(str).str.strip().ne("")
            parse_errors = group["parse_error"].fillna(False).astype(bool)
            predictions = group["predicted_category"].map(clean_value).str.lower()
            invalid = predictions.ne("") & ~predictions.isin(EXPECTED_LABELS)
            coverage.append(
                {
                    "provider": str(key[0]),
                    "model": str(key[1]),
                    "condition": str(key[2]),
                    "variety": str(key[3]),
                    "rows": len(group),
                    "unique_items": int(group["id"].nunique()),
                    "api_errors": int(errors.sum()),
                    "parse_errors": int(parse_errors.sum()),
                    "blank_predictions": int(predictions.eq("").sum()),
                    "invalid_predictions": int(invalid.sum()),
                    "valid_predictions": int((~errors & ~parse_errors & predictions.isin(EXPECTED_LABELS)).sum()),
                }
            )

        predictions = frame["predicted_category"].map(clean_value).str.lower()
        invalid_predictions = predictions[~predictions.isin(EXPECTED_LABELS) & predictions.ne("")].value_counts().to_dict()

    if "id" in frame:
        ids, invalid_ids = numeric_ids(frame["id"])
        unexpected_ids = sorted(set(ids.tolist()) - set(benchmark["id"].astype(int).tolist()))

    if {"id", "variety", "text"}.issubset(frame.columns):
        expected_text: dict[tuple[int, str], str] = {}
        for _, row in benchmark.iterrows():
            for variety, column in VARIETY_COLUMNS.items():
                expected_text[(int(row["id"]), variety)] = clean_value(row[column])
        for _, row in frame.iterrows():
            try:
                key = (int(row["id"]), str(row["variety"]))
            except (TypeError, ValueError):
                continue
            expected = expected_text.get(key)
            actual = clean_value(row["text"])
            if expected is not None and actual != expected:
                text_mismatches.append({"id": key[0], "variety": key[1]})

    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "invalid_json_lines": invalid_lines,
        "missing_required_columns": missing_columns,
        "duplicate_key_rows": duplicate_keys,
        "invalid_ids": invalid_ids,
        "unexpected_ids": unexpected_ids,
        "invalid_prediction_counts": invalid_predictions,
        "text_mismatches": text_mismatches,
        "coverage": coverage,
    }


def load_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def audit_legacy_file(path: Path, expected_ids: set[int]) -> dict[str, Any]:
    try:
        frame = load_tabular(path)
    except Exception as error:  # noqa: BLE001 - preserve file-specific audit errors
        return {"path": str(path), "sha256": sha256(path), "read_error": str(error)}

    id_column = "id" if "id" in frame.columns else None
    invalid_ids: list[str] = []
    duplicate_ids: list[int] = []
    missing_ids: list[int] = []
    unexpected_ids: list[int] = []
    valid_id_rows = 0
    if id_column:
        ids, invalid_ids = numeric_ids(frame[id_column])
        valid_id_rows = len(ids)
        duplicate_ids = sorted(ids[ids.duplicated(keep=False)].unique().tolist())
        observed = set(ids.tolist())
        missing_ids = sorted(expected_ids - observed)
        unexpected_ids = sorted(observed - expected_ids)

    prediction_columns = [
        column
        for column in frame.columns
        if "category" in str(column).lower() and str(column).lower() != "category"
    ]
    prediction_coverage: dict[str, Any] = {}
    for column in prediction_columns:
        values = frame[column].map(clean_value).str.lower()
        prediction_coverage[str(column)] = {
            "nonblank": int(values.ne("").sum()),
            "valid_label": int(values.isin(EXPECTED_LABELS).sum()),
            "invalid_values": values[values.ne("") & ~values.isin(EXPECTED_LABELS)].value_counts().head(20).to_dict(),
        }

    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(frame),
        "columns": [str(c) for c in frame.columns],
        "unnamed_columns": [str(c) for c in frame.columns if str(c).startswith("Unnamed:")],
        "has_id_column": bool(id_column),
        "valid_id_rows": valid_id_rows,
        "invalid_ids": invalid_ids,
        "duplicate_ids": duplicate_ids,
        "missing_expected_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "fully_blank_rows": int(frame.isna().all(axis=1).sum()),
        "prediction_columns": prediction_coverage,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def status_lines(report: dict[str, Any]) -> list[str]:
    benchmark = report["benchmark"]
    lines = [
        "# Study Input Audit",
        "",
        "## Benchmark",
        "",
        f"- Rows: {benchmark['rows']}",
        f"- SHA-256: `{benchmark['sha256']}`",
        f"- Missing required columns: {benchmark['missing_required_columns'] or 'none'}",
        f"- Duplicate IDs: {benchmark['duplicate_ids'] or 'none'}",
        f"- Missing IDs: {benchmark['missing_expected_ids'] or 'none'}",
        f"- Unknown labels: {benchmark['unknown_labels'] or 'none'}",
        f"- Label counts: {benchmark['label_counts']}",
        f"- Dataset types: {benchmark['dataset_type_counts']}",
        f"- PII-pattern candidates requiring human review: {len(benchmark['pii_candidates'])}",
        "",
        "### Word counts",
        "",
        "| Variety | Mean | Median | Min | Max | 6–10 words |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variety, details in benchmark["text"].items():
        counts = details["word_count"]
        lines.append(
            f"| {variety} | {counts['mean']} | {counts['median']} | {counts['min']} | {counts['max']} | {counts['in_6_to_10_band']} |"
        )

    lines.extend(["", "## Current JSONL results"])
    for results in report["current_results"]:
        lines.extend(
            [
                "",
                f"### `{results['path']}`",
                "",
                f"- SHA-256: `{results['sha256']}`",
                f"- Rows: {results['rows']}",
                f"- Invalid JSON lines: {results['invalid_json_lines'] or 'none'}",
                f"- Duplicate evaluation keys: {len(results['duplicate_key_rows'])}",
                f"- Text mismatches against benchmark: {len(results['text_mismatches'])}",
                "",
                "| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for cell in results["coverage"]:
            lines.append(
                "| {provider} | {model} | {condition} | {variety} | {rows} | {valid_predictions} | {api_errors} | {parse_errors} |".format(
                    **cell
                )
            )

    lines.extend(
        [
            "",
            "## Legacy files",
            "",
            "| File | Rows | ID rows | Duplicate IDs | Missing IDs | Prediction columns | Read error |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for entry in report["legacy_files"]:
        lines.append(
            f"| `{entry['path']}` | {entry.get('rows', '')} | {entry.get('valid_id_rows', '')} | "
            f"{len(entry.get('duplicate_ids', []))} | {len(entry.get('missing_expected_ids', []))} | "
            f"{len(entry.get('prediction_columns', {}))} | {entry.get('read_error', '')} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("data/language_variety_audit_benchmark.csv"))
    parser.add_argument(
        "--current-results",
        type=Path,
        action="append",
        help="Repeat for selected files; defaults to all five fresh and two probability runs.",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("data/source-results"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/study/audits"))
    args = parser.parse_args()

    benchmark_report = audit_benchmark(args.benchmark)
    benchmark_frame = pd.read_csv(args.benchmark)
    result_paths = args.current_results or [
        Path("results/study/raw") / name for name in CURRENT_RESULT_SPECS
    ]
    current_results = [audit_jsonl_results(path, benchmark_frame) for path in result_paths]
    for path in result_paths:
        if path.name in CURRENT_RESULT_SPECS:
            model, rows = CURRENT_RESULT_SPECS[path.name]
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("validate_evaluation_output.py")),
                    str(path),
                    "--expected-model", model,
                    "--expected-rows", str(rows),
                ],
                check=True,
            )
    model_dir = Path("results/study/model_manifests")
    if model_dir.is_dir():
        model_paths = sorted(path for path in model_dir.glob("*.json") if path.name != "summary.json")
        (model_dir / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {portable_source(path)}\n" for path in model_paths),
            encoding="utf-8",
        )
    expected_ids = set(benchmark_frame["id"].astype(int).tolist())
    legacy_paths = sorted(
        path for path in args.legacy_root.rglob("*") if path.suffix.lower() in {".csv", ".xlsx"}
    )
    legacy_reports = [audit_legacy_file(path, expected_ids) for path in legacy_paths]
    for entry in [benchmark_report, *current_results, *legacy_reports]:
        entry["path"] = portable_source(entry["path"])

    report = json_safe(
        {
            "benchmark": benchmark_report,
            "current_results": current_results,
            "legacy_files": legacy_reports,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "input_audit.json"
    markdown_path = args.out_dir / "input_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text("\n".join(status_lines(report)) + "\n", encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
