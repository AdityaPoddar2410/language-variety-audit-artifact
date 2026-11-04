"""I/O helpers for the Language-Variety Audit pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Columns expected in the released benchmark CSV.
BENCHMARK_COLUMNS = [
    "id",
    "word_or_phrase",
    "hinglish_example",
    "indian_english_translation",
    "indian_meaning",
    "category",
    "sae_example",
    "dataset_type",
]

# Language conditions and their benchmark columns.
VARIETIES = ["sae", "indian_english", "hinglish"]
VARIETY_TO_COLUMN = {
    "sae": "sae_example",
    "indian_english": "indian_english_translation",
    "hinglish": "hinglish_example",
}


def load_benchmark(path: str | Path) -> pd.DataFrame:
    """Load the benchmark CSV and validate required columns are present."""
    df = pd.read_csv(path)
    missing = [c for c in BENCHMARK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Benchmark file {path} is missing required columns: {missing}"
        )
    return df


def write_results_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of result records to JSON-lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines results file."""
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_results(path: str | Path) -> pd.DataFrame:
    """Read results from .jsonl, .csv, or .xlsx into a DataFrame."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return pd.DataFrame(read_results_jsonl(p))
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in {".xls", ".xlsx"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported results file extension: {p.suffix}")
