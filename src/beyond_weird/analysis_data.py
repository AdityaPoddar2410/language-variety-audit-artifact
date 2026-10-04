"""Canonical study analysis tables from historical and fresh result artifacts."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .prompts import CONTENT_CATEGORIES
from .run_records import result_key

HISTORICAL_FOUR_MODEL_SOURCE = Path(
    "results/historical/historical-four-model-results.csv"
)
HISTORICAL_GEMMA_SOURCE = Path("results/historical/historical-gemma-results.csv")

_HISTORICAL_MODELS = [
    {
        "model": "GPT-4o-mini",
        "model_id": "gpt-4o-mini",
        "provider": "OpenAI",
        "source": HISTORICAL_FOUR_MODEL_SOURCE,
        "column_suffix": "openai",
    },
    {
        "model": "Claude Haiku 3.5",
        "model_id": "claude-3-5-haiku-20241022",
        "provider": "Anthropic",
        "source": HISTORICAL_FOUR_MODEL_SOURCE,
        "column_suffix": "anthropic",
    },
    {
        "model": "Llama 3.1 8B",
        "model_id": "llama3.1:8b",
        "provider": "Meta/Ollama",
        "source": HISTORICAL_FOUR_MODEL_SOURCE,
        "column_suffix": "ollama",
    },
    {
        "model": "Gemma 3 12B",
        "model_id": "gemma3:12b",
        "provider": "Google/Ollama",
        "source": HISTORICAL_GEMMA_SOURCE,
        "column_suffix": "ollama",
    },
]
_VARIETY_PREFIX = {
    "sae": "sae",
    "indian_english": "indian_english",
    "hinglish": "hinglish",
}
_CONDITION_COLUMN = {
    "direct": "category_only",
    "cot": "category_from_meaning",
}
_MODEL_DISPLAY = {
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "llama3.1:8b": "Llama 3.1 8B",
    "gemma3:12b": "Gemma 3 12B",
    "mistral:7b": "Mistral 7B Instruct",
}
_PROVIDER_DISPLAY = {
    "openai": "OpenAI",
    "gemini": "Google Vertex AI",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
}
_VALID_LABELS = set(CONTENT_CATEGORIES)


def _normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def load_historical_accuracy() -> pd.DataFrame:
    sources = {
        path: pd.read_csv(path).loc[lambda frame: pd.to_numeric(frame["id"], errors="coerce").notna()].copy()
        for path in {model["source"] for model in _HISTORICAL_MODELS}
    }
    rows: list[dict[str, Any]] = []
    for model in _HISTORICAL_MODELS:
        source = model["source"]
        frame = sources[source]
        frame["id"] = pd.to_numeric(frame["id"], errors="raise").astype(int)
        if len(frame) != 200 or frame["id"].nunique() != 200:
            raise ValueError(
                f"Historical source does not contain 200 unique IDs: {source}"
            )
        for variety, prefix in _VARIETY_PREFIX.items():
            for condition, column_kind in _CONDITION_COLUMN.items():
                prediction_column = f"{prefix}_{column_kind}_{model['column_suffix']}"
                if prediction_column not in frame:
                    raise ValueError(f"Missing historical column {prediction_column!r} in {source}")
                for _, record in frame.iterrows():
                    gold = _normalize_label(record["category"])
                    prediction = _normalize_label(record[prediction_column])
                    rows.append(
                        {
                            "id": int(record["id"]),
                            "model": model["model"],
                            "model_id": model["model_id"],
                            "system_id": f"{model['model_id']}::historical-no-cue",
                            "provider": model["provider"],
                            "cohort": "historical",
                            "prompt_cue": "none",
                            "condition": condition,
                            "variety": variety,
                            "gold_category": gold,
                            "predicted_category": prediction,
                            "correct": prediction == gold,
                            "valid": prediction in _VALID_LABELS,
                            "source": str(source),
                            "run_id": "historical-single-run",
                        }
                    )
    return pd.DataFrame(rows)


def _read_latest_jsonl(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            latest[result_key(record)] = record
    return list(latest.values())


def load_fresh_accuracy(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        for record in _read_latest_jsonl(path):
            model_id = str(record.get("model", ""))
            gold = _normalize_label(record.get("gold_category"))
            prediction = _normalize_label(record.get("predicted_category"))
            valid = (
                not record.get("error")
                and not record.get("parse_error")
                and prediction in _VALID_LABELS
                and gold in _VALID_LABELS
            )
            rows.append(
                {
                    "id": int(record["id"]),
                    "model": _MODEL_DISPLAY.get(model_id, model_id),
                    "model_id": model_id,
                    "system_id": f"{model_id}::fresh-no-cue",
                    "provider": _PROVIDER_DISPLAY.get(
                        str(record.get("provider", "")),
                        str(record.get("provider", "")),
                    ),
                    "cohort": "fresh",
                    "prompt_cue": "none",
                    "condition": str(record.get("condition", "")),
                    "variety": str(record.get("variety", "")),
                    "gold_category": gold,
                    "predicted_category": prediction,
                    "correct": prediction == gold if valid else False,
                    "valid": valid,
                    "source": str(path),
                    "run_id": str(record.get("run_id", "")),
                }
            )
    return pd.DataFrame(rows)


def build_accuracy_table(current_paths: Iterable[str | Path]) -> pd.DataFrame:
    frame = pd.concat(
        [load_historical_accuracy(), load_fresh_accuracy(current_paths)],
        ignore_index=True,
    )
    key = ["id", "system_id", "condition", "variety"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, key].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate canonical accuracy cells: {examples}")
    return frame.sort_values(key).reset_index(drop=True)


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "cohort",
                "provider",
                "model",
                "model_id",
                "system_id",
                "prompt_cue",
                "condition",
                "variety",
            ],
            dropna=False,
        )
        .agg(rows=("id", "size"), unique_items=("id", "nunique"), valid=("valid", "sum"))
        .reset_index()
        .sort_values(["cohort", "model", "condition", "variety"])
    )
