"""Calibration metrics and paired variety comparisons."""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prompts import CONTENT_CATEGORIES
from .run_records import result_key

_EPSILON = 1e-15


def _read_latest(path: Path) -> list[dict[str, Any]]:
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


def load_probability_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_labels = set(CONTENT_CATEGORIES)
    for value in paths:
        path = Path(value)
        for record in _read_latest(path):
            if record.get("error") or record.get("parse_error"):
                continue
            probabilities = record.get("class_probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != expected_labels:
                raise ValueError(f"Invalid class probability keys in {path}, id={record.get('id')}")
            vector = np.array([float(probabilities[label]) for label in CONTENT_CATEGORIES])
            if not np.isfinite(vector).all() or (vector < 0).any():
                raise ValueError(f"Invalid probability values in {path}, id={record.get('id')}")
            if not math.isclose(float(vector.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Probabilities do not sum to one in {path}, id={record.get('id')}")

            gold = str(record["gold_category"]).strip().lower()
            prediction = str(record["predicted_category"]).strip().lower()
            if gold not in expected_labels or prediction not in expected_labels:
                raise ValueError(f"Invalid gold/prediction in {path}, id={record.get('id')}")
            gold_index = CONTENT_CATEGORIES.index(gold)
            target = np.zeros(len(CONTENT_CATEGORIES), dtype=float)
            target[gold_index] = 1.0
            confidence = float(vector.max())
            row = {
                "id": int(record["id"]),
                "model_id": str(record["model"]),
                "resolved_model": str(record.get("resolved_model") or record["model"]),
                "provider": str(record["provider"]),
                "variety": str(record["variety"]),
                "condition": str(record["condition"]),
                "gold_category": gold,
                "predicted_category": prediction,
                "correct": prediction == gold,
                "confidence": confidence,
                "nll": -math.log(max(float(vector[gold_index]), _EPSILON)),
                "brier": float(np.square(vector - target).sum()),
                "source": str(path),
                "run_id": str(record.get("run_id", "")),
            }
            row.update({f"p_{label}": vector[index] for index, label in enumerate(CONTENT_CATEGORIES)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    key = ["id", "model_id", "condition", "variety"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate probability cells: {frame.loc[duplicates, key].head().to_dict(orient='records')}")
    return frame.sort_values(key).reset_index(drop=True)


def expected_calibration_error(
    correctness: np.ndarray,
    confidence: np.ndarray,
    bins: int = 10,
    equal_mass: bool = False,
) -> float:
    correctness = np.asarray(correctness, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if len(correctness) == 0:
        return float("nan")
    if equal_mass:
        groups = np.array_split(np.argsort(confidence), min(bins, len(confidence)))
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        indices = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
        groups = [np.flatnonzero(indices == index) for index in range(bins)]
    error = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        weight = len(group) / len(correctness)
        error += weight * abs(float(correctness[group].mean() - confidence[group].mean()))
    return error


def calibration_summary(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(["provider", "model_id", "condition", "variety"]):
        correctness = group["correct"].to_numpy(dtype=float)
        confidence = group["confidence"].to_numpy(dtype=float)
        rows.append(
            {
                "provider": key[0],
                "model_id": key[1],
                "condition": key[2],
                "variety": key[3],
                "n": len(group),
                "accuracy": float(correctness.mean()),
                "mean_confidence": float(confidence.mean()),
                "confidence_minus_accuracy": float(confidence.mean() - correctness.mean()),
                "nll": float(group["nll"].mean()),
                "brier": float(group["brier"].mean()),
                "ece": expected_calibration_error(correctness, confidence, bins=bins),
                "ace": expected_calibration_error(
                    correctness,
                    confidence,
                    bins=bins,
                    equal_mass=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def reliability_table(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for key, group in frame.groupby(["provider", "model_id", "condition", "variety"]):
        confidence = group["confidence"].to_numpy(dtype=float)
        correctness = group["correct"].to_numpy(dtype=float)
        indices = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
        for index in range(bins):
            selected = indices == index
            if not selected.any():
                continue
            rows.append(
                {
                    "provider": key[0],
                    "model_id": key[1],
                    "condition": key[2],
                    "variety": key[3],
                    "bin": index + 1,
                    "lower": edges[index],
                    "upper": edges[index + 1],
                    "n": int(selected.sum()),
                    "mean_confidence": float(confidence[selected].mean()),
                    "accuracy": float(correctness[selected].mean()),
                }
            )
    return pd.DataFrame(rows)


def paired_calibration_differences(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    bins: int = 10,
    seed: int = 2027,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ("correct", "confidence", "nll", "brier")
    targets = ("indian_english", "hinglish")
    for model_index, (model_id, model_frame) in enumerate(frame.groupby("model_id")):
        for target_index, target_variety in enumerate(targets):
            sae = model_frame[model_frame["variety"] == "sae"].set_index("id").sort_index()
            target = model_frame[model_frame["variety"] == target_variety].set_index("id").sort_index()
            common_ids = sae.index.intersection(target.index)
            sae = sae.loc[common_ids]
            target = target.loc[common_ids]
            if len(common_ids) == 0:
                continue
            rng = np.random.default_rng(seed + model_index * 10 + target_index)
            draws = rng.integers(0, len(common_ids), size=(n_bootstrap, len(common_ids)))
            for metric in metrics:
                sae_values = sae[metric].to_numpy(dtype=float)
                target_values = target[metric].to_numpy(dtype=float)
                differences = target_values - sae_values
                estimates = differences[draws].mean(axis=1)
                lower, upper = np.quantile(estimates, [0.025, 0.975])
                rows.append(
                    {
                        "model_id": model_id,
                        "target_variety": target_variety,
                        "metric": metric,
                        "n_paired": len(common_ids),
                        "target_minus_sae": float(differences.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )

            sae_correct = sae["correct"].to_numpy(dtype=float)
            sae_confidence = sae["confidence"].to_numpy(dtype=float)
            target_correct = target["correct"].to_numpy(dtype=float)
            target_confidence = target["confidence"].to_numpy(dtype=float)
            for metric, equal_mass in (("ece", False), ("ace", True)):
                point = expected_calibration_error(
                    target_correct,
                    target_confidence,
                    bins=bins,
                    equal_mass=equal_mass,
                ) - expected_calibration_error(
                    sae_correct,
                    sae_confidence,
                    bins=bins,
                    equal_mass=equal_mass,
                )
                estimates = np.empty(n_bootstrap, dtype=float)
                for index, draw in enumerate(draws):
                    estimates[index] = expected_calibration_error(
                        target_correct[draw],
                        target_confidence[draw],
                        bins=bins,
                        equal_mass=equal_mass,
                    ) - expected_calibration_error(
                        sae_correct[draw],
                        sae_confidence[draw],
                        bins=bins,
                        equal_mass=equal_mass,
                    )
                lower, upper = np.quantile(estimates, [0.025, 0.975])
                rows.append(
                    {
                        "model_id": model_id,
                        "target_variety": target_variety,
                        "metric": metric,
                        "n_paired": len(common_ids),
                        "target_minus_sae": point,
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
    return pd.DataFrame(rows)
