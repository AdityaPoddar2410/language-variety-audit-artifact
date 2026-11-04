"""Repeated item-split conformal prediction by language variety."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .prompts import CONTENT_CATEGORIES


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if len(values) == 0:
        raise ValueError("Calibration scores cannot be empty")
    rank = min(math.ceil((len(values) + 1) * (1 - alpha)), len(values))
    return float(values[rank - 1])


def _evaluate_sets(frame: pd.DataFrame, threshold: float) -> tuple[float, float]:
    probability_columns = [f"p_{label}" for label in CONTENT_CATEGORIES]
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    gold_indices = np.array(
        [CONTENT_CATEGORIES.index(label) for label in frame["gold_category"]],
        dtype=int,
    )
    included = 1.0 - probabilities <= threshold
    coverage = included[np.arange(len(frame)), gold_indices].mean()
    mean_size = included.sum(axis=1).mean()
    return float(coverage), float(mean_size)


def repeated_conformal_evaluation(
    frame: pd.DataFrame,
    alpha: float = 0.1,
    calibration_fraction: float = 0.5,
    repetitions: int = 200,
    seed: int = 2027,
) -> pd.DataFrame:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for model_id, model_frame in frame.groupby("model_id"):
        item_ids = np.sort(model_frame["id"].unique())
        calibration_size = round(calibration_fraction * len(item_ids))
        if calibration_size < 1 or calibration_size >= len(item_ids):
            raise ValueError("Calibration split leaves an empty calibration or test set")

        for repetition in range(repetitions):
            shuffled = rng.permutation(item_ids)
            calibration_ids = set(shuffled[:calibration_size].tolist())
            calibration = model_frame[model_frame["id"].isin(calibration_ids)].copy()
            test = model_frame[~model_frame["id"].isin(calibration_ids)].copy()
            calibration["score"] = 1.0 - calibration.apply(
                lambda record: record[f"p_{record['gold_category']}"],
                axis=1,
            )
            pooled_threshold = conformal_quantile(
                calibration["score"].to_numpy(),
                alpha,
            )
            group_thresholds = {
                variety: conformal_quantile(group["score"].to_numpy(), alpha)
                for variety, group in calibration.groupby("variety")
            }

            for variety, test_group in test.groupby("variety"):
                pooled_coverage, pooled_size = _evaluate_sets(
                    test_group,
                    pooled_threshold,
                )
                conditional_coverage, conditional_size = _evaluate_sets(
                    test_group,
                    group_thresholds[variety],
                )
                rows.extend(
                    [
                        {
                            "model_id": model_id,
                            "repetition": repetition,
                            "method": "pooled_marginal",
                            "variety": variety,
                            "n_calibration_items": calibration_size,
                            "n_test_items": len(test_group),
                            "alpha": alpha,
                            "threshold": pooled_threshold,
                            "coverage": pooled_coverage,
                            "mean_set_size": pooled_size,
                        },
                        {
                            "model_id": model_id,
                            "repetition": repetition,
                            "method": "variety_conditional",
                            "variety": variety,
                            "n_calibration_items": calibration_size,
                            "n_test_items": len(test_group),
                            "alpha": alpha,
                            "threshold": group_thresholds[variety],
                            "coverage": conditional_coverage,
                            "mean_set_size": conditional_size,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def conformal_summary(split_results: pd.DataFrame) -> pd.DataFrame:
    grouped = split_results.groupby(["model_id", "method", "variety"], dropna=False)
    summary = grouped.agg(
        repetitions=("repetition", "nunique"),
        mean_coverage=("coverage", "mean"),
        sd_coverage=("coverage", "std"),
        mean_set_size=("mean_set_size", "mean"),
        sd_set_size=("mean_set_size", "std"),
        mean_threshold=("threshold", "mean"),
    ).reset_index()
    quantiles = grouped["coverage"].quantile([0.025, 0.975]).unstack().reset_index()
    quantiles = quantiles.rename(columns={0.025: "coverage_q025", 0.975: "coverage_q975"})
    return summary.merge(quantiles, on=["model_id", "method", "variety"], how="left")
