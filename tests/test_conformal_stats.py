from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beyond_weird.conformal_stats import (
    conformal_quantile,
    conformal_summary,
    repeated_conformal_evaluation,
)
from beyond_weird.prompts import CONTENT_CATEGORIES


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1, 11, dtype=float) / 10
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(0.9)


def test_repeated_conformal_split_keeps_items_grouped() -> None:
    rows = []
    for item_id in range(1, 11):
        for variety in ("sae", "indian_english", "hinglish"):
            probabilities = {label: 0.02 for label in CONTENT_CATEGORIES}
            probabilities["normal"] = 0.9
            rows.append(
                {
                    "id": item_id,
                    "model_id": "model-a",
                    "variety": variety,
                    "gold_category": "normal",
                    **{f"p_{label}": value for label, value in probabilities.items()},
                }
            )
    results = repeated_conformal_evaluation(
        pd.DataFrame(rows),
        alpha=0.1,
        calibration_fraction=0.5,
        repetitions=3,
        seed=7,
    )
    assert len(results) == 3 * 2 * 3
    assert set(results["n_calibration_items"]) == {5}
    assert set(results["n_test_items"]) == {5}
    assert set(results["coverage"]) == {1.0}
    summary = conformal_summary(results)
    assert set(summary["repetitions"]) == {3}
