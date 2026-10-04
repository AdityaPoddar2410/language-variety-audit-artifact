from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beyond_weird.calibration_stats import (
    expected_calibration_error,
    paired_calibration_differences,
)


def test_expected_calibration_error_matches_weighted_bins() -> None:
    correctness = np.array([1.0, 0.0, 1.0, 1.0])
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    value = expected_calibration_error(correctness, confidence, bins=4)
    assert value == pytest.approx(0.35)


def test_paired_calibration_difference_uses_shared_items() -> None:
    rows = []
    for item_id in range(1, 5):
        rows.append(
            {
                "id": item_id,
                "model_id": "model-a",
                "variety": "sae",
                "correct": True,
                "confidence": 0.9,
                "nll": 0.1,
                "brier": 0.02,
            }
        )
        rows.append(
            {
                "id": item_id,
                "model_id": "model-a",
                "variety": "indian_english",
                "correct": item_id <= 2,
                "confidence": 0.85,
                "nll": 0.3,
                "brier": 0.2,
            }
        )
        rows.append(
            {
                "id": item_id,
                "model_id": "model-a",
                "variety": "hinglish",
                "correct": item_id == 1,
                "confidence": 0.8,
                "nll": 0.5,
                "brier": 0.4,
            }
        )
    result = paired_calibration_differences(
        pd.DataFrame(rows),
        n_bootstrap=100,
        bins=2,
        seed=7,
    )
    accuracy = result[
        (result["target_variety"] == "hinglish")
        & (result["metric"] == "correct")
    ].iloc[0]
    confidence = result[
        (result["target_variety"] == "hinglish")
        & (result["metric"] == "confidence")
    ].iloc[0]
    assert accuracy["target_minus_sae"] == pytest.approx(-0.75)
    assert confidence["target_minus_sae"] == pytest.approx(-0.1)
