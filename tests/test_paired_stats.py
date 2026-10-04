from __future__ import annotations

import pandas as pd
import pytest

from beyond_weird.paired_stats import (
    aggregate_paired_effects,
    aggregate_reasoning_effects,
    holm_adjust,
    model_paired_effects,
)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    outcomes = {
        "sae": [1, 1, 1, 1],
        "indian_english": [1, 1, 0, 0],
        "hinglish": [1, 0, 0, 0],
    }
    for model in ("model-a", "model-b"):
        for variety, values in outcomes.items():
            for item_id, correct in enumerate(values, start=1):
                rows.append(
                    {
                        "id": item_id,
                        "cohort": "fresh",
                        "provider": "provider",
                        "model": model,
                        "model_id": model,
                        "system_id": f"{model}::test",
                        "prompt_cue": "none",
                        "condition": "direct",
                        "variety": variety,
                        "correct": bool(correct),
                        "valid": True,
                    }
                )
    return pd.DataFrame(rows)


def test_holm_adjustment_is_monotone_in_rank() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_model_paired_effects_use_within_item_differences() -> None:
    result = model_paired_effects(
        _synthetic_frame(),
        n_bootstrap=200,
        n_permutations=500,
        seed=7,
    )
    row = result[
        (result["model_id"] == "model-a")
        & (result["target_variety"] == "hinglish")
    ].iloc[0]
    assert row["n_paired"] == 4
    assert row["gap_pp"] == pytest.approx(75.0)
    assert row["sae_only"] == 3
    assert row["target_only"] == 0


def test_aggregate_effect_balances_models_and_clusters_items() -> None:
    result = aggregate_paired_effects(
        _synthetic_frame(),
        n_bootstrap=200,
        n_permutations=500,
        seed=7,
    )
    row = result[result["target_variety"] == "indian_english"].iloc[0]
    assert row["n_items"] == 4
    assert row["n_systems"] == 2
    assert row["gap_pp"] == pytest.approx(50.0)


def test_aggregate_reasoning_effect_is_cot_minus_direct() -> None:
    rows = []
    for item_id in range(1, 5):
        for model in ("model-a", "model-b"):
            for condition, correct in (("direct", item_id == 1), ("cot", item_id <= 3)):
                rows.append(
                    {
                        "id": item_id,
                        "cohort": "fresh",
                        "provider": "provider",
                        "model": model,
                        "model_id": model,
                        "system_id": f"{model}::test",
                        "prompt_cue": "none",
                        "condition": condition,
                        "variety": "hinglish",
                        "correct": correct,
                        "valid": True,
                    }
                )
    result = aggregate_reasoning_effects(
        pd.DataFrame(rows),
        n_bootstrap=200,
        n_permutations=500,
        seed=7,
    )
    row = result.iloc[0]
    assert row["reasoning_effect_pp"] == pytest.approx(50.0)
    assert row["accuracy_direct"] == pytest.approx(0.25)
    assert row["accuracy_cot"] == pytest.approx(0.75)
