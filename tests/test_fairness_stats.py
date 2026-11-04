from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from language_variety_audit.fairness_stats import (
    aggregate_fairness_effects,
    aggregate_fairness_effects_by_stratum,
    aggregate_reasoning_fairness_effects,
    fairness_indicators,
    fairness_summary,
    overflag_relative_risk,
    system_overflag_consistency,
)


def _synthetic_frame() -> pd.DataFrame:
    gold = {
        1: "normal",
        2: "normal",
        3: "harassment",
        4: "offensive",
    }
    predictions = {
        "direct": {
            "sae": ["normal", "normal", "harassment", "normal"],
            "indian_english": ["offensive", "normal", "normal", "normal"],
            "hinglish": ["offensive", "offensive", "normal", "offensive"],
        },
        "cot": {
            "sae": ["normal", "normal", "normal", "normal"],
            "indian_english": ["normal", "normal", "normal", "normal"],
            "hinglish": ["normal", "normal", "harassment", "normal"],
        },
    }
    rows = []
    for system in ("model-a", "model-b"):
        for condition, by_variety in predictions.items():
            for variety, values in by_variety.items():
                for item_id, prediction in enumerate(values, start=1):
                    rows.append(
                        {
                            "id": item_id,
                            "cohort": "new",
                            "provider": "provider",
                            "model": system,
                            "model_id": system,
                            "system_id": f"{system}::test",
                            "prompt_cue": "none",
                            "condition": condition,
                            "variety": variety,
                            "gold_category": gold[item_id],
                            "predicted_category": prediction,
                            "correct": prediction == gold[item_id],
                            "valid": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_fairness_indicators_apply_only_to_relevant_items() -> None:
    indicators = fairness_indicators(_synthetic_frame())
    normal = indicators[indicators["gold_category"] == "normal"]
    harmful = indicators[indicators["gold_category"] != "normal"]

    assert normal["overflag"].notna().all()
    assert normal["harm_miss"].isna().all()
    assert normal["harm_category_error"].isna().all()
    assert harmful["overflag"].isna().all()
    assert harmful["harm_miss"].notna().all()
    assert harmful["harm_category_error"].notna().all()


def test_fairness_summary_uses_documented_error_definitions() -> None:
    summary = fairness_summary(_synthetic_frame())
    row = summary[
        (summary["system_id"] == "model-a::test")
        & (summary["condition"] == "direct")
        & (summary["variety"] == "indian_english")
    ].iloc[0]

    assert row["n_normal"] == 2
    assert row["overflag_rate"] == pytest.approx(0.5)
    assert row["n_harmful"] == 2
    assert row["harm_miss_rate"] == pytest.approx(1.0)
    assert row["harm_category_error_rate"] == pytest.approx(1.0)


def test_aggregate_variety_effect_is_target_minus_sae() -> None:
    result = aggregate_fairness_effects(
        _synthetic_frame(),
        n_bootstrap=100,
        n_permutations=200,
        seed=7,
    )
    overflag = result[
        (result["condition"] == "direct")
        & (result["metric"] == "overflag")
        & (result["target_variety"] == "hinglish")
    ].iloc[0]

    assert overflag["n_items"] == 2
    assert overflag["n_systems"] == 2
    assert overflag["sae_rate"] == pytest.approx(0.0)
    assert overflag["target_rate"] == pytest.approx(1.0)
    assert overflag["difference_pp"] == pytest.approx(100.0)
    assert overflag["cluster_permutation_p_holm"] >= overflag["cluster_permutation_p"]


def test_reasoning_effect_is_cot_minus_direct() -> None:
    result = aggregate_reasoning_fairness_effects(
        _synthetic_frame(),
        n_bootstrap=100,
        n_permutations=200,
        seed=7,
    )
    overflag = result[
        (result["variety"] == "hinglish")
        & (result["metric"] == "overflag")
    ].iloc[0]
    sae_miss = result[
        (result["variety"] == "sae") & (result["metric"] == "harm_miss")
    ].iloc[0]

    assert overflag["difference_pp"] == pytest.approx(-100.0)
    assert sae_miss["difference_pp"] == pytest.approx(50.0)


def test_duplicate_cells_are_rejected() -> None:
    frame = _synthetic_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate fairness cells"):
        fairness_summary(duplicate)


def test_missing_cells_are_rejected() -> None:
    frame = _synthetic_frame().drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="Incomplete fairness cells"):
        fairness_summary(frame)


def test_invalid_cells_are_rejected() -> None:
    frame = _synthetic_frame()
    frame.loc[0, "valid"] = False
    with pytest.raises(ValueError, match="requires every canonical cell to be valid"):
        fairness_summary(frame)


def test_frozen_fairness_headlines_match_canonical_accuracy_table() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "results/study/validated/accuracy_long.csv")
    variety = aggregate_fairness_effects(
        frame,
        n_bootstrap=100,
        n_permutations=200,
        seed=7,
    )
    reasoning = aggregate_reasoning_fairness_effects(
        frame,
        n_bootstrap=100,
        n_permutations=200,
        seed=7,
    )

    direct_indian = variety[
        (variety["condition"] == "direct")
        & (variety["metric"] == "overflag")
        & (variety["target_variety"] == "indian_english")
    ].iloc[0]
    direct_hinglish = variety[
        (variety["condition"] == "direct")
        & (variety["metric"] == "overflag")
        & (variety["target_variety"] == "hinglish")
    ].iloc[0]
    reasoning_indian = reasoning[
        (reasoning["variety"] == "indian_english")
        & (reasoning["metric"] == "harm_miss")
    ].iloc[0]
    reasoning_hinglish = reasoning[
        (reasoning["variety"] == "hinglish")
        & (reasoning["metric"] == "harm_miss")
    ].iloc[0]

    assert direct_indian["difference_pp"] == pytest.approx(10.1851851852)
    assert direct_hinglish["difference_pp"] == pytest.approx(11.6809116809)
    assert reasoning_indian["difference_pp"] == pytest.approx(11.1111111111)
    assert reasoning_hinglish["difference_pp"] == pytest.approx(14.8989898990)


def test_equal_model_sensitivity_uses_seven_model_units() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "results/study/validated/accuracy_long.csv")
    result = aggregate_fairness_effects(
        frame,
        n_bootstrap=50,
        n_permutations=100,
        seed=7,
        unit_column="model",
    )

    assert set(result["n_models"]) == {7}


def _flag_frame() -> pd.DataFrame:
    """Two normal items, two systems; SAE flags exist so ratios are finite."""
    flags = {
        # (item, system): (sae, indian_english, hinglish) over-flag indicators
        (1, "a"): (1, 1, 1),
        (1, "b"): (0, 1, 0),
        (2, "a"): (0, 1, 1),
        (2, "b"): (1, 0, 1),
    }
    rows = []
    for (item_id, system), values in flags.items():
        for condition in ("direct", "cot"):
            for variety, flagged in zip(
                ("sae", "indian_english", "hinglish"), values, strict=True
            ):
                prediction = "offensive" if flagged else "normal"
                rows.append(
                    {
                        "id": item_id,
                        "cohort": "new",
                        "provider": "provider",
                        "model": system,
                        "model_id": system,
                        "system_id": f"{system}::test",
                        "prompt_cue": "none",
                        "condition": condition,
                        "variety": variety,
                        "gold_category": "normal",
                        "predicted_category": prediction,
                        "correct": prediction == "normal",
                        "valid": True,
                        "family": "x" if item_id == 1 else "y",
                    }
                )
    return pd.DataFrame(rows)


def test_relative_risk_and_discordance_follow_definitions() -> None:
    result = overflag_relative_risk(_flag_frame(), n_bootstrap=50, seed=3)
    row = result[
        (result["condition"] == "direct")
        & (result["target_variety"] == "indian_english")
    ].iloc[0]

    assert row["n_item_system_pairs"] == 4
    assert row["sae_rate"] == pytest.approx(0.5)
    assert row["target_rate"] == pytest.approx(0.75)
    assert row["rate_ratio"] == pytest.approx(1.5)
    # Discordant pairs: (1,b) and (2,a) flagged only in target; (2,b) only in SAE.
    assert row["target_only_flagged"] == 2
    assert row["sae_only_flagged"] == 1
    assert row["discordance_ratio"] == pytest.approx(2.0)


def test_relative_risk_handles_zero_sae_flags() -> None:
    result = overflag_relative_risk(_synthetic_frame(), n_bootstrap=20, seed=3)
    row = result[
        (result["condition"] == "direct") & (result["target_variety"] == "hinglish")
    ].iloc[0]

    assert row["sae_rate"] == 0
    assert row["rate_ratio"] == float("inf")
    assert row["sae_only_flagged"] == 0


def test_system_consistency_counts_direction_per_system() -> None:
    frame = _synthetic_frame()
    result = system_overflag_consistency(frame)
    row = result[
        (result["condition"] == "direct") & (result["target_variety"] == "hinglish")
    ].iloc[0]

    assert row["n_systems"] == 2
    assert row["n_systems_higher"] == 2
    assert row["n_systems_lower"] == 0


def test_stratified_effects_keep_items_within_one_stratum() -> None:
    frame = _flag_frame()
    result = aggregate_fairness_effects_by_stratum(
        frame, "family", n_bootstrap=20, n_permutations=50, seed=3
    )
    assert set(result["family"]) == {"x", "y"}
    assert set(result["metric"]) == {"overflag"}
    assert set(result["n_items"]) == {1}

    broken = frame.copy()
    broken.loc[broken.index[0], "family"] = "y"
    with pytest.raises(ValueError, match="exactly one stratum"):
        aggregate_fairness_effects_by_stratum(broken, "family")


def test_item_family_headlines_match_canonical_accuracy_table() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "results/study/validated/accuracy_long.csv")
    families = pd.read_csv(
        root / "data/language_variety_audit_benchmark.csv", usecols=["id", "dataset_type"]
    )
    frame = frame.merge(families, on="id", validate="many_to_one")
    result = aggregate_fairness_effects_by_stratum(
        frame, "dataset_type", n_bootstrap=50, n_permutations=100, seed=7
    )
    standard = result[
        (result["dataset_type"] == "standard_indianisms")
        & (result["condition"] == "direct")
    ]

    assert set(standard["n_items"]) == {98}
    assert standard["difference_pp"].tolist() == pytest.approx(
        [5.5555555556, 5.5555555556]
    )
