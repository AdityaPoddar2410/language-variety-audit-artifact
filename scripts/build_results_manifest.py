#!/usr/bin/env python3
"""Freeze headline study numbers in JSON and LaTeX macro files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from language_variety_audit.run_records import file_sha256


def _row(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _round(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _p_text(value: Any) -> str:
    """Format a p-value with its relation, for use inside LaTeX math mode."""
    value = float(value)
    if value < 0.001:
        return "p<.001"
    return f"p={value:.3f}".replace("0.", ".", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/study"),
    )
    args = parser.parse_args()

    paired_root = args.root / "metrics/paired"
    calibration_root = args.root / "metrics/calibration"
    conformal_root = args.root / "metrics/conformal"
    fairness_root = args.root / "metrics/fairness"
    mixed_root = args.root / "metrics/mixed_effects"
    accuracy_table = pd.read_csv(args.root / "validated/accuracy_long.csv")
    aggregate = pd.read_csv(paired_root / "aggregate_paired_effects.csv")
    reasoning = pd.read_csv(paired_root / "aggregate_reasoning_effects.csv")
    calibration = pd.read_csv(calibration_root / "paired_calibration_differences.csv")
    prompt_agreement = pd.read_csv(calibration_root / "prompt_agreement.csv")
    proper_score_contrasts = calibration[calibration["metric"].isin(["nll", "brier"])]
    conformal = pd.read_csv(conformal_root / "conformal_summary.csv")
    fairness = pd.read_csv(fairness_root / "aggregate_fairness_effects.csv")
    fairness_reasoning = pd.read_csv(
        fairness_root / "aggregate_reasoning_fairness_effects.csv"
    )
    relative_risk = pd.read_csv(fairness_root / "overflag_relative_risk.csv")
    model_paired = pd.read_csv(paired_root / "model_paired_effects.csv")
    model_paired_direct = model_paired[model_paired["condition"] == "direct"]
    consistency = pd.read_csv(fairness_root / "system_overflag_consistency.csv")
    by_family = pd.read_csv(
        fairness_root / "aggregate_fairness_effects_by_item_family.csv"
    )
    hierarchical = pd.read_csv(mixed_root / "hierarchical_fixed_effects.csv")
    diagnostics = json.loads((mixed_root / "diagnostics.json").read_text())

    direct_indian = _row(
        aggregate,
        condition="direct",
        target_variety="indian_english",
    )
    direct_hinglish = _row(
        aggregate,
        condition="direct",
        target_variety="hinglish",
    )
    cot_indian = _row(
        aggregate,
        condition="cot",
        target_variety="indian_english",
    )
    cot_hinglish = _row(
        aggregate,
        condition="cot",
        target_variety="hinglish",
    )
    reasoning_hinglish = _row(reasoning, variety="hinglish")
    reasoning_sae = _row(reasoning, variety="sae")
    direct_indian_overflag = _row(
        fairness,
        condition="direct",
        metric="overflag",
        target_variety="indian_english",
    )
    direct_hinglish_overflag = _row(
        fairness,
        condition="direct",
        metric="overflag",
        target_variety="hinglish",
    )
    reasoning_indian_overflag = _row(
        fairness_reasoning,
        variety="indian_english",
        metric="overflag",
    )
    reasoning_hinglish_overflag = _row(
        fairness_reasoning,
        variety="hinglish",
        metric="overflag",
    )
    reasoning_indian_harm_miss = _row(
        fairness_reasoning,
        variety="indian_english",
        metric="harm_miss",
    )
    reasoning_hinglish_harm_miss = _row(
        fairness_reasoning,
        variety="hinglish",
        metric="harm_miss",
    )

    robustness_rows = {
        f"{condition}_{target}": {
            "overflag": _row(
                fairness, condition=condition, metric="overflag", target_variety=target
            ),
            "relative_risk": _row(
                relative_risk, condition=condition, target_variety=target
            ),
            "families": {
                family: _row(
                    by_family,
                    item_family=family,
                    condition=condition,
                    target_variety=target,
                )
                for family in ("standard_indianisms", "genz_slang")
            },
        }
        for condition in ("direct", "cot")
        for target in ("indian_english", "hinglish")
    }

    calibration_ranges: dict[str, dict[str, float]] = {}
    for metric in ("correct", "confidence", "nll", "brier", "ece", "ace"):
        values = calibration[calibration["metric"] == metric]["target_minus_sae"]
        calibration_ranges[metric] = {
            "minimum": _round(values.min()),
            "maximum": _round(values.max()),
        }

    conformal_selected = conformal[
        conformal["variety"].isin(["sae", "indian_english", "hinglish"])
    ]
    conformal_ranges: dict[str, dict[str, dict[str, float]]] = {}
    for method in ("pooled_marginal", "variety_conditional"):
        conformal_ranges[method] = {}
        for variety in ("sae", "indian_english", "hinglish"):
            values = conformal_selected[
                (conformal_selected["method"] == method)
                & (conformal_selected["variety"] == variety)
            ]
            conformal_ranges[method][variety] = {
                "minimum_mean_coverage": _round(values["mean_coverage"].min()),
                "maximum_mean_coverage": _round(values["mean_coverage"].max()),
                "minimum_mean_set_size": _round(values["mean_set_size"].min()),
                "maximum_mean_set_size": _round(values["mean_set_size"].max()),
            }

    selected_cost_files = [
        args.root / "raw/gpt41-evaluations.costs.csv",
        args.root / "raw/gemini25-evaluations.costs.csv",
        args.root / "raw/gpt4o-probabilities-v2.costs.csv",
        args.root / "raw/gpt41-probabilities-v2.costs.csv",
    ]
    selected_cost = sum(
        pd.read_csv(path)["estimated_cost_usd"].fillna(0).sum()
        for path in selected_cost_files
    )

    variety_terms = hierarchical[
        hierarchical["term"].isin(
            [
                'C(variety, Treatment(reference="sae"))[T.hinglish]',
                'C(variety, Treatment(reference="sae"))[T.indian_english]',
            ]
        )
    ]
    manifest = {
        "paper_title": "Matched Meaning, Unequal Moderation: Auditing Linguistic Bias in LLMs",
        "dataset": {
            "items": 200,
            "varieties": 3,
            "classes": 6,
            "sha256": file_sha256("data/language_variety_audit_benchmark.csv"),
        },
        "models": {
            "count": int(accuracy_table["model_id"].nunique()),
            "system_count": int(accuracy_table["system_id"].nunique()),
            "original_count": int(
                accuracy_table.loc[accuracy_table["cohort"] == "original", "model_id"].nunique()
            ),
            "fresh_count": int(
                accuracy_table.loc[accuracy_table["cohort"] == "new", "model_id"].nunique()
            ),
            "probability_model_count": int(calibration["model_id"].nunique()),
        },
        "accuracy": {
            "direct_indian_english": direct_indian.to_dict(),
            "direct_hinglish": direct_hinglish.to_dict(),
            "reasoning_indian_english": cot_indian.to_dict(),
            "reasoning_hinglish": cot_hinglish.to_dict(),
        },
        "reasoning_effect": {
            "hinglish": reasoning_hinglish.to_dict(),
            "sae": reasoning_sae.to_dict(),
        },
        "fairness": {
            "direct_indian_english_overflag": direct_indian_overflag.to_dict(),
            "direct_hinglish_overflag": direct_hinglish_overflag.to_dict(),
            "reasoning_indian_english_overflag": reasoning_indian_overflag.to_dict(),
            "reasoning_hinglish_overflag": reasoning_hinglish_overflag.to_dict(),
            "reasoning_indian_english_harm_miss": (
                reasoning_indian_harm_miss.to_dict()
            ),
            "reasoning_hinglish_harm_miss": reasoning_hinglish_harm_miss.to_dict(),
        },
        "fairness_robustness": {
            key: {
                "overflag": rows["overflag"].to_dict(),
                "relative_risk": rows["relative_risk"].to_dict(),
                "item_families": {
                    family: row.to_dict() for family, row in rows["families"].items()
                },
            }
            for key, rows in robustness_rows.items()
        },
        "system_overflag_consistency": consistency.to_dict(orient="records"),
        "calibration_target_minus_sae_ranges": calibration_ranges,
        "ordinary_vs_coded_prompt_agreement": prompt_agreement[
            prompt_agreement["variety"] == "all"
        ].to_dict(orient="records"),
        "conformal_ranges": conformal_ranges,
        "hierarchical_variety_effects": variety_terms.to_dict(orient="records"),
        "mixed_effects_diagnostics": diagnostics,
        "selected_experiment_cost_usd": _round(selected_cost),
        "source_hashes": {
            str(path): file_sha256(path)
            for path in [
                paired_root / "aggregate_paired_effects.csv",
                paired_root / "aggregate_reasoning_effects.csv",
                paired_root / "model_paired_effects.csv",
                calibration_root / "paired_calibration_differences.csv",
                calibration_root / "prompt_agreement.csv",
                conformal_root / "conformal_summary.csv",
                fairness_root / "fairness_summary.csv",
                fairness_root / "aggregate_fairness_effects.csv",
                fairness_root / "aggregate_reasoning_fairness_effects.csv",
                fairness_root / "aggregate_fairness_effects_equal_models.csv",
                fairness_root
                / "aggregate_reasoning_fairness_effects_equal_models.csv",
                fairness_root / "aggregate_fairness_effects_by_cohort.csv",
                fairness_root
                / "aggregate_reasoning_fairness_effects_by_cohort.csv",
                fairness_root / "overflag_relative_risk.csv",
                fairness_root / "system_overflag_consistency.csv",
                fairness_root / "aggregate_fairness_effects_by_item_family.csv",
                fairness_root / "overflag_relative_risk_by_item_family.csv",
                mixed_root / "hierarchical_fixed_effects.csv",
            ]
        },
    }

    output_dir = args.root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    macros = {
        "ResultNumModels": str(manifest["models"]["count"]),
        "ResultNumSystems": str(manifest["models"]["system_count"]),
        "ResultNumItems": "200",
        "ResultNumClasses": "6",
        "ResultCalibNLLBrierComparisons": str(len(proper_score_contrasts)),
        "ResultCalibNLLBrierExcludeZero": str(
            int(
                (
                    (proper_score_contrasts["ci_lower"] > 0)
                    | (proper_score_contrasts["ci_upper"] < 0)
                ).sum()
            )
        ),
        "ResultNumSystemComparisons": str(len(model_paired)),
        "ResultNumDirectSystemGaps": str(len(model_paired_direct)),
        "ResultNumDirectSystemGapsSig": str(
            int((model_paired_direct["mcnemar_p_holm"] < 0.05).sum())
        ),
        "ResultDirectIndGap": f"{direct_indian['gap_pp']:.1f}",
        "ResultDirectHingGap": f"{direct_hinglish['gap_pp']:.1f}",
        "ResultReasonHingGain": f"{reasoning_hinglish['reasoning_effect_pp']:.1f}",
        "ResultReasonSAEEffect": f"{reasoning_sae['reasoning_effect_pp']:.1f}",
        "ResultDirectIndOverflagGap": (
            f"{direct_indian_overflag['difference_pp']:.1f}"
        ),
        "ResultDirectHingOverflagGap": (
            f"{direct_hinglish_overflag['difference_pp']:.1f}"
        ),
        "ResultReasonIndOverflagChange": (
            f"{reasoning_indian_overflag['difference_pp']:.1f}"
        ),
        "ResultReasonHingOverflagChange": (
            f"{reasoning_hinglish_overflag['difference_pp']:.1f}"
        ),
        "ResultReasonIndHarmMissChange": (
            f"{reasoning_indian_harm_miss['difference_pp']:.1f}"
        ),
        "ResultReasonHingHarmMissChange": (
            f"{reasoning_hinglish_harm_miss['difference_pp']:.1f}"
        ),
        "ResultPooledNonSAECoverageLow": f"{100 * min(conformal_ranges['pooled_marginal'][v]['minimum_mean_coverage'] for v in ('indian_english', 'hinglish')):.1f}",
        "ResultPooledNonSAECoverageHigh": f"{100 * max(conformal_ranges['pooled_marginal'][v]['maximum_mean_coverage'] for v in ('indian_english', 'hinglish')):.1f}",
        "ResultConditionalNonSAECoverageLow": f"{100 * min(conformal_ranges['variety_conditional'][v]['minimum_mean_coverage'] for v in ('indian_english', 'hinglish')):.1f}",
        "ResultConditionalNonSAECoverageHigh": f"{100 * max(conformal_ranges['variety_conditional'][v]['maximum_mean_coverage'] for v in ('indian_english', 'hinglish')):.1f}",
    }
    short = {"indian_english": "Ind", "hinglish": "Hing"}
    family_short = {"standard_indianisms": "Std", "genz_slang": "Slang"}
    for condition, prefix in (("direct", "Direct"), ("cot", "Reason")):
        for target, name in short.items():
            rows = robustness_rows[f"{condition}_{target}"]
            overflag = rows["overflag"]
            risk = rows["relative_risk"]
            macros[f"Result{prefix}{name}OverflagRate"] = f"{100 * overflag['target_rate']:.1f}"
            macros[f"Result{prefix}SAEOverflagRate"] = f"{100 * overflag['sae_rate']:.1f}"
            macros[f"Result{prefix}{name}OverflagDisparity"] = (
                f"{overflag['difference_pp']:.1f}"
            )
            macros[f"Result{prefix}{name}OverflagHolmP"] = (
                _p_text(overflag['cluster_permutation_p_holm'])
            )
            macros[f"Result{prefix}{name}OverflagRatio"] = f"{risk['rate_ratio']:.1f}"
            macros[f"Result{prefix}{name}OverflagRatioLow"] = (
                f"{risk['rate_ratio_ci_lower']:.1f}"
            )
            macros[f"Result{prefix}{name}OverflagRatioHigh"] = (
                f"{risk['rate_ratio_ci_upper']:.1f}"
            )
            macros[f"Result{prefix}{name}TargetOnly"] = str(int(risk["target_only_flagged"]))
            macros[f"Result{prefix}{name}SAEOnly"] = str(int(risk["sae_only_flagged"]))
            macros[f"Result{prefix}{name}DiscordanceRatio"] = (
                f"{risk['discordance_ratio']:.1f}"
            )
            macros[f"Result{prefix}{name}DiscordanceRatioLow"] = (
                f"{risk['discordance_ratio_ci_lower']:.1f}"
            )
            macros[f"Result{prefix}{name}DiscordanceRatioHigh"] = (
                f"{risk['discordance_ratio_ci_upper']:.1f}"
            )
            for family, family_name in family_short.items():
                row = rows["families"][family]
                stem = f"Result{prefix}{family_name}{name}Overflag"
                macros[f"{stem}Disparity"] = f"{row['difference_pp']:.1f}"
                macros[f"{stem}DisparityLow"] = f"{row['ci_lower_pp']:.1f}"
                macros[f"{stem}DisparityHigh"] = f"{row['ci_upper_pp']:.1f}"
                macros[f"{stem}Rate"] = f"{100 * row['target_rate']:.1f}"
                macros[f"Result{prefix}{family_name}SAEOverflagRate"] = (
                    f"{100 * row['sae_rate']:.1f}"
                )
                macros[f"{stem}HolmP"] = (
                    _p_text(row['cluster_permutation_p_holm'])
                )
    macros["ResultReasonIndAccGap"] = f"{cot_indian['gap_pp']:.1f}"
    macros["ResultReasonHingAccGap"] = f"{cot_hinglish['gap_pp']:.1f}"
    macros["ResultReasonAccGapHolmPMin"] = (
        _p_text(
            min(
                cot_indian["cluster_permutation_p_holm"],
                cot_hinglish["cluster_permutation_p_holm"],
            )
        )
    )
    macros["ResultMinSystemsHigherOverflag"] = str(
        int(consistency["n_systems_higher"].min())
    )
    macros["ResultNumNormalItems"] = str(
        int(_row(fairness, condition="direct", metric="overflag", target_variety="hinglish")["n_items"])
    )
    macros["ResultNumHarmfulItems"] = str(
        int(_row(fairness, condition="direct", metric="harm_miss", target_variety="hinglish")["n_items"])
    )
    macro_path = output_dir / "results_macros.tex"
    macro_path.write_text(
        "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()),
        encoding="utf-8",
    )
    print(manifest_path)
    print(macro_path)


if __name__ == "__main__":
    main()
