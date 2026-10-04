#!/usr/bin/env python3
"""Generate LaTeX result tables directly from analysis CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _escape(value: str) -> str:
    return (
        value.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def accuracy_table(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> str:
    direct = summary[summary["condition"] == "direct"].pivot(
        index=["model", "system_id", "cohort"],
        columns="variety",
        values="accuracy",
    )
    gaps = paired[paired["condition"] == "direct"].pivot(
        index="system_id",
        columns="target_variety",
        values="gap_pp",
    )
    hinglish_reasoning = reasoning[reasoning["variety"] == "hinglish"].set_index(
        "system_id"
    )["reasoning_effect_pp"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Accuracy by linguistic condition under direct prompting, paired gaps relative to SAE, and the Hinglish reasoning-prompt effect. All entries are percentage points except accuracies, which are percentages.}",
        r"\label{tab:accuracy-main}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"System & Cohort & SAE & IndEng & Hinglish & $\Delta_{\mathrm{Ind}}$ & $\Delta_{\mathrm{Hing}}$ & Hing. reasoning gain \\",
        r"\midrule",
    ]
    direct_rows = direct.reset_index()
    direct_rows["cohort_order"] = direct_rows["cohort"].map(
        {"fresh": 0, "historical": 1}
    )
    direct_rows = direct_rows.sort_values(["cohort_order", "model"])
    previous_cohort = None
    for _, row in direct_rows.iterrows():
        cohort_name = "Historical" if row["cohort"] == "historical" else "Fresh"
        if previous_cohort is not None and row["cohort"] != previous_cohort:
            lines.append(r"\addlinespace")
        lines.append(
            f"{_escape(str(row['model']))} & {cohort_name} & "
            f"{100 * row['sae']:.1f} & {100 * row['indian_english']:.1f} & "
            f"{100 * row['hinglish']:.1f} & {gaps.loc[row['system_id'], 'indian_english']:.1f} & "
            f"{gaps.loc[row['system_id'], 'hinglish']:.1f} & "
            f"{hinglish_reasoning.loc[row['system_id']]:+.1f} \\\\"
        )
        previous_cohort = row["cohort"]
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def model_system_table(summary: pd.DataFrame) -> str:
    systems = (
        summary[["model", "model_id", "system_id", "provider", "cohort", "prompt_cue"]]
        .drop_duplicates()
        .sort_values(["cohort", "model", "prompt_cue"])
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Evaluated model systems. All prompts omit the linguistic-condition label; local model digests and quantization details are included in the supplementary manifest.}",
        r"\label{tab:model-systems}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Model & Provider & Cohort \\",
        r"\midrule",
    ]
    for _, row in systems.iterrows():
        cohort = "Historical" if row["cohort"] == "historical" else "Fresh"
        lines.append(
            f"{_escape(str(row['model']))} & {_escape(str(row['provider']))} & "
            f"{cohort} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def prompt_agreement_table(agreement: pd.DataFrame) -> str:
    display = {
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
    }
    overall = agreement[agreement["variety"] == "all"].sort_values(
        "probability_model_id"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Agreement between ordinary direct classification and the grammar-constrained probability prompt over 600 matched item--variety outputs per model.}",
        r"\label{tab:prompt-agreement}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Agreement & $\kappa$ & Ordinary acc. & Coded acc. \\",
        r"\midrule",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"{display.get(row['probability_model_id'], row['probability_model_id'])} & "
            f"{100 * row['prediction_agreement']:.1f} & {row['cohen_kappa']:.3f} & "
            f"{100 * row['ordinary_accuracy']:.1f} & "
            f"{100 * row['probability_prompt_accuracy']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def calibration_table(summary: pd.DataFrame) -> str:
    display = {
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
    }
    variety_display = {
        "sae": "SAE",
        "indian_english": "IndEng",
        "hinglish": "Hinglish",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Calibration metrics for the grammar-constrained six-class coded classifier. Lower is better for NLL, Brier, ECE, and ACE.}",
        r"\label{tab:calibration-main}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Variety & Accuracy & Confidence & NLL & Brier & ECE & ACE \\",
        r"\midrule",
    ]
    ordered = summary.assign(
        model_order=summary["model_id"].map(
            {"gpt-4o-mini-2024-07-18": 0, "gpt-4.1-mini-2025-04-14": 1}
        ),
        variety_order=summary["variety"].map(
            {"sae": 0, "indian_english": 1, "hinglish": 2}
        ),
    ).sort_values(["model_order", "variety_order"])
    previous_model = None
    for _, row in ordered.iterrows():
        model = display.get(row["model_id"], row["model_id"])
        if previous_model is not None and model != previous_model:
            lines.append(r"\addlinespace")
        lines.append(
            f"{_escape(str(model))} & {variety_display[row['variety']]} & "
            f"{100 * row['accuracy']:.1f} & {100 * row['mean_confidence']:.1f} & "
            f"{row['nll']:.3f} & {row['brier']:.3f} & "
            f"{100 * row['ece']:.1f} & {100 * row['ace']:.1f} \\\\"
        )
        previous_model = model
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def conformal_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean conformal coverage and set size over 200 item-level splits, averaged across probability-producing models.}",
        r"\label{tab:conformal-summary}",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Method & Variety & Coverage & Set size \\",
        r"\midrule",
    ]
    grouped = summary.groupby(["method", "variety"], as_index=False).agg(
        coverage=("mean_coverage", "mean"),
        set_size=("mean_set_size", "mean"),
    )
    for method in ("pooled_marginal", "variety_conditional"):
        subset = grouped[grouped["method"] == method].set_index("variety")
        method_name = "Pooled" if method == "pooled_marginal" else "Conditional"
        for index, variety in enumerate(("sae", "indian_english", "hinglish")):
            shown_method = method_name if index == 0 else ""
            variety_name = {"sae": "SAE", "indian_english": "IndEng", "hinglish": "Hinglish"}[variety]
            lines.append(
                f"{shown_method} & {variety_name} & "
                f"{100 * subset.loc[variety, 'coverage']:.1f} & "
                f"{subset.loc[variety, 'set_size']:.2f} \\\\"
            )
        if method == "pooled_marginal":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/metrics"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tables"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paired_root = args.metrics_root / "paired"
    accuracy = pd.read_csv(paired_root / "accuracy_summary.csv")
    paired = pd.read_csv(paired_root / "model_paired_effects.csv")
    reasoning = pd.read_csv(paired_root / "model_reasoning_effects.csv")
    calibration = pd.read_csv(args.metrics_root / "calibration/calibration_summary.csv")
    agreement = pd.read_csv(args.metrics_root / "calibration/prompt_agreement.csv")
    conformal = pd.read_csv(args.metrics_root / "conformal/conformal_summary.csv")

    outputs = {
        "accuracy_table.tex": accuracy_table(accuracy, paired, reasoning),
        "model_system_table.tex": model_system_table(accuracy),
        "prompt_agreement_table.tex": prompt_agreement_table(agreement),
        "calibration_table.tex": calibration_table(calibration),
        "conformal_table.tex": conformal_table(conformal),
    }
    for name, content in outputs.items():
        path = args.out_dir / name
        path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
