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


def _p_cell(value: float) -> str:
    """Format a p-value for a table cell: three decimals, floored at .001."""
    value = float(value)
    if value < 0.001:
        return "$<$.001"
    return f"{value:.3f}".replace("0.", ".", 1)


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
    mcnemar = paired[paired["condition"] == "direct"].pivot(
        index="system_id",
        columns="target_variety",
        values="mcnemar_p_holm",
    )
    n_comparisons = len(paired)
    hinglish_reasoning = reasoning[reasoning["variety"] == "hinglish"].set_index(
        "system_id"
    )["reasoning_effect_pp"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Direct-prompt accuracy, paired gaps from SAE, and the change in Hinglish accuracy under reasoning. Accuracies are percentages; gaps and changes are percentage points. "
        rf"The $p$ columns give exact McNemar tests, Holm-adjusted across all {n_comparisons} per-system comparisons (nine systems, two prompts, two target conditions).}}",
        r"\label{tab:accuracy-main}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"System & Cohort & SAE & IndEng & Hinglish & $\Delta_{\mathrm{Ind}}$ & $p_{\mathrm{Ind}}$ & $\Delta_{\mathrm{Hing}}$ & $p_{\mathrm{Hing}}$ & Hing. reasoning gain \\",
        r"\midrule",
    ]
    direct_rows = direct.reset_index()
    direct_rows["cohort_order"] = direct_rows["cohort"].map({"new": 0, "original": 1})
    direct_rows = direct_rows.sort_values(["cohort_order", "model"])
    previous_cohort = None
    for _, row in direct_rows.iterrows():
        cohort_name = "Historical" if row["cohort"] == "original" else "Fresh"
        if previous_cohort is not None and row["cohort"] != previous_cohort:
            lines.append(r"\addlinespace")
        lines.append(
            f"{_escape(str(row['model']))} & {cohort_name} & "
            f"{100 * row['sae']:.1f} & {100 * row['indian_english']:.1f} & "
            f"{100 * row['hinglish']:.1f} & {gaps.loc[row['system_id'], 'indian_english']:.1f} & "
            f"{_p_cell(mcnemar.loc[row['system_id'], 'indian_english'])} & "
            f"{gaps.loc[row['system_id'], 'hinglish']:.1f} & "
            f"{_p_cell(mcnemar.loc[row['system_id'], 'hinglish'])} & "
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
        r"\caption{Systems included in the analysis. Prompts do not identify the language condition. The supplement records local-model digests and quantization.}",
        r"\label{tab:model-systems}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Model & Provider & Cohort \\",
        r"\midrule",
    ]
    for _, row in systems.iterrows():
        cohort = "Historical" if row["cohort"] == "original" else "Fresh"
        lines.append(
            f"{_escape(str(row['model']))} & {_escape(str(row['provider']))} & "
            f"{cohort} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def calibration_contrast_table(differences: pd.DataFrame) -> str:
    display = {
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
    }
    targets = {"indian_english": "IndEng", "hinglish": "Hinglish"}

    def cell(row: pd.Series, scale: float, digits: int) -> str:
        values = [scale * row[name] for name in ("target_minus_sae", "ci_lower", "ci_upper")]
        point, lower, upper = (
            f"{value:+.{digits}f}" if index == 0 else f"{value:.{digits}f}"
            for index, value in enumerate(values)
        )
        return f"{point} [{lower}, {upper}]".replace("-", "$-$")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched differences for the grammar-constrained classifier, target condition minus SAE on the same 200 items. Accuracy and confidence are in percentage points; NLL and Brier are in their own units, and higher values are worse. Brackets give 95\% item-bootstrap intervals (10,000 draws).}",
        r"\label{tab:calibration-contrasts}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Model & Target & Accuracy & Confidence & NLL & Brier \\",
        r"\midrule",
    ]
    for index, model_id in enumerate(display):
        if index:
            lines.append(r"\addlinespace")
        for target, target_name in targets.items():
            subset = differences[
                (differences["model_id"] == model_id)
                & (differences["target_variety"] == target)
            ].set_index("metric")
            lines.append(
                f"{display[model_id]} & {target_name} & "
                f"{cell(subset.loc['correct'], 100, 1)} & "
                f"{cell(subset.loc['confidence'], 100, 2)} & "
                f"{cell(subset.loc['nll'], 1, 3)} & "
                f"{cell(subset.loc['brier'], 1, 3)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
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
        r"\caption{Agreement between the ordinary direct prompt and the grammar-constrained probability prompt, measured over 600 item--variety outputs per model.}",
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
        r"\caption{Accuracy, confidence, and calibration of the grammar-constrained six-class classifier. Lower values are better for NLL, Brier, ECE, and ACE.}",
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
        r"\caption{Mean empirical coverage and prediction-set size across 200 item-level splits, averaged over the two probability-producing models.}",
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


VARIETY_NAMES = {
    "sae": "SAE",
    "indian_english": "IndEng",
    "hinglish": "Hinglish",
}
PROMPT_NAMES = {"direct": "Direct", "cot": "Reasoning"}


def _number(value: float, signed: bool = False) -> str:
    text = f"{value:+.1f}" if signed else f"{value:.1f}"
    return text.replace("-", "$-$")


def _signed_interval(row: pd.Series) -> str:
    return (
        f"{_number(row['difference_pp'], signed=True)} "
        f"[{_number(row['ci_lower_pp'])}, {_number(row['ci_upper_pp'])}]"
    )


def fairness_table(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    relative_risk: pd.DataFrame,
) -> str:
    grouped = summary.groupby(["condition", "variety"], as_index=False).agg(
        overflag_rate=("overflag_rate", "mean"),
        harm_miss_rate=("harm_miss_rate", "mean"),
    )
    effects = effects.set_index(["condition", "metric", "target_variety"])
    relative_risk = relative_risk.set_index(["condition", "target_variety"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Error-rate parity by language condition, averaged equally across nine systems. Over-flagging is a non-normal prediction for a normal item ($n=156$), so its disparity is a false-positive-rate gap. A harm miss is a \textit{normal} prediction for a non-normal item ($n=44$), so its disparity is a false-negative-rate gap. Differences are target minus SAE in percentage points with 95\% item-clustered bootstrap intervals; the ratio is the target over-flagging rate divided by the SAE rate.}",
        r"\label{tab:fairness-errors}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Over-flagging (false positives)} & \multicolumn{2}{c}{Harm misses (false negatives)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        "Prompt & Variety & Rate & Gap vs.\\ SAE & Ratio & Rate & Gap vs.\\ SAE \\\\",
        r"\midrule",
    ]
    for condition in ("direct", "cot"):
        subset = grouped[grouped["condition"] == condition].set_index("variety")
        for index, variety in enumerate(("sae", "indian_english", "hinglish")):
            shown_prompt = PROMPT_NAMES[condition] if index == 0 else ""
            if variety == "sae":
                overflag_gap = harm_gap = ratio = "---"
            else:
                overflag_gap = _signed_interval(
                    effects.loc[(condition, "overflag", variety)]
                )
                harm_gap = _signed_interval(
                    effects.loc[(condition, "harm_miss", variety)]
                )
                ratio = f"{relative_risk.loc[(condition, variety), 'rate_ratio']:.1f}$\\times$"
            lines.append(
                f"{shown_prompt} & {VARIETY_NAMES[variety]} & "
                f"{100 * subset.loc[variety, 'overflag_rate']:.1f} & {overflag_gap} & {ratio} & "
                f"{100 * subset.loc[variety, 'harm_miss_rate']:.1f} & {harm_gap} \\\\"
            )
        if condition == "direct":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def item_family_table(by_family: pd.DataFrame, risk_by_family: pd.DataFrame) -> str:
    by_family = by_family[by_family["metric"] == "overflag"]
    max_holm = by_family["cluster_permutation_p_holm"].max()
    by_family = by_family.set_index(
        ["item_family", "condition", "target_variety"]
    )
    risk_by_family = risk_by_family.set_index(
        ["item_family", "condition", "target_variety"]
    )
    family_names = {
        "standard_indianisms": "Standard Indian English",
        "genz_slang": "Youth/internet slang",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Over-flagging of normal items within the two authored item families. Rates are percentages averaged equally across nine systems; gaps are target minus SAE in percentage points with 95\% item-clustered bootstrap intervals. Each family is its own Holm family; every gap remains significant ($p\le "
        + f"{max_holm:.3f}".replace("0.", ".", 1)
        + r"$). Ratios are IndEng/Hinglish over-flagging rates divided by the SAE rate.}",
        r"\label{tab:item-family}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        "Item family (normal $n$) & Prompt & SAE & IndEng & Hinglish & IndEng gap & Hinglish gap & Ratios \\\\",
        r"\midrule",
    ]
    for family_index, family in enumerate(("standard_indianisms", "genz_slang")):
        for condition_index, condition in enumerate(("direct", "cot")):
            ind = by_family.loc[(family, condition, "indian_english")]
            hing = by_family.loc[(family, condition, "hinglish")]
            ratios = "/".join(
                f"{risk_by_family.loc[(family, condition, target), 'rate_ratio']:.1f}"
                for target in ("indian_english", "hinglish")
            )
            label = (
                f"{family_names[family]} ({int(ind['n_items'])})"
                if condition_index == 0
                else ""
            )
            lines.append(
                f"{label} & {PROMPT_NAMES[condition]} & {100 * ind['sae_rate']:.1f} & "
                f"{100 * ind['target_rate']:.1f} & {100 * hing['target_rate']:.1f} & "
                f"{_signed_interval(ind)} & {_signed_interval(hing)} & {ratios} \\\\"
            )
        if family_index == 0:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def system_overflag_table(summary: pd.DataFrame) -> str:
    cohort_names = {"new": "Fresh", "original": "Historical"}
    wide = summary.pivot_table(
        index=["cohort", "model", "system_id"],
        columns=["condition", "variety"],
        values="overflag_rate",
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Over-flagging rates (\%) on the 156 normal items for each system. Every system over-flags Indian English and Hinglish renditions more often than matched SAE renditions under both prompts.}",
        r"\label{tab:system-overflag}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Direct} & \multicolumn{3}{c}{Reasoning} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        "System & Cohort & SAE & IndEng & Hinglish & SAE & IndEng & Hinglish \\\\",
        r"\midrule",
    ]
    for cohort in ("new", "original"):
        subset = wide.loc[cohort]
        for (model, _system), row in subset.iterrows():
            values = " & ".join(
                f"{100 * row[(condition, variety)]:.1f}"
                for condition in ("direct", "cot")
                for variety in ("sae", "indian_english", "hinglish")
            )
            lines.append(f"{_escape(model)} & {cohort_names[cohort]} & {values} \\\\")
        if cohort == "new":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


EXAMPLE_ITEM_IDS = (23, 22, 45, 46, 72, 85, 142, 164)


def example_items_table(benchmark: pd.DataFrame, accuracy: pd.DataFrame) -> str:
    """Show how many systems flag each rendition of selected normal items."""
    direct = accuracy[
        (accuracy["condition"] == "direct") & (accuracy["gold_category"] == "normal")
    ]
    flags = (
        direct.assign(flagged=direct["predicted_category"] != "normal")
        .groupby(["id", "variety"])["flagged"]
        .sum()
        .unstack("variety")
    )
    n_systems = direct["system_id"].nunique()
    family_names = {"standard_indianisms": "Std.", "genz_slang": "Slang"}
    items = benchmark.set_index("id")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Illustrative normal items and the number of systems (out of "
        + str(n_systems)
        + r") that assign a non-normal label to each rendition under direct prompting. Std.\ = standard Indian English family; Slang = youth/internet slang family. Hinglish renditions are in the released benchmark.}",
        r"\label{tab:example-items}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.34\linewidth}p{0.34\linewidth}lccc}",
        r"\toprule",
        "Indian English rendition & SAE rendition & Family & IndEng & Hing. & SAE \\\\",
        r"\midrule",
    ]
    for item_id in EXAMPLE_ITEM_IDS:
        item = items.loc[item_id]
        if item["category"] != "normal":
            raise ValueError(f"Example item {item_id} must be a normal item")
        row = flags.loc[item_id]
        lines.append(
            f"{_escape(str(item['indian_english_translation']))} & "
            f"{_escape(str(item['sae_example']))} & "
            f"{family_names[item['dataset_type']]} & "
            f"{int(row['indian_english'])} & {int(row['hinglish'])} & {int(row['sae'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/study/metrics"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/language_variety_audit_benchmark.csv"),
    )
    parser.add_argument(
        "--accuracy",
        type=Path,
        default=Path("results/study/validated/accuracy_long.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/facct2027/generated"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paired_root = args.metrics_root / "paired"
    accuracy = pd.read_csv(paired_root / "accuracy_summary.csv")
    paired = pd.read_csv(paired_root / "model_paired_effects.csv")
    reasoning = pd.read_csv(paired_root / "model_reasoning_effects.csv")
    calibration = pd.read_csv(args.metrics_root / "calibration/calibration_summary.csv")
    agreement = pd.read_csv(args.metrics_root / "calibration/prompt_agreement.csv")
    calibration_differences = pd.read_csv(
        args.metrics_root / "calibration/paired_calibration_differences.csv"
    )
    conformal = pd.read_csv(args.metrics_root / "conformal/conformal_summary.csv")
    fairness_root = args.metrics_root / "fairness"
    fairness = pd.read_csv(fairness_root / "fairness_summary.csv")
    fairness_effects = pd.read_csv(fairness_root / "aggregate_fairness_effects.csv")
    relative_risk = pd.read_csv(fairness_root / "overflag_relative_risk.csv")
    by_family = pd.read_csv(
        fairness_root / "aggregate_fairness_effects_by_item_family.csv"
    )
    risk_by_family = pd.read_csv(
        fairness_root / "overflag_relative_risk_by_item_family.csv"
    )

    outputs = {
        "accuracy_table.tex": accuracy_table(accuracy, paired, reasoning),
        "model_system_table.tex": model_system_table(accuracy),
        "prompt_agreement_table.tex": prompt_agreement_table(agreement),
        "calibration_contrast_table.tex": calibration_contrast_table(
            calibration_differences
        ),
        "calibration_table.tex": calibration_table(calibration),
        "conformal_table.tex": conformal_table(conformal),
        "fairness_table.tex": fairness_table(
            fairness, fairness_effects, relative_risk
        ),
        "item_family_table.tex": item_family_table(by_family, risk_by_family),
        "system_overflag_table.tex": system_overflag_table(fairness),
        "example_items_table.tex": example_items_table(
            pd.read_csv(args.benchmark), pd.read_csv(args.accuracy)
        ),
    }
    for name, content in outputs.items():
        path = args.out_dir / name
        path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
