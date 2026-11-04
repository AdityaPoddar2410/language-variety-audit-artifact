#!/usr/bin/env python3
"""Generate a plain-language Markdown report of the frozen study results."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/study"
OUTPUT = ROOT / "results/study/RESULTS-EXPLAINED.md"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    number = 100 * float(value)
    return f"{number:+.{digits}f}%" if signed else f"{number:.{digits}f}%"


def pp(value: Any, digits: int = 1, signed: bool = False) -> str:
    number = float(value)
    return f"{number:+.{digits}f} pp" if signed else f"{number:.{digits}f} pp"


def pvalue(value: Any) -> str:
    number = float(value)
    return f"{number:.2e}" if number < 0.001 else f"{number:.4f}"


def cue_name(value: str) -> str:
    return "Explicit variety cue" if value == "explicit_variety_label" else "No cue"


def target_name(value: str) -> str:
    return {"indian_english": "Indian English", "hinglish": "Hinglish", "sae": "SAE"}[value]


def model_name(value: str) -> str:
    return {
        "claude-3-5-haiku-20241022": "Claude Haiku 3.5",
        "llama3.1:8b": "Llama 3.1 8B",
        "gemma3:12b": "Gemma 3 12B",
        "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
        "mistral:7b": "Mistral 7B Instruct",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    }.get(value, value)


def main() -> None:
    manifest = json.loads((RESULTS / "manifests/results_manifest.json").read_text())
    accuracy = pd.read_csv(RESULTS / "metrics/paired/accuracy_summary.csv")
    model_effects = pd.read_csv(RESULTS / "metrics/paired/model_paired_effects.csv")
    aggregate = pd.read_csv(RESULTS / "metrics/paired/aggregate_paired_effects.csv")
    aggregate_reasoning = pd.read_csv(RESULTS / "metrics/paired/aggregate_reasoning_effects.csv")
    cohort_effects = pd.read_csv(RESULTS / "metrics/paired/aggregate_paired_effects_by_cohort.csv")
    calibration = pd.read_csv(RESULTS / "metrics/calibration/calibration_summary.csv")
    calibration_diffs = pd.read_csv(RESULTS / "metrics/calibration/paired_calibration_differences.csv")
    prompt_agreement = pd.read_csv(RESULTS / "metrics/calibration/prompt_agreement.csv")
    conformal = pd.read_csv(RESULTS / "metrics/conformal/conformal_summary.csv")
    fairness_summary = pd.read_csv(RESULTS / "metrics/fairness/fairness_summary.csv")
    fairness_effects = pd.read_csv(
        RESULTS / "metrics/fairness/aggregate_fairness_effects.csv"
    )
    fairness_reasoning = pd.read_csv(
        RESULTS / "metrics/fairness/aggregate_reasoning_fairness_effects.csv"
    )
    mixed = pd.read_csv(RESULTS / "metrics/mixed_effects/hierarchical_fixed_effects.csv")
    mixed_variance = pd.read_csv(RESULTS / "metrics/mixed_effects/hierarchical_variance_components.csv")
    diagnostics = json.loads((RESULTS / "metrics/mixed_effects/diagnostics.json").read_text())

    lines: list[str] = [
        "# Study Results Explained",
        "",
        f"**Paper:** *{manifest['paper_title']}*  ",
        "**Generated from the frozen result manifest and analysis tables**  ",
        "**Benchmark:** 200 matched items × 3 language conditions × 2 prompt conditions",
        "",
        "## Executive summary",
        "",
        f"- The final analysis contains **{manifest['models']['count']} distinct LLMs**, **{manifest['models']['system_count']} model–prompt systems**, and **10,800 complete classification observations**.",
        f"- Under direct prompting, accuracy is **{manifest['accuracy']['direct_indian_english']['gap_pp']:.1f} percentage points lower on Indian English** and **{manifest['accuracy']['direct_hinglish']['gap_pp']:.1f} points lower on Hinglish** than on matched SAE items.",
        f"- Normal Indian English and Hinglish content is over-flagged **{manifest['fairness']['direct_indian_english_overflag']['difference_pp']:.1f}** and **{manifest['fairness']['direct_hinglish_overflag']['difference_pp']:.1f} percentage points more often** than matched SAE content under direct prompting.",
        f"- Explicit reasoning raises Hinglish accuracy by **{manifest['reasoning_effect']['hinglish']['reasoning_effect_pp']:.1f} points**, raises Indian English by **{float(aggregate_reasoning.loc[aggregate_reasoning['variety']=='indian_english','reasoning_effect_pp'].iloc[0]):.1f} points**, and changes SAE by only **{manifest['reasoning_effect']['sae']['reasoning_effect_pp']:.1f} points**.",
        f"- Reasoning reduces Indian English and Hinglish over-flagging by **{abs(manifest['fairness']['reasoning_indian_english_overflag']['difference_pp']):.1f}** and **{abs(manifest['fairness']['reasoning_hinglish_overflag']['difference_pp']):.1f} points**, but increases harmful-content misses by **{manifest['fairness']['reasoning_indian_english_harm_miss']['difference_pp']:.1f}** and **{manifest['fairness']['reasoning_hinglish_harm_miss']['difference_pp']:.1f} points**.",
        "- A stratified manual audit covered 420 reasoning responses across models and varieties; one reviewer per response required both semantic interpretation and final category to be acceptable, and more than 90% were accepted.",
        "- In a separate two-model case study, grammar-constrained label-token confidence barely falls as non-SAE accuracy declines. This coded classifier therefore becomes more overconfident on non-SAE input.",
        f"- In a descriptive repeated-split uncertainty-set diagnostic, pooled 90%-nominal thresholds cover non-SAE input at only **{100 * min(manifest['conformal_ranges']['pooled_marginal'][v]['minimum_mean_coverage'] for v in ('indian_english','hinglish')):.1f}–{100 * max(manifest['conformal_ranges']['pooled_marginal'][v]['maximum_mean_coverage'] for v in ('indian_english','hinglish')):.1f}%** on average. Variety-specific thresholds raise average coverage to **{100 * min(manifest['conformal_ranges']['variety_conditional'][v]['minimum_mean_coverage'] for v in ('indian_english','hinglish')):.1f}–{100 * max(manifest['conformal_ranges']['variety_conditional'][v]['maximum_mean_coverage'] for v in ('indian_english','hinglish')):.1f}%**, but produce larger sets. This is not claimed as a finite-sample pooled conformal guarantee.",
        "",
        "## 1. Experimental design",
        "",
        "Every benchmark item has three semantically matched forms:",
        "",
        "- **SAE:** Standard American English, used as the reference condition.",
        "- **Indian English:** an English variety used widely in India.",
        "- **Hinglish:** Hindi–English code-mixing; it is not treated as an English dialect.",
        "",
        "Each form shares the same human-authored gold label from six moderation classes. Each model is evaluated with a direct classification prompt and an explicit five-step reasoning prompt. All prompts omit the variety label. Four systems come from historical CSVs without request-level prompt hashes; five are fresh runs with complete request metadata, so cohort sensitivity is reported.",
        "",
        "### Evaluated systems",
        "",
    ]

    systems = (
        accuracy[["model", "model_id", "system_id", "provider", "cohort", "prompt_cue"]]
        .drop_duplicates()
        .sort_values(["cohort", "model", "prompt_cue"])
    )
    lines.append(
        md_table(
            ["Model", "Provider", "Cohort", "Prompt regime", "System ID"],
            [
                [
                    row.model,
                    row.provider,
                    "Historical" if row.cohort == "original" else "Fresh",
                    cue_name(row.prompt_cue),
                    f"`{row.system_id}`",
                ]
                for row in systems.itertuples()
            ],
        )
    )

    lines.extend(
        [
            "",
            "### Sampled human audit of model responses",
            "",
            "After the benchmark itself was finalized, humans manually audited a stratified sample of 420 reasoning responses spanning models and varieties. One reviewer assessed each response and required both the semantic interpretation and final category to be acceptable. More than 90% were accepted.",
            "",
            "This is a sampled descriptive audit, not exhaustive validation of all outputs. There was one reviewer per response, no inter-rater reliability statistic, and no retained response-level rating export; the audit is therefore not used in hypothesis tests or headline estimates.",
            "",
            "## 2. Aggregate paired accuracy results",
            "",
        ]
    )
    aggregate_sorted = aggregate.assign(
        condition_order=aggregate["condition"].map({"direct": 0, "cot": 1}),
        variety_order=aggregate["target_variety"].map({"indian_english": 0, "hinglish": 1}),
    ).sort_values(["condition_order", "variety_order"])
    lines.append(
        md_table(
            ["Prompt", "Comparison", "Gap", "95% item-bootstrap CI", "Holm-adjusted permutation p", "Interpretation"],
            [
                [
                    "Direct" if row.condition == "direct" else "Reasoning",
                    f"SAE − {target_name(row.target_variety)}",
                    pp(row.gap_pp),
                    f"[{row.ci_lower_pp:.1f}, {row.ci_upper_pp:.1f}] pp",
                    pvalue(row.cluster_permutation_p_holm),
                    "Lower non-SAE accuracy" if row.gap_pp > 0 else "No SAE advantage",
                ]
                for row in aggregate_sorted.itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "A positive gap means the classifier is more accurate on SAE than on the matched target-variety item. Under direct prompting, both intervals exclude zero and both multiplicity-adjusted tests are below 0.05. Under reasoning, the remaining gaps are smaller and their adjusted tests are not below 0.05.",
            "",
            "## 3. Reasoning-prompt effects",
            "",
        ]
    )
    lines.append(
        md_table(
            ["Variety", "Direct accuracy", "Reasoning accuracy", "Reasoning effect", "95% CI", "Adjusted p"],
            [
                [
                    target_name(row.variety),
                    pct(row.accuracy_direct),
                    pct(row.accuracy_cot),
                    pp(row.reasoning_effect_pp, signed=True),
                    f"[{row.ci_lower_pp:.1f}, {row.ci_upper_pp:.1f}] pp",
                    pvalue(row.cluster_permutation_p_holm),
                ]
                for row in aggregate_reasoning.sort_values("variety").itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "Aggregate accuracy improves under reasoning for the non-SAE conditions, but the error decomposition below shows that this is not an unqualified fairness improvement.",
            "",
            "## Fairness-oriented moderation errors",
            "",
        ]
    )
    fairness_rates = fairness_summary.groupby(
        ["condition", "variety"], as_index=False
    ).agg(
        overflag_rate=("overflag_rate", "mean"),
        harm_miss_rate=("harm_miss_rate", "mean"),
        harm_category_error_rate=("harm_category_error_rate", "mean"),
    )
    lines.append(
        md_table(
            ["Prompt", "Variety", "Over-flag", "Harm miss", "Harm-category error"],
            [
                [
                    "Direct" if row.condition == "direct" else "Reasoning",
                    target_name(row.variety),
                    pct(row.overflag_rate),
                    pct(row.harm_miss_rate),
                    pct(row.harm_category_error_rate),
                ]
                for row in fairness_rates.assign(
                    condition_order=fairness_rates["condition"].map(
                        {"direct": 0, "cot": 1}
                    ),
                    variety_order=fairness_rates["variety"].map(
                        {"sae": 0, "indian_english": 1, "hinglish": 2}
                    ),
                )
                .sort_values(["condition_order", "variety_order"])
                .itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "Under direct prompting, normal Indian English and Hinglish items are over-flagged substantially more often than matched SAE items. Reasoning lowers over-flagging, but it raises harmful-content misses and harmful-category errors. Because 156 of the 200 items are normal, the over-flagging reduction dominates aggregate accuracy even though performance on the harmful subset worsens.",
            "",
            "### Direct-prompt over-flagging disparities",
            "",
        ]
    )
    direct_overflag = fairness_effects[
        (fairness_effects["condition"] == "direct")
        & (fairness_effects["metric"] == "overflag")
    ].sort_values("target_variety")
    lines.append(
        md_table(
            ["Comparison", "Difference", "95% item-bootstrap CI", "Adjusted p"],
            [
                [
                    f"{target_name(row.target_variety)} − SAE",
                    pp(row.difference_pp, signed=True),
                    f"[{row.ci_lower_pp:.1f}, {row.ci_upper_pp:.1f}] pp",
                    pvalue(row.cluster_permutation_p_holm),
                ]
                for row in direct_overflag.itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Reasoning changes the error trade-off",
            "",
        ]
    )
    non_sae_reasoning = fairness_reasoning[
        fairness_reasoning["variety"].isin(["indian_english", "hinglish"])
        & fairness_reasoning["metric"].isin(["overflag", "harm_miss"])
    ].sort_values(["metric", "variety"])
    lines.append(
        md_table(
            ["Variety", "Metric", "Direct", "Reasoning", "Change", "95% CI"],
            [
                [
                    target_name(row.variety),
                    "Over-flag" if row.metric == "overflag" else "Harm miss",
                    pct(row.direct_rate),
                    pct(row.reasoning_rate),
                    pp(row.difference_pp, signed=True),
                    f"[{row.ci_lower_pp:.1f}, {row.ci_upper_pp:.1f}] pp",
                ]
                for row in non_sae_reasoning.itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "A negative over-flagging change is beneficial, while a positive harmful-content-miss change is harmful. The reasoning prompt moves both metrics at once and should therefore be described as redistributing errors, not simply fixing linguistic bias.",
            "",
            "## 4. Historical-versus-fresh cohort sensitivity",
            "",
        ]
    )
    cohort_direct = cohort_effects[cohort_effects["condition"] == "direct"].sort_values(
        ["cohort", "target_variety"]
    )
    lines.append(
        md_table(
            ["Cohort", "Comparison", "Systems", "Gap", "95% CI", "Adjusted p"],
            [
                [
                    "Historical" if row.cohort == "original" else "Fresh",
                    f"SAE − {target_name(row.target_variety)}",
                    row.n_systems,
                    pp(row.gap_pp),
                    f"[{row.ci_lower_pp:.1f}, {row.ci_upper_pp:.1f}] pp",
                    pvalue(row.cluster_permutation_p_holm),
                ]
                for row in cohort_direct.itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "The direct-prompt gap has the same direction in both cohorts. Fresh systems show a 6.7-point Indian English gap and a 9.4-point Hinglish gap; historical systems show 6.2 and 8.8 points, respectively.",
            "",
            "## 5. Per-system direct-prompt results",
            "",
        ]
    )

    direct_accuracy = accuracy[accuracy["condition"] == "direct"].pivot(
        index=["model", "system_id", "cohort"], columns="variety", values="accuracy"
    )
    direct_effects = model_effects[model_effects["condition"] == "direct"].pivot(
        index="system_id", columns="target_variety", values="gap_pp"
    )
    per_system_rows = []
    for (model, system_id, cohort), row in direct_accuracy.sort_index().iterrows():
        per_system_rows.append(
            [
                model,
                "Historical" if cohort == "original" else "Fresh",
                pct(row["sae"]),
                pct(row["indian_english"]),
                pct(row["hinglish"]),
                pp(direct_effects.loc[system_id, "indian_english"]),
                pp(direct_effects.loc[system_id, "hinglish"]),
            ]
        )
    lines.append(
        md_table(
            ["System", "Cohort", "SAE", "Indian English", "Hinglish", "IndEng gap", "Hinglish gap"],
            per_system_rows,
        )
    )

    lines.extend(["", "## 6. Complete model-specific paired inference", ""])
    effect_rows = []
    ordered_effects = model_effects.assign(
        condition_order=model_effects["condition"].map({"direct": 0, "cot": 1}),
        target_order=model_effects["target_variety"].map({"indian_english": 0, "hinglish": 1}),
    ).sort_values(["model", "system_id", "condition_order", "target_order"])
    for row in ordered_effects.itertuples():
        effect_rows.append(
            [
                row.model,
                "Historical" if row.cohort == "original" else "Fresh",
                "Direct" if row.condition == "direct" else "Reasoning",
                target_name(row.target_variety),
                pp(row.gap_pp),
                f"[{row.ci_lower_pp:.1f}, {row.ci_upper_pp:.1f}]",
                pvalue(row.permutation_p_holm),
            ]
        )
    lines.append(
        md_table(
            ["Model", "Cohort", "Prompt", "Target", "Gap", "95% CI (pp)", "Adjusted p"],
            effect_rows,
        )
    )

    lines.extend(["", "## 7. Constrained-probability case study", ""])
    lines.append(
        "The strict A–F output grammar defines a separate coded classifier. Before interpreting calibration, its predictions are compared with ordinary direct-label output."
    )
    lines.append("")
    overall_agreement = prompt_agreement[prompt_agreement["variety"] == "all"].sort_values(
        "probability_model_id"
    )
    lines.append(
        md_table(
            ["Model", "Prediction agreement", "95% CI", "Cohen's kappa", "Ordinary accuracy", "Coded accuracy", "Accuracy difference"],
            [
                [
                    model_name(row.probability_model_id),
                    pct(row.prediction_agreement),
                    f"{pct(row.agreement_ci_lower)}–{pct(row.agreement_ci_upper)}",
                    f"{row.cohen_kappa:.3f}",
                    pct(row.ordinary_accuracy),
                    pct(row.probability_prompt_accuracy),
                    pct(row.accuracy_difference, signed=True),
                ]
                for row in overall_agreement.itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "Agreement is high but not perfect, and the coded prompt is about four percentage points less accurate overall. The calibration results therefore characterize the coded classifier, not an interchangeable native-confidence interface.",
            "",
            "Calibration is reported only for GPT-4o-mini and GPT-4.1 Mini because these APIs exposed comparable grammar-constrained six-class token probabilities. Accuracy and confidence are percentages; lower NLL, Brier, ECE, and ACE are better.",
            "",
        ]
    )
    calibration_sorted = calibration.assign(
        model_order=calibration["model_id"].map(
            {"gpt-4o-mini-2024-07-18": 0, "gpt-4.1-mini-2025-04-14": 1}
        ),
        variety_order=calibration["variety"].map({"sae": 0, "indian_english": 1, "hinglish": 2}),
    ).sort_values(["model_order", "variety_order"])
    lines.append(
        md_table(
            ["Model", "Variety", "Accuracy", "Mean confidence", "Confidence − accuracy", "NLL", "Brier", "ECE", "ACE"],
            [
                [
                    model_name(row.model_id),
                    target_name(row.variety),
                    pct(row.accuracy),
                    pct(row.mean_confidence),
                    pct(row.confidence_minus_accuracy, signed=True),
                    f"{row.nll:.3f}",
                    f"{row.brier:.3f}",
                    pct(row.ece),
                    pct(row.ace),
                ]
                for row in calibration_sorted.itertuples()
            ],
        )
    )

    diff_pivot = calibration_diffs.pivot(
        index=["model_id", "target_variety"], columns="metric", values="target_minus_sae"
    ).reset_index()
    lines.extend(["", "### Change relative to SAE", ""])
    lines.append(
        md_table(
            ["Model", "Target", "Accuracy change", "Confidence change", "NLL change", "Brier change", "ECE change", "ACE change"],
            [
                [
                    model_name(row.model_id),
                    target_name(row.target_variety),
                    pp(100 * row.correct, signed=True),
                    pp(100 * row.confidence, signed=True),
                    f"{row.nll:+.3f}",
                    f"{row.brier:+.3f}",
                    pp(100 * row.ece, signed=True),
                    pp(100 * row.ace, signed=True),
                ]
                for row in diff_pivot.itertuples()
            ],
        )
    )
    lines.extend(
        [
            "",
            "The central calibration finding is visible here: accuracy drops by 7–11 points, but mean confidence drops by only about 0.2–1.7 points. Consequently, proper scoring rules and calibration errors worsen.",
            "",
            "## 8. Descriptive uncertainty-set coverage diagnostic",
            "",
            "The conformal-style diagnostic uses a 90% nominal target. Coverage is the percentage of test items whose prediction set contains the true label. Set size is the average number of labels retained; smaller sets are more informative. Because pooled calibration scores remain correlated within matched triplets, these repeated-split values are descriptive and no finite-sample pooled guarantee is claimed.",
            "",
        ]
    )
    conformal_sorted = conformal.assign(
        model_order=conformal["model_id"].map(
            {"gpt-4o-mini-2024-07-18": 0, "gpt-4.1-mini-2025-04-14": 1}
        ),
        method_order=conformal["method"].map({"pooled_marginal": 0, "variety_conditional": 1}),
        variety_order=conformal["variety"].map({"sae": 0, "indian_english": 1, "hinglish": 2}),
    ).sort_values(["model_order", "method_order", "variety_order"])
    lines.append(
        md_table(
            ["Model", "Method", "Variety", "Mean coverage", "Split SD", "Mean set size", "2.5–97.5% split range"],
            [
                [
                    model_name(row.model_id),
                    "Pooled" if row.method == "pooled_marginal" else "Variety-conditional",
                    target_name(row.variety),
                    pct(row.mean_coverage),
                    pct(row.sd_coverage),
                    f"{row.mean_set_size:.2f}",
                    f"{pct(row.coverage_q025)}–{pct(row.coverage_q975)}",
                ]
                for row in conformal_sorted.itertuples()
            ],
        )
    )

    lines.extend(["", "## 9. Hierarchical mixed-effects results", ""])
    lines.append(
        "The model uses SAE, direct prompting, and no variety cue as reference levels. Negative variety coefficients mean lower odds of a correct answer than SAE. Positive variety × reasoning interactions mean reasoning offsets part of that loss."
    )
    lines.append("")
    mixed_rows = []
    for row in mixed.itertuples():
        mixed_rows.append(
            [
                str(row.term).replace('C(variety, Treatment(reference="sae"))[T.', '').replace('C(condition, Treatment(reference="direct"))[T.', '').replace('C(prompt_cue, Treatment(reference="none"))[T.', '').replace(']', ''),
                f"{row.posterior_mean_log_odds:+.3f}",
                f"[{row.credible_lower:+.3f}, {row.credible_upper:+.3f}]",
                f"{row.odds_ratio:.3f}",
            ]
        )
    lines.append(md_table(["Term", "Posterior log-odds", "95% credible interval", "Odds ratio"], mixed_rows))
    lines.extend(
        [
            "",
            f"The optimizer converged successfully after {diagnostics['hierarchical_iterations']} iterations. Estimated random-intercept standard deviations were "
            + ", ".join(f"{row.component}: {row.random_effect_sd:.3f}" for row in mixed_variance.itertuples())
            + ". A GEE model with system fixed effects and item-clustered standard errors reached the same qualitative conclusions.",
            "",
            "## 10. Definitions and mathematics in simple terms",
            "",
            "### Accuracy",
            "",
            "Accuracy is the fraction of predictions equal to the gold label:",
            "",
            "$$",
            "\\operatorname{Accuracy}=\\frac{\\text{number correct}}{\\text{number evaluated}}.",
            "$$",
            "",
            "### Percentage points (pp)",
            "",
            "Percentage points measure an absolute difference between percentages. A change from 84% to 75% is a drop of 9 percentage points, not a 9% relative reduction.",
            "",
            "### Paired linguistic gap",
            "",
            "For item $i$, model-system $m$, prompt condition $c$, and target variety $v$, let $Y_{imcv}=1$ when the prediction is correct and $0$ otherwise. The paired gap is:",
            "",
            "$$",
            "\\widehat{\\Delta}_{mvc}=\\frac{1}{n}\\sum_{i=1}^{n}\\left(Y_{imc,\\mathrm{SAE}}-Y_{imcv}\\right).",
            "$$",
            "",
            "Each subtraction compares two versions of the **same item**. A positive gap means SAE was more accurate. Pairing removes item difficulty as an explanation for the difference.",
            "",
            "### Reasoning effect",
            "",
            "The reasoning effect is reasoning accuracy minus direct accuracy:",
            "",
            "$$",
            "\\widehat{R}_{v}=\\operatorname{Accuracy}_{v,\\mathrm{reasoning}}-\\operatorname{Accuracy}_{v,\\mathrm{direct}}.",
            "$$",
            "",
            "A positive value means the five-step prompt improved classification for that variety.",
            "",
            "### Bootstrap confidence interval",
            "",
            "The item-clustered bootstrap repeatedly resamples the 200 item IDs with replacement. Whenever an item is selected, all its varieties, models, and prompt conditions move together. The middle 95% of the resulting effect estimates forms the confidence interval. This preserves the matched structure.",
            "",
            "### Permutation test and Holm correction",
            "",
            "Under a no-effect null hypothesis, the direction of each paired difference can be randomly flipped. The permutation p-value is the fraction of randomized effects at least as large as the observed effect. Holm correction raises p-values when several related hypotheses are tested, reducing false discoveries.",
            "",
            "### Exact McNemar test",
            "",
            "For one model and two matched varieties, McNemar's test ignores pairs where both predictions are correct or both are wrong. It compares the number of SAE-only correct pairs with the number of target-only correct pairs using an exact binomial test. It is another way to test whether errors are asymmetric within matched items.",
            "",
            "### Log probabilities and class probabilities",
            "",
            "The API supplies one log score $\\ell_k$ for each class. Scores are converted to probabilities with a softmax:",
            "",
            "$$",
            "p_k=\\frac{e^{\\ell_k}}{\\sum_{j=1}^{K}e^{\\ell_j}}.",
            "$$",
            "",
            "The largest $p_k$ is the model's top-label confidence.",
            "",
            "### Negative log-likelihood (NLL)",
            "",
            "For the true class $y$:",
            "",
            "$$",
            "\\operatorname{NLL}=-\\log p_y.",
            "$$",
            "",
            "Confident wrong answers receive a large penalty. Lower NLL is better.",
            "",
            "### Multiclass Brier score",
            "",
            "With one-hot target $y_k$:",
            "",
            "$$",
            "\\operatorname{Brier}=\\sum_{k=1}^{K}(p_k-y_k)^2.",
            "$$",
            "",
            "It measures the squared distance between the full predicted distribution and the correct answer. Lower is better.",
            "",
            "### Calibration, ECE, and ACE",
            "",
            "A calibrated classifier is correct about 80% of the time on predictions made with 80% confidence. ECE groups predictions into confidence bins and averages the absolute confidence–accuracy difference:",
            "",
            "$$",
            "\\operatorname{ECE}=\\sum_{b=1}^{B}\\frac{|I_b|}{n}\\left|\\operatorname{accuracy}(I_b)-\\operatorname{confidence}(I_b)\\right|.",
            "$$",
            "",
            "ACE uses bins containing approximately equal numbers of examples. Both are reported as supplements because their values depend on binning; NLL and Brier are the primary proper scoring rules.",
            "",
            "### Conformal prediction",
            "",
            "The nonconformity score for true class $y$ is:",
            "",
            "$$",
            "s=1-p_y.",
            "$$",
            "",
            "A calibration set determines a threshold $q$. For a new item, all labels satisfying $1-p_k\\le q$ enter the prediction set. **Coverage** is how often this set contains the true label. **Set size** measures how many labels remain. Pooled calibration uses one threshold for all varieties; variety-specific calibration estimates one threshold per variety. In this study, pooled calibration scores within a triplet are correlated, so the repeated-split numbers are empirical diagnostics rather than a formal finite-sample pooled guarantee.",
            "",
            "### Mixed-effects logistic model",
            "",
            "The hierarchical model represents correctness probability using log odds:",
            "",
            "$$",
            "\\operatorname{logit}\\Pr(Y=1)=X\\beta+u_{\\mathrm{item}}+u_{\\mathrm{system}}.",
            "$$",
            "",
            "$X\\beta$ contains the average effects of variety, prompt condition, and their planned interaction. The random intercepts $u$ allow some items and systems to be intrinsically easier or harder. Exponentiating a coefficient gives an odds ratio: values below 1 mean lower odds of correctness than the reference condition.",
            "",
            "## 11. Important limitations",
            "",
            "- The dataset contains only 200 matched items and is heavily weighted toward the normal class (156/200).",
            "- Utterances are short; this does not establish performance on dialogue or long documents.",
            "- Hinglish is code-mixing, and no pure-Hindi comparison is included.",
            "- All runs are recorded as no-cue prompts. Fresh outputs retain prompt hashes, while historical no-cue provenance relies on the authors' preserved execution record because legacy CSVs lack request-level metadata.",
            "- The 420-response human audit used one reviewer per response and has no retained response-level rating export, so it supports only a descriptive aggregate acceptance statement.",
            "- Historical result files contain one stored prediction per cell and do not support the earlier claim that five independent runs were averaged.",
            "- Calibration and uncertainty-set diagnostics cover only two OpenAI models and a distinct coded output grammar; they do not measure unconstrained semantic confidence.",
            "- Pooled calibration scores remain correlated within triplets, so no formal pooled conformal coverage guarantee is claimed.",
            "- Local Llama, Gemma, and Mistral models use Q4_K_M quantization; results apply to those exact model digests.",
            "- Findings concern this six-class moderation task, not all language-model behavior.",
            "",
            "## 12. Validation, cost, and artifact locations",
            "",
            "- Hosted API expenditure recorded for the full development session: **$0.470130** of the $50 cap.",
            "- Every fresh hosted and local result file has one canonical record per evaluation key and zero failed final cells.",
            "- Local models were identified by exact Ollama digests and validated before analysis.",
            "- Run `pytest -q` for the current test result; this generated report does not execute tests or lint checks.",
            "- The current manuscript targets ACM FAccT 2027.",
            "",
            "### Key files",
            "",
            "- [Paper tables and figures](../../paper/facct2027/)",
            "- [Frozen results manifest](./manifests/results_manifest.json)",
            "- [Canonical accuracy table](./validated/accuracy_long.csv)",
            "- [Paired-effect tables](./metrics/paired/)",
            "- [Calibration tables](./metrics/calibration/)",
            "- [Conformal tables](./metrics/conformal/)",
            "- [Mixed-effects outputs](./metrics/mixed_effects/)",
            "- [Exact local model manifests](./model_manifests/summary.json)",
            "",
            "## Bottom line",
            "",
            "The paired design shows a reproducible accuracy disadvantage for Indian English and Hinglish under direct prompting. Explicit reasoning substantially narrows the gap without helping SAE. In a separate coded-output case study, grammar-constrained label-token confidence declines much less than accuracy, so calibration worsens. The descriptive repeated-split diagnostic finds lower average non-SAE coverage under pooled thresholds; variety-specific thresholds raise that average only by returning larger and less informative sets.",
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
