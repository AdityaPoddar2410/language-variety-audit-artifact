#!/usr/bin/env python3
"""Generate publication-ready study figures from frozen result tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = {
    "sae": "#4C78A8",
    "indian_english": "#F58518",
    "hinglish": "#54A24B",
}
_LABELS = {
    "sae": "SAE",
    "indian_english": "Indian English",
    "hinglish": "Hinglish",
}
_MODEL_NAMES = {
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
        }
    )


def paired_gap_figure(paired: pd.DataFrame, output: Path) -> None:
    direct = paired[paired["condition"] == "direct"].copy()
    direct["display"] = direct["model"] + direct["cohort"].map(
        {"historical": " (historical)", "fresh": ""}
    ).fillna("")
    system_order = (
        direct[["display", "cohort", "model"]]
        .drop_duplicates()
        .assign(
            cohort_order=lambda frame: frame["cohort"].map(
                {"fresh": 0, "historical": 1}
            )
        )
        .sort_values(["cohort_order", "model"])["display"]
        .tolist()
    )
    y = np.arange(len(system_order))
    offsets = {"indian_english": -0.14, "hinglish": 0.14}
    markers = {"indian_english": "o", "hinglish": "s"}
    figure, axis = plt.subplots(figsize=(7.0, 3.55))
    for target in ("indian_english", "hinglish"):
        subset = (
            direct[direct["target_variety"] == target]
            .set_index("display")
            .loc[system_order]
        )
        estimates = subset["gap_pp"].to_numpy()
        lower = estimates - subset["ci_lower_pp"].to_numpy()
        upper = subset["ci_upper_pp"].to_numpy() - estimates
        axis.errorbar(
            estimates,
            y + offsets[target],
            xerr=np.vstack([lower, upper]),
            fmt=markers[target],
            markersize=4.2,
            color=_COLORS[target],
            ecolor=_COLORS[target],
            elinewidth=1.1,
            capsize=2.2,
            label=_LABELS[target],
            zorder=3,
        )
    axis.axvline(0, color="#333333", linewidth=0.9)
    axis.axhline(4.5, color="#999999", linestyle=":", linewidth=0.8)
    axis.text(
        0.99,
        0.48,
        "fresh systems",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=7,
    )
    axis.text(
        0.99,
        0.45,
        "historical systems",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#555555",
        fontsize=7,
    )
    axis.set_yticks(y, system_order)
    axis.invert_yaxis()
    axis.set_xlim(-8, 20)
    axis.set_xlabel("SAE accuracy minus target-variety accuracy (percentage points)")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handletextpad=0.5,
        columnspacing=1.5,
    )
    figure.subplots_adjust(left=0.27, right=0.98, bottom=0.16, top=0.88)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _accuracy_confidence_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    model_id: str,
    panel_label: str,
) -> None:
    varieties = ["sae", "indian_english", "hinglish"]
    subset = frame[frame["model_id"] == model_id].set_index("variety").loc[varieties]
    x = np.arange(len(varieties))
    width = 0.36
    accuracy = subset["accuracy"].to_numpy()
    confidence = subset["mean_confidence"].to_numpy()
    accuracy_bars = axis.bar(
        x - width / 2,
        accuracy,
        width,
        label="Accuracy",
        color="#4C78A8",
        edgecolor="white",
    )
    confidence_bars = axis.bar(
        x + width / 2,
        confidence,
        width,
        label="Constrained confidence",
        color="#B8B8B8",
        edgecolor="#555555",
        linewidth=0.5,
        hatch="//",
    )
    axis.bar_label(accuracy_bars, labels=[f"{100 * value:.0f}" for value in accuracy], padding=2, fontsize=6.5)
    axis.bar_label(confidence_bars, labels=[f"{100 * value:.0f}" for value in confidence], padding=2, fontsize=6.5)
    axis.set_ylim(0.65, 1.02)
    axis.set_xticks(x, [_LABELS[variety] for variety in varieties], rotation=12)
    axis.set_title(f"{panel_label} {_MODEL_NAMES[model_id]}: accuracy vs. confidence", loc="left")
    axis.set_ylabel("Proportion")
    axis.grid(axis="y", color="#E1E1E1", linewidth=0.6)


def uncertainty_diagnostics_figure(
    calibration: pd.DataFrame,
    conformal: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
    model_ids = ["gpt-4o-mini-2024-07-18", "gpt-4.1-mini-2025-04-14"]
    _accuracy_confidence_panel(axes[0, 0], calibration, model_ids[0], "(a)")
    _accuracy_confidence_panel(axes[0, 1], calibration, model_ids[1], "(b)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
    )

    aggregated = (
        conformal.groupby(["method", "variety"], as_index=False)
        .agg(coverage=("mean_coverage", "mean"), set_size=("mean_set_size", "mean"))
    )
    methods = ["pooled_marginal", "variety_conditional"]
    method_labels = {
        "pooled_marginal": "Pooled",
        "variety_conditional": "Variety-specific",
    }
    varieties = ["sae", "indian_english", "hinglish"]
    x = np.arange(len(varieties))
    width = 0.36
    method_styles = {
        "pooled_marginal": {"color": "#72B7B2", "hatch": None},
        "variety_conditional": {"color": "#ECA82C", "hatch": "//"},
    }
    for method_index, method in enumerate(methods):
        subset = aggregated[aggregated["method"] == method].set_index("variety").loc[varieties]
        offset = (method_index - 0.5) * width
        coverage_bars = axes[1, 0].bar(
            x + offset,
            subset["coverage"],
            width,
            label=method_labels[method],
            color=method_styles[method]["color"],
            edgecolor="#555555",
            linewidth=0.4,
            hatch=method_styles[method]["hatch"],
        )
        set_bars = axes[1, 1].bar(
            x + offset,
            subset["set_size"],
            width,
            label=method_labels[method],
            color=method_styles[method]["color"],
            edgecolor="#555555",
            linewidth=0.4,
            hatch=method_styles[method]["hatch"],
        )
        axes[1, 0].bar_label(
            coverage_bars,
            labels=[f"{100 * value:.1f}" for value in subset["coverage"]],
            padding=2,
            fontsize=6.3,
            rotation=90,
        )
        axes[1, 1].bar_label(
            set_bars,
            labels=[f"{value:.2f}" for value in subset["set_size"]],
            padding=2,
            fontsize=6.3,
            rotation=90,
        )
    axes[1, 0].axhline(0.9, color="#333333", linestyle="--", linewidth=0.9)
    axes[1, 0].text(2.47, 0.901, "90% target", ha="right", va="bottom", fontsize=6.5)
    axes[1, 0].set_ylim(0.84, 0.955)
    axes[1, 1].set_ylim(0, 2.55)
    axes[1, 0].set_title("(c) Repeated-split empirical coverage", loc="left")
    axes[1, 1].set_title("(d) Mean prediction-set size", loc="left")
    axes[1, 0].set_ylabel("Coverage")
    axes[1, 1].set_ylabel("Number of labels")
    for axis in axes[1, :]:
        axis.set_xticks(x, [_LABELS[variety] for variety in varieties], rotation=12)
        axis.grid(axis="y", color="#E1E1E1", linewidth=0.6)
    axes[1, 0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.91, wspace=0.28, hspace=0.48)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired",
        type=Path,
        default=Path("results/metrics/paired/model_paired_effects.csv"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/metrics/calibration/calibration_summary.csv"),
    )
    parser.add_argument(
        "--conformal",
        type=Path,
        default=Path("results/metrics/conformal/conformal_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    paired_gap_figure(pd.read_csv(args.paired), args.out_dir / "paired_gaps.pdf")
    uncertainty_diagnostics_figure(
        pd.read_csv(args.calibration),
        pd.read_csv(args.conformal),
        args.out_dir / "uncertainty_diagnostics.pdf",
    )
    for obsolete in ("reliability.pdf", "conformal_coverage.pdf"):
        path = args.out_dir / obsolete
        if path.exists():
            path.unlink()
    for path in sorted(args.out_dir.glob("*.pdf")):
        print(path)


if __name__ == "__main__":
    main()
