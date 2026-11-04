"""Compute headline metrics from a results file:

  * Per-(model, condition, variety) classification accuracy.
  * Linguistic Gap (Delta_v) per model and condition: SAE - IndEng,
    SAE - Hinglish.
  * Per-(model, condition) item-paired McNemar tests for SAE vs IndEng and
    SAE vs Hinglish, with Bonferroni correction across the resulting family
    of tests.

The script accepts either:
  * Classification results (predicted_category) -- correctness is computed
    against the gold ``category`` column in the benchmark, OR
  * Judge results (judge_match) -- correctness is read directly from the
    judge's binary ``match`` flag.

Use ``--correctness category`` for the former and ``--correctness judge``
for the latter.
"""
from __future__ import annotations

import math
from pathlib import Path

import click
import pandas as pd

from .io_utils import load_benchmark, read_results


def _mcnemar_exact(b: int, c: int) -> tuple[float, float, float]:
    """McNemar test on the (b, c) discordant pairs.

    Returns ``(chi2, chi2_cc, p_value)``:

      * ``chi2``    -- the standard (uncorrected) McNemar statistic
                       ``(b - c)**2 / (b + c)``. This is the quantity
                       reported in the paper (Section 6, Appendix K).
      * ``chi2_cc`` -- Edwards' continuity-corrected statistic
                       ``(|b - c| - 1)**2 / (b + c)``. Statistically
                       preferred for moderate ``n``; reported alongside
                       for transparency.
      * ``p_value`` -- exact two-sided binomial when ``b + c <= 25``
                       (does not depend on either statistic), otherwise
                       the asymptotic chi^2_1 survival of ``chi2`` (the
                       uncorrected statistic, to match the paper).
    """
    n = b + c
    if n == 0:
        return 0.0, 0.0, 1.0
    chi2 = (b - c) ** 2 / n
    chi2_cc = (abs(b - c) - 1) ** 2 / n
    if n <= 25:
        from math import comb
        k = min(b, c)
        p_one = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p_value = min(1.0, 2 * p_one)
    else:
        # P(chi^2_1 > x) = erfc(sqrt(x)/sqrt(2))
        p_value = math.erfc(math.sqrt(chi2) / math.sqrt(2))
    return chi2, chi2_cc, p_value


def _per_cell_correct(df: pd.DataFrame, bench: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return per-cell correctness, using authoritative benchmark labels in category mode."""
    if mode == "category":
        gold = bench[["category"]].rename(columns={"category": "gold_category"})
        merged = df.drop(columns=["gold_category"], errors="ignore").merge(
            gold, left_on="id", right_index=True, how="left"
        )
        merged["correct"] = (
            merged["predicted_category"].astype(str).str.strip().str.lower()
            == merged["gold_category"].astype(str).str.strip().str.lower()
        )
        return merged
    if mode == "judge":
        if "judge_match" not in df.columns:
            raise ValueError("--correctness judge requires a results file with judge_match column")
        df = df.copy()
        df["correct"] = df["judge_match"].astype(bool)
        return df
    raise ValueError(f"Unknown correctness mode: {mode}")


def _accuracy_table(long: pd.DataFrame) -> pd.DataFrame:
    """Group by (provider, model, condition, variety) and compute accuracy."""
    agg = (
        long.groupby(["provider", "model", "condition", "variety"], dropna=False)
        ["correct"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "n_correct", "count": "n_items"})
    )
    agg["accuracy"] = agg["n_correct"] / agg["n_items"]
    return agg


def _gap_table(acc: pd.DataFrame) -> pd.DataFrame:
    """Compute Delta_v = acc(sae) - acc(variety) for indian_english and hinglish."""
    pivot = acc.pivot_table(
        index=["provider", "model", "condition"],
        columns="variety",
        values="accuracy",
    ).reset_index()
    out_rows = []
    for _, row in pivot.iterrows():
        sae = row.get("sae", float("nan"))
        for v in ("indian_english", "hinglish"):
            if v in pivot.columns:
                out_rows.append({
                    "provider": row["provider"],
                    "model": row["model"],
                    "condition": row["condition"],
                    "variety": v,
                    "acc_sae": sae,
                    "acc_variety": row[v],
                    "delta_v_pp": (sae - row[v]) * 100.0,
                })
    return pd.DataFrame(out_rows)


def _mcnemar_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per (provider, model, condition) and (sae vs target_variety), compute McNemar."""
    rows = []
    for (provider, model, condition), grp in long.groupby(
        ["provider", "model", "condition"], dropna=False
    ):
        wide = grp.pivot_table(
            index="id", columns="variety", values="correct", aggfunc="first"
        )
        if "sae" not in wide.columns:
            continue
        for v in ("indian_english", "hinglish"):
            if v not in wide.columns:
                continue
            paired = wide[["sae", v]].dropna()
            if paired.empty:
                continue
            sae = paired["sae"].astype(bool)
            other = paired[v].astype(bool)
            both_correct = int((sae & other).sum())
            sae_only = int((sae & ~other).sum())
            other_only = int((~sae & other).sum())
            both_wrong = int((~sae & ~other).sum())
            chi2, chi2_cc, p = _mcnemar_exact(sae_only, other_only)
            rows.append({
                "provider": provider,
                "model": model,
                "condition": condition,
                "comparison": f"sae_vs_{v}",
                "n_paired": int(len(paired)),
                "both_correct": both_correct,
                "sae_only": sae_only,
                "other_only": other_only,
                "both_wrong": both_wrong,
                "chi2": chi2,
                "chi2_cc": chi2_cc,
                "p_value": p,
            })
    return pd.DataFrame(rows)


def _bonferroni(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    df = df.copy()
    m = max(len(df), 1)
    df["bonferroni_m"] = m
    df["p_bonferroni"] = (df[p_col] * m).clip(upper=1.0)
    df["sig_bonferroni_0_05"] = df["p_bonferroni"] < 0.05
    return df


@click.command()
@click.option(
    "--benchmark",
    type=click.Path(exists=True, dir_okay=False),
    default="data/language_variety_audit_benchmark.csv",
    show_default=True,
)
@click.option(
    "--results",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Results file (.jsonl/.csv/.xlsx) from run_evaluations or run_judge.",
)
@click.option(
    "--correctness",
    type=click.Choice(["category", "judge"], case_sensitive=False),
    default="category",
    show_default=True,
    help="How to derive per-item correctness.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False),
    default="results/metrics",
    show_default=True,
)
def main(benchmark, results, correctness, out_dir):
    """Compute accuracy, Delta_v, and Bonferroni-corrected McNemar tests."""
    bench = load_benchmark(benchmark).set_index("id")
    df = read_results(results)
    long = _per_cell_correct(df, bench, mode=correctness)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = _accuracy_table(long)
    acc_path = out_dir / "accuracy.csv"
    acc.to_csv(acc_path, index=False)

    gaps = _gap_table(acc)
    gaps_path = out_dir / "linguistic_gap.csv"
    gaps.to_csv(gaps_path, index=False)

    mcn = _mcnemar_table(long)
    mcn = _bonferroni(mcn, p_col="p_value")
    mcn_path = out_dir / "mcnemar.csv"
    mcn.to_csv(mcn_path, index=False)

    click.echo("\n=== Accuracy (provider x model x condition x variety) ===")
    click.echo(acc.to_string(index=False))
    click.echo("\n=== Linguistic Gap (Delta_v, percentage points) ===")
    click.echo(gaps.to_string(index=False))
    click.echo("\n=== McNemar (Bonferroni-corrected over family) ===")
    click.echo(mcn.to_string(index=False))
    click.echo(f"\nWrote: {acc_path}\n       {gaps_path}\n       {mcn_path}")


if __name__ == "__main__":
    main()
