#!/usr/bin/env python3
"""Compute fairness-oriented moderation error rates and paired effects."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from language_variety_audit.fairness_stats import (
    aggregate_fairness_effects,
    aggregate_fairness_effects_by_cohort,
    aggregate_fairness_effects_by_stratum,
    aggregate_reasoning_fairness_effects,
    aggregate_reasoning_fairness_effects_by_cohort,
    fairness_summary,
    overflag_relative_risk,
    system_overflag_consistency,
)

ITEM_FAMILY_COLUMN = "item_family"


def _with_item_family(frame: pd.DataFrame, benchmark: Path) -> pd.DataFrame:
    items = pd.read_csv(benchmark, usecols=["id", "dataset_type"]).rename(
        columns={"dataset_type": ITEM_FAMILY_COLUMN}
    )
    merged = frame.merge(items, on="id", how="left", validate="many_to_one")
    if merged[ITEM_FAMILY_COLUMN].isna().any():
        raise ValueError("Every evaluated item must have a benchmark item family")
    return merged


def _relative_risk_by_family(
    frame: pd.DataFrame, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    tables = []
    for index, (family, subset) in enumerate(frame.groupby(ITEM_FAMILY_COLUMN)):
        table = overflag_relative_risk(
            subset, n_bootstrap=n_bootstrap, seed=seed + index * 1_000
        )
        table.insert(0, ITEM_FAMILY_COLUMN, family)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accuracy",
        type=Path,
        default=Path("results/study/validated/accuracy_long.csv"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/language_variety_audit_benchmark.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/study/metrics/fairness"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    frame = pd.read_csv(args.accuracy)
    family_frame = _with_item_family(frame, args.benchmark)
    outputs = {
        "fairness_summary.csv": fairness_summary(frame),
        "aggregate_fairness_effects.csv": aggregate_fairness_effects(
            frame,
            n_bootstrap=args.bootstrap,
            n_permutations=args.permutations,
            seed=args.seed,
        ),
        "aggregate_reasoning_fairness_effects.csv": (
            aggregate_reasoning_fairness_effects(
                frame,
                n_bootstrap=args.bootstrap,
                n_permutations=args.permutations,
                seed=args.seed,
            )
        ),
        "aggregate_fairness_effects_equal_models.csv": aggregate_fairness_effects(
            frame,
            n_bootstrap=args.bootstrap,
            n_permutations=args.permutations,
            seed=args.seed,
            unit_column="model",
        ),
        "aggregate_reasoning_fairness_effects_equal_models.csv": (
            aggregate_reasoning_fairness_effects(
                frame,
                n_bootstrap=args.bootstrap,
                n_permutations=args.permutations,
                seed=args.seed,
                unit_column="model",
            )
        ),
        "aggregate_fairness_effects_by_cohort.csv": (
            aggregate_fairness_effects_by_cohort(
                frame,
                n_bootstrap=args.bootstrap,
                n_permutations=args.permutations,
                seed=args.seed,
            )
        ),
        "aggregate_reasoning_fairness_effects_by_cohort.csv": (
            aggregate_reasoning_fairness_effects_by_cohort(
                frame,
                n_bootstrap=args.bootstrap,
                n_permutations=args.permutations,
                seed=args.seed,
            )
        ),
        "overflag_relative_risk.csv": overflag_relative_risk(
            frame,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        ),
        "system_overflag_consistency.csv": system_overflag_consistency(frame),
        "aggregate_fairness_effects_by_item_family.csv": (
            aggregate_fairness_effects_by_stratum(
                family_frame,
                ITEM_FAMILY_COLUMN,
                n_bootstrap=args.bootstrap,
                n_permutations=args.permutations,
                seed=args.seed,
            )
        ),
        "overflag_relative_risk_by_item_family.csv": _relative_risk_by_family(
            family_frame,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        path = args.out_dir / name
        table.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
