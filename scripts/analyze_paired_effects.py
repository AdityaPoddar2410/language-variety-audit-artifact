#!/usr/bin/env python3
"""Compute paired study accuracy estimates and inference tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from beyond_weird.paired_stats import (
    accuracy_summary,
    aggregate_paired_effects,
    aggregate_paired_effects_by_cohort,
    aggregate_paired_effects_by_prompt_cue,
    aggregate_reasoning_effects,
    aggregate_reasoning_effects_by_cohort,
    aggregate_reasoning_effects_by_prompt_cue,
    model_paired_effects,
    model_reasoning_effects,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accuracy",
        type=Path,
        default=Path("results/validated/accuracy_long.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/metrics/paired"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    frame = pd.read_csv(args.accuracy)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = accuracy_summary(frame)
    per_model = model_paired_effects(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    aggregate = aggregate_paired_effects(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    model_reasoning = model_reasoning_effects(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    aggregate_reasoning = aggregate_reasoning_effects(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    by_cohort = aggregate_paired_effects_by_cohort(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    by_prompt_cue = aggregate_paired_effects_by_prompt_cue(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    reasoning_by_cohort = aggregate_reasoning_effects_by_cohort(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    reasoning_by_prompt_cue = aggregate_reasoning_effects_by_prompt_cue(
        frame,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )

    outputs = {
        "accuracy_summary.csv": summary,
        "model_paired_effects.csv": per_model,
        "aggregate_paired_effects.csv": aggregate,
        "model_reasoning_effects.csv": model_reasoning,
        "aggregate_reasoning_effects.csv": aggregate_reasoning,
        "aggregate_paired_effects_by_cohort.csv": by_cohort,
        "aggregate_reasoning_effects_by_cohort.csv": reasoning_by_cohort,
        "aggregate_paired_effects_by_prompt_cue.csv": by_prompt_cue,
        "aggregate_reasoning_effects_by_prompt_cue.csv": reasoning_by_prompt_cue,
    }
    for name, table in outputs.items():
        path = args.out_dir / name
        table.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
