#!/usr/bin/env python3
"""Evaluate pooled and variety-conditional conformal prediction sets."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from beyond_weird.conformal_stats import (
    conformal_summary,
    repeated_conformal_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probability-table",
        type=Path,
        default=Path("results/metrics/calibration/probability_long.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/metrics/conformal"),
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    frame = pd.read_csv(args.probability_table)
    splits = repeated_conformal_evaluation(
        frame,
        alpha=args.alpha,
        calibration_fraction=args.calibration_fraction,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    summary = conformal_summary(splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.out_dir / "repeated_split_results.csv"
    summary_path = args.out_dir / "conformal_summary.csv"
    splits.to_csv(splits_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(splits_path)
    print(summary_path)


if __name__ == "__main__":
    main()
