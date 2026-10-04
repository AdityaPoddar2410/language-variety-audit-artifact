#!/usr/bin/env python3
"""Build study calibration tables from probability JSONL files."""
from __future__ import annotations

import argparse
from pathlib import Path

from beyond_weird.calibration_stats import (
    calibration_summary,
    load_probability_results,
    paired_calibration_differences,
    reliability_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probability-result",
        action="append",
        type=Path,
        required=True,
        help="Probability JSONL file; repeat for each model.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/metrics/calibration"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    frame = load_probability_results(args.probability_result)
    summary = calibration_summary(frame, bins=args.bins)
    reliability = reliability_table(frame, bins=args.bins)
    differences = paired_calibration_differences(
        frame,
        n_bootstrap=args.bootstrap,
        bins=args.bins,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "probability_long.csv": frame,
        "calibration_summary.csv": summary,
        "reliability_bins.csv": reliability,
        "paired_calibration_differences.csv": differences,
    }
    for name, table in outputs.items():
        path = args.out_dir / name
        table.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
