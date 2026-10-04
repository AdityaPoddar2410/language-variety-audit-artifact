#!/usr/bin/env python3
"""Compare ordinary and constrained-probability classification prompts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from beyond_weird.prompt_agreement import prompt_agreement_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accuracy",
        type=Path,
        default=Path("results/validated/accuracy_long.csv"),
    )
    parser.add_argument(
        "--probability-table",
        type=Path,
        default=Path("results/metrics/calibration/probability_long.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/metrics/calibration/prompt_agreement.csv"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    result = prompt_agreement_table(
        pd.read_csv(args.accuracy),
        pd.read_csv(args.probability_table),
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
