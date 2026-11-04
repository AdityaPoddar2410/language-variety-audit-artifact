#!/usr/bin/env python3
"""Build the canonical study accuracy table and coverage report."""
from __future__ import annotations

import argparse
from pathlib import Path

from language_variety_audit.study_data import build_accuracy_table, coverage_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-result",
        action="append",
        type=Path,
        required=True,
        help="Current-run JSONL file; repeat for each new model.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/study/validated"),
    )
    args = parser.parse_args()

    frame = build_accuracy_table(args.current_result)
    coverage = coverage_table(frame)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    accuracy_path = args.out_dir / "accuracy_long.csv"
    coverage_path = args.out_dir / "accuracy_coverage.csv"
    frame.to_csv(accuracy_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    print(accuracy_path)
    print(coverage_path)


if __name__ == "__main__":
    main()
