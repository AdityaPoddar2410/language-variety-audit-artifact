"""Score model-generated meanings against ground-truth definitions using
GPT-5.2 as an optional semantic-equivalence judge.

Reads classification results from ``run_evaluations.py`` (JSON-lines), looks
up the ground-truth ``indian_meaning`` for each item from the benchmark CSV,
and writes per-(item, variety, condition, provider) judge scores.

Only records with a non-empty ``predicted_meaning`` are scored (i.e. the
``cot`` condition; the ``direct`` prompt does not produce a free-text
meaning).
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

from .io_utils import load_benchmark, read_results, write_results_jsonl
from .llm_utils import chat_completion
from .prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt


def _judge_one(sentence_a: str, sentence_b: str, model: str, retries: int = 2) -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = chat_completion(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=judge_user_prompt(sentence_a, sentence_b),
                provider="openai",
                model=model,
                json_output=True,
            )
            try:
                parsed = json.loads(raw)
                return {
                    "judge_score": float(parsed.get("score", 0.0)),
                    "judge_match": bool(parsed.get("match", False)),
                    "judge_relation": str(parsed.get("relation", "")),
                    "judge_threshold": float(parsed.get("match_threshold", 0.90)),
                    "judge_reason": str(parsed.get("reason", "")),
                    "judge_raw": raw,
                    "judge_error": None,
                }
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return {
        "judge_score": float("nan"),
        "judge_match": False,
        "judge_relation": "",
        "judge_threshold": 0.90,
        "judge_reason": "",
        "judge_raw": "",
        "judge_error": str(last_error),
    }


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
    default="results/raw_classifications.jsonl",
    show_default=True,
    help="Classification results from run_evaluations (.jsonl/.csv/.xlsx).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="results/judge_scores.jsonl",
    show_default=True,
)
@click.option(
    "--judge-model",
    type=str,
    default="gpt-5.2",
    show_default=True,
    help="OpenAI model to use as judge (paper uses gpt-5.2 with default reasoning).",
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Score only the first N records (smoke test).",
)
@click.option(
    "--workers",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of parallel judge calls.",
)
@click.option("--dotenv/--no-dotenv", default=True, show_default=True)
def main(benchmark, results, out, judge_model, max_rows, workers, dotenv):
    """Run the LLM-as-judge over a classification results file."""
    if dotenv:
        load_dotenv()

    bench = load_benchmark(benchmark).set_index("id")
    df = read_results(results)

    # Only score records that carry a non-empty model-generated meaning
    # (i.e. the cot condition; direct produces a label only).
    if "predicted_meaning" not in df.columns:
        raise click.UsageError(
            "Results file does not contain a 'predicted_meaning' column; "
            "the judge requires output from the cot condition."
        )
    df = df[df["predicted_meaning"].astype(str).str.strip().ne("")].copy()
    if max_rows is not None:
        df = df.head(max_rows)

    rows = list(df.itertuples(index=False))
    out_records: list[dict] = [None] * len(rows)
    lock = threading.Lock()
    completed = [0]
    skipped = [0]

    def _judge_row(idx: int, row) -> None:
        rec = row._asdict()
        item_id = int(rec["id"])
        if item_id not in bench.index:
            with lock:
                skipped[0] += 1
                click.echo(f"  ! id {item_id} missing from benchmark; skipping", err=True)
            out_records[idx] = None
            return
        sentence_b = str(bench.loc[item_id, "indian_meaning"])
        sentence_a = str(rec.get("predicted_meaning", ""))
        judge = _judge_one(sentence_a, sentence_b, model=judge_model)
        rec.update(judge)
        out_records[idx] = rec
        status = "ok" if judge["judge_error"] is None else "ERR"
        with lock:
            completed[0] += 1
            click.echo(
                f"[{completed[0]:5d}/{len(rows)}] id={item_id:3d} "
                f"{rec['variety']:15s} {rec['condition']:6s} {rec['provider']:9s} "
                f"score={judge['judge_score']:.2f} match={judge['judge_match']} {status}"
            )

    click.echo(f"Judging {len(rows)} records with {workers} parallel workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_judge_row, i, row): i for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            fut.result()

    final = [r for r in out_records if r is not None]
    write_results_jsonl(final, out)
    n_err = sum(1 for r in final if r.get("judge_error"))
    click.echo(f"\nWrote {len(final)} judged records to {out} ({n_err} errors).")


if __name__ == "__main__":
    main()
