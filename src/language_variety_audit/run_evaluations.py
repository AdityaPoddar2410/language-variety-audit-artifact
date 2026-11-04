"""Run classification evaluations across providers and linguistic varieties.

For each item in the benchmark, each (provider, model) pair is queried under
two reasoning conditions:

  * ``direct``  -- the no-reasoning prompt.
  * ``cot``     -- the structured reasoning prompt.

For each of the three language conditions (SAE, Indian English, Hinglish)
the corresponding text column is sent to the model. Results are written as
JSON-lines so they can be fed to ``compute_metrics`` and ``run_judge``
without re-running inference.

Usage:
    python -m language_variety_audit.run_evaluations \\
        --benchmark data/language_variety_audit_benchmark.csv \\
        --out results/raw_classifications.jsonl \\
        --provider openai --model gpt-4o-mini-2024-07-18 \\
        --condition direct cot

Set ``--max-rows N`` to dry-run on the first N items.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import pandas as pd
from dotenv import load_dotenv

from .io_utils import VARIETIES, VARIETY_TO_COLUMN, load_benchmark
from .llm_utils import PROVIDER_CHOICES, chat_completion_with_metadata
from .pricing import estimate_cost_usd
from .prompts import (
    CONTENT_CATEGORIES,
    cot_classification_prompt,
    direct_classification_prompt,
)
from .run_records import (
    append_record,
    compact_records,
    default_run_id,
    file_sha256,
    read_latest_records,
    result_key,
    successful,
    text_sha256,
    utc_timestamp,
)

# Historical defaults; fresh configurations are selected explicitly via the CLI.
DEFAULT_MATRIX: list[tuple[str, str]] = [
    ("openai", "gpt-4o-mini-2024-07-18"),
    ("anthropic", "claude-3-5-haiku-20241022"),
    ("ollama", "llama3.1:8b"),
    ("ollama", "gemma3:12b"),
]

_DIRECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": CONTENT_CATEGORIES}},
    "required": ["category"],
    "additionalProperties": False,
}
_COT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "meaning": {"type": "string"},
        "category": {"type": "string", "enum": CONTENT_CATEGORIES},
    },
    "required": ["reasoning", "meaning", "category"],
    "additionalProperties": False,
}
_VALID_LABELS = set(CONTENT_CATEGORIES)


@dataclass(frozen=True)
class EvalCell:
    item_id: int
    variety: str
    text: str
    gold_category: str
    condition: str
    provider: str
    model: str


def _iter_cells(
    df: pd.DataFrame,
    matrix: list[tuple[str, str]],
    conditions: list[str],
    varieties: list[str],
) -> Iterable[EvalCell]:
    for _, row in df.iterrows():
        for variety in varieties:
            text = row.get(VARIETY_TO_COLUMN[variety])
            if text is None or (isinstance(text, float) and pd.isna(text)):
                continue
            for condition in conditions:
                for provider, model in matrix:
                    yield EvalCell(
                        item_id=int(row["id"]),
                        variety=variety,
                        text=str(text),
                        gold_category=str(row["category"]).strip().lower(),
                        condition=condition,
                        provider=provider,
                        model=model,
                    )


def _cell_key(cell: EvalCell) -> tuple[Any, ...]:
    return (cell.item_id, cell.variety, cell.condition, cell.provider, cell.model)


def _prompts_for(cell: EvalCell) -> tuple[str, str]:
    if cell.condition == "direct":
        return direct_classification_prompt(cell.text)
    if cell.condition == "cot":
        return cot_classification_prompt(cell.text)
    raise ValueError(f"Unknown condition: {cell.condition}")


def _conservative_batch_cost(cells: Iterable[EvalCell]) -> float:
    total = 0.0
    for cell in cells:
        system, user = _prompts_for(cell)
        estimated_input_tokens = max((len(system) + len(user) + 3) // 4, 1)
        estimate = estimate_cost_usd(
            cell.model,
            input_tokens=estimated_input_tokens,
            output_tokens=4096,
        )
        if estimate is not None:
            total += estimate
    return total


def _run_cell(
    cell: EvalCell,
    run_id: str,
    benchmark_sha256: str,
    retries: int = 2,
) -> dict[str, Any]:
    system, user = _prompts_for(cell)
    schema = _DIRECT_SCHEMA if cell.condition == "direct" else _COT_SCHEMA

    required = set(schema["required"])
    prompt_sha256 = text_sha256(system, user)
    last_error: Exception | None = None
    last_raw = ""
    parse_error = False
    resolved_model: str | None = None
    request_ids: list[str] = []
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    total_latency_seconds = 0.0
    total_cost = 0.0
    cost_known = True

    for attempt in range(1, retries + 2):
        try:
            completion = chat_completion_with_metadata(
                system_prompt=system,
                user_prompt=user,
                provider=cell.provider,
                model=cell.model,
                json_output=True,
                max_tokens=4096,
                json_schema=schema,
            )
            last_raw = completion.text
            resolved_model = completion.model
            if completion.request_id:
                request_ids.append(completion.request_id)
            total_input_tokens += completion.input_tokens
            total_cached_input_tokens += completion.cached_input_tokens
            total_output_tokens += completion.output_tokens
            total_latency_seconds += completion.latency_seconds
            attempt_cost = estimate_cost_usd(
                completion.model,
                completion.input_tokens,
                completion.output_tokens,
                completion.cached_input_tokens,
            )
            if attempt_cost is None:
                cost_known = False
            else:
                total_cost += attempt_cost
            clean = re.sub(r"^```(?:json)?\s*", "", last_raw.strip(), flags=re.IGNORECASE)
            clean = re.sub(r"\s*```$", "", clean)
            parsed = json.loads(clean)
            if not isinstance(parsed, dict):
                raise TypeError("Response JSON is not an object")
            missing = required - parsed.keys()
            if missing:
                raise ValueError(f"Missing required keys: {sorted(missing)}")
            prediction = str(parsed.get("category", "")).strip().lower()
            if prediction not in _VALID_LABELS:
                raise ValueError(f"Invalid category: {prediction!r}")

            return {
                "run_id": run_id,
                "timestamp_utc": utc_timestamp(),
                "benchmark_sha256": benchmark_sha256,
                "prompt_sha256": prompt_sha256,
                "prompt_version": "v1-no-variety-cue",
                "id": cell.item_id,
                "variety": cell.variety,
                "condition": cell.condition,
                "provider": cell.provider,
                "model": cell.model,
                "resolved_model": resolved_model,
                "request_id": request_ids[-1] if request_ids else None,
                "attempt_request_ids": request_ids,
                "text": cell.text,
                "gold_category": cell.gold_category,
                "predicted_category": prediction,
                "predicted_meaning": str(parsed.get("meaning", "")),
                "predicted_reasoning": str(parsed.get("reasoning", "")),
                "raw_response": last_raw,
                "input_tokens": total_input_tokens,
                "cached_input_tokens": total_cached_input_tokens,
                "output_tokens": total_output_tokens,
                "estimated_cost_usd": total_cost if cost_known else None,
                "latency_seconds": total_latency_seconds,
                "attempt_count": attempt,
                "parse_error": False,
                "error": None,
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            parse_error = True
        except Exception as exc:  # noqa: BLE001 -- provider errors are recorded verbatim
            last_error = exc
            parse_error = False
        if attempt <= retries:
            time.sleep(1.5 * attempt)

    return {
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "benchmark_sha256": benchmark_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_version": "v1-no-variety-cue",
        "id": cell.item_id,
        "variety": cell.variety,
        "condition": cell.condition,
        "provider": cell.provider,
        "model": cell.model,
        "resolved_model": resolved_model,
        "request_id": request_ids[-1] if request_ids else None,
        "attempt_request_ids": request_ids,
        "text": cell.text,
        "gold_category": cell.gold_category,
        "predicted_category": "",
        "predicted_meaning": "",
        "predicted_reasoning": "",
        "raw_response": last_raw,
        "input_tokens": total_input_tokens,
        "cached_input_tokens": total_cached_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": total_cost if cost_known else None,
        "latency_seconds": total_latency_seconds,
        "attempt_count": retries + 1,
        "parse_error": parse_error,
        "error": str(last_error),
    }


@click.command()
@click.option(
    "--benchmark",
    type=click.Path(exists=True, dir_okay=False),
    default="data/language_variety_audit_benchmark.csv",
    show_default=True,
    help="Path to the benchmark CSV.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="results/raw_classifications.jsonl",
    show_default=True,
    help="Output JSON-lines file.",
)
@click.option(
    "--provider",
    type=click.Choice(PROVIDER_CHOICES, case_sensitive=False),
    default=None,
    help="Restrict to a single provider (otherwise the default matrix is used).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model identifier to pair with --provider.",
)
@click.option(
    "--skip-provider",
    type=click.Choice(PROVIDER_CHOICES, case_sensitive=False),
    multiple=True,
    help="Exclude one or more providers from the default matrix "
         "(ignored when --provider is given).",
)
@click.option(
    "--condition",
    type=click.Choice(["direct", "cot"], case_sensitive=False),
    multiple=True,
    default=("direct", "cot"),
    show_default=True,
)
@click.option(
    "--variety",
    type=click.Choice(VARIETIES, case_sensitive=False),
    multiple=True,
    default=tuple(VARIETIES),
    show_default=True,
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Limit to the first N benchmark rows (useful for smoke tests).",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Maximum number of parallel API calls.",
)
@click.option("--retries", type=click.IntRange(min=0), default=2, show_default=True)
@click.option("--run-id", type=str, default=None, help="Stable identifier stored with every record.")
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option(
    "--budget-usd",
    type=click.FloatRange(min=0),
    default=50.0,
    show_default=True,
    help="Abort before execution if the conservative batch estimate exceeds this amount.",
)
@click.option(
    "--dotenv/--no-dotenv",
    default=True,
    show_default=True,
    help="Load environment variables from a local .env file.",
)
def main(
    benchmark,
    out,
    provider,
    model,
    skip_provider,
    condition,
    variety,
    max_rows,
    workers,
    retries,
    run_id,
    resume,
    budget_usd,
    dotenv,
):
    """Run the classification evaluation and write JSON-lines results."""
    if dotenv:
        load_dotenv()

    if (provider is None) ^ (model is None):
        raise click.UsageError("Provide both --provider and --model, or neither.")
    if provider:
        matrix = [(provider, model)]
    else:
        skip = {p.lower() for p in skip_provider}
        matrix = [(p, m) for (p, m) in DEFAULT_MATRIX if p.lower() not in skip]
        if not matrix:
            raise click.UsageError("--skip-provider excluded every default provider.")

    output_path = Path(out)
    if output_path.exists() and not resume:
        raise click.UsageError(f"Output already exists: {output_path}. Use --resume or a new path.")

    existing = read_latest_records(output_path) if resume else {}
    existing_run_ids = {str(record.get("run_id")) for record in existing.values() if record.get("run_id")}
    if run_id is None:
        run_id = next(iter(existing_run_ids)) if len(existing_run_ids) == 1 else default_run_id()

    df = load_benchmark(benchmark)
    if max_rows is not None:
        df = df.head(max_rows)

    conditions = list(condition)
    varieties = list(variety)
    cells = list(_iter_cells(df, matrix, conditions, varieties))
    pending = [
        cell
        for cell in cells
        if not successful(existing.get(_cell_key(cell), {}), _VALID_LABELS)
    ]
    prior_cost = sum(
        float(record.get("estimated_cost_usd") or 0.0) for record in existing.values()
    )
    conservative_new_cost = _conservative_batch_cost(pending)
    if prior_cost + conservative_new_cost > budget_usd:
        raise click.UsageError(
            f"Conservative cost estimate ${prior_cost + conservative_new_cost:.2f} "
            f"exceeds --budget-usd ${budget_usd:.2f}."
        )

    click.echo(
        f"Run {run_id}: {len(pending)} pending of {len(cells)} cells "
        f"({len(df)} items x {len(varieties)} varieties x "
        f"{len(conditions)} conditions x {len(matrix)} models) with {workers} workers"
    )
    click.echo(
        f"Existing estimated cost: ${prior_cost:.6f}; conservative new ceiling: "
        f"${conservative_new_cost:.6f}"
    )
    if not pending:
        click.echo("All requested cells already have valid results.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_sha256 = file_sha256(benchmark)
    lock = threading.Lock()
    completed = [0]

    with output_path.open("a", encoding="utf-8") as output_handle:
        def _run_and_track(cell: EvalCell) -> None:
            record = _run_cell(cell, run_id, benchmark_sha256, retries=retries)
            status = "ok" if successful(record, _VALID_LABELS) else "ERR"
            with lock:
                append_record(output_handle, record)
                existing[result_key(record)] = record
                completed[0] += 1
                click.echo(
                    f"[{completed[0]:5d}/{len(pending)}] {cell.provider:9s} "
                    f"{cell.model:32s} {cell.variety:15s} {cell.condition:6s} "
                    f"id={cell.item_id:3d} {status}"
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_and_track, cell): cell for cell in pending}
            for future in as_completed(futures):
                future.result()

    compact_records(output_path, existing.values())
    target_keys = {_cell_key(cell) for cell in cells}
    target_records = [existing[key] for key in target_keys if key in existing]
    errors = sum(1 for record in target_records if not successful(record, _VALID_LABELS))
    total_cost = sum(float(record.get("estimated_cost_usd") or 0.0) for record in target_records)

    cost_path = output_path.with_suffix(".costs.csv")
    cost_columns = [
        "run_id",
        "timestamp_utc",
        "provider",
        "model",
        "resolved_model",
        "request_id",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "latency_seconds",
        "attempt_count",
    ]
    pd.DataFrame(target_records).reindex(columns=cost_columns).to_csv(cost_path, index=False)
    click.echo(
        f"\nWrote {len(target_records)} canonical records to {output_path} "
        f"({errors} errors; estimated cost ${total_cost:.6f})."
    )
    click.echo(f"Wrote request cost ledger to {cost_path}.")


if __name__ == "__main__":
    main()
