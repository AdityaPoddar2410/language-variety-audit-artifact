"""Run six-class OpenAI evaluations with token-level class probabilities."""
from __future__ import annotations

import json
import math
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
from .llm_utils import chat_completion_with_metadata
from .pricing import estimate_cost_usd
from .prompts import CATEGORY_CODE_TO_LABEL, probability_classification_prompt
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

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(CATEGORY_CODE_TO_LABEL)},
    },
    "required": ["label"],
    "additionalProperties": False,
}
_VALID_LABELS = set(CATEGORY_CODE_TO_LABEL.values())


@dataclass(frozen=True)
class ProbabilityCell:
    item_id: int
    variety: str
    text: str
    gold_category: str
    provider: str
    model: str
    condition: str = "direct"


def _iter_cells(
    frame: pd.DataFrame,
    model: str,
    varieties: list[str],
) -> Iterable[ProbabilityCell]:
    for _, row in frame.iterrows():
        for variety in varieties:
            text = row.get(VARIETY_TO_COLUMN[variety])
            if text is None or (isinstance(text, float) and pd.isna(text)):
                continue
            yield ProbabilityCell(
                item_id=int(row["id"]),
                variety=variety,
                text=str(text),
                gold_category=str(row["category"]).strip().lower(),
                provider="openai",
                model=model,
            )


def _cell_key(cell: ProbabilityCell) -> tuple[Any, ...]:
    return (cell.item_id, cell.variety, cell.condition, cell.provider, cell.model)


def _extract_label_distribution(
    token_logprobs: list[dict[str, Any]] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    if not token_logprobs:
        raise ValueError("OpenAI response did not include token log probabilities")

    label_token = next(
        (token for token in token_logprobs if token.get("token") in CATEGORY_CODE_TO_LABEL),
        None,
    )
    if label_token is None:
        raise ValueError("Could not locate the class-label token in the response")

    code_logprobs = {
        str(candidate.get("token")): float(candidate.get("logprob"))
        for candidate in label_token.get("top_logprobs", [])
        if candidate.get("token") in CATEGORY_CODE_TO_LABEL
    }
    missing = set(CATEGORY_CODE_TO_LABEL) - set(code_logprobs)
    if missing:
        raise ValueError(f"Missing label alternatives in top log probabilities: {sorted(missing)}")

    maximum = max(code_logprobs.values())
    weights = {code: math.exp(value - maximum) for code, value in code_logprobs.items()}
    denominator = sum(weights.values())
    probabilities = {
        CATEGORY_CODE_TO_LABEL[code]: weights[code] / denominator
        for code in CATEGORY_CODE_TO_LABEL
    }
    label_logprobs = {
        CATEGORY_CODE_TO_LABEL[code]: code_logprobs[code]
        for code in CATEGORY_CODE_TO_LABEL
    }
    return probabilities, label_logprobs


def _run_cell(
    cell: ProbabilityCell,
    run_id: str,
    benchmark_sha256: str,
    retries: int = 2,
) -> dict[str, Any]:
    system, user = probability_classification_prompt(cell.text)
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
                provider="openai",
                model=cell.model,
                json_output=True,
                max_tokens=10,
                json_schema=_SCHEMA,
                include_logprobs=True,
                top_logprobs=10,
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
            parsed = json.loads(last_raw)
            if not isinstance(parsed, dict):
                raise TypeError("Response JSON is not an object")
            code = str(parsed.get("label", "")).strip()
            if code not in CATEGORY_CODE_TO_LABEL:
                raise ValueError(f"Invalid class code: {code!r}")
            probabilities, label_logprobs = _extract_label_distribution(
                completion.token_logprobs
            )
            prediction = CATEGORY_CODE_TO_LABEL[code]
            return {
                "run_id": run_id,
                "timestamp_utc": utc_timestamp(),
                "benchmark_sha256": benchmark_sha256,
                "prompt_sha256": prompt_sha256,
                "prompt_version": "probabilities-v2",
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
                "predicted_code": code,
                "class_probabilities": probabilities,
                "label_logprobs": label_logprobs,
                "raw_token_logprobs": completion.token_logprobs,
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
        "prompt_version": "probabilities-v2",
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
        "predicted_code": "",
        "class_probabilities": {},
        "label_logprobs": {},
        "raw_token_logprobs": None,
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


def _conservative_batch_cost(cells: Iterable[ProbabilityCell]) -> float:
    total = 0.0
    for cell in cells:
        system, user = probability_classification_prompt(cell.text)
        estimated_input_tokens = max((len(system) + len(user) + 3) // 4, 1)
        estimate = estimate_cost_usd(
            cell.model,
            input_tokens=estimated_input_tokens,
            output_tokens=10,
        )
        if estimate is not None:
            total += estimate
    return total


@click.command()
@click.option(
    "--benchmark",
    type=click.Path(exists=True, dir_okay=False),
    default="data/language_variety_audit_benchmark.csv",
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="results/study/raw/probabilities.jsonl",
    show_default=True,
)
@click.option(
    "--model",
    type=str,
    default="gpt-4o-mini-2024-07-18",
    show_default=True,
)
@click.option(
    "--variety",
    type=click.Choice(VARIETIES, case_sensitive=False),
    multiple=True,
    default=tuple(VARIETIES),
    show_default=True,
)
@click.option("--max-rows", type=click.IntRange(min=1), default=None)
@click.option("--workers", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--retries", type=click.IntRange(min=0), default=2, show_default=True)
@click.option("--run-id", type=str, default=None)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--budget-usd", type=click.FloatRange(min=0), default=50.0, show_default=True)
@click.option("--dotenv/--no-dotenv", default=True, show_default=True)
def main(
    benchmark,
    out,
    model,
    variety,
    max_rows,
    workers,
    retries,
    run_id,
    resume,
    budget_usd,
    dotenv,
):
    """Write resumable six-class probability records for one OpenAI model."""
    if dotenv:
        load_dotenv()

    output_path = Path(out)
    if output_path.exists() and not resume:
        raise click.UsageError(f"Output already exists: {output_path}. Use --resume or a new path.")
    existing = read_latest_records(output_path) if resume else {}
    existing_run_ids = {str(record.get("run_id")) for record in existing.values() if record.get("run_id")}
    if run_id is None:
        run_id = next(iter(existing_run_ids)) if len(existing_run_ids) == 1 else default_run_id()

    frame = load_benchmark(benchmark)
    if max_rows is not None:
        frame = frame.head(max_rows)
    cells = list(_iter_cells(frame, model=model, varieties=list(variety)))
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
        f"Run {run_id}: {len(pending)} pending of {len(cells)} probability cells "
        f"with {workers} workers"
    )
    click.echo(
        f"Existing estimated cost: ${prior_cost:.6f}; conservative new ceiling: "
        f"${conservative_new_cost:.6f}"
    )
    if not pending:
        click.echo("All requested cells already have valid probability records.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_sha256 = file_sha256(benchmark)
    lock = threading.Lock()
    completed = [0]

    with output_path.open("a", encoding="utf-8") as output_handle:
        def _run_and_track(cell: ProbabilityCell) -> None:
            record = _run_cell(cell, run_id, benchmark_sha256, retries=retries)
            status = "ok" if successful(record, _VALID_LABELS) else "ERR"
            with lock:
                append_record(output_handle, record)
                existing[result_key(record)] = record
                completed[0] += 1
                click.echo(
                    f"[{completed[0]:4d}/{len(pending)}] {cell.model:32s} "
                    f"{cell.variety:15s} id={cell.item_id:3d} {status}"
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
        f"\nWrote {len(target_records)} canonical probability records to {output_path} "
        f"({errors} errors; estimated cost ${total_cost:.6f})."
    )
    click.echo(f"Wrote request cost ledger to {cost_path}.")


if __name__ == "__main__":
    main()
