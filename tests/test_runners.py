from __future__ import annotations

import json
import math

import pytest

from language_variety_audit import run_evaluations
from language_variety_audit.llm_utils import CompletionResult
from language_variety_audit.pricing import estimate_cost_usd
from language_variety_audit.prompts import (
    CATEGORY_CODE_TO_LABEL,
    cot_classification_prompt,
    probability_classification_prompt,
)
from language_variety_audit.run_probabilities import _extract_label_distribution
from language_variety_audit.run_records import (
    append_record,
    compact_records,
    read_latest_records,
    result_key,
)
from language_variety_audit.study_data import load_legacy_accuracy


def test_legacy_accuracy_sources_reproduce_paper_cells() -> None:
    frame = load_legacy_accuracy()
    assert len(frame) == 4 * 200 * 3 * 2
    grouped = frame.groupby(["model", "condition", "variety"])["correct"].mean()
    assert grouped.loc[("GPT-4o-mini", "direct", "sae")] == pytest.approx(0.845)
    assert grouped.loc[("Claude Haiku 3.5", "direct", "hinglish")] == pytest.approx(0.735)
    assert grouped.loc[("Llama 3.1 8B", "cot", "indian_english")] == pytest.approx(0.775)
    assert grouped.loc[("Gemma 3 12B", "direct", "hinglish")] == pytest.approx(0.750)


def test_original_reasoning_prompt_is_preserved_without_variety_cue() -> None:
    system, _ = cot_classification_prompt("sample text")
    for step in (
        "1. Literal reading",
        "2. Contextual reading",
        "3. Pragmatic context",
        "4. Harm signals",
        "5. Category decision",
    ):
        assert step in system
    assert "analyzing hinglish" not in system.lower()
    assert "analyzing indian english" not in system.lower()


def test_probability_prompt_uses_single_token_codes() -> None:
    system, user = probability_classification_prompt("sample text")
    assert set(CATEGORY_CODE_TO_LABEL) == set("ABCDEF")
    assert all(f"{code} = {label}" in system for code, label in CATEGORY_CODE_TO_LABEL.items())
    assert "sample text" in user


def test_extract_label_distribution_returns_all_classes() -> None:
    alternatives = [
        {"token": code, "logprob": -float(index)}
        for index, code in enumerate(CATEGORY_CODE_TO_LABEL)
    ]
    token_logprobs = [
        {"token": '{"label":"', "logprob": 0.0, "top_logprobs": []},
        {"token": "A", "logprob": 0.0, "top_logprobs": alternatives},
    ]
    probabilities, logprobs = _extract_label_distribution(token_logprobs)
    assert set(probabilities) == set(CATEGORY_CODE_TO_LABEL.values())
    assert set(logprobs) == set(CATEGORY_CODE_TO_LABEL.values())
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert max(probabilities, key=probabilities.get) == "normal"


def test_extract_label_distribution_rejects_missing_class() -> None:
    alternatives = [
        {"token": code, "logprob": -1.0}
        for code in list(CATEGORY_CODE_TO_LABEL)[:-1]
    ]
    with pytest.raises(ValueError, match="Missing label alternatives"):
        _extract_label_distribution(
            [{"token": "A", "logprob": 0.0, "top_logprobs": alternatives}]
        )


def test_pricing_accounts_for_cached_tokens() -> None:
    cost = estimate_cost_usd(
        "gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        cached_input_tokens=250_000,
        output_tokens=100_000,
    )
    assert cost == pytest.approx(0.485)


def test_record_compaction_keeps_latest_attempt(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    base = {
        "id": 1,
        "variety": "sae",
        "condition": "direct",
        "provider": "openai",
        "model": "gpt-4.1-mini-2025-04-14",
    }
    with path.open("a", encoding="utf-8") as handle:
        append_record(handle, {**base, "error": "temporary", "predicted_category": ""})
        append_record(handle, {**base, "error": None, "predicted_category": "normal"})
    latest = read_latest_records(path)
    assert latest[result_key(base)]["predicted_category"] == "normal"
    compact_records(path, latest.values())
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_evaluation_record_contains_usage_and_provenance(monkeypatch) -> None:
    completion = CompletionResult(
        text=json.dumps({"category": "normal"}),
        provider="openai",
        model="gpt-4.1-mini-2025-04-14",
        request_id="request-1",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=5,
        latency_seconds=0.25,
    )
    monkeypatch.setattr(
        run_evaluations,
        "chat_completion_with_metadata",
        lambda **_: completion,
    )
    cell = run_evaluations.EvalCell(
        item_id=1,
        variety="sae",
        text="After graduating, I went to college.",
        gold_category="normal",
        condition="direct",
        provider="openai",
        model="gpt-4.1-mini-2025-04-14",
    )
    record = run_evaluations._run_cell(
        cell,
        run_id="test-run",
        benchmark_sha256="benchmark-hash",
        retries=0,
    )
    assert record["predicted_category"] == "normal"
    assert record["request_id"] == "request-1"
    assert record["input_tokens"] == 100
    assert record["estimated_cost_usd"] == pytest.approx(0.000048)
    assert record["benchmark_sha256"] == "benchmark-hash"
    assert record["prompt_sha256"]


def test_retry_usage_is_accumulated(monkeypatch) -> None:
    responses = iter(
        [
            CompletionResult(
                text="not json",
                provider="openai",
                model="gpt-4.1-mini-2025-04-14",
                request_id="request-1",
                input_tokens=10,
                cached_input_tokens=0,
                output_tokens=2,
                latency_seconds=0.1,
            ),
            CompletionResult(
                text=json.dumps({"category": "normal"}),
                provider="openai",
                model="gpt-4.1-mini-2025-04-14",
                request_id="request-2",
                input_tokens=20,
                cached_input_tokens=0,
                output_tokens=3,
                latency_seconds=0.2,
            ),
        ]
    )
    monkeypatch.setattr(
        run_evaluations,
        "chat_completion_with_metadata",
        lambda **_: next(responses),
    )
    monkeypatch.setattr(run_evaluations.time, "sleep", lambda _: None)
    cell = run_evaluations.EvalCell(
        item_id=1,
        variety="sae",
        text="Routine text",
        gold_category="normal",
        condition="direct",
        provider="openai",
        model="gpt-4.1-mini-2025-04-14",
    )
    record = run_evaluations._run_cell(
        cell,
        run_id="test-run",
        benchmark_sha256="benchmark-hash",
        retries=1,
    )
    assert record["attempt_count"] == 2
    assert record["attempt_request_ids"] == ["request-1", "request-2"]
    assert record["input_tokens"] == 30
    assert record["output_tokens"] == 5
    assert record["latency_seconds"] == pytest.approx(0.3)
    assert record["estimated_cost_usd"] == pytest.approx(0.00002)
