from __future__ import annotations

import pytest

from beyond_weird.analysis_data import load_historical_accuracy
from beyond_weird.prompts import (
    CATEGORY_CODE_TO_LABEL,
    cot_classification_prompt,
    probability_classification_prompt,
)
from beyond_weird.run_records import (
    append_record,
    compact_records,
    read_latest_records,
    result_key,
)


def test_historical_accuracy_sources_reproduce_reported_cells() -> None:
    frame = load_historical_accuracy()
    assert len(frame) == 4 * 200 * 3 * 2
    grouped = frame.groupby(["model", "condition", "variety"])["correct"].mean()
    assert grouped.loc[("GPT-4o-mini", "direct", "sae")] == pytest.approx(0.845)
    assert grouped.loc[("Claude Haiku 3.5", "direct", "hinglish")] == pytest.approx(0.735)
    assert grouped.loc[("Llama 3.1 8B", "cot", "indian_english")] == pytest.approx(0.775)
    assert grouped.loc[("Gemma 3 12B", "direct", "hinglish")] == pytest.approx(0.750)


def test_reasoning_prompt_has_no_variety_cue() -> None:
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
    assert all(
        f"{code} = {label}" in system
        for code, label in CATEGORY_CODE_TO_LABEL.items()
    )
    assert "sample text" in user


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
