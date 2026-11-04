from __future__ import annotations

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from language_variety_audit.compute_metrics import _per_cell_correct, main


@pytest.mark.parametrize("has_runner_gold", [False, True])
def test_category_correctness_uses_benchmark_labels(has_runner_gold: bool) -> None:
    benchmark = pd.DataFrame({"id": [1, 2], "category": ["normal", "harassment"]}).set_index("id")
    results = pd.DataFrame({"id": [1, 2], "predicted_category": [" NORMAL ", "normal"]})
    if has_runner_gold:
        results["gold_category"] = ["offensive", "normal"]
    original = results.copy(deep=True)

    scored = _per_cell_correct(results, benchmark, "category")

    assert scored["correct"].tolist() == [True, False]
    assert scored["gold_category"].tolist() == ["normal", "harassment"]
    assert not {"gold_category_x", "gold_category_y"} & set(scored.columns)
    pd.testing.assert_frame_equal(results, original)


def test_judge_correctness_preserves_runner_metadata() -> None:
    results = pd.DataFrame({"judge_match": [True, False], "gold_category": ["normal", "normal"]})
    scored = _per_cell_correct(results, pd.DataFrame(), "judge")
    assert scored["correct"].tolist() == [True, False]
    assert scored["gold_category"].tolist() == ["normal", "normal"]
    assert "correct" not in results


def test_metrics_cli_accepts_runner_gold_column(tmp_path) -> None:
    benchmark_path = tmp_path / "benchmark.csv"
    results_path = tmp_path / "results.jsonl"
    out_dir = tmp_path / "metrics"
    benchmark = pd.DataFrame([
        {
            "id": item_id,
            "word_or_phrase": "sample",
            "hinglish_example": "sample text",
            "indian_english_translation": "sample text",
            "indian_meaning": "sample meaning",
            "category": category,
            "sae_example": "sample text",
            "dataset_type": "standard_indianisms",
        }
        for item_id, category in [(1, "normal"), (2, "harassment")]
    ])
    benchmark.to_csv(benchmark_path, index=False)
    records = [
        {
            "id": item_id,
            "variety": variety,
            "condition": "direct",
            "provider": "test",
            "model": "test-model",
            "gold_category": "offensive",
            "predicted_category": "normal" if item_id == 1 or variety == "hinglish" else "harassment",
        }
        for item_id in (1, 2)
        for variety in ("sae", "indian_english", "hinglish")
    ]
    results_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = CliRunner().invoke(main, [
        "--benchmark", str(benchmark_path),
        "--results", str(results_path),
        "--out-dir", str(out_dir),
    ])

    assert result.exit_code == 0, result.output
    accuracy = pd.read_csv(out_dir / "accuracy.csv").set_index("variety")
    assert accuracy["accuracy"].to_dict() == {"hinglish": 0.5, "indian_english": 1.0, "sae": 1.0}
    gaps = pd.read_csv(out_dir / "linguistic_gap.csv").set_index("variety")
    assert gaps.loc["hinglish", "delta_v_pp"] == pytest.approx(50.0)
    comparisons = pd.read_csv(out_dir / "mcnemar.csv")
    assert len(comparisons) == 2
    assert set(comparisons["n_paired"]) == {2}
