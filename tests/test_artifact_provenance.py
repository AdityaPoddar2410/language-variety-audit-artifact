from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_auditor():
    spec = importlib.util.spec_from_file_location("study_input_audit", ROOT / "scripts/audit_study_inputs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_record() -> dict:
    return {
        "id": 1,
        "variety": "sae",
        "condition": "direct",
        "provider": "test",
        "model": "test-model",
        "predicted_category": "normal",
        "run_id": "test-run",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "benchmark_sha256": "benchmark-hash",
        "prompt_sha256": "prompt-hash",
        "prompt_version": "v1-no-variety-cue",
        "input_tokens": 1,
        "output_tokens": 1,
        "latency_seconds": 0.0,
        "attempt_count": 1,
        "raw_response": '{"category":"normal"}',
        "error": None,
        "parse_error": False,
    }


def test_default_audit_covers_all_preserved_current_runs() -> None:
    specs = load_auditor().CURRENT_RESULT_SPECS
    assert len(specs) == 7
    assert sum(rows == 1200 for _, rows in specs.values()) == 5
    assert sum(rows == 600 for _, rows in specs.values()) == 2
    assert all("probabilities-v2" in name for name, (_, rows) in specs.items() if rows == 600)


def test_audit_cli_keeps_repeated_result_files_and_portable_paths(tmp_path) -> None:
    benchmark = ROOT / "data/language_variety_audit_benchmark.csv"
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for path in paths:
        path.write_text(json.dumps(result_record()) + "\n")
    command = [sys.executable, str(ROOT / "scripts/audit_study_inputs.py"), "--benchmark", str(benchmark)]
    for path in paths:
        command.extend(["--current-results", str(path)])
    command.extend(["--legacy-root", "unused", "--out-dir", "audit"])
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "audit/input_audit.json").read_text())
    assert [entry["path"] for entry in report["current_results"]] == [path.name for path in paths]
    text = (tmp_path / "audit/input_audit.md").read_text()
    assert all(path.name in text for path in paths)
    assert str(tmp_path) not in json.dumps(report)


def test_validation_hash_matches_packaged_file_after_metadata_change(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    record = result_record()
    command = [
        sys.executable, str(ROOT / "scripts/validate_evaluation_output.py"),
        str(path), "--expected-rows", "1", "--expected-model", "test-model",
    ]
    hashes = []
    for run_id in ["original-run", "neutral-run"]:
        record["run_id"] = run_id
        path.write_text(json.dumps(record) + "\n")
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
        report = json.loads(path.with_suffix(".validation.json").read_text())
        assert report["path"] == path.name
        assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert report["run_ids"] == [run_id]
        hashes.append(report["sha256"])
    assert hashes[0] != hashes[1]
