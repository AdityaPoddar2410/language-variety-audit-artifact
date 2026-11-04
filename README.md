# Language-Variety Moderation Audit: Benchmark and Analysis Artifact

Anonymous artifact for the submission *Matched Meaning, Unequal Moderation: Auditing Linguistic Bias in LLMs*.

> **Content warning.** The benchmark contains offensive, sexual, harassment, self-harm, and violence examples written to test moderation behavior. Provide content warnings to human readers and do not use the items as templates for bypassing moderation systems.

## What is here

| Path | Contents |
|---|---|
| `data/language_variety_audit_benchmark.csv` | 200 hand-authored matched items, each with SAE, Indian English, and Hinglish renditions, one gold moderation label, and an item family (`standard_indianisms` or `genz_slang`) |
| `data/croissant.json` | Croissant 1.0 metadata with responsible-AI fields |
| `data/source-results/` | Preserved historical result tables used to reconstruct the four historical systems |
| `results/study/raw/` | Request-level outputs for the five fresh systems and the two probability-producing snapshots |
| `results/study/validated/accuracy_long.csv` | Canonical table of 10,800 classifications (200 items x 3 conditions x 2 prompts x 9 systems) |
| `results/study/metrics/` | Paired accuracy, fairness (over-flagging, harm misses, category errors, rate ratios, matched discordance, item-family and per-system results), mixed-effects, calibration, and conformal outputs |
| `results/study/manifests/` | Frozen headline results, SHA-256 hashes of the result files, and the LaTeX macros used in the paper |
| `results/study/model_manifests/` | Local model digests and quantization records |
| `results/study/audits/` | Combined audit of five fresh classification runs and two probability runs, current model inventory, and historical provenance |
| `paper/facct2027/generated/`, `paper/facct2027/figures/` | Tables and figures generated from the outputs above |
| `src/language_variety_audit/`, `scripts/`, `tests/` | Evaluation runners, statistics, analysis scripts, and tests |

## Reproduce the analyses

Python 3.10 or later is required.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[analysis,dev]'
PYTHON="$PWD/.venv/bin/python" sh scripts/reproduce.sh
pytest -q
```

`scripts/reproduce.sh` rebuilds the canonical accuracy table from the preserved outputs and regenerates every analysis, the manifest, the macros, the tables, and the figures. It makes no model calls. The pipeline also refreshes validation reports for all seven raw files, the combined input audit, and local-model manifest checksums. CSV outputs can be compared against `CHECKSUMS.sha256`; figure PDFs and `gee_summary.txt` contain timestamps and can differ after regeneration. The artifact contains generated tables and figures, not the manuscript source, so manuscript compilation is skipped.

Run IDs and prompt-version labels are venue-neutral metadata. This normalization does not change request IDs, prompt hashes, model outputs, or predictions.

Re-running models requires provider API keys (see `.env.example`) or a local Ollama installation, and produces new outputs. Hosted model behavior can change over time, so new runs are not expected to match the frozen outputs exactly.

## Measures

- **Over-flagging:** any non-`normal` prediction for a `normal` item (156 items).
- **Harm miss:** a `normal` prediction for a non-`normal` item (44 items).
- **Harmful-category error:** any incorrect prediction for a non-`normal` item.
- **Matched disparity:** target-condition rate minus SAE rate on the same items, averaged equally over systems, with item-clustered bootstrap intervals and sign-flip randomization tests (Holm-corrected within each output family).

## Licenses

Code: MIT (`LICENSE`). Benchmark and metadata: CC BY 4.0 (`LICENSE-DATA`).
