#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

cd "$ROOT"

"$PYTHON" scripts/build_accuracy.py \
  --current-result results/raw/gpt41-evaluations.jsonl \
  --current-result results/raw/gemini25-evaluations.jsonl \
  --current-result results/raw/llama31-local-evaluations.jsonl \
  --current-result results/raw/gemma3-local-evaluations.jsonl \
  --current-result results/raw/mistral7b-local-evaluations.jsonl

"$PYTHON" scripts/analyze_paired_effects.py
"$PYTHON" scripts/analyze_mixed_effects.py
"$PYTHON" scripts/analyze_calibration.py \
  --probability-result results/raw/gpt4o-probabilities-v2.jsonl \
  --probability-result results/raw/gpt41-probabilities-v2.jsonl
"$PYTHON" scripts/analyze_prompt_agreement.py
"$PYTHON" scripts/analyze_conformal.py
"$PYTHON" scripts/generate_figures.py
"$PYTHON" scripts/generate_tables.py

printf '%s\n' 'Analysis reproduced successfully.'
