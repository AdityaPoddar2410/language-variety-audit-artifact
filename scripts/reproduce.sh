#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-"$ROOT/../.venv/bin/python"}
TEXBIN="$ROOT/.tools/TinyTeX/bin/universal-darwin"
RESULTS="$ROOT/results/study"
PAPER="$ROOT/paper/facct2027"

if [ ! -x "$PYTHON" ]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

set -- \
  --current-result "$RESULTS/raw/gpt41-evaluations.jsonl" \
  --current-result "$RESULTS/raw/gemini25-evaluations.jsonl"
for result in \
  "$RESULTS/raw/llama31-local-evaluations.jsonl" \
  "$RESULTS/raw/gemma3-local-evaluations.jsonl" \
  "$RESULTS/raw/mistral7b-local-evaluations.jsonl"
do
  if [ -f "$result" ] && [ "$(wc -l < "$result" | tr -d ' ')" -eq 1200 ]; then
    set -- "$@" --current-result "$result"
  fi
done

cd "$ROOT"
"$PYTHON" scripts/audit_study_inputs.py
"$PYTHON" scripts/build_accuracy_dataset.py "$@"
"$PYTHON" scripts/analyze_paired_effects.py
"$PYTHON" scripts/analyze_fairness.py
"$PYTHON" scripts/analyze_mixed_effects.py
"$PYTHON" scripts/analyze_calibration.py \
  --probability-result "$RESULTS/raw/gpt4o-probabilities-v2.jsonl" \
  --probability-result "$RESULTS/raw/gpt41-probabilities-v2.jsonl"
"$PYTHON" scripts/analyze_prompt_agreement.py
"$PYTHON" scripts/analyze_conformal.py
"$PYTHON" scripts/build_results_manifest.py
cp "$RESULTS/manifests/results_macros.tex" "$PAPER/generated/results_macros.tex"
"$PYTHON" scripts/generate_paper_figures.py
"$PYTHON" scripts/generate_paper_tables.py
"$PYTHON" scripts/generate_results_explainer.py

if [ -f "$PAPER/main.tex" ] && [ -x "$TEXBIN/latexmk" ]; then
  cd "$PAPER"
  PATH="$TEXBIN:$PATH" latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
fi
