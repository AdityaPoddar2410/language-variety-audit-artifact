# Historical Result Provenance

## Scope

This note documents where the four historical systems in the analysis come from. It does not certify provider-side execution metadata that is absent from the historical files.

## Source files

The historical predictions are read from two CSV files in `data/source-results/`:

1. `evaluation_results_with_sae_20260113_022333.csv`
   - `*_openai` columns give the GPT-4o-mini predictions.
   - `*_anthropic` columns give the Claude Haiku 3.5 predictions.
   - `*_ollama` columns give the Llama 3.1 8B predictions.
2. `meaning_match_results_20260113_195052.csv`
   - `*_ollama` columns give the Gemma 3 12B predictions.

Each file contains the 200 benchmark IDs and one prediction column per variety and prompt condition.

## Model identity

The columns store provider names (`openai`, `anthropic`, `ollama`), not model IDs. The mapping to `gpt-4o-mini-2024-07-18`, `claude-3-5-haiku-20241022`, `llama3.1:8b`, and `gemma3:12b` is recorded in `src/language_variety_audit/study_data.py`. The files do not record request IDs, per-request timestamps, token usage, or prompt hashes, so their provenance is weaker than that of the fresh runs.

## Prompt regime

An intermediate version of the prompt code included variety labels. The authors' execution record is the evidence that the preserved historical files were produced with the no-cue prompt, so the analysis records them as `prompt_cue=none`. The fresh runs keep prompt hashes directly.

## Repetitions

The files contain one prediction per item, variety, and prompt condition, with no repetition index. A later GPT-4o-mini rerun did not reproduce every historical prediction, so the historical results should be read as a single preserved run.
