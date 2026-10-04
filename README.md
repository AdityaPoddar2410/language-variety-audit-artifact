# Language-Variety Classification Audit

This artifact contains the benchmark, model outputs, analysis code, and derived results for a matched-item study of classification behavior across Standard American English (SAE), Indian English, and Hinglish.

Each of the 200 benchmark items has three renditions that share one intended meaning and one moderation label. The paired structure permits within-item comparisons while holding the underlying intent fixed.

## Contents

- `data/`: benchmark CSV and Croissant metadata
- `src/beyond_weird/`: statistical-analysis modules
- `scripts/`: command-line analysis and reproduction scripts
- `results/raw/`: fresh evaluation and constrained-probability outputs
- `results/historical/`: preserved historical classification outputs used in the analysis
- `results/validated/`: canonical long-form analysis tables
- `results/metrics/`: paired, calibration, conformal, and mixed-effects results
- `results/manifests/`: frozen summary of the reported numerical results
- `results/model_manifests/`: local-model identities and quantization details
- `figures/`: figures generated from the metric tables
- `tests/`: unit tests for the statistical and evaluation utilities

No credentials, environment files, model weights, participant identifiers, or author-identifying metadata are included.

## Benchmark

The dataset contains 200 matched triplets and six labels: `normal`, `offensive`, `sexually_suggestive`, `harassment`, `violence`, and `self_harm`. The class distribution is imbalanced: 156 items are normal and 44 are spread across the five sensitive categories. The benchmark is intended for controlled diagnostic evaluation, not model training or population-level prevalence estimation.

The dataset was written specifically for this study rather than collected from social media or other user-generated sources. Thirty-two adult Indian English speakers who regularly use Hinglish reviewed the triplets for naturalness, semantic equivalence, and label agreement. Further provenance and limitation information is recorded in `data/croissant.json`.

## Reproduce the analysis

Python 3.10 or later is required. From the artifact root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
sh scripts/reproduce_analysis.sh
pytest -q
```

The complete statistical run uses 10,000 bootstrap draws and 100,000 randomization draws and can take several minutes. For a quick code check, run the tests only.

The reproduction script rebuilds the canonical accuracy table, paired estimates, reasoning-prompt contrasts, mixed-effects and GEE analyses, calibration metrics, repeated-split uncertainty-set diagnostics, figures, and LaTeX tables. It uses only the preserved outputs in this artifact and does not issue API requests.

## Raw-result provenance

Fresh JSONL records contain benchmark and prompt hashes, requested and resolved model identifiers, timestamps, usage counts, request identifiers when available, retries, latency, and parsing status. Historical CSVs predate that logging format and do not contain request-level prompt hashes. They are retained separately and identified as the historical cohort in the derived tables.

The constrained-probability files represent a separate coded classifier that restricts the response to six single-token alternatives. They should not be interpreted as unconstrained semantic confidence from the ordinary output interface.

## Scope and limitations

- Inference is conditional on the evaluated systems and benchmark items.
- Rare labels have insufficient sample sizes for precise category-specific conclusions.
- The utterances are short and do not represent dialogue or long-form text.
- Hinglish is Hindi-English code-mixing, not an English dialect.
- The probability and uncertainty-set analyses cover two model snapshots.
- The historical cohort has weaker request-level provenance than the fresh cohort.

## Licensing and citation

Code is released under the MIT License. The benchmark and Croissant metadata are released under CC BY 4.0. During anonymous review, cite the dataset as:

> Anonymous authors (2026). *Beyond WEIRD English: A Matched-Guise Benchmark for Evaluating LLM Comprehension Across Standard American English, Indian English, and Hinglish*. Version 1.0.0.
