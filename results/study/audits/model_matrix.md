# Evaluated Model Configurations

The frozen study contains seven model names and nine configurations: five fresh runs and four historical configurations. Gemma and Llama occur in both cohorts. Each configuration contributes 1,200 classifications (200 items × three language conditions × two prompts).

| Cohort | Provider | Model | Recorded identifier |
|---|---|---|---|
| Fresh | OpenAI | GPT-4.1 Mini | `gpt-4.1-mini-2025-04-14` |
| Fresh | Google Vertex AI | Gemini 2.5 Flash-Lite | `gemini-2.5-flash-lite` |
| Fresh | Ollama | Gemma 3 12B | `gemma3:12b` |
| Fresh | Ollama | Llama 3.1 8B | `llama3.1:8b` |
| Fresh | Ollama | Mistral 7B Instruct | `mistral:7b` |
| Historical | Anthropic | Claude Haiku 3.5 | `claude-3-5-haiku-20241022` |
| Historical | OpenAI | GPT-4o-mini | `gpt-4o-mini-2024-07-18` |
| Historical | Ollama | Gemma 3 12B | `gemma3:12b` |
| Historical | Ollama | Llama 3.1 8B | `llama3.1:8b` |

Two additional 600-record probability files cover GPT-4.1 Mini and GPT-4o-mini under a separate coded-label classifier. They are not extra configurations in the nine-configuration accuracy panel.

Fresh records retain request metadata and prompt and benchmark hashes. Historical identifiers rely on the documented mapping in [historical provenance](./legacy_provenance.md), not request-level metadata. [Local-model manifests](../model_manifests/) retain model digests and quantization. Headline estimates weight configurations equally; equal-model and cohort sensitivities are in [fairness metrics](../metrics/fairness/).

Recorded API behavior describes the preserved runs, not current endpoint availability or authentication status.
