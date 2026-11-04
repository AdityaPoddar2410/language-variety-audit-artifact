# Study Input Audit

## Benchmark

- Rows: 200
- SHA-256: `49a924e33a1d3bcaa36a0be1f9c284678907a68c7a433b6f8f41606b6e0a7c6f`
- Missing required columns: none
- Duplicate IDs: none
- Missing IDs: none
- Unknown labels: none
- Label counts: {'harassment': 15, 'normal': 156, 'offensive': 16, 'self_harm': 4, 'sexually_suggestive': 7, 'violence': 2}
- Dataset types: {'genz_slang': 100, 'standard_indianisms': 100}
- PII-pattern candidates requiring human review: 0

### Word counts

| Variety | Mean | Median | Min | Max | 6–10 words |
|---|---:|---:|---:|---:|---:|
| sae | 7.475 | 7.0 | 2 | 14 | 145 |
| indian_english | 6.355 | 6.0 | 2 | 11 | 133 |
| hinglish | 6.72 | 7.0 | 4 | 12 | 163 |

## Current JSONL results

### `results/study/raw/gpt41-evaluations.jsonl`

- SHA-256: `74a9f8cc0d555c73e65a6812306fb398f66f8ef4e2924bb11acc1624d28663b6`
- Rows: 1200
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| openai | gpt-4.1-mini-2025-04-14 | cot | hinglish | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | cot | indian_english | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | cot | sae | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | direct | hinglish | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | direct | indian_english | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | direct | sae | 200 | 200 | 0 | 0 |

### `results/study/raw/gemini25-evaluations.jsonl`

- SHA-256: `b11a697d1013ead7fdfb4416e15aad0567039f095227a2c94f528e2bfe31c768`
- Rows: 1200
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| gemini | gemini-2.5-flash-lite | cot | hinglish | 200 | 200 | 0 | 0 |
| gemini | gemini-2.5-flash-lite | cot | indian_english | 200 | 200 | 0 | 0 |
| gemini | gemini-2.5-flash-lite | cot | sae | 200 | 200 | 0 | 0 |
| gemini | gemini-2.5-flash-lite | direct | hinglish | 200 | 200 | 0 | 0 |
| gemini | gemini-2.5-flash-lite | direct | indian_english | 200 | 200 | 0 | 0 |
| gemini | gemini-2.5-flash-lite | direct | sae | 200 | 200 | 0 | 0 |

### `results/study/raw/gemma3-local-evaluations.jsonl`

- SHA-256: `843371dbcd4f692559329f75113f2735411da28821e3b215dbe3e30f41469b0b`
- Rows: 1200
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| ollama | gemma3:12b | cot | hinglish | 200 | 200 | 0 | 0 |
| ollama | gemma3:12b | cot | indian_english | 200 | 200 | 0 | 0 |
| ollama | gemma3:12b | cot | sae | 200 | 200 | 0 | 0 |
| ollama | gemma3:12b | direct | hinglish | 200 | 200 | 0 | 0 |
| ollama | gemma3:12b | direct | indian_english | 200 | 200 | 0 | 0 |
| ollama | gemma3:12b | direct | sae | 200 | 200 | 0 | 0 |

### `results/study/raw/llama31-local-evaluations.jsonl`

- SHA-256: `ecc0ef09054293b40d8f8f54444045e054c1e8339a42927478f6a42088bff8e7`
- Rows: 1200
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| ollama | llama3.1:8b | cot | hinglish | 200 | 200 | 0 | 0 |
| ollama | llama3.1:8b | cot | indian_english | 200 | 200 | 0 | 0 |
| ollama | llama3.1:8b | cot | sae | 200 | 200 | 0 | 0 |
| ollama | llama3.1:8b | direct | hinglish | 200 | 200 | 0 | 0 |
| ollama | llama3.1:8b | direct | indian_english | 200 | 200 | 0 | 0 |
| ollama | llama3.1:8b | direct | sae | 200 | 200 | 0 | 0 |

### `results/study/raw/mistral7b-local-evaluations.jsonl`

- SHA-256: `b300faa1f5c5ae145cebd1210265682bc16869120bcb32b4b4a7c5282c2e58c8`
- Rows: 1200
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| ollama | mistral:7b | cot | hinglish | 200 | 200 | 0 | 0 |
| ollama | mistral:7b | cot | indian_english | 200 | 200 | 0 | 0 |
| ollama | mistral:7b | cot | sae | 200 | 200 | 0 | 0 |
| ollama | mistral:7b | direct | hinglish | 200 | 200 | 0 | 0 |
| ollama | mistral:7b | direct | indian_english | 200 | 200 | 0 | 0 |
| ollama | mistral:7b | direct | sae | 200 | 200 | 0 | 0 |

### `results/study/raw/gpt41-probabilities-v2.jsonl`

- SHA-256: `aded8a35315b49334722dc495a57d8686c649cf662aaaf9b35cbaef0dc2eb1be`
- Rows: 600
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| openai | gpt-4.1-mini-2025-04-14 | direct | hinglish | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | direct | indian_english | 200 | 200 | 0 | 0 |
| openai | gpt-4.1-mini-2025-04-14 | direct | sae | 200 | 200 | 0 | 0 |

### `results/study/raw/gpt4o-probabilities-v2.jsonl`

- SHA-256: `f4fce53fdb57d6586690a628fc5d0482a711769cf00ecee34641d9a9bcdf1957`
- Rows: 600
- Invalid JSON lines: none
- Duplicate evaluation keys: 0
- Text mismatches against benchmark: 0

| Provider | Model | Condition | Variety | Rows | Valid | API errors | Parse errors |
|---|---|---|---|---:|---:|---:|---:|
| openai | gpt-4o-mini-2024-07-18 | direct | hinglish | 200 | 200 | 0 | 0 |
| openai | gpt-4o-mini-2024-07-18 | direct | indian_english | 200 | 200 | 0 | 0 |
| openai | gpt-4o-mini-2024-07-18 | direct | sae | 200 | 200 | 0 | 0 |

## Legacy files

| File | Rows | ID rows | Duplicate IDs | Missing IDs | Prediction columns | Read error |
|---|---:|---:|---:|---:|---:|---|
| `data/source-results/evaluation_results_with_sae_20260113_022333.csv` | 200 | 200 | 0 | 0 | 18 |  |
| `data/source-results/meaning_match_results_20260113_195052.csv` | 200 | 200 | 0 | 0 | 19 |  |
