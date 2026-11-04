# Study Results Explained

**Paper:** *Matched Meaning, Unequal Moderation: Auditing Linguistic Bias in LLMs*  
**Generated from the frozen result manifest and analysis tables**  
**Benchmark:** 200 matched items × 3 language conditions × 2 prompt conditions

## Executive summary

- The final analysis contains **7 distinct LLMs**, **9 model–prompt systems**, and **10,800 complete classification observations**.
- Under direct prompting, accuracy is **6.5 percentage points lower on Indian English** and **9.1 points lower on Hinglish** than on matched SAE items.
- Normal Indian English and Hinglish content is over-flagged **10.2** and **11.7 percentage points more often** than matched SAE content under direct prompting.
- Explicit reasoning raises Hinglish accuracy by **6.2 points**, raises Indian English by **3.6 points**, and changes SAE by only **-0.2 points**.
- Reasoning reduces Indian English and Hinglish over-flagging by **7.2** and **9.8 points**, but increases harmful-content misses by **11.1** and **14.9 points**.
- A stratified manual audit covered 420 reasoning responses across models and varieties; one reviewer per response required both semantic interpretation and final category to be acceptable, and more than 90% were accepted.
- In a separate two-model case study, grammar-constrained label-token confidence barely falls as non-SAE accuracy declines. This coded classifier therefore becomes more overconfident on non-SAE input.
- In a descriptive repeated-split uncertainty-set diagnostic, pooled 90%-nominal thresholds cover non-SAE input at only **87.0–88.9%** on average. Variety-specific thresholds raise average coverage to **90.1–90.4%**, but produce larger sets. This is not claimed as a finite-sample pooled conformal guarantee.

## 1. Experimental design

Every benchmark item has three semantically matched forms:

- **SAE:** Standard American English, used as the reference condition.
- **Indian English:** an English variety used widely in India.
- **Hinglish:** Hindi–English code-mixing; it is not treated as an English dialect.

Each form shares the same human-authored gold label from six moderation classes. Each model is evaluated with a direct classification prompt and an explicit five-step reasoning prompt. All prompts omit the variety label. Four systems come from historical CSVs without request-level prompt hashes; five are fresh runs with complete request metadata, so cohort sensitivity is reported.

### Evaluated systems

| Model | Provider | Cohort | Prompt regime | System ID |
|---|---|---|---|---|
| GPT-4.1 Mini | OpenAI | Fresh | No cue | `gpt-4.1-mini-2025-04-14::fresh-no-cue` |
| Gemini 2.5 Flash-Lite | Google Vertex AI | Fresh | No cue | `gemini-2.5-flash-lite::fresh-no-cue` |
| Gemma 3 12B | Ollama | Fresh | No cue | `gemma3:12b::fresh-no-cue` |
| Llama 3.1 8B | Ollama | Fresh | No cue | `llama3.1:8b::fresh-no-cue` |
| Mistral 7B Instruct | Ollama | Fresh | No cue | `mistral:7b::fresh-no-cue` |
| Claude Haiku 3.5 | Anthropic | Historical | No cue | `claude-3-5-haiku-20241022::legacy-no-cue` |
| GPT-4o-mini | OpenAI | Historical | No cue | `gpt-4o-mini-2024-07-18::legacy-no-cue` |
| Gemma 3 12B | Ollama | Historical | No cue | `gemma3:12b::legacy-no-cue` |
| Llama 3.1 8B | Ollama | Historical | No cue | `llama3.1:8b::legacy-no-cue` |

### Sampled human audit of model responses

After the benchmark itself was finalized, humans manually audited a stratified sample of 420 reasoning responses spanning models and varieties. One reviewer assessed each response and required both the semantic interpretation and final category to be acceptable. More than 90% were accepted.

This is a sampled descriptive audit, not exhaustive validation of all outputs. There was one reviewer per response, no inter-rater reliability statistic, and no retained response-level rating export; the audit is therefore not used in hypothesis tests or headline estimates.

## 2. Aggregate paired accuracy results

| Prompt | Comparison | Gap | 95% item-bootstrap CI | Holm-adjusted permutation p | Interpretation |
|---|---|---|---|---|---|
| Direct | SAE − Indian English | 6.5 pp | [2.4, 10.8] pp | 0.0098 | Lower non-SAE accuracy |
| Direct | SAE − Hinglish | 9.1 pp | [5.2, 13.3] pp | 4.00e-05 | Lower non-SAE accuracy |
| Reasoning | SAE − Indian English | 2.7 pp | [-0.4, 5.7] pp | 0.1226 | Lower non-SAE accuracy |
| Reasoning | SAE − Hinglish | 2.7 pp | [-0.1, 5.4] pp | 0.1226 | Lower non-SAE accuracy |

A positive gap means the classifier is more accurate on SAE than on the matched target-variety item. Under direct prompting, both intervals exclude zero and both multiplicity-adjusted tests are below 0.05. Under reasoning, the remaining gaps are smaller and their adjusted tests are not below 0.05.

## 3. Reasoning-prompt effects

| Variety | Direct accuracy | Reasoning accuracy | Reasoning effect | 95% CI | Adjusted p |
|---|---|---|---|---|---|
| Hinglish | 73.7% | 79.9% | +6.2 pp | [3.6, 9.0] pp | 3.00e-05 |
| Indian English | 76.3% | 79.9% | +3.6 pp | [0.8, 6.4] pp | 0.0263 |
| SAE | 82.8% | 82.6% | -0.2 pp | [-2.2, 1.7] pp | 0.8602 |

Aggregate accuracy improves under reasoning for the non-SAE conditions, but the error decomposition below shows that this is not an unqualified fairness improvement.

## Fairness-oriented moderation errors

| Prompt | Variety | Over-flag | Harm miss | Harm-category error |
|---|---|---|---|---|
| Direct | SAE | 5.0% | 50.0% | 60.4% |
| Direct | Indian English | 15.2% | 43.2% | 53.8% |
| Direct | Hinglish | 16.7% | 45.7% | 60.4% |
| Reasoning | SAE | 2.2% | 66.9% | 71.2% |
| Reasoning | Indian English | 8.0% | 54.3% | 62.9% |
| Reasoning | Hinglish | 6.8% | 60.6% | 66.9% |

Under direct prompting, normal Indian English and Hinglish items are over-flagged substantially more often than matched SAE items. Reasoning lowers over-flagging, but it raises harmful-content misses and harmful-category errors. Because 156 of the 200 items are normal, the over-flagging reduction dominates aggregate accuracy even though performance on the harmful subset worsens.

### Direct-prompt over-flagging disparities

| Comparison | Difference | 95% item-bootstrap CI | Adjusted p |
|---|---|---|---|
| Hinglish − SAE | +11.7 pp | [7.7, 16.0] pp | 1.20e-04 |
| Indian English − SAE | +10.2 pp | [5.9, 14.7] pp | 1.20e-04 |

### Reasoning changes the error trade-off

| Variety | Metric | Direct | Reasoning | Change | 95% CI |
|---|---|---|---|---|---|
| Hinglish | Harm miss | 45.7% | 60.6% | +14.9 pp | [9.6, 20.2] pp |
| Indian English | Harm miss | 43.2% | 54.3% | +11.1 pp | [4.3, 18.4] pp |
| Hinglish | Over-flag | 16.7% | 6.8% | -9.8 pp | [-12.9, -7.1] pp |
| Indian English | Over-flag | 15.2% | 8.0% | -7.2 pp | [-10.0, -4.5] pp |

A negative over-flagging change is beneficial, while a positive harmful-content-miss change is harmful. The reasoning prompt moves both metrics at once and should therefore be described as redistributing errors, not simply fixing linguistic bias.

## 4. Historical-versus-fresh cohort sensitivity

| Cohort | Comparison | Systems | Gap | 95% CI | Adjusted p |
|---|---|---|---|---|---|
| Fresh | SAE − Hinglish | 5 | 9.4 pp | [5.2, 13.7] pp | 8.00e-05 |
| Fresh | SAE − Indian English | 5 | 6.7 pp | [2.4, 11.1] pp | 0.0093 |
| Historical | SAE − Hinglish | 4 | 8.8 pp | [4.6, 13.1] pp | 2.40e-04 |
| Historical | SAE − Indian English | 4 | 6.2 pp | [1.8, 10.9] pp | 0.0260 |

The direct-prompt gap has the same direction in both cohorts. Fresh systems show a 6.7-point Indian English gap and a 9.4-point Hinglish gap; historical systems show 6.2 and 8.8 points, respectively.

## 5. Per-system direct-prompt results

| System | Cohort | SAE | Indian English | Hinglish | IndEng gap | Hinglish gap |
|---|---|---|---|---|---|---|
| Claude Haiku 3.5 | Historical | 83.0% | 78.5% | 73.5% | 4.5 pp | 9.5 pp |
| GPT-4.1 Mini | Fresh | 83.5% | 79.0% | 74.5% | 4.5 pp | 9.0 pp |
| GPT-4o-mini | Historical | 84.5% | 79.0% | 76.0% | 5.5 pp | 8.5 pp |
| Gemini 2.5 Flash-Lite | Fresh | 85.5% | 76.5% | 81.5% | 9.0 pp | 4.0 pp |
| Gemma 3 12B | Fresh | 83.5% | 72.0% | 72.0% | 11.5 pp | 11.5 pp |
| Gemma 3 12B | Historical | 84.0% | 76.0% | 75.0% | 8.0 pp | 9.0 pp |
| Llama 3.1 8B | Fresh | 80.0% | 75.5% | 69.0% | 4.5 pp | 11.0 pp |
| Llama 3.1 8B | Historical | 81.5% | 74.5% | 73.5% | 7.0 pp | 8.0 pp |
| Mistral 7B Instruct | Fresh | 80.0% | 76.0% | 68.5% | 4.0 pp | 11.5 pp |

## 6. Complete model-specific paired inference

| Model | Cohort | Prompt | Target | Gap | 95% CI (pp) | Adjusted p |
|---|---|---|---|---|---|---|
| Claude Haiku 3.5 | Historical | Direct | Indian English | 4.5 pp | [-1.0, 10.0] | 1.0000 |
| Claude Haiku 3.5 | Historical | Direct | Hinglish | 9.5 pp | [3.5, 15.5] | 0.0970 |
| Claude Haiku 3.5 | Historical | Reasoning | Indian English | 3.0 pp | [-1.5, 7.5] | 1.0000 |
| Claude Haiku 3.5 | Historical | Reasoning | Hinglish | 5.5 pp | [1.5, 10.0] | 0.4971 |
| GPT-4.1 Mini | Fresh | Direct | Indian English | 4.5 pp | [-1.0, 10.5] | 1.0000 |
| GPT-4.1 Mini | Fresh | Direct | Hinglish | 9.0 pp | [3.5, 14.5] | 0.0909 |
| GPT-4.1 Mini | Fresh | Reasoning | Indian English | 4.0 pp | [-1.5, 9.5] | 1.0000 |
| GPT-4.1 Mini | Fresh | Reasoning | Hinglish | 2.0 pp | [-3.0, 7.0] | 1.0000 |
| GPT-4o-mini | Historical | Direct | Indian English | 5.5 pp | [-0.5, 11.5] | 1.0000 |
| GPT-4o-mini | Historical | Direct | Hinglish | 8.5 pp | [3.0, 14.0] | 0.1218 |
| GPT-4o-mini | Historical | Reasoning | Indian English | 3.0 pp | [-1.5, 7.5] | 1.0000 |
| GPT-4o-mini | Historical | Reasoning | Hinglish | 3.0 pp | [-1.5, 7.5] | 1.0000 |
| Gemini 2.5 Flash-Lite | Fresh | Direct | Indian English | 9.0 pp | [3.0, 15.0] | 0.1690 |
| Gemini 2.5 Flash-Lite | Fresh | Direct | Hinglish | 4.0 pp | [-1.5, 9.5] | 1.0000 |
| Gemini 2.5 Flash-Lite | Fresh | Reasoning | Indian English | 2.5 pp | [-3.0, 8.5] | 1.0000 |
| Gemini 2.5 Flash-Lite | Fresh | Reasoning | Hinglish | -1.0 pp | [-6.5, 4.0] | 1.0000 |
| Gemma 3 12B | Fresh | Direct | Indian English | 11.5 pp | [5.0, 18.0] | 0.0275 |
| Gemma 3 12B | Fresh | Direct | Hinglish | 11.5 pp | [4.5, 18.0] | 0.0436 |
| Gemma 3 12B | Fresh | Reasoning | Indian English | 2.0 pp | [-4.0, 8.0] | 1.0000 |
| Gemma 3 12B | Fresh | Reasoning | Hinglish | 1.0 pp | [-4.5, 6.5] | 1.0000 |
| Gemma 3 12B | Historical | Direct | Indian English | 8.0 pp | [2.0, 14.0] | 0.4971 |
| Gemma 3 12B | Historical | Direct | Hinglish | 9.0 pp | [3.5, 15.0] | 0.1151 |
| Gemma 3 12B | Historical | Reasoning | Indian English | 4.0 pp | [-0.5, 8.5] | 1.0000 |
| Gemma 3 12B | Historical | Reasoning | Hinglish | 2.0 pp | [-2.0, 6.0] | 1.0000 |
| Llama 3.1 8B | Fresh | Direct | Indian English | 4.5 pp | [-0.5, 9.5] | 1.0000 |
| Llama 3.1 8B | Fresh | Direct | Hinglish | 11.0 pp | [5.0, 17.0] | 0.0192 |
| Llama 3.1 8B | Fresh | Reasoning | Indian English | -0.5 pp | [-5.0, 4.0] | 1.0000 |
| Llama 3.1 8B | Fresh | Reasoning | Hinglish | 3.5 pp | [0.5, 6.5] | 1.0000 |
| Llama 3.1 8B | Historical | Direct | Indian English | 7.0 pp | [1.5, 12.5] | 0.6038 |
| Llama 3.1 8B | Historical | Direct | Hinglish | 8.0 pp | [3.0, 13.5] | 0.1038 |
| Llama 3.1 8B | Historical | Reasoning | Indian English | 3.0 pp | [-1.0, 7.0] | 1.0000 |
| Llama 3.1 8B | Historical | Reasoning | Hinglish | 4.0 pp | [0.5, 7.5] | 1.0000 |
| Mistral 7B Instruct | Fresh | Direct | Indian English | 4.0 pp | [-1.5, 9.5] | 1.0000 |
| Mistral 7B Instruct | Fresh | Direct | Hinglish | 11.5 pp | [5.5, 17.5] | 0.0166 |
| Mistral 7B Instruct | Fresh | Reasoning | Indian English | 3.0 pp | [-1.5, 7.5] | 1.0000 |
| Mistral 7B Instruct | Fresh | Reasoning | Hinglish | 4.0 pp | [-2.0, 10.0] | 1.0000 |

## 7. Constrained-probability case study

The strict A–F output grammar defines a separate coded classifier. Before interpreting calibration, its predictions are compared with ordinary direct-label output.

| Model | Prediction agreement | 95% CI | Cohen's kappa | Ordinary accuracy | Coded accuracy | Accuracy difference |
|---|---|---|---|---|---|---|
| GPT-4.1 Mini | 89.0% | 85.7%–92.2% | 0.761 | 79.0% | 75.0% | -4.0% |
| GPT-4o-mini | 88.2% | 85.0%–91.2% | 0.721 | 79.8% | 75.5% | -4.3% |

Agreement is high but not perfect, and the coded prompt is about four percentage points less accurate overall. The calibration results therefore characterize the coded classifier, not an interchangeable native-confidence interface.

Calibration is reported only for GPT-4o-mini and GPT-4.1 Mini because these APIs exposed comparable grammar-constrained six-class token probabilities. Accuracy and confidence are percentages; lower NLL, Brier, ECE, and ACE are better.

| Model | Variety | Accuracy | Mean confidence | Confidence − accuracy | NLL | Brier | ECE | ACE |
|---|---|---|---|---|---|---|---|---|
| GPT-4o-mini | SAE | 82.0% | 98.0% | +16.0% | 1.675 | 0.316 | 16.0% | 16.0% |
| GPT-4o-mini | Indian English | 73.5% | 97.8% | +24.3% | 3.575 | 0.498 | 24.6% | 24.3% |
| GPT-4o-mini | Hinglish | 71.0% | 96.2% | +25.2% | 3.234 | 0.522 | 25.2% | 25.2% |
| GPT-4.1 Mini | SAE | 80.0% | 96.7% | +16.7% | 1.576 | 0.373 | 17.8% | 16.7% |
| GPT-4.1 Mini | Indian English | 73.0% | 95.5% | +22.5% | 2.282 | 0.479 | 23.2% | 22.5% |
| GPT-4.1 Mini | Hinglish | 72.0% | 96.1% | +24.1% | 2.310 | 0.504 | 24.3% | 24.1% |

### Change relative to SAE

| Model | Target | Accuracy change | Confidence change | NLL change | Brier change | ECE change | ACE change |
|---|---|---|---|---|---|---|---|
| GPT-4.1 Mini | Hinglish | -8.0 pp | -0.6 pp | +0.734 | +0.131 | +6.5 pp | +7.4 pp |
| GPT-4.1 Mini | Indian English | -7.0 pp | -1.2 pp | +0.706 | +0.106 | +5.4 pp | +5.8 pp |
| GPT-4o-mini | Hinglish | -11.0 pp | -1.7 pp | +1.559 | +0.207 | +9.3 pp | +9.3 pp |
| GPT-4o-mini | Indian English | -8.5 pp | -0.2 pp | +1.900 | +0.182 | +8.7 pp | +8.3 pp |

The central calibration finding is visible here: accuracy drops by 7–11 points, but mean confidence drops by only about 0.2–1.7 points. Consequently, proper scoring rules and calibration errors worsen.

## 8. Descriptive uncertainty-set coverage diagnostic

The conformal-style diagnostic uses a 90% nominal target. Coverage is the percentage of test items whose prediction set contains the true label. Set size is the average number of labels retained; smaller sets are more informative. Because pooled calibration scores remain correlated within matched triplets, these repeated-split values are descriptive and no finite-sample pooled guarantee is claimed.

| Model | Method | Variety | Mean coverage | Split SD | Mean set size | 2.5–97.5% split range |
|---|---|---|---|---|---|---|
| GPT-4o-mini | Pooled | SAE | 94.3% | 2.2% | 1.52 | 90.0%–98.0% |
| GPT-4o-mini | Pooled | Indian English | 87.0% | 3.9% | 1.64 | 78.0%–94.0% |
| GPT-4o-mini | Pooled | Hinglish | 88.5% | 4.0% | 1.99 | 81.0%–94.0% |
| GPT-4o-mini | Variety-conditional | SAE | 89.7% | 4.3% | 1.21 | 80.0%–97.0% |
| GPT-4o-mini | Variety-conditional | Indian English | 90.2% | 4.4% | 2.21 | 79.0%–97.0% |
| GPT-4o-mini | Variety-conditional | Hinglish | 90.4% | 4.1% | 2.30 | 82.0%–97.0% |
| GPT-4.1 Mini | Pooled | SAE | 92.7% | 2.7% | 1.35 | 87.0%–97.0% |
| GPT-4.1 Mini | Pooled | Indian English | 88.9% | 3.7% | 1.55 | 81.0%–95.0% |
| GPT-4.1 Mini | Pooled | Hinglish | 88.7% | 3.9% | 1.62 | 79.0%–95.0% |
| GPT-4.1 Mini | Variety-conditional | SAE | 90.6% | 4.0% | 1.28 | 83.0%–97.0% |
| GPT-4.1 Mini | Variety-conditional | Indian English | 90.1% | 3.8% | 1.65 | 82.0%–96.0% |
| GPT-4.1 Mini | Variety-conditional | Hinglish | 90.3% | 4.5% | 1.75 | 80.0%–98.0% |

## 9. Hierarchical mixed-effects results

The model uses SAE, direct prompting, and no variety cue as reference levels. Negative variety coefficients mean lower odds of a correct answer than SAE. Positive variety × reasoning interactions mean reasoning offsets part of that loss.

| Term | Posterior log-odds | 95% credible interval | Odds ratio |
|---|---|---|---|
| Intercept | +3.716 | [+3.647, +3.785] | 41.085 |
| hinglish | -1.206 | [-1.350, -1.061] | 0.300 |
| indian_english | -0.878 | [-1.024, -0.733] | 0.416 |
| cot | -0.031 | [-0.169, +0.107] | 0.970 |
| hinglish:cot | +0.832 | [+0.646, +1.017] | 2.297 |
| indian_english:cot | +0.504 | [+0.319, +0.690] | 1.656 |

The optimizer converged successfully after 264 iterations. Estimated random-intercept standard deviations were item: 3.867, system: 0.228. A GEE model with system fixed effects and item-clustered standard errors reached the same qualitative conclusions.

## 10. Definitions and mathematics in simple terms

### Accuracy

Accuracy is the fraction of predictions equal to the gold label:

$$
\operatorname{Accuracy}=\frac{\text{number correct}}{\text{number evaluated}}.
$$

### Percentage points (pp)

Percentage points measure an absolute difference between percentages. A change from 84% to 75% is a drop of 9 percentage points, not a 9% relative reduction.

### Paired linguistic gap

For item $i$, model-system $m$, prompt condition $c$, and target variety $v$, let $Y_{imcv}=1$ when the prediction is correct and $0$ otherwise. The paired gap is:

$$
\widehat{\Delta}_{mvc}=\frac{1}{n}\sum_{i=1}^{n}\left(Y_{imc,\mathrm{SAE}}-Y_{imcv}\right).
$$

Each subtraction compares two versions of the **same item**. A positive gap means SAE was more accurate. Pairing removes item difficulty as an explanation for the difference.

### Reasoning effect

The reasoning effect is reasoning accuracy minus direct accuracy:

$$
\widehat{R}_{v}=\operatorname{Accuracy}_{v,\mathrm{reasoning}}-\operatorname{Accuracy}_{v,\mathrm{direct}}.
$$

A positive value means the five-step prompt improved classification for that variety.

### Bootstrap confidence interval

The item-clustered bootstrap repeatedly resamples the 200 item IDs with replacement. Whenever an item is selected, all its varieties, models, and prompt conditions move together. The middle 95% of the resulting effect estimates forms the confidence interval. This preserves the matched structure.

### Permutation test and Holm correction

Under a no-effect null hypothesis, the direction of each paired difference can be randomly flipped. The permutation p-value is the fraction of randomized effects at least as large as the observed effect. Holm correction raises p-values when several related hypotheses are tested, reducing false discoveries.

### Exact McNemar test

For one model and two matched varieties, McNemar's test ignores pairs where both predictions are correct or both are wrong. It compares the number of SAE-only correct pairs with the number of target-only correct pairs using an exact binomial test. It is another way to test whether errors are asymmetric within matched items.

### Log probabilities and class probabilities

The API supplies one log score $\ell_k$ for each class. Scores are converted to probabilities with a softmax:

$$
p_k=\frac{e^{\ell_k}}{\sum_{j=1}^{K}e^{\ell_j}}.
$$

The largest $p_k$ is the model's top-label confidence.

### Negative log-likelihood (NLL)

For the true class $y$:

$$
\operatorname{NLL}=-\log p_y.
$$

Confident wrong answers receive a large penalty. Lower NLL is better.

### Multiclass Brier score

With one-hot target $y_k$:

$$
\operatorname{Brier}=\sum_{k=1}^{K}(p_k-y_k)^2.
$$

It measures the squared distance between the full predicted distribution and the correct answer. Lower is better.

### Calibration, ECE, and ACE

A calibrated classifier is correct about 80% of the time on predictions made with 80% confidence. ECE groups predictions into confidence bins and averages the absolute confidence–accuracy difference:

$$
\operatorname{ECE}=\sum_{b=1}^{B}\frac{|I_b|}{n}\left|\operatorname{accuracy}(I_b)-\operatorname{confidence}(I_b)\right|.
$$

ACE uses bins containing approximately equal numbers of examples. Both are reported as supplements because their values depend on binning; NLL and Brier are the primary proper scoring rules.

### Conformal prediction

The nonconformity score for true class $y$ is:

$$
s=1-p_y.
$$

A calibration set determines a threshold $q$. For a new item, all labels satisfying $1-p_k\le q$ enter the prediction set. **Coverage** is how often this set contains the true label. **Set size** measures how many labels remain. Pooled calibration uses one threshold for all varieties; variety-specific calibration estimates one threshold per variety. In this study, pooled calibration scores within a triplet are correlated, so the repeated-split numbers are empirical diagnostics rather than a formal finite-sample pooled guarantee.

### Mixed-effects logistic model

The hierarchical model represents correctness probability using log odds:

$$
\operatorname{logit}\Pr(Y=1)=X\beta+u_{\mathrm{item}}+u_{\mathrm{system}}.
$$

$X\beta$ contains the average effects of variety, prompt condition, and their planned interaction. The random intercepts $u$ allow some items and systems to be intrinsically easier or harder. Exponentiating a coefficient gives an odds ratio: values below 1 mean lower odds of correctness than the reference condition.

## 11. Important limitations

- The dataset contains only 200 matched items and is heavily weighted toward the normal class (156/200).
- Utterances are short; this does not establish performance on dialogue or long documents.
- Hinglish is code-mixing, and no pure-Hindi comparison is included.
- All runs are recorded as no-cue prompts. Fresh outputs retain prompt hashes, while historical no-cue provenance relies on the authors' preserved execution record because legacy CSVs lack request-level metadata.
- The 420-response human audit used one reviewer per response and has no retained response-level rating export, so it supports only a descriptive aggregate acceptance statement.
- Historical result files contain one stored prediction per cell and do not support the earlier claim that five independent runs were averaged.
- Calibration and uncertainty-set diagnostics cover only two OpenAI models and a distinct coded output grammar; they do not measure unconstrained semantic confidence.
- Pooled calibration scores remain correlated within triplets, so no formal pooled conformal coverage guarantee is claimed.
- Local Llama, Gemma, and Mistral models use Q4_K_M quantization; results apply to those exact model digests.
- Findings concern this six-class moderation task, not all language-model behavior.

## 12. Validation, cost, and artifact locations

- Hosted API expenditure recorded for the full development session: **$0.470130** of the $50 cap.
- Every fresh hosted and local result file has one canonical record per evaluation key and zero failed final cells.
- Local models were identified by exact Ollama digests and validated before analysis.
- Run `pytest -q` for the current test result; this generated report does not execute tests or lint checks.
- The current manuscript targets ACM FAccT 2027.

### Key files

- [Paper tables and figures](../../paper/facct2027/)
- [Frozen results manifest](./manifests/results_manifest.json)
- [Canonical accuracy table](./validated/accuracy_long.csv)
- [Paired-effect tables](./metrics/paired/)
- [Calibration tables](./metrics/calibration/)
- [Conformal tables](./metrics/conformal/)
- [Mixed-effects outputs](./metrics/mixed_effects/)
- [Exact local model manifests](./model_manifests/summary.json)

## Bottom line

The paired design shows a reproducible accuracy disadvantage for Indian English and Hinglish under direct prompting. Explicit reasoning substantially narrows the gap without helping SAE. In a separate coded-output case study, grammar-constrained label-token confidence declines much less than accuracy, so calibration worsens. The descriptive repeated-split diagnostic finds lower average non-SAE coverage under pooled thresholds; variety-specific thresholds raise that average only by returning larger and less informative sets.
