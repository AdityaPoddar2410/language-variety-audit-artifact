"""Agreement between ordinary labels and constrained probability prompts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prompts import CONTENT_CATEGORIES

_PROBABILITY_TO_ORDINARY_MODEL = {
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini-2025-04-14",
}


def _cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    observed = float((first == second).mean())
    expected = 0.0
    for label in CONTENT_CATEGORIES:
        expected += float((first == label).mean() * (second == label).mean())
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def prompt_agreement_table(
    accuracy: pd.DataFrame,
    probabilities: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 2027,
) -> pd.DataFrame:
    ordinary = accuracy[
        (accuracy["condition"] == "direct") & accuracy["valid"]
    ].copy()
    probability = probabilities.copy()
    rows: list[dict[str, object]] = []

    for model_index, (probability_model, ordinary_model) in enumerate(
        _PROBABILITY_TO_ORDINARY_MODEL.items()
    ):
        ordinary_model_frame = ordinary[ordinary["model_id"] == ordinary_model]
        probability_model_frame = probability[
            probability["model_id"] == probability_model
        ]
        merged = ordinary_model_frame.merge(
            probability_model_frame[
                ["id", "variety", "predicted_category", "correct"]
            ],
            on=["id", "variety"],
            suffixes=("_ordinary", "_probability"),
            validate="one_to_one",
        )
        if len(merged) != 600:
            raise ValueError(
                f"Expected 600 matched prompt outputs for {probability_model}, found {len(merged)}"
            )
        for variety_index, variety in enumerate(("all", "sae", "indian_english", "hinglish")):
            subset = merged if variety == "all" else merged[merged["variety"] == variety]
            item_ids = subset["id"].unique()
            rng = np.random.default_rng(seed + model_index * 10 + variety_index)
            agreement_by_item = (
                subset.assign(
                    agreement=subset["predicted_category_ordinary"]
                    == subset["predicted_category_probability"],
                    accuracy_difference=subset["correct_probability"].astype(float)
                    - subset["correct_ordinary"].astype(float),
                )
                .groupby("id")[["agreement", "accuracy_difference"]]
                .mean()
                .reindex(item_ids)
            )
            draws = rng.integers(0, len(item_ids), size=(n_bootstrap, len(item_ids)))
            agreement_bootstrap = agreement_by_item["agreement"].to_numpy()[draws].mean(axis=1)
            difference_bootstrap = (
                agreement_by_item["accuracy_difference"].to_numpy()[draws].mean(axis=1)
            )
            agreement_lower, agreement_upper = np.quantile(
                agreement_bootstrap, [0.025, 0.975]
            )
            difference_lower, difference_upper = np.quantile(
                difference_bootstrap, [0.025, 0.975]
            )
            rows.append(
                {
                    "probability_model_id": probability_model,
                    "ordinary_model_id": ordinary_model,
                    "variety": variety,
                    "n": len(subset),
                    "prediction_agreement": float(
                        (
                            subset["predicted_category_ordinary"]
                            == subset["predicted_category_probability"]
                        ).mean()
                    ),
                    "agreement_ci_lower": float(agreement_lower),
                    "agreement_ci_upper": float(agreement_upper),
                    "cohen_kappa": _cohen_kappa(
                        subset["predicted_category_ordinary"],
                        subset["predicted_category_probability"],
                    ),
                    "ordinary_accuracy": float(subset["correct_ordinary"].mean()),
                    "probability_prompt_accuracy": float(
                        subset["correct_probability"].mean()
                    ),
                    "accuracy_difference": float(
                        subset["correct_probability"].mean()
                        - subset["correct_ordinary"].mean()
                    ),
                    "accuracy_difference_ci_lower": float(difference_lower),
                    "accuracy_difference_ci_upper": float(difference_upper),
                }
            )
    return pd.DataFrame(rows)
