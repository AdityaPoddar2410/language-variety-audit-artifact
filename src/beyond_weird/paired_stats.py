"""Paired accuracy estimators and item-clustered inference."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

TARGET_VARIETIES = ("indian_english", "hinglish")


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min((count - rank) * values[index], 1.0)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _bootstrap_interval(
    differences: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    estimates = differences[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _paired_permutation_pvalue(
    differences: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(nonzero)))
    permuted = np.abs((signs * np.abs(nonzero)).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (n_permutations + 1))


def model_paired_effects(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.loc[frame["valid"]].groupby(
        [
            "cohort",
            "provider",
            "model",
            "model_id",
            "system_id",
            "prompt_cue",
            "condition",
        ],
        dropna=False,
    )
    for group_index, (key, group) in enumerate(grouped):
        wide = group.pivot(index="id", columns="variety", values="correct")
        for target_index, target in enumerate(TARGET_VARIETIES):
            paired = wide[["sae", target]].dropna().astype(float)
            differences = paired["sae"].to_numpy() - paired[target].to_numpy()
            rng = np.random.default_rng(seed + group_index * 10 + target_index)
            lower, upper = _bootstrap_interval(differences, n_bootstrap, rng)
            permutation_p = _paired_permutation_pvalue(
                differences,
                n_permutations,
                rng,
            )
            sae_correct = paired["sae"].astype(bool)
            target_correct = paired[target].astype(bool)
            sae_only = int((sae_correct & ~target_correct).sum())
            target_only = int((~sae_correct & target_correct).sum())
            discordant = sae_only + target_only
            mcnemar_p = (
                float(binomtest(sae_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "cohort": key[0],
                    "provider": key[1],
                    "model": key[2],
                    "model_id": key[3],
                    "system_id": key[4],
                    "prompt_cue": key[5],
                    "condition": key[6],
                    "target_variety": target,
                    "n_paired": len(paired),
                    "accuracy_sae": float(paired["sae"].mean()),
                    "accuracy_target": float(paired[target].mean()),
                    "gap": float(differences.mean()),
                    "gap_pp": float(100 * differences.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_lower_pp": 100 * lower,
                    "ci_upper_pp": 100 * upper,
                    "sae_only": sae_only,
                    "target_only": target_only,
                    "permutation_p": permutation_p,
                    "mcnemar_exact_p": mcnemar_p,
                }
            )
    result = pd.DataFrame(rows)
    result["permutation_p_holm"] = holm_adjust(result["permutation_p"])
    result["mcnemar_p_holm"] = holm_adjust(result["mcnemar_exact_p"])
    return result


def aggregate_paired_effects(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = frame.loc[frame["valid"]]
    for condition_index, (condition, group) in enumerate(valid.groupby("condition")):
        for target_index, target in enumerate(TARGET_VARIETIES):
            wide = group.pivot(index=["id", "system_id"], columns="variety", values="correct")
            paired = wide[["sae", target]].dropna().astype(float)
            differences = (paired["sae"] - paired[target]).rename("difference").reset_index()
            item_model = differences.pivot(index="id", columns="system_id", values="difference")
            if item_model.isna().any().any():
                raise ValueError(
                    f"Aggregate effect requires complete item-model pairs for {condition}/{target}"
                )
            matrix = item_model.to_numpy(dtype=float)
            rng = np.random.default_rng(seed + condition_index * 10 + target_index)
            bootstrap_rows = rng.integers(0, matrix.shape[0], size=(n_bootstrap, matrix.shape[0]))
            estimates = matrix[bootstrap_rows].mean(axis=(1, 2))
            lower, upper = np.quantile(estimates, [0.025, 0.975])

            item_differences = matrix.mean(axis=1)
            observed = abs(float(item_differences.mean()))
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(n_permutations, matrix.shape[0]),
            )
            permuted = np.abs((signs * item_differences).mean(axis=1))
            permutation_p = float(
                (np.count_nonzero(permuted >= observed) + 1) / (n_permutations + 1)
            )
            rows.append(
                {
                    "condition": condition,
                    "target_variety": target,
                    "n_items": matrix.shape[0],
                    "n_systems": matrix.shape[1],
                    "gap": float(matrix.mean()),
                    "gap_pp": 100 * float(matrix.mean()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "ci_lower_pp": 100 * float(lower),
                    "ci_upper_pp": 100 * float(upper),
                    "cluster_permutation_p": permutation_p,
                }
            )
    result = pd.DataFrame(rows)
    result["cluster_permutation_p_holm"] = holm_adjust(
        result["cluster_permutation_p"]
    )
    return result


def aggregate_paired_effects_by_cohort(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    tables = []
    for index, (cohort, subset) in enumerate(frame.groupby("cohort")):
        table = aggregate_paired_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 100,
        )
        table.insert(0, "cohort", cohort)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def aggregate_paired_effects_by_prompt_cue(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    tables = []
    for index, (prompt_cue, subset) in enumerate(frame.groupby("prompt_cue")):
        table = aggregate_paired_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 100,
        )
        table.insert(0, "prompt_cue", prompt_cue)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def model_reasoning_effects(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.loc[frame["valid"]].groupby(
        [
            "cohort",
            "provider",
            "model",
            "model_id",
            "system_id",
            "prompt_cue",
            "variety",
        ],
        dropna=False,
    )
    for group_index, (key, group) in enumerate(grouped):
        wide = group.pivot(index="id", columns="condition", values="correct")
        paired = wide[["direct", "cot"]].dropna().astype(float)
        differences = paired["cot"].to_numpy() - paired["direct"].to_numpy()
        rng = np.random.default_rng(seed + group_index)
        lower, upper = _bootstrap_interval(differences, n_bootstrap, rng)
        permutation_p = _paired_permutation_pvalue(
            differences,
            n_permutations,
            rng,
        )
        rows.append(
            {
                "cohort": key[0],
                "provider": key[1],
                "model": key[2],
                "model_id": key[3],
                "system_id": key[4],
                "prompt_cue": key[5],
                "variety": key[6],
                "n_paired": len(paired),
                "accuracy_direct": float(paired["direct"].mean()),
                "accuracy_cot": float(paired["cot"].mean()),
                "reasoning_effect": float(differences.mean()),
                "reasoning_effect_pp": 100 * float(differences.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_lower_pp": 100 * lower,
                "ci_upper_pp": 100 * upper,
                "permutation_p": permutation_p,
            }
        )
    result = pd.DataFrame(rows)
    result["permutation_p_holm"] = holm_adjust(result["permutation_p"])
    return result


def aggregate_reasoning_effects(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = frame.loc[frame["valid"]]
    for variety_index, (variety, group) in enumerate(valid.groupby("variety")):
        wide = group.pivot(index=["id", "system_id"], columns="condition", values="correct")
        paired = wide[["direct", "cot"]].dropna().astype(float)
        differences = (paired["cot"] - paired["direct"]).rename("difference").reset_index()
        item_model = differences.pivot(index="id", columns="system_id", values="difference")
        if item_model.isna().any().any():
            raise ValueError(f"Reasoning effect requires complete item-model pairs for {variety}")
        matrix = item_model.to_numpy(dtype=float)
        rng = np.random.default_rng(seed + variety_index)
        bootstrap_rows = rng.integers(0, matrix.shape[0], size=(n_bootstrap, matrix.shape[0]))
        estimates = matrix[bootstrap_rows].mean(axis=(1, 2))
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        item_differences = matrix.mean(axis=1)
        observed = abs(float(item_differences.mean()))
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(n_permutations, matrix.shape[0]),
        )
        permuted = np.abs((signs * item_differences).mean(axis=1))
        permutation_p = float(
            (np.count_nonzero(permuted >= observed) + 1) / (n_permutations + 1)
        )
        rows.append(
            {
                "variety": variety,
                "n_items": matrix.shape[0],
                "n_systems": matrix.shape[1],
                "accuracy_direct": float(
                    group[group["condition"] == "direct"]["correct"].mean()
                ),
                "accuracy_cot": float(group[group["condition"] == "cot"]["correct"].mean()),
                "reasoning_effect": float(matrix.mean()),
                "reasoning_effect_pp": 100 * float(matrix.mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "ci_lower_pp": 100 * float(lower),
                "ci_upper_pp": 100 * float(upper),
                "cluster_permutation_p": permutation_p,
            }
        )
    result = pd.DataFrame(rows)
    result["cluster_permutation_p_holm"] = holm_adjust(result["cluster_permutation_p"])
    return result


def aggregate_reasoning_effects_by_cohort(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    tables = []
    for index, (cohort, subset) in enumerate(frame.groupby("cohort")):
        table = aggregate_reasoning_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 100,
        )
        table.insert(0, "cohort", cohort)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def aggregate_reasoning_effects_by_prompt_cue(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    tables = []
    for index, (prompt_cue, subset) in enumerate(frame.groupby("prompt_cue")):
        table = aggregate_reasoning_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 100,
        )
        table.insert(0, "prompt_cue", prompt_cue)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def accuracy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "cohort",
                "provider",
                "model",
                "model_id",
                "system_id",
                "prompt_cue",
                "condition",
                "variety",
            ],
            dropna=False,
        )
        .agg(
            n=("id", "size"),
            valid_n=("valid", "sum"),
            n_correct=("correct", "sum"),
        )
        .reset_index()
        .assign(accuracy=lambda value: value["n_correct"] / value["valid_n"])
    )
