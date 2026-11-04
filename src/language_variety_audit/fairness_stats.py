"""Fairness-oriented moderation error metrics and paired inference."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .paired_stats import holm_adjust

FAIRNESS_METRICS = ("overflag", "harm_miss", "harm_category_error")
TARGET_VARIETIES = ("indian_english", "hinglish")
VARIETIES = ("sae", "indian_english", "hinglish")
CONDITIONS = ("direct", "cot")
_REQUIRED_COLUMNS = {
    "id",
    "cohort",
    "provider",
    "model",
    "model_id",
    "system_id",
    "prompt_cue",
    "condition",
    "variety",
    "gold_category",
    "predicted_category",
    "valid",
}


def _validate_accuracy_frame(frame: pd.DataFrame) -> None:
    missing_columns = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required fairness columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Fairness analysis requires at least one observation")
    if not frame["valid"].astype(bool).all():
        raise ValueError("Fairness analysis requires every canonical cell to be valid")

    key = ["id", "system_id", "condition", "variety"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, key].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate fairness cells: {examples}")

    unknown_conditions = sorted(set(frame["condition"]).difference(CONDITIONS))
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    unknown_varieties = sorted(set(frame["variety"]).difference(VARIETIES))
    if unknown_varieties:
        raise ValueError(f"Unknown varieties: {unknown_varieties}")

    expected_ids = set(frame["id"].unique())
    coverage = frame.groupby(["system_id", "condition", "variety"])["id"].agg(
        lambda values: set(values)
    )
    incomplete = [index for index, ids in coverage.items() if ids != expected_ids]
    if incomplete:
        raise ValueError(f"Incomplete fairness cells for groups: {incomplete[:10]}")


def fairness_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add moderation-error indicators with inapplicable rows set to missing."""
    _validate_accuracy_frame(frame)
    result = frame.copy()
    normal = result["gold_category"].eq("normal")
    predicted_normal = result["predicted_category"].eq("normal")
    correct = result["predicted_category"].eq(result["gold_category"])

    result["overflag"] = np.where(normal, ~predicted_normal, np.nan)
    result["harm_miss"] = np.where(~normal, predicted_normal, np.nan)
    result["harm_category_error"] = np.where(~normal, ~correct, np.nan)
    return result


def fairness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-system moderation error rates by condition and variety."""
    indicators = fairness_indicators(frame)
    group_columns = [
        "cohort",
        "provider",
        "model",
        "model_id",
        "system_id",
        "prompt_cue",
        "condition",
        "variety",
    ]
    rows: list[dict[str, object]] = []
    for key, group in indicators.groupby(group_columns, dropna=False):
        normal = group[group["overflag"].notna()]
        harmful = group[group["harm_miss"].notna()]
        if normal.empty or harmful.empty:
            raise ValueError(f"Fairness summary requires normal and harmful items for {key}")
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            {
                "n_normal": int(normal["id"].nunique()),
                "overflag_rate": float(normal["overflag"].mean()),
                "n_harmful": int(harmful["id"].nunique()),
                "harm_miss_rate": float(harmful["harm_miss"].mean()),
                "harm_category_error_rate": float(
                    harmful["harm_category_error"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _cluster_inference(
    matrix: np.ndarray,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if n_bootstrap < 1 or n_permutations < 1:
        raise ValueError("Bootstrap and permutation counts must be positive")
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("Fairness inference requires a non-empty item-by-unit matrix")

    bootstrap_rows = rng.integers(
        0,
        matrix.shape[0],
        size=(n_bootstrap, matrix.shape[0]),
    )
    estimates = matrix[bootstrap_rows].mean(axis=(1, 2))
    lower, upper = np.quantile(estimates, [0.025, 0.975])

    item_differences = matrix.mean(axis=1)
    observed = abs(float(item_differences.mean()))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_permutations, matrix.shape[0]),
    )
    permuted = np.abs((signs * item_differences).mean(axis=1))
    p_value = float(
        (np.count_nonzero(permuted >= observed) + 1) / (n_permutations + 1)
    )
    return float(lower), float(upper), p_value


def _unit_values(
    frame: pd.DataFrame,
    metric: str,
    unit_column: str,
    comparison_column: str,
) -> pd.DataFrame:
    if metric not in FAIRNESS_METRICS:
        raise ValueError(f"Unknown fairness metric: {metric}")
    if unit_column not in {"system_id", "model"}:
        raise ValueError("unit_column must be 'system_id' or 'model'")
    selected = frame[frame[metric].notna()]
    return (
        selected.groupby(
            ["id", unit_column, comparison_column],
            as_index=False,
            dropna=False,
        )[metric]
        .mean()
        .sort_values(["id", unit_column, comparison_column])
    )


def _paired_matrix(
    frame: pd.DataFrame,
    metric: str,
    unit_column: str,
    comparison_column: str,
    baseline: str,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    values = _unit_values(frame, metric, unit_column, comparison_column)
    wide = values.pivot(
        index=["id", unit_column],
        columns=comparison_column,
        values=metric,
    )
    missing_levels = [value for value in (baseline, target) if value not in wide]
    if missing_levels:
        raise ValueError(
            f"Missing {comparison_column} levels for {metric}: {missing_levels}"
        )
    paired = wide[[baseline, target]]
    if paired.isna().any().any():
        raise ValueError(
            f"Incomplete paired fairness cells for {metric}/{baseline}/{target}"
        )
    differences = (paired[target] - paired[baseline]).rename("difference").reset_index()
    item_unit = differences.pivot(
        index="id",
        columns=unit_column,
        values="difference",
    )
    if item_unit.isna().any().any():
        raise ValueError(
            f"Fairness effect requires complete item-unit pairs for {metric}/{target}"
        )
    return paired, item_unit.to_numpy(dtype=float)


def aggregate_fairness_effects(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
    unit_column: str = "system_id",
    metrics: Iterable[str] = FAIRNESS_METRICS,
) -> pd.DataFrame:
    """Compare target-variety error rates with SAE using matched items.

    Holm correction is applied across every condition, metric, and target
    variety returned by one call, which defines one correction family.
    """
    indicators = fairness_indicators(frame)
    metrics = tuple(metrics)
    unknown_metrics = sorted(set(metrics).difference(FAIRNESS_METRICS))
    if unknown_metrics:
        raise ValueError(f"Unknown fairness metrics: {unknown_metrics}")
    rows: list[dict[str, object]] = []
    conditions = sorted(indicators["condition"].unique())
    for condition_index, condition in enumerate(conditions):
        condition_frame = indicators[indicators["condition"] == condition]
        for metric_index, metric in enumerate(metrics):
            for target_index, target in enumerate(TARGET_VARIETIES):
                paired, matrix = _paired_matrix(
                    condition_frame,
                    metric,
                    unit_column,
                    "variety",
                    "sae",
                    target,
                )
                rng = np.random.default_rng(
                    seed + condition_index * 100 + metric_index * 10 + target_index
                )
                lower, upper, p_value = _cluster_inference(
                    matrix,
                    n_bootstrap,
                    n_permutations,
                    rng,
                )
                rows.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "target_variety": target,
                        "n_items": matrix.shape[0],
                        "n_systems" if unit_column == "system_id" else "n_models": matrix.shape[1],
                        "sae_rate": float(paired["sae"].mean()),
                        "target_rate": float(paired[target].mean()),
                        "difference": float(matrix.mean()),
                        "difference_pp": 100 * float(matrix.mean()),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_lower_pp": 100 * lower,
                        "ci_upper_pp": 100 * upper,
                        "cluster_permutation_p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["cluster_permutation_p_holm"] = holm_adjust(
        result["cluster_permutation_p"]
    )
    return result


def aggregate_reasoning_fairness_effects(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
    unit_column: str = "system_id",
) -> pd.DataFrame:
    """Compare reasoning and direct error rates within matched items."""
    indicators = fairness_indicators(frame)
    required_conditions = set(CONDITIONS)
    if not required_conditions.issubset(set(indicators["condition"])):
        raise ValueError("Reasoning fairness effects require direct and cot conditions")

    rows: list[dict[str, object]] = []
    for variety_index, variety in enumerate(sorted(indicators["variety"].unique())):
        variety_frame = indicators[indicators["variety"] == variety]
        for metric_index, metric in enumerate(FAIRNESS_METRICS):
            paired, matrix = _paired_matrix(
                variety_frame,
                metric,
                unit_column,
                "condition",
                "direct",
                "cot",
            )
            rng = np.random.default_rng(seed + variety_index * 10 + metric_index)
            lower, upper, p_value = _cluster_inference(
                matrix,
                n_bootstrap,
                n_permutations,
                rng,
            )
            rows.append(
                {
                    "variety": variety,
                    "metric": metric,
                    "n_items": matrix.shape[0],
                    "n_systems" if unit_column == "system_id" else "n_models": matrix.shape[1],
                    "direct_rate": float(paired["direct"].mean()),
                    "reasoning_rate": float(paired["cot"].mean()),
                    "difference": float(matrix.mean()),
                    "difference_pp": 100 * float(matrix.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_lower_pp": 100 * lower,
                    "ci_upper_pp": 100 * upper,
                    "cluster_permutation_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["cluster_permutation_p_holm"] = holm_adjust(
        result["cluster_permutation_p"]
    )
    return result


def aggregate_fairness_effects_by_cohort(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    tables = []
    for index, (cohort, subset) in enumerate(frame.groupby("cohort")):
        table = aggregate_fairness_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 1_000,
        )
        table.insert(0, "cohort", cohort)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def aggregate_reasoning_fairness_effects_by_cohort(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
) -> pd.DataFrame:
    tables = []
    for index, (cohort, subset) in enumerate(frame.groupby("cohort")):
        table = aggregate_reasoning_fairness_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 1_000,
        )
        table.insert(0, "cohort", cohort)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def aggregate_fairness_effects_by_stratum(
    frame: pd.DataFrame,
    stratum_column: str,
    n_bootstrap: int = 10_000,
    n_permutations: int = 100_000,
    seed: int = 2027,
    metrics: Iterable[str] = ("overflag",),
) -> pd.DataFrame:
    """Repeat matched variety comparisons within item strata.

    Each stratum is a separate Holm family. Strata are defined at the item
    level, so every rendition, system, and condition of an item stays in the
    same stratum.
    """
    if stratum_column not in frame:
        raise ValueError(f"Missing stratum column: {stratum_column}")
    per_item = frame.groupby("id")[stratum_column].nunique()
    if (per_item != 1).any():
        raise ValueError("Each item must belong to exactly one stratum")
    tables = []
    for index, (stratum, subset) in enumerate(frame.groupby(stratum_column)):
        table = aggregate_fairness_effects(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + index * 1_000,
            metrics=metrics,
        )
        table.insert(0, stratum_column, stratum)
        table.insert(1, "n_stratum_items", int(subset["id"].nunique()))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf") if numerator > 0 else float("nan")
    return float(numerator / denominator)


def _finite_interval(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if not finite.size:
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper)


def _normal_flag_arrays(
    indicators: pd.DataFrame,
    condition: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = indicators[
        (indicators["condition"] == condition) & indicators["overflag"].notna()
    ]
    wide = selected.pivot(
        index=["id", "system_id"],
        columns="variety",
        values="overflag",
    )
    if wide[["sae", target]].isna().any().any():
        raise ValueError(f"Incomplete normal-item pairs for {condition}/{target}")
    sae = wide["sae"].unstack("system_id").to_numpy(dtype=float)
    other = wide[target].unstack("system_id").to_numpy(dtype=float)
    return sae, other


def overflag_relative_risk(
    frame: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 2027,
) -> pd.DataFrame:
    """Return target/SAE over-flagging ratios and matched discordant pairs.

    A discordant pair is one normal item and system for which exactly one of
    the SAE and target renditions is flagged. Intervals resample whole items.
    """
    if n_bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    indicators = fairness_indicators(frame)
    rows: list[dict[str, object]] = []
    for condition_index, condition in enumerate(sorted(indicators["condition"].unique())):
        for target_index, target in enumerate(TARGET_VARIETIES):
            sae, other = _normal_flag_arrays(indicators, condition, target)
            target_only = other * (1 - sae)
            sae_only = sae * (1 - other)
            rng = np.random.default_rng(seed + condition_index * 10 + target_index)
            draws = rng.integers(0, sae.shape[0], size=(n_bootstrap, sae.shape[0]))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_draws = other[draws].mean(axis=(1, 2)) / sae[draws].mean(axis=(1, 2))
                discordance_draws = target_only[draws].sum(axis=(1, 2)) / sae_only[
                    draws
                ].sum(axis=(1, 2))
            ratio_lower, ratio_upper = _finite_interval(ratio_draws)
            disc_lower, disc_upper = _finite_interval(discordance_draws)
            n_target_only = int(target_only.sum())
            n_sae_only = int(sae_only.sum())
            rows.append(
                {
                    "condition": condition,
                    "target_variety": target,
                    "n_items": sae.shape[0],
                    "n_systems": sae.shape[1],
                    "n_item_system_pairs": int(sae.size),
                    "sae_rate": float(sae.mean()),
                    "target_rate": float(other.mean()),
                    "rate_ratio": _safe_ratio(float(other.mean()), float(sae.mean())),
                    "rate_ratio_ci_lower": float(ratio_lower),
                    "rate_ratio_ci_upper": float(ratio_upper),
                    "target_only_flagged": n_target_only,
                    "sae_only_flagged": n_sae_only,
                    "discordance_ratio": _safe_ratio(n_target_only, n_sae_only),
                    "discordance_ratio_ci_lower": float(disc_lower),
                    "discordance_ratio_ci_upper": float(disc_upper),
                }
            )
    return pd.DataFrame(rows)


def system_overflag_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    """Count systems whose target over-flagging rate exceeds their SAE rate."""
    summary = fairness_summary(frame)
    wide = summary.pivot(
        index=["system_id", "condition"],
        columns="variety",
        values="overflag_rate",
    )
    rows: list[dict[str, object]] = []
    for condition, subset in wide.groupby(level="condition"):
        for target in TARGET_VARIETIES:
            differences = subset[target] - subset["sae"]
            rows.append(
                {
                    "condition": condition,
                    "target_variety": target,
                    "n_systems": len(differences),
                    "n_systems_higher": int((differences > 0).sum()),
                    "n_systems_equal": int((differences == 0).sum()),
                    "n_systems_lower": int((differences < 0).sum()),
                    "min_difference_pp": 100 * float(differences.min()),
                    "max_difference_pp": 100 * float(differences.max()),
                }
            )
    return pd.DataFrame(rows)


def metric_names() -> Iterable[str]:
    return FAIRNESS_METRICS
