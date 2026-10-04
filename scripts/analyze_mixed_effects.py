#!/usr/bin/env python3
"""Fit hierarchical and fixed-model robustness analyses for study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Exchangeable

_FORMULA = (
    'correct_int ~ C(variety, Treatment(reference="sae")) * '
    'C(condition, Treatment(reference="direct"))'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accuracy",
        type=Path,
        default=Path("results/validated/accuracy_long.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/metrics/mixed_effects"),
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    frame = pd.read_csv(args.accuracy)
    frame = frame[frame["valid"]].copy()
    frame["correct_int"] = frame["correct"].astype(int)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hierarchical_model = BinomialBayesMixedGLM.from_formula(
        _FORMULA,
        {"item": "0 + C(id)", "system": "0 + C(system_id)"},
        frame,
        vcp_p=1,
        fe_p=2,
    )
    hierarchical = hierarchical_model.fit_vb(
        fit_method="BFGS",
        minim_opts={"maxiter": 1_000},
        scale_fe=True,
        rng=np.random.default_rng(args.seed),
    )
    fixed_rows = []
    for name, mean, standard_deviation in zip(
        hierarchical.model.exog_names,
        hierarchical.fe_mean,
        hierarchical.fe_sd,
        strict=True,
    ):
        fixed_rows.append(
            {
                "term": name,
                "posterior_mean_log_odds": mean,
                "posterior_sd": standard_deviation,
                "credible_lower": mean - 1.96 * standard_deviation,
                "credible_upper": mean + 1.96 * standard_deviation,
                "odds_ratio": math.exp(mean),
            }
        )
    pd.DataFrame(fixed_rows).to_csv(
        args.out_dir / "hierarchical_fixed_effects.csv",
        index=False,
    )

    variance_rows = []
    for name, log_sd, posterior_sd in zip(
        hierarchical.model.vcp_names,
        hierarchical.vcp_mean,
        hierarchical.vcp_sd,
        strict=True,
    ):
        variance_rows.append(
            {
                "component": name,
                "posterior_mean_log_sd": log_sd,
                "posterior_sd_log_sd": posterior_sd,
                "random_effect_sd": math.exp(log_sd),
                "sd_lower": math.exp(log_sd - 1.96 * posterior_sd),
                "sd_upper": math.exp(log_sd + 1.96 * posterior_sd),
            }
        )
    pd.DataFrame(variance_rows).to_csv(
        args.out_dir / "hierarchical_variance_components.csv",
        index=False,
    )

    gee_formula = (
        'correct_int ~ C(variety, Treatment(reference="sae")) * '
        'C(condition, Treatment(reference="direct")) + C(system_id)'
    )
    gee = smf.gee(
        gee_formula,
        groups="id",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    ).fit()
    confidence = gee.conf_int()
    gee_table = pd.DataFrame(
        {
            "term": gee.params.index,
            "coefficient": gee.params.to_numpy(),
            "standard_error": gee.bse.to_numpy(),
            "z": gee.tvalues.to_numpy(),
            "p_value": gee.pvalues.to_numpy(),
            "ci_lower": confidence[0].to_numpy(),
            "ci_upper": confidence[1].to_numpy(),
            "odds_ratio": np.exp(gee.params.to_numpy()),
        }
    )
    gee_table.to_csv(args.out_dir / "gee_model_fixed_effects.csv", index=False)

    diagnostics = {
        "n_rows": len(frame),
        "n_items": int(frame["id"].nunique()),
        "n_models": int(frame["model_id"].nunique()),
        "n_systems": int(frame["system_id"].nunique()),
        "hierarchical_formula": _FORMULA,
        "hierarchical_optimizer_success": bool(hierarchical.optim_retvals.get("success")),
        "hierarchical_optimizer_message": str(hierarchical.optim_retvals.get("message")),
        "hierarchical_iterations": int(hierarchical.optim_retvals.get("nit", 0)),
        "hierarchical_objective": float(hierarchical.optim_retvals.get("fun", float("nan"))),
        "gee_formula": gee_formula,
        "gee_converged": bool(gee.converged),
    }
    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "hierarchical_summary.txt").write_text(
        str(hierarchical.summary()) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "gee_summary.txt").write_text(
        str(gee.summary()) + "\n",
        encoding="utf-8",
    )
    for path in sorted(args.out_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
