#!/usr/bin/env python
"""Random-effects meta-analysis, figures and report for WP27."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t

TABLES = ROOT / "output" / "tables" / "wp27_consensus_prognosis"
FIGURES = ROOT / "figures" / "wp27_consensus_prognosis"
DOCS = ROOT / "docs"
FIGURES.mkdir(parents=True, exist_ok=True)


def random_effects(data: pd.DataFrame, score_type: str, model: str) -> dict:
    yi = data.logHR.to_numpy(float)
    vi = data.se.to_numpy(float) ** 2
    wi = 1 / vi
    fixed = np.sum(wi * yi) / np.sum(wi)
    q = float(np.sum(wi * (yi - fixed) ** 2))
    df = len(yi) - 1
    c_value = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
    tau2 = max(0.0, (q - df) / c_value) if c_value > 0 else 0.0
    wr = 1 / (vi + tau2)
    pooled = float(np.sum(wr * yi) / np.sum(wr))
    se_normal = math.sqrt(1 / np.sum(wr))
    scale_hk = max(1.0, float(np.sum(wr * (yi - pooled) ** 2) / df)) if df > 0 else 1.0
    se_mkh = math.sqrt(scale_hk / np.sum(wr))
    critical_t = t.ppf(0.975, df)
    return {
        "score_type": score_type, "model": model, "cohorts_n": len(data),
        "logHR": pooled, "HR": math.exp(pooled),
        "CI_low_normal": math.exp(pooled - 1.96 * se_normal),
        "CI_high_normal": math.exp(pooled + 1.96 * se_normal),
        "p_value_normal": 2 * norm.sf(abs(pooled / se_normal)),
        "CI_low_mKH": math.exp(pooled - critical_t * se_mkh),
        "CI_high_mKH": math.exp(pooled + critical_t * se_mkh),
        "p_value_mKH": 2 * t.sf(abs(pooled / se_mkh), df),
        "tau2_DL": tau2, "Q": q, "Q_p_value": chi2.sf(q, df),
        "I2_percent": max(0.0, (q - df) / q) * 100 if q > 0 else 0.0,
        "direction_concordant": bool(np.all(yi > 0) or np.all(yi < 0)),
    }


def forest_plot(results: pd.DataFrame, meta: pd.DataFrame) -> None:
    data = results[(results.score_type == "consensus") & (results.model == "clinical")].copy()
    pooled = meta[(meta.score_type == "consensus") & (meta.model == "clinical")].iloc[0]
    labels = data.cohort.tolist() + ["Random-effects meta (mKH)"]
    hr = data.HR.tolist() + [pooled.HR]
    low = data.CI_low.tolist() + [pooled.CI_low_mKH]
    high = data.CI_high.tolist() + [pooled.CI_high_mKH]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    ax.axvline(1, color="grey", ls="--", lw=1)
    ax.errorbar(hr, y, xerr=[np.asarray(hr) - np.asarray(low), np.asarray(high) - np.asarray(hr)],
                fmt="o", color="#8b2f5d", ecolor="#8b2f5d", capsize=3)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("Overall-survival HR per 1 SD consensus-state residual")
    ax.set_title("WP27 clinically adjusted prognostic validation")
    fig.savefig(FIGURES / "wp27_clinical_adjusted_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "wp27_clinical_adjusted_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = pd.read_csv(TABLES / "cohort_cox_models.csv")
    increment = pd.read_csv(TABLES / "incremental_concordance.csv")
    nonlinear = pd.read_csv(TABLES / "nonlinearity_tests.csv")
    ph_robust = pd.read_csv(TABLES / "ph_robust_sensitivity.csv")
    coverage = pd.read_csv(TABLES / "gene_coverage.csv")
    meta_rows = []
    for score_type in ["consensus", "common"]:
        for model in ["unadjusted", "abundance_adjusted", "clinical"]:
            subset = results[(results.score_type == score_type) & (results.model == model)]
            meta_rows.append(random_effects(subset, score_type, model))
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(TABLES / "random_effects_meta.csv", index=False)
    robust_meta_rows = []
    for score_type in ["consensus", "common"]:
        robust_meta_rows.append(random_effects(ph_robust[ph_robust.score_type == score_type], score_type, "ph_robust_clinical"))
    robust_meta = pd.DataFrame(robust_meta_rows)
    robust_meta.to_csv(TABLES / "ph_robust_random_effects_meta.csv", index=False)
    forest_plot(results, meta)

    primary = meta[(meta.score_type == "consensus") & (meta.model == "clinical")].iloc[0]
    abundance = meta[(meta.score_type == "consensus") & (meta.model == "abundance_adjusted")].iloc[0]
    common = meta[(meta.score_type == "common") & (meta.model == "clinical")].iloc[0]
    robust_primary = robust_meta[robust_meta.score_type == "consensus"].iloc[0]
    clinical_rows = results[(results.score_type == "consensus") & (results.model == "clinical")]
    summary = {
        "endpoint": "overall survival",
        "primary_predictor": "frozen WP23 consensus signature residualized on generic macrophage abundance",
        "primary_meta_HR": float(primary.HR),
        "primary_meta_CI_mKH": [float(primary.CI_low_mKH), float(primary.CI_high_mKH)],
        "primary_meta_p_mKH": float(primary.p_value_mKH),
        "primary_meta_CI_normal": [float(primary.CI_low_normal), float(primary.CI_high_normal)],
        "primary_meta_p_normal": float(primary.p_value_normal),
        "primary_meta_I2_percent": float(primary.I2_percent),
        "primary_direction_concordant": bool(primary.direction_concordant),
        "abundance_adjusted_meta_HR": float(abundance.HR),
        "abundance_adjusted_meta_p_mKH": float(abundance.p_value_mKH),
        "common_gene_clinical_meta_HR": float(common.HR),
        "common_gene_clinical_meta_p_mKH": float(common.p_value_mKH),
        "ph_robust_meta_HR": float(robust_primary.HR),
        "ph_robust_meta_CI_mKH": [float(robust_primary.CI_low_mKH), float(robust_primary.CI_high_mKH)],
        "ph_robust_meta_p_mKH": float(robust_primary.p_value_mKH),
        "cohort_PH_violations_p_lt_0_05": clinical_rows.loc[clinical_rows.ph_p_value < 0.05, "cohort"].tolist(),
        "clinical_models_min_events_per_parameter": float(clinical_rows.events_per_parameter.min()),
        "upgrade_gate_pass": bool(primary.p_value_mKH < 0.05 and primary.direction_concordant and common.p_value_mKH < 0.05),
    }
    (TABLES / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cohort_lines = []
    for _, row in clinical_rows.iterrows():
        cohort_lines.append(
            f"| {row.cohort} | {int(row.n)} | {int(row.events)} | {row.HR:.3f} | "
            f"{row.CI_low:.3f}-{row.CI_high:.3f} | {row.p_value:.3g} | {row.ph_p_value:.3g} |"
        )
    common_genes_n = int(coverage.loc[coverage.gene_set == "common_signature", "requested_n"].iloc[0])
    report = f"""# WP27 fixed consensus-state prognostic validation

## Design

The WP23 30-gene consensus signature was frozen before survival analysis. Scores were computed continuously within cohort/platform, residualized on an eight-gene generic macrophage score, and analysed without outcome-derived cut-offs. The primary endpoint was overall survival. Clinical Cox models were stratified by available breast-cancer subtype and adjusted for available age, stage/grade, tumour size and nodal variables. Proportional hazards were assessed with `cox.zph`. Meta-analysis reports modified Knapp-Hartung inference because only three cohorts were available.

## Clinically adjusted cohort results

| Cohort | N | Events | HR per SD | 95% CI | P | PH P |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cohort_lines)}

## Primary meta-analysis

- Modified Knapp-Hartung random-effects HR: {primary.HR:.3f}.
- 95% CI: {primary.CI_low_mKH:.3f}-{primary.CI_high_mKH:.3f}.
- P={primary.p_value_mKH:.3g}; I2={primary.I2_percent:.1f}%.
- Cohort directions concordant: {'yes' if primary.direction_concordant else 'no'}.
- Upgrade gate: {'passed' if summary['upgrade_gate_pass'] else 'not passed'}.

The normal-approximation random-effects interval is also retained in the machine-readable table, but the modified Knapp-Hartung result is the primary small-k inference.

## Sensitivity and diagnostics

- Cross-platform common signature: {common_genes_n} genes.
- Common-gene clinical meta HR={common.HR:.3f}, modified Knapp-Hartung P={common.p_value_mKH:.3g}.
- Cohorts with state-specific PH test P<0.05: {', '.join(summary['cohort_PH_violations_p_lt_0_05']) or 'none'}.
- PH-robust sensitivity meta HR={robust_primary.HR:.3f}, 95% CI {robust_primary.CI_low_mKH:.3f}-{robust_primary.CI_high_mKH:.3f}, modified Knapp-Hartung P={robust_primary.p_value_mKH:.3g}.
- Minimum events per fitted parameter among primary clinical models: {summary['clinical_models_min_events_per_parameter']:.1f}.
- Nonlinearity and incremental concordance results are provided as secondary diagnostics; they do not replace the prespecified continuous Cox estimand.

## Interpretation rule

Only a significant, directionally concordant primary meta-analysis supported by the common-gene sensitivity analysis qualifies the state as an independent prognostic marker. Otherwise it remains a reproducible macrophage biological state whose survival association overlaps standard clinicopathologic context.
"""
    (DOCS / "wp27_consensus_state_prognostic_validation.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nClinical cohort models:\n", clinical_rows.to_string(index=False))
    print("\nMeta-analysis:\n", meta.to_string(index=False))
    print("\nIncremental concordance:\n", increment.to_string(index=False))
    print("\nNonlinearity:\n", nonlinear.to_string(index=False))
    print("\nPH-robust sensitivity:\n", ph_robust.to_string(index=False))
    print("\nPH-robust meta:\n", robust_meta.to_string(index=False))


if __name__ == "__main__":
    main()
