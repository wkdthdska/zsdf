#!/usr/bin/env python
"""WP28: fixed consensus macrophage-state remodeling during aromatase inhibition.

The WP23 signature is frozen. ARL11 is not used in the score. POETIC provides
the untreated control arm; GSE59515 provides independent paired consistency.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdata
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("poetic_legacy_io", ROOT / "scripts" / "111_analyze_poetic_endocrine.py")
poetic = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(poetic)

RAW_POETIC = ROOT / "data" / "raw" / "endocrine" / "POETIC"
RAW_GSE59515 = ROOT / "data" / "raw" / "endocrine" / "GSE59515"
PROCESSED = ROOT / "data" / "processed" / "wp28_treatment_remodeling"
TABLES = ROOT / "output" / "tables" / "wp28_treatment_remodeling"
FIGURES = ROOT / "figures" / "wp28_treatment_remodeling"
DOCS = ROOT / "docs"
for folder in (PROCESSED, TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

SIGNATURE = pd.read_csv(
    ROOT / "output" / "tables" / "wp23_consensus_state" / "consensus_macrophage_remodeling_signature.csv"
).gene.astype(str).head(30).tolist()
GENERIC = ["C1QA", "C1QB", "C1QC", "CD68", "LST1", "AIF1", "FCER1G", "TYROBP"]
PROLIFERATION = ["MKI67", "AURKA", "AURKB", "CCNB1", "CCNB2", "CDK1", "BUB1", "BUB1B",
                 "TOP2A", "UBE2C", "CDC20", "MCM2", "MCM4", "PCNA", "PLK1"]
SCORE_GENES = {
    "consensus_state": SIGNATURE,
    "generic_macrophage": GENERIC,
    "proliferation": PROLIFERATION,
}
poetic.SCORE_GENES = SCORE_GENES
SCORES = ["consensus_state", "state_residual", "generic_macrophage"]
LABELS = {
    "consensus_state": "Consensus remodelling state",
    "state_residual": "State residual (macrophage-adjusted)",
    "generic_macrophage": "Generic macrophage",
}


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def add_baseline_anchored_residual(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = scores.copy()
    baseline = scores.timepoint.eq("baseline")
    fit_data = scores.loc[baseline, ["consensus_state", "generic_macrophage"]].dropna()
    fit = sm.OLS(fit_data.consensus_state,
                 sm.add_constant(fit_data[["generic_macrophage"]], has_constant="add")).fit()
    predicted = fit.params["const"] + fit.params["generic_macrophage"] * scores.generic_macrophage
    residual = scores.consensus_state - predicted
    mean = residual.loc[baseline].mean()
    sd = residual.loc[baseline].std(ddof=1)
    scores["state_residual"] = (residual - mean) / sd
    return scores, {
        "baseline_n": int(len(fit_data)),
        "state_generic_slope": float(fit.params["generic_macrophage"]),
        "state_generic_r_squared": float(fit.rsquared),
        "baseline_residual_sd": float(sd),
    }


def prepare_poetic() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    annotation = poetic.load_annotation()
    expression, pheno = poetic.load_eset(RAW_POETIC / "GSE105777_eSet.Rdata")
    meta = poetic.parse_sample_metadata(pheno)
    genes = poetic.extract_target_genes(expression, annotation)
    scores, coverage = poetic.compute_scores(genes, meta, "GSE105777")
    scores, residual_audit = add_baseline_anchored_residual(scores)
    clinical = poetic.load_clinical()
    values = SCORES + ["proliferation"]
    wide = scores.reset_index(names="sample").pivot(index="clinical_key", columns="timepoint", values=values)
    wide.columns = [f"{score}_{timepoint}" for score, timepoint in wide.columns]
    wide = wide.reset_index().merge(clinical, on="clinical_key", how="inner", validate="one_to_one")
    for score in SCORES + ["proliferation"]:
        wide[f"delta_{score}"] = wide[f"{score}_post"] - wide[f"{score}_baseline"]
    return wide, coverage, residual_audit


def analyze_poetic(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    change_rows = []
    for score in SCORES:
        columns = [f"{score}_post", f"{score}_baseline", "proliferation_baseline", "her2", "ai"]
        frame = wide[columns].dropna().copy()
        formula = f"{score}_post ~ {score}_baseline + proliferation_baseline + C(her2) + ai"
        fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
        low, high = fit.conf_int().loc["ai"]
        change_rows.append({
            "analysis": "AI_vs_control_ANCOVA", "score": score, "n": int(fit.nobs),
            "ai_n": int(frame.ai.sum()), "control_n": int((frame.ai == 0).sum()),
            "beta_ai": fit.params["ai"], "ci_low": low, "ci_high": high,
            "p_value": fit.pvalues["ai"],
        })
    change = pd.DataFrame(change_rows)
    change["FDR"] = bh_adjust(change.p_value)

    # Explicit adjustment of state change for contemporaneous generic macrophage change.
    columns = ["delta_consensus_state", "consensus_state_baseline", "delta_generic_macrophage",
               "generic_macrophage_baseline", "proliferation_baseline", "her2", "ai"]
    frame = wide[columns].dropna().copy()
    fit = smf.ols(
        "delta_consensus_state ~ consensus_state_baseline + generic_macrophage_baseline + "
        "delta_generic_macrophage + proliferation_baseline + C(her2) + ai", data=frame
    ).fit(cov_type="HC3")
    low, high = fit.conf_int().loc["ai"]
    explicit = pd.DataFrame([{
        "analysis": "AI_state_change_adjusted_for_generic_change", "n": int(fit.nobs),
        "beta_ai": fit.params["ai"], "ci_low": low, "ci_high": high,
        "p_value": fit.pvalues["ai"],
    }])

    resistance_rows = []
    treated = wide.loc[wide.ai.eq(1)].copy()
    for score in SCORES:
        columns = ["log_post_ki67", "log_baseline_ki67", "her2", "proliferation_baseline",
                   f"{score}_baseline", f"delta_{score}"]
        frame = treated[columns].dropna().copy()
        formula = (f"log_post_ki67 ~ log_baseline_ki67 + proliferation_baseline + C(her2) + "
                   f"{score}_baseline + delta_{score}")
        fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
        term = f"delta_{score}"
        low, high = fit.conf_int().loc[term]
        resistance_rows.append({
            "score": score, "n": int(fit.nobs), "beta_delta": fit.params[term],
            "ci_low": low, "ci_high": high, "p_value": fit.pvalues[term],
        })
    resistance = pd.DataFrame(resistance_rows)
    resistance["FDR"] = bh_adjust(resistance.p_value)
    return change, explicit, resistance


def prepare_gse59515() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = rdata.read_rda(RAW_GSE59515 / "GSE59515_eSet.Rdata")
    eset = next(iter(loaded["gset"].values()))
    annotation = poetic.load_annotation()
    genes = poetic.extract_target_genes(eset.assayData["exprs"], annotation)
    pheno = eset.phenoData.data.copy()
    meta = pd.DataFrame({
        "sample": pheno["geo_accession"].astype(str),
        "patient": pheno["patient id:ch1"].astype(str),
        "response": pheno["clinical response:ch1"].astype(str),
        "time_text": pheno["time point:ch1"].astype(str),
    }).set_index("sample")
    meta["timepoint"] = np.select(
        [meta.time_text.str.contains("pre-treatment", case=False),
         meta.time_text.str.contains("2 wks", case=False)],
        ["baseline", "week2"], default="month3",
    )
    meta["responder"] = meta.response.eq("Responder").astype(int)
    common = genes.index.intersection(meta.index)
    genes, meta = genes.loc[common], meta.loc[common]
    baseline = meta.timepoint.eq("baseline")
    means = genes.loc[baseline].mean(axis=0)
    sds = genes.loc[baseline].std(axis=0, ddof=1).replace(0, np.nan)
    z = (genes - means) / sds
    scores = meta.copy()
    coverage_rows = []
    for score, gene_set in SCORE_GENES.items():
        present = [gene for gene in gene_set if gene in z.columns and z[gene].notna().any()]
        scores[score] = z[present].mean(axis=1)
        coverage_rows.append({"cohort": "GSE59515", "score": score, "expected": len(gene_set),
                              "present": len(present), "genes": ";".join(present)})
    scores, residual_audit = add_baseline_anchored_residual(scores)
    return scores, pd.DataFrame(coverage_rows), residual_audit


def analyze_gse59515(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = SCORES + ["proliferation"]
    wide = scores.reset_index(names="sample").pivot(
        index=["patient", "response", "responder"], columns="timepoint", values=values
    )
    wide.columns = [f"{score}_{timepoint}" for score, timepoint in wide.columns]
    wide = wide.reset_index()
    paired_rows, response_rows, adjusted_rows = [], [], []
    for score in SCORES:
        for timepoint in ["week2", "month3"]:
            delta = f"delta_{timepoint}_{score}"
            wide[delta] = wide[f"{score}_{timepoint}"] - wide[f"{score}_baseline"]
            values_delta = wide[delta].dropna()
            test = stats.wilcoxon(values_delta, alternative="two-sided", method="auto")
            paired_rows.append({
                "score": score, "timepoint": timepoint, "n": len(values_delta),
                "median_change": values_delta.median(), "p_value": test.pvalue,
                "positive_fraction": float((values_delta > 0).mean()),
            })
        metric = f"delta_week2_{score}"
        frame = wide[[metric, "responder"]].dropna()
        responder = frame.loc[frame.responder.eq(1), metric]
        nonresponder = frame.loc[frame.responder.eq(0), metric]
        test = stats.mannwhitneyu(responder, nonresponder, alternative="two-sided", method="exact")
        response_rows.append({
            "score": score, "n": len(frame), "responders": len(responder), "nonresponders": len(nonresponder),
            "median_difference_responder_minus_nonresponder": responder.median() - nonresponder.median(),
            "auc_responder": roc_auc_score(frame.responder, frame[metric]), "p_value": test.pvalue,
        })
        model_frame = wide[["responder", metric, f"{score}_baseline", "proliferation_baseline"]].dropna()
        try:
            fit = smf.glm(
                f"responder ~ {metric} + {score}_baseline + proliferation_baseline",
                data=model_frame, family=sm.families.Binomial(),
            ).fit()
            low, high = fit.conf_int().loc[metric]
            adjusted_rows.append({
                "score": score, "n": int(fit.nobs), "odds_ratio_week2_delta": np.exp(fit.params[metric]),
                "ci_low": np.exp(low), "ci_high": np.exp(high), "p_value": fit.pvalues[metric],
            })
        except Exception:
            adjusted_rows.append({"score": score, "n": len(model_frame), "odds_ratio_week2_delta": np.nan,
                                  "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan})
    paired = pd.DataFrame(paired_rows)
    paired["FDR"] = paired.groupby("timepoint").p_value.transform(bh_adjust)
    response = pd.DataFrame(response_rows)
    response["FDR"] = bh_adjust(response.p_value)
    adjusted = pd.DataFrame(adjusted_rows)
    adjusted["FDR"] = bh_adjust(adjusted.p_value.fillna(1.0))
    return wide, paired, response.merge(adjusted, on=["score", "n"], how="left", suffixes=("_unadjusted", "_adjusted"))


def make_figures(poetic_change: pd.DataFrame, gse_wide: pd.DataFrame) -> None:
    plot = poetic_change.copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    y = np.arange(len(plot))
    colors = np.where(plot.FDR < 0.05, "#9c2f55", "#8aa6b8")
    for position, (_, row), color in zip(y, plot.iterrows(), colors):
        ax.errorbar(row.beta_ai, position,
                    xerr=[[row.beta_ai - row.ci_low], [row.ci_high - row.beta_ai]],
                    fmt="o", color=color, ecolor=color, capsize=3, markersize=7)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(y, [LABELS[x] for x in plot.score])
    ax.set_xlabel("AI-specific post-treatment score difference (baseline-adjusted)")
    ax.set_title("POETIC: fixed macrophage-state remodelling")
    fig.savefig(FIGURES / "poetic_fixed_state_change.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "poetic_fixed_state_change.pdf", bbox_inches="tight")
    plt.close(fig)

    long_rows = []
    for score in SCORES:
        for timepoint in ["baseline", "week2", "month3"]:
            column = f"{score}_{timepoint}"
            for _, row in gse_wide[["patient", column]].dropna().iterrows():
                long_rows.append({"patient": row.patient, "score": LABELS[score],
                                  "timepoint": timepoint, "value": row[column]})
    long = pd.DataFrame(long_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, layout="constrained")
    for ax, score in zip(axes, LABELS.values()):
        frame = long[long.score.eq(score)]
        sns.boxplot(data=frame, x="timepoint", y="value", color="#b8d6d8", showfliers=False, ax=ax)
        sns.stripplot(data=frame, x="timepoint", y="value", color="#444444", size=3, alpha=0.6, ax=ax)
        ax.set_title(score)
        ax.set_xlabel("")
        ax.set_ylabel("Baseline-standardized score")
    fig.suptitle("GSE59515: fixed-state trajectory during letrozole")
    fig.savefig(FIGURES / "gse59515_fixed_state_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "gse59515_fixed_state_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    poetic_wide, poetic_coverage, poetic_residual = prepare_poetic()
    poetic_change, poetic_explicit, poetic_resistance = analyze_poetic(poetic_wide)
    gse_scores, gse_coverage, gse_residual = prepare_gse59515()
    gse_wide, gse_paired, gse_response = analyze_gse59515(gse_scores)

    poetic_wide.to_csv(PROCESSED / "poetic_paired_scores_clinical.csv", index=False)
    gse_scores.reset_index().to_csv(PROCESSED / "gse59515_sample_scores.csv", index=False)
    gse_wide.to_csv(PROCESSED / "gse59515_patient_changes.csv", index=False)
    pd.concat([poetic_coverage, gse_coverage], ignore_index=True).to_csv(TABLES / "gene_coverage.csv", index=False)
    poetic_change.to_csv(TABLES / "poetic_ai_vs_control_change.csv", index=False)
    poetic_explicit.to_csv(TABLES / "poetic_generic_change_adjusted_state.csv", index=False)
    poetic_resistance.to_csv(TABLES / "poetic_change_vs_residual_ki67.csv", index=False)
    gse_paired.to_csv(TABLES / "gse59515_paired_changes.csv", index=False)
    gse_response.to_csv(TABLES / "gse59515_change_vs_response.csv", index=False)
    pd.DataFrame([{"cohort": "POETIC", **poetic_residual},
                  {"cohort": "GSE59515", **gse_residual}]).to_csv(TABLES / "state_residual_audit.csv", index=False)
    make_figures(poetic_change, gse_wide)

    poetic_primary = poetic_change.loc[poetic_change.score.eq("state_residual")].iloc[0]
    poetic_explicit_primary = poetic_explicit.iloc[0]
    gse_week2 = gse_paired[(gse_paired.score == "state_residual") & (gse_paired.timepoint == "week2")].iloc[0]
    gse_month3 = gse_paired[(gse_paired.score == "state_residual") & (gse_paired.timepoint == "month3")].iloc[0]
    poetic_ki67 = poetic_resistance.loc[poetic_resistance.score.eq("state_residual")].iloc[0]
    gse_response_primary = gse_response.loc[gse_response.score.eq("state_residual")].iloc[0]
    summary = {
        "signature_frozen": True,
        "arl11_in_signature": False,
        "poetic_paired_patients_n": int(len(poetic_wide)),
        "poetic_ai_n": int(poetic_wide.ai.sum()),
        "poetic_control_n": int((poetic_wide.ai == 0).sum()),
        "poetic_state_residual_ai_beta": float(poetic_primary.beta_ai),
        "poetic_state_residual_ai_CI": [float(poetic_primary.ci_low), float(poetic_primary.ci_high)],
        "poetic_state_residual_ai_FDR": float(poetic_primary.FDR),
        "poetic_state_change_generic_delta_adjusted_beta": float(poetic_explicit_primary.beta_ai),
        "poetic_state_change_generic_delta_adjusted_CI": [float(poetic_explicit_primary.ci_low), float(poetic_explicit_primary.ci_high)],
        "poetic_state_change_generic_delta_adjusted_p": float(poetic_explicit_primary.p_value),
        "gse59515_patients_n": int(len(gse_wide)),
        "gse59515_week2_state_residual_median_change": float(gse_week2.median_change),
        "gse59515_week2_state_residual_FDR": float(gse_week2.FDR),
        "gse59515_month3_state_residual_median_change": float(gse_month3.median_change),
        "gse59515_month3_state_residual_FDR": float(gse_month3.FDR),
        "poetic_state_change_vs_residual_ki67_FDR": float(poetic_ki67.FDR),
        "gse59515_state_change_vs_response_FDR": float(gse_response_primary.FDR_unadjusted),
        "remodeling_replication_gate_pass": bool(
            poetic_primary.beta_ai > 0 and poetic_primary.FDR < 0.05
            and gse_week2.median_change > 0 and gse_week2.FDR < 0.05
        ),
        "adaptive_resistance_gate_pass": bool(
            poetic_ki67.beta_delta > 0 and poetic_ki67.FDR < 0.05
            and gse_response_primary.auc_responder < 0.5 and gse_response_primary.FDR_unadjusted < 0.05
        ),
    }
    (TABLES / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# WP28 fixed consensus-state remodeling during aromatase inhibition

## Design

The WP23 consensus signature was frozen and does not contain ARL11. Scores were anchored to pretreatment expression. A state residual was generated by fitting the consensus score on generic macrophage abundance in baseline samples and applying the frozen relation to all timepoints. POETIC estimates AI-specific change relative to untreated controls; GSE59515 provides independent paired consistency but no untreated control.

## Main results

- POETIC paired patients: {summary['poetic_paired_patients_n']} ({summary['poetic_ai_n']} AI; {summary['poetic_control_n']} control).
- POETIC AI-specific state-residual difference: beta={summary['poetic_state_residual_ai_beta']:.3f}, 95% CI {summary['poetic_state_residual_ai_CI'][0]:.3f} to {summary['poetic_state_residual_ai_CI'][1]:.3f}, FDR={summary['poetic_state_residual_ai_FDR']:.3g}.
- Explicit adjustment for contemporaneous generic-macrophage change: beta={summary['poetic_state_change_generic_delta_adjusted_beta']:.3f}, 95% CI {summary['poetic_state_change_generic_delta_adjusted_CI'][0]:.3f} to {summary['poetic_state_change_generic_delta_adjusted_CI'][1]:.3f}, P={summary['poetic_state_change_generic_delta_adjusted_p']:.3g}.
- GSE59515 week-2 state-residual change: median={summary['gse59515_week2_state_residual_median_change']:.3f}, FDR={summary['gse59515_week2_state_residual_FDR']:.3g}.
- GSE59515 month-3 state-residual change: median={summary['gse59515_month3_state_residual_median_change']:.3f}, FDR={summary['gse59515_month3_state_residual_FDR']:.3g}.
- Remodelling replication gate: {'passed' if summary['remodeling_replication_gate_pass'] else 'not passed'}.

## Resistance boundary

- POETIC state change versus residual Ki67 FDR: {summary['poetic_state_change_vs_residual_ki67_FDR']:.3g}.
- GSE59515 week-2 state change versus clinical response FDR: {summary['gse59515_state_change_vs_response_FDR']:.3g}.
- Adaptive-resistance gate: {'passed' if summary['adaptive_resistance_gate_pass'] else 'not passed'}.

Treatment-associated remodelling and treatment resistance are separate claims. A reproducible increase in the fixed macrophage state does not establish that the state causes or predicts endocrine resistance unless the change is independently associated with residual proliferation or response.

## Scope limitation

The locally available GSE130788 files contain CIBERSORT abundances rather than whole-transcriptome paired expression. They cannot calculate the frozen 30-gene state and are not presented as direct anti-HER2 state validation.
"""
    (DOCS / "wp28_fixed_state_treatment_remodeling.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nPOETIC change:\n", poetic_change.to_string(index=False))
    print("\nPOETIC resistance:\n", poetic_resistance.to_string(index=False))
    print("\nGSE59515 paired:\n", gse_paired.to_string(index=False))
    print("\nGSE59515 response:\n", gse_response.to_string(index=False))


if __name__ == "__main__":
    main()
