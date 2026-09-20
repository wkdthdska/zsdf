#!/usr/bin/env python
"""WP18: donor-level ARL11-specific macrophage state analysis.

The discovery-derived 30-gene signature is kept fixed.  Analyses are performed
within cohort and only cohort-specific effect estimates are meta-analysed.
"""

from __future__ import annotations

import json
import math
import re
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm, pearsonr

TABLE_DIR = ROOT / "output" / "tables" / "wp18_state_residual"
FIG_DIR = ROOT / "figures" / "wp18_state_residual"
PROCESSED_DIR = ROOT / "data" / "processed" / "macrophage_state_residual"
for directory in (TABLE_DIR, FIG_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260818

SIGNATURE_PATH = (ROOT / "output" / "tables" / "macrophage_reclustering" /
                  "gse176078_arl11_high_cluster_signature.csv")

GENERIC_MACROPHAGE = ["C1QA", "C1QB", "C1QC", "CD68", "LST1", "AIF1", "FCER1G", "TYROBP"]

# Prespecified before running WP18.  MAPK sets are transcriptional-response
# proxies and must not be interpreted as direct measurements of phosphorylation.
PATHWAYS = {
    "lipid_metabolism": ["LPL", "LIPA", "APOE", "APOC1", "FABP5", "FABP3", "PPARG", "CYP27A1",
                         "ABCA1", "ABCG1", "CD36", "PLIN2", "SOAT1", "NCEH1", "MGLL", "PLA2G7"],
    "phagolysosome": ["CTSB", "CTSD", "CTSL", "CTSK", "CTSZ", "LGMN", "ACP5", "GPNMB", "LAMP1",
                      "LAMP2", "ATP6V1A", "ATP6V0D1", "HEXA", "HEXB", "GM2A", "FUCA1"],
    "ECM_remodelling": ["SPP1", "MMP9", "MMP12", "CTSK", "CTSB", "CTSD", "LGMN", "COL6A1",
                        "COL6A2", "FN1", "VCAN", "TGFBI", "SEMA3C"],
    "T_cell_exclusion_like": ["SPP1", "TGFB1", "VEGFA", "CXCL12", "CCL2", "CCL18", "LGALS9",
                              "CD274", "VSIR", "TREM2", "MARCO", "APOE"],
    "antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA", "B2M", "TAP1"],
    "efferocytosis": ["MERTK", "AXL", "TYRO3", "GAS6", "PROS1", "TIMD4", "STAB1", "MARCO",
                      "CD36", "LRP1", "MFGE8", "APOE", "ABCA1"],
    "MAPK_ERK_immediate_early": ["DUSP1", "DUSP2", "DUSP4", "DUSP5", "DUSP6", "ETV4", "ETV5", "FOS",
                                 "FOSB", "JUN", "JUNB", "EGR1", "EGR2", "IER2", "IER3", "SPRY1", "SPRY2", "ATF3"],
    "p38_stress_response": ["DUSP1", "DUSP2", "FOS", "JUN", "JUNB", "ATF3", "EGR1", "HSPB1", "NFKBIA",
                             "TNFAIP3", "IL1B", "TNF"],
    "STING_typeI_IFN_response": ["TMEM173", "TBK1", "IRF3", "IRF7", "STAT1", "STAT2", "ISG15", "IFIT1",
                                 "IFIT2", "IFIT3", "MX1", "OAS1", "OAS2", "IFI6", "IFI27", "CXCL10"],
    "dysfunction_tolerization": ["SPP1", "GPNMB", "TREM2", "APOE", "APOC1", "LGALS9", "VSIR", "CD274",
                                  "MARCO", "MSR1", "IL10", "TGFB1", "CCL18", "CD163"],
}

PRIMARY_PATHWAYS = ["lipid_metabolism", "phagolysosome", "ECM_remodelling",
                    "T_cell_exclusion_like", "antigen_presentation"]

COHORTS = {
    "GSE176078": {
        "pseudobulk": ROOT / "data" / "processed" / "gse176078_macrophage_pseudobulk_logcpm.tsv.gz",
        "mac_h5ad": ROOT / "data" / "processed" / "macrophage_reclustering" / "gse176078_macrophage_reclustered.h5ad",
        "source_h5ad": ROOT / "data" / "raw" / "GSE176078_cellxgene_22a27631.h5ad",
        "high_cluster": "7", "sampling_frame": "macrophage/all_cells",
    },
    "GSE161529": {
        "pseudobulk": ROOT / "data" / "processed" / "gse161529_macrophage_pseudobulk_logcpm.tsv.gz",
        "mac_h5ad": ROOT / "data" / "processed" / "macrophage_reclustering" / "gse161529_macrophage_reclustered.h5ad",
        "source_h5ad": ROOT / "data" / "raw" / "breast_atlas_immune.h5ad",
        "high_cluster": "6", "sampling_frame": "macrophage/pal_2021_immune_cells",
    },
}


def normalize_patient(value: object) -> str:
    return re.sub(r"(pal_Patient 0029)-[79]C$", r"\1", str(value))


def bh(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    values = p[valid]
    if not len(values):
        return result
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    restored = np.empty(len(values))
    restored[order] = np.minimum(adjusted, 1)
    result[np.where(valid)[0]] = restored
    return result


def zmean(expression: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, list[str]]:
    present = [g for g in genes if g in expression.columns and expression[g].std(ddof=0) > 0]
    if not present:
        return pd.Series(np.nan, index=expression.index), []
    z = expression[present].apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)
    return z.mean(axis=1), present


def rank_residual(values: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    frame = pd.concat([values.rename("y"), covariates], axis=1).dropna()
    ranked = frame.rank(method="average")
    design = sm.add_constant(ranked.drop(columns="y"), has_constant="add")
    fit = sm.OLS(ranked["y"], design).fit()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[frame.index] = fit.resid
    return result


def linear_residual(values: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    frame = pd.concat([values.rename("y"), covariates], axis=1).dropna()
    design = sm.add_constant(frame.drop(columns="y"), has_constant="add")
    fit = sm.OLS(frame["y"], design).fit()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[frame.index] = fit.resid
    return result


def source_fraction(cohort: str, config: dict) -> pd.DataFrame:
    source = ad.read_h5ad(config["source_h5ad"], backed="r")
    obs = source.obs.copy()
    source.file.close()
    obs["patient_id"] = obs["donor_id"].map(normalize_patient)
    if cohort == "GSE176078":
        obs["is_macrophage"] = obs["celltype_minor"].astype(str).eq("Macrophage")
    else:
        obs = obs[obs["batch"].astype(str).eq("pal_2021")].copy()
        obs["is_macrophage"] = obs["cell_type"].astype(str).eq("macrophage")
    result = obs.groupby("patient_id").agg(all_sampled_cell_n=("patient_id", "size"),
                                            sampled_macrophage_n=("is_macrophage", "sum"))
    result["macrophage_fraction"] = result["sampled_macrophage_n"] / result["all_sampled_cell_n"]
    result["sampling_frame"] = config["sampling_frame"]
    return result


def build_scores(cohort: str, config: dict, signature: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = pd.read_csv(config["pseudobulk"], sep="\t").set_index("patient_id")
    expression.index = expression.index.map(normalize_patient)
    mac = ad.read_h5ad(config["mac_h5ad"], backed="r")
    obs = mac.obs.copy()
    mac.file.close()
    obs["patient_id"] = obs["patient_id"].map(normalize_patient)
    donor = obs.groupby("patient_id").agg(
        macrophage_cell_n=("patient_id", "size"),
        high_cluster_fraction=("cluster", lambda x: x.astype(str).eq(config["high_cluster"]).mean()),
        ARL11_detection_fraction=("arl11_detected", "mean"),
        ARL11_mean_log1p10k=("arl11_log1p10k", "mean"),
        median_log1p_total_counts=("log1p_total_counts", "median"),
        median_log1p_n_genes=("log1p_n_genes_by_counts", "median"),
    )
    donor = donor.join(source_fraction(cohort, config), how="left")
    scores = donor.reindex(expression.index).copy()
    scores["ARL11_logCPM"] = expression["ARL11"]
    scores["signature_score"], signature_present = zmean(expression, signature)
    scores["generic_macrophage_score"], generic_present = zmean(expression, GENERIC_MACROPHAGE)
    coverage = [{"cohort": cohort, "gene_set": "fixed_ARL11_TAM_signature", "analysis": "fixed",
                 "requested_n": len(signature), "present_n": len(signature_present),
                 "present_genes": ";".join(signature_present)},
                {"cohort": cohort, "gene_set": "generic_macrophage", "analysis": "fixed",
                 "requested_n": len(GENERIC_MACROPHAGE), "present_n": len(generic_present),
                 "present_genes": ";".join(generic_present)}]
    signature_set = set(signature)
    for pathway, genes in PATHWAYS.items():
        nonoverlap = [g for g in genes if g not in signature_set]
        scores[pathway], present = zmean(expression, genes)
        scores[f"{pathway}__drop_signature_overlap"], present_nonoverlap = zmean(expression, nonoverlap)
        coverage.extend([
            {"cohort": cohort, "gene_set": pathway, "analysis": "full", "requested_n": len(genes),
             "present_n": len(present), "present_genes": ";".join(present)},
            {"cohort": cohort, "gene_set": pathway, "analysis": "drop_signature_overlap",
             "requested_n": len(nonoverlap), "present_n": len(present_nonoverlap),
             "present_genes": ";".join(present_nonoverlap)},
        ])

    # Collapse depth and detected-gene metrics to one QC dimension so the
    # 25/29-donor models remain parsimonious.
    tech = scores[["median_log1p_total_counts", "median_log1p_n_genes"]]
    tech = tech.apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    _, _, vt = np.linalg.svd(tech.to_numpy(), full_matrices=False)
    loading = vt[0]
    if loading.sum() < 0:
        loading = -loading
    scores["technical_QC_PC1"] = tech.to_numpy() @ loading
    covariates = scores[["generic_macrophage_score", "macrophage_fraction", "technical_QC_PC1"]]
    scores["state_residual"] = linear_residual(scores["signature_score"], covariates)
    detected = scores["ARL11_detection_fraction"] * scores["macrophage_cell_n"]
    prevalence = (detected + 0.5) / (scores["macrophage_cell_n"] + 1.0)
    scores["ARL11_prevalence_logit"] = np.log(prevalence / (1 - prevalence))
    scores["ARL11_prevalence_residual"] = linear_residual(scores["ARL11_prevalence_logit"], covariates)
    fraction = (scores["high_cluster_fraction"] * scores["macrophage_cell_n"] + 0.5) / (scores["macrophage_cell_n"] + 1)
    scores["high_cluster_logit_residual"] = linear_residual(np.log(fraction / (1 - fraction)), covariates)
    scores["cohort"] = cohort
    scores.index.name = "patient_id"
    return scores.reset_index(), pd.DataFrame(coverage)


def partial_associations(cohort: str, scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.set_index("patient_id")
    covariates = scores[["generic_macrophage_score", "macrophage_fraction", "technical_QC_PC1"]]
    predictors = ["state_residual", "ARL11_prevalence_residual", "high_cluster_logit_residual"]
    rows = []
    for predictor in predictors:
        x = rank_residual(scores[predictor], covariates)
        for pathway in PATHWAYS:
            outcome_col = f"{pathway}__drop_signature_overlap"
            y = rank_residual(scores[outcome_col], covariates)
            frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
            rho, p = pearsonr(frame["x"], frame["y"])
            # HC3 standardized linear-model sensitivity using unranked variables.
            lm_frame = pd.concat([scores[[predictor, outcome_col]], covariates], axis=1).dropna()
            standardized = lm_frame.apply(lambda v: (v - v.mean()) / v.std(ddof=0))
            design = sm.add_constant(standardized[[predictor] + list(covariates.columns)], has_constant="add")
            fit = sm.OLS(standardized[outcome_col], design).fit(cov_type="HC3")
            rows.append({"cohort": cohort, "predictor": predictor, "pathway": pathway,
                         "family": "primary" if pathway in PRIMARY_PATHWAYS else "secondary",
                         "analysis": "drop_signature_overlap", "n": len(frame), "partial_rho": rho,
                         "p_value": p, "hc3_beta": fit.params[predictor],
                         "hc3_se": fit.bse[predictor], "hc3_p_value": fit.pvalues[predictor]})
    result = pd.DataFrame(rows)
    result["fdr_bh"] = result.groupby(["cohort", "predictor", "family"])["p_value"].transform(bh)
    result["hc3_fdr_bh"] = result.groupby(["cohort", "predictor", "family"])["hc3_p_value"].transform(bh)
    return result


def random_effects_meta(associations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (predictor, pathway, family), group in associations.groupby(["predictor", "pathway", "family"]):
        valid = group[(group["n"] > 3) & group["partial_rho"].notna()].copy()
        z = np.arctanh(np.clip(valid["partial_rho"].to_numpy(), -0.999999, 0.999999))
        variance = 1 / (valid["n"].to_numpy() - 3)
        fixed_w = 1 / variance
        fixed_z = np.sum(fixed_w * z) / np.sum(fixed_w)
        q = float(np.sum(fixed_w * (z - fixed_z) ** 2))
        df = len(z) - 1
        c_value = np.sum(fixed_w) - np.sum(fixed_w ** 2) / np.sum(fixed_w)
        tau2 = max(0.0, (q - df) / c_value) if df > 0 and c_value > 0 else 0.0
        weights = 1 / (variance + tau2)
        pooled_z = np.sum(weights * z) / np.sum(weights)
        se = math.sqrt(1 / np.sum(weights))
        p = 2 * norm.sf(abs(pooled_z / se))
        i2 = max(0.0, (q - df) / q) * 100 if q > 0 and df > 0 else 0.0
        rows.append({"predictor": predictor, "pathway": pathway, "family": family,
                     "cohorts_n": len(valid), "meta_partial_rho": math.tanh(pooled_z),
                     "ci_low": math.tanh(pooled_z - 1.96 * se),
                     "ci_high": math.tanh(pooled_z + 1.96 * se), "p_value": p,
                     "tau2_DL": tau2, "Q": q, "Q_df": df,
                     "Q_p_value": chi2.sf(q, df) if df > 0 else np.nan,
                     "I2_percent": i2,
                     "direction_concordant": bool(valid["partial_rho"].gt(0).all() or valid["partial_rho"].lt(0).all())})
    result = pd.DataFrame(rows)
    result["fdr_bh"] = result.groupby(["predictor", "family"])["p_value"].transform(bh)
    return result.sort_values(["predictor", "family", "fdr_bh", "p_value"])


def nonlinear_analysis(score_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    for cohort, data in score_tables.items():
        frame = data.set_index("patient_id").copy()
        covariates = frame[["generic_macrophage_score", "macrophage_fraction", "technical_QC_PC1"]]
        for outcome in ["MAPK_ERK_immediate_early", "dysfunction_tolerization"]:
            col = f"{outcome}__drop_signature_overlap"
            frame[f"{outcome}_resid"] = linear_residual(frame[col], covariates)
        for col in ["state_residual", "ARL11_prevalence_residual",
                    "MAPK_ERK_immediate_early_resid", "dysfunction_tolerization_resid"]:
            frame[col] = (frame[col] - frame[col].mean()) / frame[col].std(ddof=0)
        frame["cohort"] = cohort
        pieces.append(frame.reset_index())
    pooled = pd.concat(pieces, ignore_index=True)
    rows = []
    for predictor in ["state_residual", "ARL11_prevalence_residual"]:
        for outcome in ["MAPK_ERK_immediate_early", "dysfunction_tolerization"]:
            ycol = f"{outcome}_resid"
            frame = pooled[[predictor, ycol, "cohort"]].dropna().copy()
            frame["x2"] = frame[predictor] ** 2
            cohort_dummy = pd.get_dummies(frame["cohort"], drop_first=True, dtype=float)
            design = pd.concat([frame[[predictor, "x2"]], cohort_dummy], axis=1)
            fit = sm.OLS(frame[ycol], sm.add_constant(design, has_constant="add")).fit(cov_type="HC3")
            rows.append({"predictor": predictor, "outcome": outcome, "n": len(frame),
                         "linear_beta": fit.params[predictor], "linear_p": fit.pvalues[predictor],
                         "quadratic_beta": fit.params["x2"], "quadratic_p": fit.pvalues["x2"],
                         "aic": fit.aic})
    result = pd.DataFrame(rows)
    result["quadratic_fdr_bh"] = bh(result["quadratic_p"])
    return pooled, result


def make_figures(associations: pd.DataFrame, meta: pd.DataFrame, pooled: pd.DataFrame) -> None:
    pathways = list(PATHWAYS)
    predictors = ["state_residual", "ARL11_prevalence_residual"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), sharey=True, layout="constrained")
    image = None
    for ax, predictor in zip(axes, predictors):
        subset = associations[associations["predictor"].eq(predictor)]
        matrix = subset.pivot(index="pathway", columns="cohort", values="partial_rho").reindex(pathways)
        matrix["Random-effects meta"] = meta[meta["predictor"].eq(predictor)].set_index("pathway")["meta_partial_rho"]
        image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pathways)), pathways)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8)
        ax.set_title(predictor.replace("_", " "))
    fig.colorbar(image, ax=axes, shrink=0.75, label="Partial Spearman rho")
    fig.suptitle("ARL11 macrophage state after abundance and QC adjustment")
    fig.savefig(FIG_DIR / "wp18_partial_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "wp18_partial_correlation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for row, predictor in enumerate(predictors):
        for col, outcome in enumerate(["MAPK_ERK_immediate_early", "dysfunction_tolerization"]):
            ax = axes[row, col]
            ycol = f"{outcome}_resid"
            for cohort, group in pooled.groupby("cohort"):
                ax.scatter(group[predictor], group[ycol], s=38, alpha=0.75, label=cohort)
            frame = pooled[[predictor, ycol]].dropna().sort_values(predictor)
            fit = np.polyfit(frame[predictor], frame[ycol], 2)
            xx = np.linspace(frame[predictor].min(), frame[predictor].max(), 150)
            ax.plot(xx, np.polyval(fit, xx), color="black", lw=1.5)
            ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
            ax.set_xlabel(predictor.replace("_", " ")); ax.set_ylabel(outcome.replace("_", " "))
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    fig.suptitle("Exploratory quadratic rheostat analysis (cohort-standardized residuals)")
    fig.savefig(FIG_DIR / "wp18_rheostat_quadratic.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "wp18_rheostat_quadratic.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    signature = pd.read_csv(SIGNATURE_PATH)["names"].drop_duplicates().tolist()
    scores_by_cohort, coverage_tables, association_tables = {}, [], []
    for cohort, config in COHORTS.items():
        scores, coverage = build_scores(cohort, config, signature)
        scores_by_cohort[cohort] = scores
        coverage_tables.append(coverage)
        association_tables.append(partial_associations(cohort, scores))
        scores.to_csv(PROCESSED_DIR / f"{cohort.lower()}_donor_state_residuals.csv", index=False)
    coverage = pd.concat(coverage_tables, ignore_index=True)
    associations = pd.concat(association_tables, ignore_index=True)
    meta = random_effects_meta(associations)
    pooled, nonlinear = nonlinear_analysis(scores_by_cohort)
    coverage.to_csv(TABLE_DIR / "gene_set_coverage.csv", index=False)
    associations.to_csv(TABLE_DIR / "cohort_partial_associations.csv", index=False)
    meta.to_csv(TABLE_DIR / "random_effects_meta.csv", index=False)
    nonlinear.to_csv(TABLE_DIR / "rheostat_quadratic_tests.csv", index=False)
    pooled.to_csv(PROCESSED_DIR / "pooled_standardized_residuals.csv", index=False)
    make_figures(associations, meta, pooled)

    primary_state = meta[(meta["predictor"].eq("state_residual")) & (meta["family"].eq("primary"))]
    gate = primary_state[(primary_state["fdr_bh"] < 0.05) & primary_state["direction_concordant"]]
    summary = {
        "analysis_date": "2026-08-18", "seed": SEED,
        "cohort_donor_n": {cohort: int(len(data)) for cohort, data in scores_by_cohort.items()},
        "primary_covariates": ["generic_macrophage_score", "macrophage_fraction", "technical_QC_PC1"],
        "primary_endpoint_analysis": "partial Spearman; pathway genes overlapping fixed signature removed",
        "meta_model": "DerSimonian-Laird random effects on Fisher-z partial correlations",
        "primary_state_fdr_concordant_n": int(len(gate)),
        "primary_state_fdr_concordant_pathways": gate["pathway"].tolist(),
        "upgrade_gate_pass": bool(len(gate) >= 2),
        "nonlinear_fdr_0_05_n": int((nonlinear["quadratic_fdr_bh"] < 0.05).sum()),
        "limitations": ["Only two cohorts; random-effects heterogeneity estimates are imprecise",
                        "GSE161529 denominator is immune cells, not all tumor cells",
                        "MAPK/p38 scores are transcriptional response proxies, not phosphoprotein measurements"],
    }
    (TABLE_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nState residual meta:\n", primary_state[["pathway", "meta_partial_rho", "ci_low", "ci_high", "p_value", "fdr_bh", "I2_percent", "direction_concordant"]].to_string(index=False))
    print("\nNonlinear tests:\n", nonlinear.to_string(index=False))


if __name__ == "__main__":
    main()
