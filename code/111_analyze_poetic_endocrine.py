#!/usr/bin/env python
"""POETIC validation of ARL11-associated macrophage scores and early AI response."""

from __future__ import annotations

import json
import re
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

RAW = ROOT / "data" / "raw" / "endocrine" / "POETIC"
PROCESSED = ROOT / "data" / "processed" / "endocrine" / "POETIC"
OUT = ROOT / "output" / "tables" / "endocrine_validation_poetic"
FIG = ROOT / "figures" / "endocrine_validation_poetic"
for directory in (PROCESSED, OUT, FIG):
    directory.mkdir(parents=True, exist_ok=True)

ARL11_HIGH = [
    "CHIT1", "TM4SF19", "RBP4", "CES1", "MATK", "LPL", "CHI3L1",
    "SULT1C2", "GCHFR", "CYP27A1", "DCSTAMP", "PHLDA3", "SDC2", "CCL7",
    "ACOT4", "CTSK", "STEAP3", "ACP5", "MYG1-AS1", "SEMA3C", "COL6A2",
    "FABP5", "PPARG", "RMDN3", "FBP1", "HTRA4", "SPP1", "CD52", "GPNMB",
    "SLC16A6",
]
GENERIC_MACROPHAGE = ["C1QA", "C1QB", "C1QC", "CD68", "LST1", "AIF1", "FCER1G", "TYROBP"]
LIPID = [
    "LPL", "LIPA", "APOE", "APOC1", "FABP5", "FABP3", "PPARG", "CYP27A1",
    "ABCA1", "ABCG1", "CD36", "PLIN2", "SOAT1", "NCEH1", "MGLL", "PLA2G7",
]
PHAGOLYSOSOME = [
    "CTSB", "CTSD", "CTSL", "CTSK", "CTSZ", "LGMN", "ACP5", "GPNMB", "LAMP1",
    "LAMP2", "ATP6V1A", "ATP6V0D1", "HEXA", "HEXB", "GM2A", "FUCA1",
]
LIPID_PHAGOLYSOSOMAL = list(dict.fromkeys(LIPID + PHAGOLYSOSOME))
PROLIFERATION = [
    "MKI67", "AURKA", "AURKB", "CCNB1", "CCNB2", "CDK1", "BUB1", "BUB1B",
    "TOP2A", "UBE2C", "CDC20", "MCM2", "MCM4", "PCNA", "PLK1",
]
SCORE_GENES = {
    "generic_macrophage_z": GENERIC_MACROPHAGE,
    "arl11_high_state_z": ARL11_HIGH,
    "lipid_phagolysosomal_z": LIPID_PHAGOLYSOSOMAL,
    "proliferation_z": PROLIFERATION,
}
PREDICTORS = ["arl11_z", "generic_macrophage_z", "arl11_high_state_z", "lipid_phagolysosomal_z"]
LABELS = {
    "arl11_z": "ARL11",
    "generic_macrophage_z": "Generic macrophage",
    "arl11_high_state_z": "ARL11-high state",
    "lipid_phagolysosomal_z": "Lipid-phagolysosomal",
}
OFFSET = 0.1


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    ranked = p[ok]
    if not ranked.size:
        return result
    order = np.argsort(ranked)
    adjusted = ranked[order] * ranked.size / np.arange(1, ranked.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1)
    result[ok] = restored
    return result


def load_eset(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = rdata.read_rda(path)
    eset = next(iter(loaded["gset"].values()))
    return eset.assayData["exprs"], eset.phenoData.data.copy()


def load_annotation() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        annotation = rdata.read_rda(RAW / "GPL10558_pipe.rda")["GPL10558_pipe"].copy()
    annotation.columns = ["probe_id", "symbol"]
    annotation["probe_id"] = annotation["probe_id"].astype(str)
    annotation["symbol"] = annotation["symbol"].astype(str).str.upper().str.strip()
    return annotation.dropna().drop_duplicates()


def extract_target_genes(expression, annotation: pd.DataFrame) -> pd.DataFrame:
    targets = {"ARL11", *[gene for genes in SCORE_GENES.values() for gene in genes]}
    probe_ids = pd.Index(expression.coords["dim_0"].values.astype(str))
    sample_ids = expression.coords["dim_1"].values.astype(str)
    lookup = annotation[annotation["symbol"].isin(targets)]
    lookup = lookup[lookup["probe_id"].isin(probe_ids)]
    positions = pd.Series(np.arange(len(probe_ids)), index=probe_ids)
    matrix = np.asarray(expression.values)
    genes = {}
    for symbol, group in lookup.groupby("symbol"):
        indices = positions.loc[group["probe_id"].drop_duplicates()].to_numpy(dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            genes[symbol] = np.nanmedian(matrix[indices, :], axis=0)
    return pd.DataFrame(genes, index=sample_ids)


def parse_sample_metadata(pheno: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pheno.iterrows():
        title = str(row["title"])
        treated = re.search(r"AI\.Treated\.(\d+)([BS])", title)
        control = re.search(r"^(\d+)([DS])", title)
        if treated:
            number, suffix = treated.groups()
            group, timepoint = "AI", "baseline" if suffix == "B" else "post"
            key = f"Treated.{int(number)}"
        elif control:
            number, suffix = control.groups()
            group, timepoint = "Control", "baseline" if suffix == "D" else "post"
            key = f"Control.{int(number)}"
        else:
            continue
        rows.append({
            "sample": str(row["geo_accession"]), "title": title,
            "clinical_key": key, "group": group, "timepoint": timepoint,
        })
    return pd.DataFrame(rows).set_index("sample")


def compute_scores(gene_expression: pd.DataFrame, sample_meta: pd.DataFrame, cohort: str):
    common = gene_expression.index.intersection(sample_meta.index)
    gene_expression = gene_expression.loc[common]
    sample_meta = sample_meta.loc[common]
    baseline = sample_meta["timepoint"].eq("baseline")
    means = gene_expression.loc[baseline].mean(axis=0)
    sds = gene_expression.loc[baseline].std(axis=0, ddof=1).replace(0, np.nan)
    z = (gene_expression - means) / sds
    scores = sample_meta.copy()
    scores["arl11_z"] = z["ARL11"] if "ARL11" in z else np.nan
    coverage = []
    coverage.append({"cohort": cohort, "score": "arl11_z", "expected": 1,
                     "present": int("ARL11" in z),
                     "nonmissing_baseline": int(scores.loc[baseline, "arl11_z"].notna().sum()),
                     "genes": "ARL11" if "ARL11" in z else ""})
    for score, genes in SCORE_GENES.items():
        present = [gene for gene in genes if gene in z.columns and z[gene].notna().any()]
        scores[score] = z[present].mean(axis=1) if present else np.nan
        coverage.append({"cohort": cohort, "score": score, "expected": len(genes),
                         "present": len(present),
                         "nonmissing_baseline": int(scores.loc[baseline, score].notna().sum()),
                         "genes": ";".join(present)})
    return scores, pd.DataFrame(coverage)


def load_clinical() -> pd.DataFrame:
    data = pd.read_excel(RAW / "Gao2019_MOESM2.xlsx", sheet_name="TableS4", header=2)
    data = data.rename(columns={
        "# 254 tumours: Control=56, AI-treated=198)": "clinical_key",
        "Patient.ID": "patient_id", "Group": "treatment",
        "HER2 status": "her2", "Baseline.Ki67": "baseline_ki67",
        "Surgery.Ki67": "post_ki67", "Change.Ki67": "percent_change_ki67",
    })
    keep = ["clinical_key", "patient_id", "treatment", "her2", "baseline_ki67",
            "post_ki67", "percent_change_ki67"]
    data = data[keep].copy()
    for column in ["baseline_ki67", "post_ki67", "percent_change_ki67"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["ai"] = data["treatment"].eq("Peri AI").astype(int)
    data["her2"] = data["her2"].replace({"Unknown/Missing": "Unknown"}).fillna("Unknown")
    data["log_baseline_ki67"] = np.log(data["baseline_ki67"] + OFFSET)
    data["log_post_ki67"] = np.log(data["post_ki67"] + OFFSET)
    data["log_change_ki67"] = data["log_post_ki67"] - data["log_baseline_ki67"]
    data["ccca"] = np.where(data["post_ki67"].notna(), (data["post_ki67"] <= 2.7).astype(int), np.nan)
    return data


def prepare_baseline(scores: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    baseline = scores[scores["timepoint"].eq("baseline")].reset_index(names="sample")
    merged = baseline.merge(clinical, on="clinical_key", how="inner", validate="one_to_one")
    return merged


def ols_interaction(data: pd.DataFrame, score: str, endpoint: str, adjustment: str, subset: str):
    columns = [endpoint, "log_baseline_ki67", "her2", score, "ai"]
    if adjustment == "full":
        columns.append("proliferation_z")
    frame = data[columns].dropna().copy()
    frame["her2"] = frame["her2"].astype(str)
    covariates = "log_baseline_ki67"
    if frame["her2"].nunique() > 1:
        covariates += " + C(her2)"
    if adjustment == "full":
        covariates += " + proliferation_z"
    formula = f"{endpoint} ~ {covariates} + {score} * ai"
    fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
    term = f"{score}:ai"
    beta, se, pvalue = fit.params[term], fit.bse[term], fit.pvalues[term]
    low, high = fit.conf_int().loc[term]
    return {
        "subset": subset, "endpoint": endpoint, "adjustment": adjustment,
        "score": score, "n": int(fit.nobs), "beta_interaction": beta,
        "se_robust": se, "ci_low": low, "ci_high": high,
        "effect_ratio": np.exp(beta), "ratio_ci_low": np.exp(low),
        "ratio_ci_high": np.exp(high), "wald_p": pvalue,
        "r_squared": fit.rsquared,
    }


def logistic_interaction(data: pd.DataFrame, score: str, subset: str):
    columns = ["ccca", "log_baseline_ki67", "proliferation_z", "her2", score, "ai"]
    frame = data[columns].dropna().copy()
    frame["her2"] = frame["her2"].astype(str)
    covariates = "log_baseline_ki67 + proliferation_z"
    if frame["her2"].nunique() > 1:
        covariates += " + C(her2)"
    formula = f"ccca ~ {covariates} + {score} * ai"
    fit = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(cov_type="HC3")
    term = f"{score}:ai"
    beta = fit.params[term]
    low, high = fit.conf_int().loc[term]
    return {
        "subset": subset, "endpoint": "CCCA", "score": score, "n": int(fit.nobs),
        "interaction_or": np.exp(beta), "or_ci_low": np.exp(low),
        "or_ci_high": np.exp(high), "wald_p": fit.pvalues[term],
    }


def treated_only_models(data: pd.DataFrame, cohort: str, predictors: list[str]) -> pd.DataFrame:
    rows = []
    treated = data[data["ai"].eq(1)].copy()
    for score in predictors:
        columns = ["log_post_ki67", "log_baseline_ki67", "proliferation_z", "her2", score]
        frame = treated[columns].dropna()
        fit = smf.ols(
            f"log_post_ki67 ~ log_baseline_ki67 + proliferation_z + C(her2) + {score}",
            data=frame,
        ).fit(cov_type="HC3")
        low, high = fit.conf_int().loc[score]
        rows.append({
            "cohort": cohort, "score": score, "n": int(fit.nobs),
            "beta": fit.params[score], "ci_low": low, "ci_high": high,
            "effect_ratio": np.exp(fit.params[score]),
            "ratio_ci_low": np.exp(low), "ratio_ci_high": np.exp(high),
            "wald_p": fit.pvalues[score],
        })
    result = pd.DataFrame(rows)
    result["fdr"] = bh_adjust(result["wald_p"])
    return result


def adaptive_models(scores: pd.DataFrame, clinical: pd.DataFrame, predictors: list[str]):
    score_columns = predictors + ["proliferation_z"]
    wide = scores.reset_index(names="sample").pivot(
        index="clinical_key", columns="timepoint", values=score_columns
    )
    wide.columns = [f"{score}_{time}" for score, time in wide.columns]
    wide = wide.reset_index().merge(clinical, on="clinical_key", how="inner", validate="one_to_one")
    rows = []
    for score in predictors:
        wide[f"delta_{score}"] = wide[f"{score}_post"] - wide[f"{score}_baseline"]
    for score in predictors:
        columns = [f"delta_{score}", f"{score}_baseline", "proliferation_z_baseline", "her2", "ai"]
        frame = wide[columns].dropna()
        fit = smf.ols(
            f"delta_{score} ~ {score}_baseline + proliferation_z_baseline + C(her2) + ai",
            data=frame,
        ).fit(cov_type="HC3")
        low, high = fit.conf_int().loc["ai"]
        rows.append({
            "analysis": "AI-induced change vs control", "score": score,
            "n": int(fit.nobs), "beta": fit.params["ai"], "ci_low": low,
            "ci_high": high, "wald_p": fit.pvalues["ai"],
        })
        treated = wide[wide["ai"].eq(1)].copy()
        columns = ["log_post_ki67", "log_baseline_ki67", "her2",
                   "proliferation_z_baseline", f"{score}_baseline", f"delta_{score}"]
        frame = treated[columns].dropna()
        formula = (
            f"log_post_ki67 ~ log_baseline_ki67 + proliferation_z_baseline + "
            f"C(her2) + {score}_baseline + delta_{score}"
        )
        fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
        term = f"delta_{score}"
        low, high = fit.conf_int().loc[term]
        rows.append({
            "analysis": "Delta association with residual Ki67 in AI arm", "score": score,
            "n": int(fit.nobs), "beta": fit.params[term], "ci_low": low,
            "ci_high": high, "wald_p": fit.pvalues[term],
        })
        if score in {"arl11_high_state_z", "lipid_phagolysosomal_z"}:
            columns = [f"delta_{score}", f"{score}_baseline", "proliferation_z_baseline",
                       "generic_macrophage_z_baseline", "delta_generic_macrophage_z", "her2", "ai"]
            frame = wide[columns].dropna()
            formula = (
                f"delta_{score} ~ {score}_baseline + proliferation_z_baseline + C(her2) + "
                "generic_macrophage_z_baseline + delta_generic_macrophage_z + ai"
            )
            fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
            low, high = fit.conf_int().loc["ai"]
            rows.append({
                "analysis": "AI-induced change adjusted for generic macrophage change",
                "score": score, "n": int(fit.nobs), "beta": fit.params["ai"],
                "ci_low": low, "ci_high": high, "wald_p": fit.pvalues["ai"],
            })
    result = pd.DataFrame(rows)
    result["fdr"] = result.groupby("analysis")["wald_p"].transform(lambda x: bh_adjust(x))
    return wide, result


def forest_plot(primary: pd.DataFrame):
    plot = primary[(primary["subset"] == "all") & (primary["endpoint"] == "log_post_ki67") &
                   (primary["adjustment"] == "full")].copy()
    plot["label"] = plot["score"].map(LABELS)
    plot = plot.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    y = np.arange(len(plot))
    ax.errorbar(plot["effect_ratio"], y,
                xerr=[plot["effect_ratio"] - plot["ratio_ci_low"],
                      plot["ratio_ci_high"] - plot["effect_ratio"]],
                fmt="o", color="#8f2d56", ecolor="#8f2d56", capsize=3)
    ax.axvline(1, color="0.45", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Interaction effect ratio per 1-SD baseline score")
    ax.set_title("POETIC: baseline score × aromatase inhibitor interaction")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "poetic_interaction_forest.png", dpi=300)
    fig.savefig(FIG / "poetic_interaction_forest.pdf")
    plt.close(fig)


def adaptive_plot(adaptive: pd.DataFrame):
    plot = adaptive[adaptive["analysis"] == "AI-induced change vs control"].copy()
    plot["label"] = plot["score"].map(LABELS)
    plot = plot.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    y = np.arange(len(plot))
    ax.errorbar(plot["beta"], y,
                xerr=[plot["beta"] - plot["ci_low"], plot["ci_high"] - plot["beta"]],
                fmt="o", color="#247ba0", ecolor="#247ba0", capsize=3)
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("AI-specific standardized score change")
    ax.set_title("POETIC paired biopsies: AI-induced macrophage-state change")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "poetic_adaptive_change_forest.png", dpi=300)
    fig.savefig(FIG / "poetic_adaptive_change_forest.pdf")
    plt.close(fig)


def main():
    annotation = load_annotation()
    clinical = load_clinical()

    expression_105, pheno_105 = load_eset(RAW / "GSE105777_eSet.Rdata")
    meta_105 = parse_sample_metadata(pheno_105)
    genes_105 = extract_target_genes(expression_105, annotation)
    scores_105, coverage_105 = compute_scores(genes_105, meta_105, "GSE105777")
    baseline_105 = prepare_baseline(scores_105, clinical)
    analysis_predictors = [score for score in PREDICTORS if baseline_105[score].notna().sum() >= 20]

    primary_rows = []
    subsets = {
        "all": baseline_105,
        "HER2-negative": baseline_105[baseline_105["her2"].eq("Negative")],
        "known-HER2": baseline_105[~baseline_105["her2"].eq("Unknown")],
    }
    for subset, frame in subsets.items():
        for score in analysis_predictors:
            primary_rows.append(ols_interaction(frame, score, "log_post_ki67", "full", subset))
            if subset == "all":
                primary_rows.append(ols_interaction(frame, score, "log_post_ki67", "baseline_Ki67_only", subset))
                primary_rows.append(ols_interaction(frame, score, "log_change_ki67", "full", subset))
    primary = pd.DataFrame(primary_rows)
    main_mask = ((primary["subset"] == "all") & (primary["endpoint"] == "log_post_ki67") &
                 (primary["adjustment"] == "full"))
    primary["fdr"] = np.nan
    primary.loc[main_mask, "fdr"] = bh_adjust(primary.loc[main_mask, "wald_p"])

    binary = pd.DataFrame([logistic_interaction(frame, score, subset)
                           for subset, frame in subsets.items() for score in analysis_predictors])
    binary["fdr"] = np.nan
    binary.loc[binary["subset"].eq("all"), "fdr"] = bh_adjust(
        binary.loc[binary["subset"].eq("all"), "wald_p"]
    )

    paired, adaptive = adaptive_models(scores_105, clinical, analysis_predictors)
    treated_105 = treated_only_models(baseline_105, "GSE105777", analysis_predictors)

    del expression_105, pheno_105, genes_105
    expression_126, pheno_126 = load_eset(RAW / "GSE126870_eSet.Rdata")
    meta_126 = parse_sample_metadata(pheno_126)
    genes_126 = extract_target_genes(expression_126, annotation)
    scores_126, coverage_126 = compute_scores(genes_126, meta_126, "GSE126870")
    baseline_126 = prepare_baseline(scores_126, clinical)
    treated_126 = treated_only_models(baseline_126, "GSE126870", analysis_predictors)

    coverage = pd.concat([coverage_105, coverage_126], ignore_index=True)
    primary.to_csv(OUT / "continuous_interaction_models.csv", index=False)
    binary.to_csv(OUT / "ccca_interaction_models.csv", index=False)
    adaptive.to_csv(OUT / "adaptive_change_models.csv", index=False)
    pd.concat([treated_105, treated_126], ignore_index=True).to_csv(
        OUT / "treated_arm_associations.csv", index=False
    )
    coverage.to_csv(OUT / "gene_coverage.csv", index=False)
    baseline_105.to_csv(PROCESSED / "gse105777_baseline_scores_clinical.csv", index=False)
    baseline_126.to_csv(PROCESSED / "gse126870_baseline_scores_clinical.csv", index=False)
    paired.to_csv(PROCESSED / "gse105777_paired_score_changes.csv", index=False)

    audit = pd.DataFrame([
        {"metric": "clinical_patients", "value": len(clinical)},
        {"metric": "GSE105777_baseline_joined", "value": len(baseline_105)},
        {"metric": "GSE105777_AI_baseline", "value": int(baseline_105["ai"].sum())},
        {"metric": "GSE105777_control_baseline", "value": int((baseline_105["ai"] == 0).sum())},
        {"metric": "GSE105777_complete_Ki67", "value": int(baseline_105[["baseline_ki67", "post_ki67"]].notna().all(axis=1).sum())},
        {"metric": "GSE126870_AI_baseline_joined", "value": len(baseline_126)},
        {"metric": "grade_available", "value": "No"},
        {"metric": "primary_offset_c", "value": OFFSET},
    ])
    audit.to_csv(OUT / "data_audit.csv", index=False)

    main_results = primary.loc[main_mask].sort_values("wald_p")
    summary = {
        "primary_n": int(main_results["n"].max()),
        "primary_results": main_results.to_dict(orient="records"),
        "any_primary_fdr_below_0_05": bool((main_results["fdr"] < 0.05).any()),
        "any_adaptive_fdr_below_0_05": bool((adaptive["fdr"] < 0.05).any()),
        "analyzable_predictors": analysis_predictors,
        "not_estimable_predictors": sorted(set(PREDICTORS).difference(analysis_predictors)),
        "grade_unavailable": True,
        "offset": OFFSET,
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    forest_plot(primary)
    adaptive_plot(adaptive)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
