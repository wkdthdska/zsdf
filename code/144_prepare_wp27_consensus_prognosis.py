#!/usr/bin/env python
"""Prepare fixed consensus-state scores and clinical data for WP27 survival models."""

from __future__ import annotations

import json
import re
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = ROOT / "data"
PROCESSED = DATA / "processed" / "wp27_consensus_prognosis"
TABLES = ROOT / "output" / "tables" / "wp27_consensus_prognosis"
for folder in (PROCESSED, TABLES):
    folder.mkdir(parents=True, exist_ok=True)

SIGNATURE = pd.read_csv(
    ROOT / "output" / "tables" / "wp23_consensus_state" / "consensus_macrophage_remodeling_signature.csv"
).gene.astype(str).head(30).tolist()
GENERIC = ["C1QA", "C1QB", "C1QC", "CD68", "LST1", "AIF1", "FCER1G", "TYROBP"]
TARGETS = set(SIGNATURE + GENERIC)


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if np.isfinite(sd) and sd > 0 else values * np.nan


def read_selected_matrix(path: Path, sep: str, gene_column: str) -> pd.DataFrame:
    selected = []
    for chunk in pd.read_csv(path, sep=sep, chunksize=1000):
        genes = chunk[gene_column].astype(str).str.strip().str.upper()
        use = genes.isin(TARGETS)
        if use.any():
            block = chunk.loc[use].copy()
            block[gene_column] = genes.loc[use]
            selected.append(block)
    matrix = pd.concat(selected, ignore_index=True)
    matrix = matrix.groupby(gene_column, sort=False).mean(numeric_only=True)
    return matrix.apply(pd.to_numeric, errors="coerce")


def score_matrix(matrix: pd.DataFrame, sample_groups: pd.Series | None = None) -> tuple[pd.DataFrame, set[str]]:
    expression = matrix.T.copy()
    expression.index = expression.index.astype(str)
    if sample_groups is None:
        sample_groups = pd.Series("all", index=expression.index)
    else:
        sample_groups = sample_groups.reindex(expression.index).fillna("unknown").astype(str)
    present = {gene for gene in matrix.index if matrix.loc[gene].notna().sum() > 2 and matrix.loc[gene].std(ddof=0) > 0}
    output = pd.DataFrame(index=expression.index)
    for name, genes in [("state_score", SIGNATURE), ("generic_score", GENERIC)]:
        available = [gene for gene in genes if gene in present]
        values = pd.Series(np.nan, index=expression.index, dtype=float)
        for _, indices in sample_groups.groupby(sample_groups).groups.items():
            gene_z = expression.loc[indices, available].apply(zscore)
            values.loc[indices] = gene_z.mean(axis=1)
        output[name] = values
    return output, present


def add_common_score(matrix: pd.DataFrame, scores: pd.DataFrame, common: list[str], groups: pd.Series | None = None) -> None:
    expression = matrix.T.reindex(scores.index)
    if groups is None:
        groups = pd.Series("all", index=scores.index)
    else:
        groups = groups.reindex(scores.index).fillna("unknown").astype(str)
    values = pd.Series(np.nan, index=scores.index, dtype=float)
    for _, indices in groups.groupby(groups).groups.items():
        values.loc[indices] = expression.loc[indices, common].apply(zscore).mean(axis=1)
    scores["common_score"] = values


def add_residuals(frame: pd.DataFrame, groups: pd.Series | None = None) -> None:
    if groups is None:
        groups = pd.Series("all", index=frame.index)
    else:
        groups = groups.reindex(frame.index).fillna("unknown").astype(str)
    for score in ["state_score", "common_score"]:
        result = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, indices in groups.groupby(groups).groups.items():
            block = frame.loc[indices, [score, "generic_score"]].dropna()
            fit = sm.OLS(block[score], sm.add_constant(block[["generic_score"]], has_constant="add")).fit()
            result.loc[block.index] = zscore(fit.resid)
        frame[score.replace("_score", "_residual")] = result
    frame["generic_score"] = frame.groupby(groups)["generic_score"].transform(zscore)


def clean_stage(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.upper().str.replace("STAGE", "", regex=False).str.strip()
    result = text.str.extract(r"^(I{1,3}|IV)", expand=False)
    return result.where(result.isin(["I", "II", "III", "IV"]))


def receptor_subtype(frame: pd.DataFrame) -> pd.Series:
    er = frame["ER_STATUS"].map({"Positive": 1, "Negative": 0})
    pr = frame["PR_STATUS"].map({"Positive": 1, "Negative": 0})
    her2 = frame["HER2_STATUS"].map({"Positive": 1, "Negative": 0})
    result = pd.Series(np.nan, index=frame.index, dtype=object)
    result[her2.eq(1)] = "HER2+"
    result[her2.eq(0) & (er.eq(1) | pr.eq(1))] = "HR+/HER2-"
    result[her2.eq(0) & er.eq(0) & pr.eq(0)] = "TNBC"
    return result


def main() -> None:
    tcga_matrix = read_selected_matrix(PROCESSED.parent / "tcga_brca_deconvolution_input.tsv.gz", "\t", "sample")
    metabric_matrix = read_selected_matrix(PROCESSED.parent / "metabric_deconvolution_input.tsv.gz", "\t", "Hugo_Symbol")
    gse_raw = pd.read_csv(PROCESSED / "gse96058_consensus_genes.csv.gz")
    gse_gene_col = gse_raw.columns[0]
    gse_raw[gse_gene_col] = gse_raw[gse_gene_col].astype(str).str.upper()
    gse_matrix = gse_raw.groupby(gse_gene_col).mean(numeric_only=True)

    gse_metadata = pd.read_csv(PROCESSED.parent / "gse96058_sample_metadata.tsv", sep="\t")
    gse_platform = gse_metadata.set_index("title")["platform"].astype(str)

    tcga_scores, tcga_present = score_matrix(tcga_matrix)
    metabric_scores, metabric_present = score_matrix(metabric_matrix)
    gse_scores, gse_present = score_matrix(gse_matrix, gse_platform)
    common = sorted(set(SIGNATURE) & tcga_present & metabric_present & gse_present)
    if len(common) < 10:
        raise RuntimeError(f"Only {len(common)} signature genes are shared across all cohorts")
    add_common_score(tcga_matrix, tcga_scores, common)
    add_common_score(metabric_matrix, metabric_scores, common)
    add_common_score(gse_matrix, gse_scores, common, gse_platform)
    add_residuals(tcga_scores)
    add_residuals(metabric_scores)
    add_residuals(gse_scores, gse_platform)

    # TCGA clinical merge.
    tcga_score_meta = pd.read_csv(PROCESSED.parent / "tam_infiltration_by_subtype" / "tcga_brca_scores_clinical.csv")
    tcga_score_meta["patient_id"] = tcga_score_meta["sample"].str[:12]
    tcga_scores.index.name = "sample"
    tcga = tcga_score_meta.merge(tcga_scores.reset_index(), on="sample", how="inner", validate="one_to_one")
    cdr = pd.read_csv(PROCESSED.parent / "tcga_brca_legacy_arl11_with_cdr.csv").drop_duplicates("patient_id")
    tcga = tcga.merge(cdr, on="patient_id", how="inner", validate="one_to_one")
    tcga["time"] = pd.to_numeric(tcga["OS.time"], errors="coerce") / 30.4375
    tcga["event"] = pd.to_numeric(tcga["OS"], errors="coerce")
    tcga["age10"] = pd.to_numeric(tcga["age_at_initial_pathologic_diagnosis"], errors="coerce") / 10
    tcga["stage"] = clean_stage(tcga["ajcc_pathologic_tumor_stage"])
    tcga["subtype"] = tcga["pam50"].where(tcga["pam50"].isin(["Basal", "Her2", "LumA", "LumB", "Normal"]))

    # METABRIC clinical merge.
    metabric_scores.index.name = "PATIENT_ID"
    metabric = metabric_scores.reset_index()
    patient = pd.read_csv(DATA / "raw" / "metabric_data_clinical_patient.txt", sep="\t", comment="#")
    sample = pd.read_csv(DATA / "raw" / "metabric_data_clinical_sample.txt", sep="\t", comment="#").drop_duplicates("PATIENT_ID")
    metabric = metabric.merge(patient, on="PATIENT_ID", how="inner", validate="one_to_one")
    metabric = metabric.merge(sample, on="PATIENT_ID", how="inner", validate="one_to_one")
    metabric["time"] = pd.to_numeric(metabric["OS_MONTHS"], errors="coerce")
    metabric["event"] = metabric["OS_STATUS"].astype(str).str.startswith("1:").astype(float)
    metabric["age10"] = pd.to_numeric(metabric["AGE_AT_DIAGNOSIS"], errors="coerce") / 10
    metabric["stage"] = pd.to_numeric(metabric["TUMOR_STAGE"], errors="coerce").round().astype("Int64").astype(str).replace("<NA>", np.nan)
    metabric["grade"] = pd.to_numeric(metabric["GRADE"], errors="coerce").round().astype("Int64").astype(str).replace("<NA>", np.nan)
    metabric["subtype"] = receptor_subtype(metabric)

    # GSE96058 clinical merge and technical-replicate removal.
    gse_scores.index.name = "title"
    gse = gse_metadata.merge(gse_scores.reset_index(), on="title", how="inner", validate="one_to_one")
    gse["is_technical_replicate"] = gse["is_technical_replicate"].astype(str).str.lower().eq("true")
    gse = gse.loc[~gse.is_technical_replicate].copy()
    gse["time"] = pd.to_numeric(gse["overall_survival_days"], errors="coerce") / 30.4375
    gse["event"] = pd.to_numeric(gse["overall_survival_event"], errors="coerce")
    gse["age10"] = pd.to_numeric(gse["age_at_diagnosis"], errors="coerce") / 10
    gse["tumor_size10"] = pd.to_numeric(gse["tumor_size"], errors="coerce") / 10
    gse["grade"] = gse["nhg"].where(gse["nhg"].isin(["G1", "G2", "G3"]))
    gse["node"] = gse["lymph_node_group"].where(gse["lymph_node_group"].notna())
    gse["subtype"] = gse["pam50_subtype"].where(gse["pam50_subtype"].isin(["Basal", "Her2", "LumA", "LumB", "Normal"]))
    gse["platform_subtype"] = gse["platform"].astype(str) + "|" + gse["subtype"].astype(str)

    output_columns = {
        "tcga_brca": ["patient_id", "time", "event", "state_score", "state_residual", "common_score", "common_residual", "generic_score", "age10", "stage", "subtype"],
        "metabric": ["PATIENT_ID", "time", "event", "state_score", "state_residual", "common_score", "common_residual", "generic_score", "age10", "stage", "grade", "subtype"],
        "gse96058": ["title", "time", "event", "state_score", "state_residual", "common_score", "common_residual", "generic_score", "age10", "tumor_size10", "grade", "node", "subtype", "platform_subtype"],
    }
    frames = {"tcga_brca": tcga, "metabric": metabric, "gse96058": gse}
    for name, frame in frames.items():
        frame[output_columns[name]].to_csv(PROCESSED / f"{name}_analysis_data.csv", index=False)

    coverage_rows = []
    for cohort, present in [("TCGA-BRCA", tcga_present), ("METABRIC", metabric_present), ("GSE96058", gse_present)]:
        for gene_set, genes in [("consensus_state", SIGNATURE), ("generic_macrophage", GENERIC), ("common_signature", common)]:
            available = [gene for gene in genes if gene in present]
            coverage_rows.append({
                "cohort": cohort, "gene_set": gene_set, "requested_n": len(genes), "present_n": len(available),
                "present_genes": ";".join(available), "missing_genes": ";".join(sorted(set(genes) - set(available))),
            })
    pd.DataFrame(coverage_rows).to_csv(TABLES / "gene_coverage.csv", index=False)
    summary = {
        "signature_frozen_before_outcome_analysis": True,
        "signature_genes_n": len(SIGNATURE),
        "common_signature_genes_n": len(common),
        "common_signature_genes": common,
        "available_rows": {name: int(len(frame)) for name, frame in frames.items()},
    }
    (TABLES / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
