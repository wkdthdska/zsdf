#!/usr/bin/env python
"""Build paired donor x macrophage-state raw-count pseudobulks in two cohorts."""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = ROOT / "data" / "raw"
RECLUSTERED = ROOT / "data" / "processed" / "macrophage_reclustering"
OUT = ROOT / "data" / "processed" / "consensus_macrophage_state"
TABLES = ROOT / "output" / "tables" / "wp23_consensus_state"
for directory in (OUT, TABLES):
    directory.mkdir(parents=True, exist_ok=True)


def patient_id(value: str) -> str:
    return re.sub(r"(pal_Patient 0029)-[79]C$", r"\1", str(value))


def collapse_gene_symbols(matrix: sp.spmatrix, symbols: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    symbols = np.asarray([str(x).strip().upper() for x in symbols])
    valid = (symbols != "") & (symbols != "NAN")
    matrix = matrix[:, valid]
    symbols = symbols[valid]
    unique, inverse = np.unique(symbols, return_inverse=True)
    mapper = sp.csr_matrix((np.ones(len(inverse)), (np.arange(len(inverse)), inverse)), shape=(len(inverse), len(unique)))
    return (matrix @ mapper).tocsr(), unique


def run(cohort: str, source_path: Path) -> None:
    source = ad.read_h5ad(source_path, backed="r")
    if cohort == "GSE176078":
        mask = source.obs["celltype_minor"].astype(str).eq("Macrophage").to_numpy()
    else:
        mask = (
            source.obs["batch"].astype(str).eq("pal_2021")
            & source.obs["cell_type"].astype(str).eq("macrophage")
        ).to_numpy()
    raw = source.raw[mask].to_adata()
    source.file.close()
    raw.obs_names_make_unique()
    raw.obs["patient_id"] = [patient_id(x) for x in raw.obs["donor_id"].astype(str)]

    clustered = ad.read_h5ad(RECLUSTERED / f"{cohort.lower()}_macrophage_reclustered.h5ad", backed="r")
    high_cluster = str(clustered.uns["arl11_high_cluster"])
    labels = clustered.obs["cluster"].astype(str).reindex(raw.obs_names)
    if labels.isna().any():
        raise RuntimeError(f"{cohort}: {labels.isna().sum()} macrophages lack cluster labels")
    raw.obs["state"] = np.where(labels.eq(high_cluster), "ARL11_marked_state", "other_macrophages")
    clustered.file.close()

    symbols = raw.var["feature_name"].astype(str).to_numpy() if "feature_name" in raw.var else raw.var_names.to_numpy()
    matrix, genes = collapse_gene_symbols(raw.X.tocsr(), symbols)
    sample_rows = []
    count_rows = []
    for (donor, state), indices in raw.obs.groupby(["patient_id", "state"], observed=True).indices.items():
        summed = np.asarray(matrix[np.asarray(indices)].sum(axis=0)).ravel().astype(np.int64)
        sample_id = f"{donor}__{'state' if state == 'ARL11_marked_state' else 'other'}"
        sample_rows.append({
            "sample_id": sample_id, "patient_id": donor, "state": state,
            "n_cells": len(indices), "library_size": int(summed.sum()), "high_cluster": high_cluster,
        })
        count_rows.append(summed)

    metadata = pd.DataFrame(sample_rows).set_index("sample_id")
    counts = pd.DataFrame(np.vstack(count_rows), index=metadata.index, columns=genes)
    counts.to_csv(OUT / f"{cohort.lower()}_state_pseudobulk_counts.tsv.gz", sep="\t", compression="gzip")
    metadata.to_csv(OUT / f"{cohort.lower()}_state_pseudobulk_metadata.csv")

    paired = metadata.pivot_table(index="patient_id", columns="state", values="n_cells", fill_value=0)
    for threshold in (3, 5, 8, 10):
        eligible = (
            (paired.get("ARL11_marked_state", 0) >= threshold)
            & (paired.get("other_macrophages", 0) >= 20)
        )
        paired[f"eligible_state_min{threshold}_other_min20"] = eligible
    paired.insert(0, "cohort", cohort)
    paired.to_csv(TABLES / f"{cohort.lower()}_paired_donor_qc.csv")


run("GSE176078", RAW / "GSE176078_cellxgene_22a27631.h5ad")
run("GSE161529", RAW / "breast_atlas_immune.h5ad")

all_qc = pd.concat([
    pd.read_csv(TABLES / "gse176078_paired_donor_qc.csv"),
    pd.read_csv(TABLES / "gse161529_paired_donor_qc.csv"),
], ignore_index=True)
summary = []
for cohort, group in all_qc.groupby("cohort"):
    for threshold in (3, 5, 8, 10):
        column = f"eligible_state_min{threshold}_other_min20"
        summary.append({"cohort": cohort, "state_min_cells": threshold, "other_min_cells": 20, "paired_donors_n": int(group[column].sum())})
pd.DataFrame(summary).to_csv(TABLES / "paired_donor_threshold_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
