#!/usr/bin/env python
"""WP19: spatial location of the frozen ARL11-associated macrophage state.

Four Wu et al. Visium tumors are analysed independently. Spots are never
treated as independent patients: inference uses spatial blocks within patient,
followed by random-effects meta-analysis of patient-level correlations.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import chi2, norm, rankdata, t

RAW = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "output" / "tables" / "wp19_spatial_interface"
FIG_DIR = ROOT / "figures" / "wp19_spatial_interface"
PROCESSED_DIR = ROOT / "data" / "processed" / "wp19_spatial_interface"
for directory in (TABLE_DIR, FIG_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SAMPLES = {"CID4290": "ER+", "CID4535": "ER+", "CID4465": "TNBC", "CID44971": "TNBC"}
GENERIC_MACROPHAGE = ["C1QA", "C1QB", "C1QC", "CD68", "LST1", "AIF1", "FCER1G", "TYROBP"]
CANCER_COLUMNS = ["Cancer Basal SC", "Cancer Cycling", "Cancer Her2 SC", "Cancer LumA SC", "Cancer LumB SC"]
CAF_COLUMNS = ["CAFs MSC iCAF-like", "CAFs myCAF-like"]

OUTCOMES = {
    "tumor_nest": ["cancer_total", "local_cancer"],
    "caf_niche": ["caf_total", "local_caf"],
    "boundary": ["tumor_caf_coexistence", "boundary_proximity", "histology_mixed_fraction"],
    "cd8_relation": ["local_cd8", "distance_to_cd8_rich"],
}


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values])


def read_column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Dataset):
        values = node[:]
        return decode(values) if values.dtype.kind in "OSU" else values
    if node.attrs.get("encoding-type") == "categorical":
        categories = decode(node["categories"][:])
        codes = node["codes"][:]
        values = np.full(len(codes), "", dtype=object)
        keep = codes >= 0
        values[keep] = categories[codes[keep]]
        return values
    raise TypeError(f"Unsupported obs column: {key}")


def bh(values) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    data = p[valid]
    if not len(data):
        return result
    order = np.argsort(data)
    ranked = data[order]
    adjusted = np.minimum.accumulate((ranked * len(data) / np.arange(1, len(data) + 1))[::-1])[::-1]
    restored = np.empty(len(data))
    restored[order] = np.minimum(adjusted, 1)
    result[np.where(valid)[0]] = restored
    return result


def score(logcpm: np.ndarray, genes: list[str], lookup: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    present = [gene for gene in genes if gene in lookup]
    values = logcpm[:, [lookup[g] for g in present]]
    variable = values.std(axis=0) > 0
    present = [g for g, keep in zip(present, variable) if keep]
    values = values[:, variable]
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    return z.mean(axis=1), present


def residual(values: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    frame = pd.concat([values.rename("y"), covariates], axis=1).dropna()
    design = sm.add_constant(frame.drop(columns="y"), has_constant="add")
    fit = sm.OLS(frame["y"], design).fit()
    result = pd.Series(np.nan, index=values.index)
    result.loc[frame.index] = fit.resid
    return result


def build_neighbors(coords: np.ndarray) -> tuple[list[np.ndarray], float]:
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=7)
    median_spacing = float(np.median(distances[:, 1]))
    neighborhoods = []
    for row_d, row_i in zip(distances, indices):
        keep = (row_d > 0) & (row_d <= 1.8 * median_spacing)
        neighborhoods.append(row_i[keep])
    return neighborhoods, median_spacing


def neighborhood_mean(values: np.ndarray, neighborhoods: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.nanmean(values[idx]) if len(idx) else np.nan for idx in neighborhoods])


def nearest_distance(coords: np.ndarray, target: np.ndarray, scale: float) -> np.ndarray:
    if target.sum() == 0:
        return np.full(len(coords), np.nan)
    distance, _ = cKDTree(coords[target]).query(coords, k=1)
    return distance / scale


def classify_histology(values: pd.Series) -> pd.Series:
    lower = values.astype(str).str.lower()
    tumor = lower.str.contains("invasive cancer|dcis", regex=True)
    stroma = lower.str.contains("stroma", regex=False)
    result = pd.Series("other", index=values.index)
    result[tumor & ~stroma] = "tumor_only"
    result[~tumor & stroma] = "stroma_only"
    result[tumor & stroma] = "mixed_tumor_stroma"
    return result


def process_sample(sample: str, subtype: str, signature: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    path = RAW / f"{sample}.cellxgene.h5ad"
    if not path.exists():
        path = RAW / f"{sample}.cellxgene.h5ad.part"
    requested = sorted(set(signature + GENERIC_MACROPHAGE + ["SPP1", "GPNMB"]))
    with h5py.File(path, "r") as handle:
        feature_names = read_column(handle["var"], "feature_name")
        source_lookup = {gene: i for i, gene in enumerate(feature_names)}
        selected = [gene for gene in requested if gene in source_lookup]
        columns = np.asarray([source_lookup[g] for g in selected])
        shape = tuple(int(x) for x in handle["X"].attrs["shape"])
        matrix = csr_matrix((handle["X"]["data"][:], handle["X"]["indices"][:],
                             handle["X"]["indptr"][:]), shape=shape)
        in_tissue = read_column(handle["obs"], "in_tissue").astype(int) == 1
        total = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
        keep = in_tissue & (total > 0)
        kept = matrix[keep]
        counts = kept[:, columns].toarray().astype(float)
        total = total[keep]
        detected = np.asarray(kept.getnnz(axis=1)).ravel().astype(float)
        coords = handle["obsm"]["spatial"][:].astype(float)[keep]
        array_col = read_column(handle["obs"], "array_col").astype(float)[keep]
        array_row = read_column(handle["obs"], "array_row").astype(float)[keep]
        obs_values = {column: read_column(handle["obs"], column)[keep] for column in
                      CANCER_COLUMNS + CAF_COLUMNS + ["Macrophage", "Monocyte", "T cells CD8+", "Classification"]}

    lookup = {gene: i for i, gene in enumerate(selected)}
    logcpm = np.log1p(counts / total[:, None] * 1e6)
    spots = pd.DataFrame({"sample": sample, "subtype": subtype, "x": coords[:, 0], "y": coords[:, 1],
                          "array_col": array_col, "array_row": array_row,
                          "log_total_counts": np.log1p(total), "detected_gene_n": detected})
    for column, values in obs_values.items():
        spots[column] = values
    numeric = CANCER_COLUMNS + CAF_COLUMNS + ["Macrophage", "Monocyte", "T cells CD8+"]
    spots[numeric] = spots[numeric].astype(float)
    spots["cancer_total"] = spots[CANCER_COLUMNS].sum(axis=1)
    spots["caf_total"] = spots[CAF_COLUMNS].sum(axis=1)
    spots["myeloid"] = spots["Macrophage"] + spots["Monocyte"]
    spots["t_cd8"] = spots["T cells CD8+"]
    spots["histology_region"] = classify_histology(spots["Classification"])
    spots["histology_mixed"] = spots["histology_region"].eq("mixed_tumor_stroma").astype(float)
    spots["signature_score"], signature_present = score(logcpm, signature, lookup)
    spots["generic_macro_score"], generic_present = score(logcpm, GENERIC_MACROPHAGE, lookup)
    spots["SPP1_GPNMB_score"], niche_present = score(logcpm, ["SPP1", "GPNMB"], lookup)

    tech = spots[["log_total_counts", "detected_gene_n"]].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    _, _, vt = np.linalg.svd(tech.to_numpy(), full_matrices=False)
    loading = vt[0]
    if loading.sum() < 0:
        loading = -loading
    spots["technical_QC_PC1"] = tech.to_numpy() @ loading
    covariates = spots[["generic_macro_score", "myeloid", "technical_QC_PC1"]]
    spots["state_residual"] = residual(spots["signature_score"], covariates)

    neighborhoods, spacing = build_neighbors(coords)
    spots["local_cancer"] = neighborhood_mean(spots["cancer_total"].to_numpy(), neighborhoods)
    spots["local_caf"] = neighborhood_mean(spots["caf_total"].to_numpy(), neighborhoods)
    spots["local_cd8"] = neighborhood_mean(spots["t_cd8"].to_numpy(), neighborhoods)
    spots["tumor_caf_coexistence"] = np.sqrt(np.maximum(spots["cancer_total"], 0) * np.maximum(spots["caf_total"], 0))

    # Define the interface on one-ring neighborhood composition rather than
    # noisy single-spot proportions.
    dominance = spots["local_cancer"] - spots["local_caf"]
    tumor_dominant = dominance >= dominance.quantile(2 / 3)
    stroma_dominant = dominance <= dominance.quantile(1 / 3)
    interface = []
    for idx in neighborhoods:
        interface.append(bool(tumor_dominant.iloc[idx].any() and stroma_dominant.iloc[idx].any()))
    spots["interface"] = np.asarray(interface, dtype=bool)
    spots["distance_to_interface"] = nearest_distance(coords, spots["interface"].to_numpy(), spacing)
    spots["boundary_proximity"] = -spots["distance_to_interface"]

    positive_cd8 = spots["t_cd8"] > 0
    threshold = spots.loc[positive_cd8, "t_cd8"].quantile(0.75) if positive_cd8.any() else np.inf
    cd8_rich = positive_cd8 & (spots["t_cd8"] >= threshold)
    spots["distance_to_cd8_rich"] = nearest_distance(coords, cd8_rich.to_numpy(), spacing)

    blocks = aggregate_blocks(spots, 4)
    blocks["sample"] = sample
    blocks["subtype"] = subtype
    coverage = [
        {"sample": sample, "gene_set": "frozen_ARL11_TAM_signature", "requested_n": len(signature),
         "present_variable_n": len(signature_present), "present_genes": ";".join(signature_present)},
        {"sample": sample, "gene_set": "generic_macrophage", "requested_n": len(GENERIC_MACROPHAGE),
         "present_variable_n": len(generic_present), "present_genes": ";".join(generic_present)},
        {"sample": sample, "gene_set": "SPP1_GPNMB_descriptive", "requested_n": 2,
         "present_variable_n": len(niche_present), "present_genes": ";".join(niche_present)},
    ]
    return spots, blocks, coverage


def aggregate_blocks(spots: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    frame = spots.copy()
    frame["x_bin"] = pd.qcut(frame["array_col"], quantiles, labels=False, duplicates="drop")
    frame["y_bin"] = pd.qcut(frame["array_row"], quantiles, labels=False, duplicates="drop")
    mean_columns = ["state_residual", "signature_score", "generic_macro_score", "myeloid", "technical_QC_PC1",
                    "cancer_total", "caf_total", "local_cancer", "local_caf", "local_cd8", "t_cd8",
                    "tumor_caf_coexistence", "boundary_proximity", "distance_to_cd8_rich", "SPP1_GPNMB_score"]
    blocks = frame.groupby(["x_bin", "y_bin"], observed=True).agg(
        **{column: (column, "mean") for column in mean_columns},
        n_spots=("state_residual", "size"),
        histology_mixed_fraction=("histology_mixed", "mean"),
        interface_fraction=("interface", "mean"),
    ).reset_index()
    blocks = blocks[blocks["n_spots"] >= 10].copy()
    return blocks


def partial_spearman(x: pd.Series, y: pd.Series, covariates: pd.DataFrame) -> tuple[int, int, float, float]:
    frame = pd.concat([x.rename("x"), y.rename("y"), covariates], axis=1).dropna()
    n, k = len(frame), covariates.shape[1]
    if n <= k + 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return n, k, np.nan, np.nan
    ranked = frame.apply(rankdata, axis=0)
    design = np.column_stack([np.ones(n), ranked[covariates.columns].to_numpy()])
    rx = ranked["x"].to_numpy() - design @ np.linalg.lstsq(design, ranked["x"], rcond=None)[0]
    ry = ranked["y"].to_numpy() - design @ np.linalg.lstsq(design, ranked["y"], rcond=None)[0]
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df = n - k - 2
    statistic = rho * math.sqrt(df / max(1e-15, 1 - rho ** 2))
    return n, k, rho, float(2 * t.sf(abs(statistic), df))


def patient_associations(sample: str, blocks: pd.DataFrame) -> pd.DataFrame:
    covariates = blocks[["generic_macro_score", "myeloid", "technical_QC_PC1"]]
    rows = []
    for family, outcomes in OUTCOMES.items():
        for outcome in outcomes:
            n, k, rho, p = partial_spearman(blocks["state_residual"], blocks[outcome], covariates)
            rows.append({"sample": sample, "family": family, "outcome": outcome, "n_blocks": n,
                         "covariates_n": k, "rho": rho, "p_value": p})
    result = pd.DataFrame(rows)
    result["fdr_bh_within_patient"] = result.groupby("family")["p_value"].transform(bh)
    return result


def random_effects(group: pd.DataFrame) -> dict:
    valid = group[(group["n_blocks"] - group["covariates_n"] - 3 > 0) & group["rho"].notna()].copy()
    z = np.arctanh(np.clip(valid["rho"].to_numpy(), -0.999999, 0.999999))
    variance = 1 / (valid["n_blocks"].to_numpy() - valid["covariates_n"].to_numpy() - 3)
    fixed_w = 1 / variance
    fixed = np.sum(fixed_w * z) / np.sum(fixed_w)
    q = float(np.sum(fixed_w * (z - fixed) ** 2))
    df = len(z) - 1
    c_value = np.sum(fixed_w) - np.sum(fixed_w ** 2) / np.sum(fixed_w)
    tau2 = max(0, (q - df) / c_value) if c_value > 0 else 0
    weights = 1 / (variance + tau2)
    pooled = np.sum(weights * z) / np.sum(weights)
    se = math.sqrt(1 / np.sum(weights))
    return {"patients_n": len(valid), "meta_rho": math.tanh(pooled),
            "ci_low": math.tanh(pooled - 1.96 * se), "ci_high": math.tanh(pooled + 1.96 * se),
            "p_value": float(2 * norm.sf(abs(pooled / se))), "tau2_DL": tau2, "Q": q,
            "Q_p_value": float(chi2.sf(q, df)) if df > 0 else np.nan,
            "I2_percent": max(0, (q - df) / q) * 100 if q > 0 else 0,
            "direction_concordant": bool(valid["rho"].gt(0).all() or valid["rho"].lt(0).all())}


def meta_analysis(associations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, outcome), group in associations.groupby(["family", "outcome"]):
        rows.append({"family": family, "outcome": outcome, **random_effects(group)})
    result = pd.DataFrame(rows)
    result["fdr_bh"] = result.groupby("family")["p_value"].transform(bh)
    return result.sort_values(["family", "fdr_bh", "p_value"])


def figures(spots: pd.DataFrame, associations: pd.DataFrame, meta: pd.DataFrame) -> None:
    features = ["state_residual", "cancer_total", "caf_total", "local_cd8", "interface"]
    titles = ["ARL11-TAM state residual", "Cancer proportion", "CAF proportion", "Local CD8", "Computed interface"]
    cmaps = ["magma", "Reds", "Greens", "Blues", "coolwarm"]
    fig, axes = plt.subplots(4, 5, figsize=(15, 11.5), layout="constrained")
    for row, sample in enumerate(SAMPLES):
        data = spots[spots["sample"].eq(sample)]
        for col, (feature, title, cmap) in enumerate(zip(features, titles, cmaps)):
            ax = axes[row, col]
            values = data[feature].astype(float)
            if feature == "interface":
                vmin, vmax = 0, 1
            else:
                vmin, vmax = np.nanpercentile(values, [2, 98])
            ax.scatter(data["x"], data["y"], c=values, s=4, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
            ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_axis_off()
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.text(-0.04, 0.5, f"{sample}\n{SAMPLES[sample]}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("WP19 spatial structure: measured Visium spots only")
    fig.savefig(FIG_DIR / "wp19_spatial_structure_maps.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "wp19_spatial_structure_maps.pdf", bbox_inches="tight")
    plt.close(fig)

    outcomes = [item for values in OUTCOMES.values() for item in values]
    matrix = associations.pivot(index="outcome", columns="sample", values="rho").reindex(outcomes)
    matrix["Random-effects meta"] = meta.set_index("outcome")["meta_rho"]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Partial Spearman rho", shrink=0.8)
    ax.set_title("Patient-level spatial-block associations with frozen state residual")
    fig.savefig(FIG_DIR / "wp19_spatial_association_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "wp19_spatial_association_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    signature = pd.read_csv(ROOT / "output" / "tables" / "macrophage_reclustering" /
                            "gse176078_arl11_high_cluster_signature.csv")["names"].drop_duplicates().tolist()
    spot_tables, block_tables, coverage_rows, association_tables = [], [], [], []
    for sample, subtype in SAMPLES.items():
        spots, blocks, coverage = process_sample(sample, subtype, signature)
        spot_tables.append(spots); block_tables.append(blocks); coverage_rows.extend(coverage)
        association_tables.append(patient_associations(sample, blocks))
    spots = pd.concat(spot_tables, ignore_index=True)
    blocks = pd.concat(block_tables, ignore_index=True)
    associations = pd.concat(association_tables, ignore_index=True)
    meta = meta_analysis(associations)
    sensitivity_rows = []
    for grid_q in [3, 5]:
        grid_associations = []
        for sample, subtype in SAMPLES.items():
            sample_spots = spots[spots["sample"].eq(sample)].copy()
            grid_blocks = aggregate_blocks(sample_spots, grid_q)
            grid_blocks["sample"] = sample
            grid_blocks["subtype"] = subtype
            current = patient_associations(sample, grid_blocks)
            current["grid_q"] = grid_q
            grid_associations.append(current)
        grid_associations = pd.concat(grid_associations, ignore_index=True)
        grid_meta = meta_analysis(grid_associations)
        grid_meta["grid_q"] = grid_q
        sensitivity_rows.append(grid_meta)
    sensitivity_meta = pd.concat(sensitivity_rows, ignore_index=True)
    spots.to_csv(PROCESSED_DIR / "wu_wp19_spot_scores.csv.gz", index=False, compression="gzip")
    blocks.to_csv(PROCESSED_DIR / "wu_wp19_spatial_blocks.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(TABLE_DIR / "gene_set_coverage.csv", index=False)
    associations.to_csv(TABLE_DIR / "patient_block_associations.csv", index=False)
    meta.to_csv(TABLE_DIR / "random_effects_meta.csv", index=False)
    sensitivity_meta.to_csv(TABLE_DIR / "block_grid_sensitivity_meta.csv", index=False)
    histology = (spots.groupby(["sample", "histology_region"], observed=True)
                 .agg(n_spots=("state_residual", "size"), mean_state_residual=("state_residual", "mean"),
                      interface_fraction=("interface", "mean"), mean_cancer=("cancer_total", "mean"),
                      mean_caf=("caf_total", "mean")).reset_index())
    histology.to_csv(TABLE_DIR / "histology_region_summary.csv", index=False)
    figures(spots, associations, meta)

    positive_boundary = meta[(meta["family"].eq("boundary")) & (meta["fdr_bh"] < 0.05) &
                             meta["direction_concordant"]]
    cd8 = meta[meta["family"].eq("cd8_relation")]
    summary = {"analysis_date": "2026-08-18", "patients_n": len(SAMPLES), "spots_n": int(len(spots)),
               "blocks_n": int(len(blocks)), "frozen_signature_n": len(signature),
               "state_adjustment": ["generic_macro_score", "myeloid", "technical_QC_PC1"],
               "boundary_fdr_concordant_outcomes": positive_boundary["outcome"].tolist(),
               "boundary_gate_pass": bool(len(positive_boundary) >= 1),
               "cd8_fdr_0_05_outcomes": cd8.loc[cd8["fdr_bh"] < 0.05, "outcome"].tolist(),
               "limitations": ["Only four patients from one study", "Visium spots are multicellular mixtures",
                               "Computed interface uses within-section deconvolution tertiles",
                               "Spatial blocks reduce but do not eliminate spatial autocorrelation"]}
    (TABLE_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nRandom-effects meta:\n", meta.to_string(index=False))


if __name__ == "__main__":
    main()
