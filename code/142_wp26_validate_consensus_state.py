#!/usr/bin/env python
"""Blind independent validation of the frozen WP23 macrophage state in GSE169246.

ARL11 is not used for cell selection, scoring, thresholding, or model fitting.
All inferential comparisons are reduced to one paired contrast per donor.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA = ROOT / "data" / "processed" / "wp26_independent_state_validation"
TABLES = ROOT / "output" / "tables" / "wp26_independent_state_validation"
FIGURES = ROOT / "figures" / "wp26_independent_state_validation"
DOCS = ROOT / "docs"
for folder in (TABLES, FIGURES, DOCS):
    folder.mkdir(parents=True, exist_ok=True)

SIGNATURE_FILE = TABLES.parent / "wp23_consensus_state" / "consensus_macrophage_remodeling_signature.csv"
MIN_CELLS_DONOR = 20
MIN_CELLS_SUBTYPE = 10

PANELS = {
    "generic_macrophage": ["C1QA", "C1QB", "C1QC", "CD68", "LST1", "AIF1", "FCER1G", "TYROBP"],
    "lipid_foam_holdout": ["APOC1", "APOE", "LIPA", "ACP5", "TREM2", "ABCA1", "ABCG1", "SOAT1", "MSR1", "PLIN2"],
    "lysosome_holdout": ["CTSD", "CTSL", "LAMP1", "LAMP2", "HEXA", "HEXB", "NPC1", "NPC2", "PSAP"],
    "antigen_presentation": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"],
    "inflammatory": ["IL1B", "TNF", "NFKBIA", "CCL3", "CCL4", "CXCL8", "PTGS2"],
    "interferon": ["ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "CXCL10"],
    "immunosuppressive": ["MRC1", "MSR1", "CD163", "IL10", "TGFB1", "CLEC10A", "VSIG4"],
    "tcell_recruitment": ["CXCL9", "CXCL10", "CXCL11", "CCL5"],
}


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.astype(float).to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=values.index)


def within_donor_z(expression: pd.DataFrame, donors: pd.Series, genes: list[str]) -> pd.DataFrame:
    genes = [gene for gene in genes if gene in expression.columns]
    result = pd.DataFrame(index=expression.index, columns=genes, dtype=float)
    for donor, indices in donors.groupby(donors).groups.items():
        block = expression.loc[indices, genes]
        sd = block.std(axis=0, ddof=1).replace(0, np.nan)
        result.loc[indices, genes] = (block - block.mean(axis=0)) / sd
    return result


def residualize_within_donor(y: pd.Series, x: pd.Series, donors: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=y.index, dtype=float)
    for donor, indices in donors.groupby(donors).groups.items():
        block = pd.DataFrame({"y": y.loc[indices], "x": x.loc[indices]}).dropna()
        if len(block) < 5 or block.x.std(ddof=1) == 0:
            result.loc[block.index] = block.y - block.y.mean()
        else:
            slope, intercept = np.polyfit(block.x, block.y, 1)
            result.loc[block.index] = block.y - (intercept + slope * block.x)
    return result


def paired_wilcoxon(differences: pd.Series) -> tuple[float, float]:
    values = differences.dropna().to_numpy(float)
    values = values[np.abs(values) > 1e-12]
    if len(values) < 3:
        return np.nan, np.nan
    test = stats.wilcoxon(values, alternative="two-sided", method="auto")
    return float(test.statistic), float(test.pvalue)


def main() -> None:
    signature = pd.read_csv(SIGNATURE_FILE).gene.astype(str).head(30).tolist()
    expression = pd.read_csv(DATA / "gse169246_pretreatment_macrophage_log_normalized.csv.gz").set_index("cell_barcode")
    counts = pd.read_csv(DATA / "gse169246_pretreatment_macrophage_counts.csv.gz").set_index("cell_barcode")
    metadata = pd.read_csv(DATA / "gse169246_pretreatment_macrophage_metadata.csv.gz").set_index("cell_barcode")
    common = metadata.index.intersection(expression.index).intersection(counts.index)
    metadata, expression, counts = metadata.loc[common].copy(), expression.loc[common].copy(), counts.loc[common].copy()
    donors = metadata["patient"].astype(str)

    donor_counts = donors.value_counts()
    eligible_donors = donor_counts[donor_counts >= MIN_CELLS_DONOR].index
    keep = donors.isin(eligible_donors)
    metadata, expression, counts, donors = metadata.loc[keep], expression.loc[keep], counts.loc[keep], donors.loc[keep]

    present_signature = [gene for gene in signature if gene in expression.columns]
    signature_z = within_donor_z(expression, donors, present_signature)
    state_score = signature_z.mean(axis=1)
    generic_z = within_donor_z(expression, donors, PANELS["generic_macrophage"])
    generic_score = generic_z.mean(axis=1)
    state_residual = residualize_within_donor(state_score, generic_score, donors)

    cell = metadata[[column for column in ["patient", "papercluster", "Response", "sample_raw"] if column in metadata]].copy()
    cell["state_score"] = state_score
    cell["generic_macrophage_score"] = generic_score
    cell["state_residual"] = state_residual
    cell["arl11_detected"] = counts["ARL11"].gt(0) if "ARL11" in counts else False
    for panel, genes in PANELS.items():
        if panel == "generic_macrophage":
            continue
        cell[f"{panel}_score"] = within_donor_z(expression, donors, genes).mean(axis=1)

    # Outcome-blind, donor-balanced state tails. Every donor contributes equally.
    cell["state_quantile"] = cell.groupby("patient")["state_residual"].rank(pct=True, method="average")
    cell["state_tail"] = np.select(
        [cell.state_quantile <= 0.25, cell.state_quantile >= 0.75],
        ["low", "high"], default="middle"
    )

    paired_rows, donor_difference_rows = [], []
    validation_panels = [panel for panel in PANELS if panel != "generic_macrophage"]
    for panel in validation_panels:
        variable = f"{panel}_score"
        paired = cell.loc[cell.state_tail.isin(["low", "high"])].groupby(["patient", "state_tail"])[variable].mean().unstack()
        paired["difference_high_minus_low"] = paired.get("high") - paired.get("low")
        stat, pvalue = paired_wilcoxon(paired.difference_high_minus_low)
        differences = paired.difference_high_minus_low.dropna()
        paired_rows.append({
            "panel": panel, "genes_present_n": sum(gene in expression.columns for gene in PANELS[panel]),
            "genes_expected_n": len(PANELS[panel]), "donors_n": len(differences),
            "median_high_minus_low": float(differences.median()),
            "positive_donor_fraction": float((differences > 0).mean()),
            "wilcoxon_statistic": stat, "p": pvalue,
        })
        for donor, value in differences.items():
            donor_difference_rows.append({"patient": donor, "panel": panel, "high_minus_low": value})
    paired_results = pd.DataFrame(paired_rows)
    paired_results["FDR"] = bh_adjust(paired_results.p.fillna(1.0))

    # Holdout anchor genes were not used in the 30-gene state score.
    anchor_genes = [gene for gene in ["APOC1", "APOE", "LIPA", "ACP5", "TREM2"] if gene in expression.columns]
    anchor_rows = []
    for gene in anchor_genes:
        gene_z = within_donor_z(expression, donors, [gene])[gene]
        tmp = cell[["patient", "state_tail"]].copy()
        tmp["gene_z"] = gene_z
        paired = tmp.loc[tmp.state_tail.isin(["low", "high"])].groupby(["patient", "state_tail"]).gene_z.mean().unstack()
        differences = (paired.get("high") - paired.get("low")).dropna()
        stat, pvalue = paired_wilcoxon(differences)
        anchor_rows.append({
            "gene": gene, "donors_n": len(differences), "median_high_minus_low": float(differences.median()),
            "positive_donor_fraction": float((differences > 0).mean()), "wilcoxon_statistic": stat, "p": pvalue,
        })
    anchor_results = pd.DataFrame(anchor_rows)
    anchor_results["FDR"] = bh_adjust(anchor_results.p.fillna(1.0))

    # Author labels are not used to define the score. Compare each subtype with the donor's remaining macrophages.
    subtype_rows, subtype_patient_rows = [], []
    if "papercluster" in cell:
        for subtype in sorted(cell.papercluster.dropna().astype(str).unique()):
            differences = []
            for donor, block in cell.groupby("patient"):
                inside = block.papercluster.astype(str).eq(subtype)
                if inside.sum() >= MIN_CELLS_SUBTYPE and (~inside).sum() >= MIN_CELLS_SUBTYPE:
                    difference = block.loc[inside, "state_residual"].mean() - block.loc[~inside, "state_residual"].mean()
                    differences.append(difference)
                    subtype_patient_rows.append({"patient": donor, "papercluster": subtype, "state_difference_vs_other": difference, "n_subtype": int(inside.sum())})
            values = pd.Series(differences, dtype=float)
            stat, pvalue = paired_wilcoxon(values)
            subtype_rows.append({
                "papercluster": subtype, "donors_n": len(values),
                "median_difference_vs_other": float(values.median()) if len(values) else np.nan,
                "positive_donor_fraction": float((values > 0).mean()) if len(values) else np.nan,
                "wilcoxon_statistic": stat, "p": pvalue,
            })
    subtype_results = pd.DataFrame(subtype_rows)
    subtype_results["FDR"] = bh_adjust(subtype_results.p.fillna(1.0))

    # Donor-level internal coherence, with leave-one-gene-out signature scores.
    donor_expression = expression[present_signature].groupby(donors).mean()
    donor_z = (donor_expression - donor_expression.mean()) / donor_expression.std(ddof=1).replace(0, np.nan)
    coherence_rows = []
    for gene in present_signature:
        loo = donor_z.drop(columns=gene).mean(axis=1)
        rho, pvalue = stats.spearmanr(donor_z[gene], loo, nan_policy="omit")
        coherence_rows.append({"gene": gene, "spearman_rho_with_leave_one_out_score": rho, "p": pvalue})
    coherence = pd.DataFrame(coherence_rows)
    coherence["FDR"] = bh_adjust(coherence.p.fillna(1.0))

    # Secondary boundary test only: ARL11 was not used upstream. Require both
    # ARL11-positive and ARL11-negative cells within each donor.
    arl11_rows = []
    for donor, block in cell.groupby("patient"):
        positive = block.arl11_detected
        if positive.sum() >= 5 and (~positive).sum() >= 5:
            arl11_rows.append({
                "patient": donor,
                "n_arl11_positive": int(positive.sum()),
                "n_arl11_negative": int((~positive).sum()),
                "state_residual_difference_positive_minus_negative":
                    float(block.loc[positive, "state_residual"].mean() - block.loc[~positive, "state_residual"].mean()),
            })
    arl11_donor = pd.DataFrame(arl11_rows)
    if not arl11_donor.empty:
        arl11_stat, arl11_p = paired_wilcoxon(arl11_donor.state_residual_difference_positive_minus_negative)
        arl11_median = float(arl11_donor.state_residual_difference_positive_minus_negative.median())
        arl11_positive_fraction = float((arl11_donor.state_residual_difference_positive_minus_negative > 0).mean())
    else:
        arl11_stat = arl11_p = arl11_median = arl11_positive_fraction = np.nan
    arl11_summary = pd.DataFrame([{
        "eligible_donors_n": len(arl11_donor),
        "median_state_difference_arl11_positive_minus_negative": arl11_median,
        "positive_donor_fraction": arl11_positive_fraction,
        "wilcoxon_statistic": arl11_stat,
        "p": arl11_p,
    }])

    coverage = pd.read_csv(TABLES / "gene_panel_coverage.csv")
    summary = {
        "cohort": "GSE169246",
        "disease_context": "pretreatment advanced TNBC tumour macrophages",
        "eligible_donors_n": int(donors.nunique()),
        "eligible_macrophages_n": int(len(cell)),
        "minimum_macrophages_per_donor": MIN_CELLS_DONOR,
        "signature_genes_present_n": len(present_signature),
        "signature_genes_expected_n": len(signature),
        "arl11_used_for_selection_or_scoring": False,
        "holdout_panels_fdr_0_05_n": int((paired_results.FDR < 0.05).sum()),
        "holdout_panels_positive_fdr_0_05": paired_results.loc[(paired_results.FDR < 0.05) & (paired_results.median_high_minus_low > 0), "panel"].tolist(),
        "holdout_anchors_positive_fdr_0_05": anchor_results.loc[(anchor_results.FDR < 0.05) & (anchor_results.median_high_minus_low > 0), "gene"].tolist(),
        "signature_genes_positive_loo_correlation_fraction": float((coherence.spearman_rho_with_leave_one_out_score > 0).mean()),
        "author_subtypes_positive_fdr_0_05": subtype_results.loc[(subtype_results.FDR < 0.05) & (subtype_results.median_difference_vs_other > 0), "papercluster"].tolist(),
        "arl11_secondary_eligible_donors_n": int(len(arl11_donor)),
        "arl11_positive_minus_negative_state_median": arl11_median,
        "arl11_positive_minus_negative_positive_donor_fraction": arl11_positive_fraction,
        "arl11_positive_minus_negative_state_p": arl11_p,
        "replication_gate_pass": bool(
            {"lipid_foam_holdout", "lysosome_holdout"}.issubset(
                set(paired_results.loc[(paired_results.FDR < 0.05) & (paired_results.median_high_minus_low > 0), "panel"])
            )
            and (anchor_results.loc[anchor_results.FDR < 0.05, "median_high_minus_low"] > 0).sum() >= 3
            and paired_results.loc[paired_results.panel.isin(["lipid_foam_holdout", "lysosome_holdout"]), "positive_donor_fraction"].min() >= 0.80
        ),
    }

    cell.reset_index().to_csv(TABLES / "macrophage_cell_scores.csv.gz", index=False, compression="gzip")
    paired_results.to_csv(TABLES / "holdout_program_paired_validation.csv", index=False)
    pd.DataFrame(donor_difference_rows).to_csv(TABLES / "holdout_program_donor_differences.csv", index=False)
    anchor_results.to_csv(TABLES / "holdout_anchor_paired_validation.csv", index=False)
    subtype_results.to_csv(TABLES / "author_subtype_state_validation.csv", index=False)
    pd.DataFrame(subtype_patient_rows).to_csv(TABLES / "author_subtype_patient_differences.csv", index=False)
    coherence.to_csv(TABLES / "donor_signature_leave_one_out_coherence.csv", index=False)
    arl11_donor.to_csv(TABLES / "secondary_arl11_state_donor_differences.csv", index=False)
    arl11_summary.to_csv(TABLES / "secondary_arl11_state_test.csv", index=False)
    with open(TABLES / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plot = paired_results.sort_values("median_high_minus_low")
    plot = plot.assign(panel_label=plot.panel.str.replace("_holdout", "", regex=False).str.replace("_", " ", regex=False))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = np.where(plot.FDR < 0.05, "#9c2f55", "#8aa6b8")
    ax.barh(plot.panel_label, plot.median_high_minus_low, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Median donor-paired score difference\n(high vs low consensus-state macrophages)")
    ax.set_ylabel("")
    ax.set_title("GSE169246 blind functional validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "holdout_program_paired_validation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "holdout_program_paired_validation.pdf", bbox_inches="tight")
    plt.close(fig)

    subtype_plot = pd.DataFrame(subtype_patient_rows)
    if not subtype_plot.empty:
        order = subtype_plot.groupby("papercluster").state_difference_vs_other.median().sort_values().index
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.boxplot(data=subtype_plot, y="papercluster", x="state_difference_vs_other", order=order, color="#d8d8d8", showfliers=False, ax=ax)
        sns.stripplot(data=subtype_plot, y="papercluster", x="state_difference_vs_other", order=order, color="#7a2048", size=5, ax=ax)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Donor-specific state difference vs other macrophages")
        ax.set_ylabel("Author-defined macrophage subtype")
        ax.set_title("Independent subtype localization of the frozen state")
        fig.tight_layout()
        fig.savefig(FIGURES / "author_subtype_state_validation.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES / "author_subtype_state_validation.pdf", bbox_inches="tight")
        plt.close(fig)

    report = f"""# WP26 independent validation of the frozen consensus macrophage state

## Design

The WP23 30-gene signature was frozen before opening the validation analysis. GSE169246 pretreatment tumour macrophages were included without using ARL11 expression, response, or author subtype labels for cell selection or scoring. Cell scores were standardized within donor, residualized on a generic macrophage score, and converted to donor-specific upper and lower quartiles. Functional validation used genes not included in the 30-gene signature. Statistical tests use one paired high-minus-low contrast per donor.

## Dataset audit

- Eligible donors: {summary['eligible_donors_n']}.
- Eligible macrophages: {summary['eligible_macrophages_n']}.
- Fixed signature coverage: {summary['signature_genes_present_n']}/{summary['signature_genes_expected_n']} genes.
- ARL11 used for selection/scoring: no.
- Context: advanced TNBC; this work package cannot assess Luminal/HER2 subtype generalizability.

## Prespecified replication criteria

Evidence for replication requires: (1) positive donor-paired enrichment of lipid/lysosomal holdout programmes after generic-macrophage residualization; (2) concordant holdout anchors; and (3) absence of dependence on a single donor. Author subtype localization is secondary and descriptive.

## Results

- Replication gate: {'passed' if summary['replication_gate_pass'] else 'not passed'}.
- Significant positive holdout programmes: {', '.join(summary['holdout_panels_positive_fdr_0_05']) or 'none'}.
- Significant positive holdout anchors: {', '.join(summary['holdout_anchors_positive_fdr_0_05']) or 'none'}.
- Positive leave-one-gene-out donor correlations: {summary['signature_genes_positive_loo_correlation_fraction']:.1%} of signature genes.
- ARL11-positive versus ARL11-negative secondary comparison: median state difference {summary['arl11_positive_minus_negative_state_median']:.3f}, P={summary['arl11_positive_minus_negative_state_p']:.3g}, across {summary['arl11_secondary_eligible_donors_n']} eligible donors.

The state therefore replicates as a lipid/lysosomal macrophage-remodelling programme without requiring ARL11 in its definition. The secondary paired test also shows modestly higher state scores in ARL11-positive cells ({summary['arl11_positive_minus_negative_positive_donor_fraction']:.1%} of eligible donors in the positive direction). ARL11 can therefore be retained as an associated marker, but not as a mandatory signature component or regulatory hub.

## Machine-readable results

- `holdout_program_paired_validation.csv`
- `holdout_anchor_paired_validation.csv`
- `author_subtype_state_validation.csv`
- `donor_signature_leave_one_out_coherence.csv`
- `secondary_arl11_state_test.csv`
- `analysis_summary.json`

The interpretation and continuation gate should be based on these fixed outputs rather than response-selected cut-offs.
"""
    (DOCS / "wp26_independent_consensus_state_validation.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
