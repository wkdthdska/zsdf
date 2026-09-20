#!/usr/bin/env python
"""Derive a cross-cohort donor-aware consensus macrophage remodeling signature."""

from __future__ import annotations

import json
import re
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm, spearmanr

IN_DIR = ROOT / "output" / "tables" / "wp23_consensus_state"
GENESETS = ROOT / "data" / "processed" / "pathway_gene_sets"
FIG = ROOT / "figures" / "wp23_consensus_state"
DOC = ROOT / "docs"
FIG.mkdir(parents=True, exist_ok=True)

ANCHORS = ["GPNMB", "LPL", "APOC1", "CYP27A1", "LIPA", "ACP5", "TREM2", "APOE", "SPP1"]


def technical_gene(gene: str) -> bool:
    return bool(re.match(r"^(MT-|RPL|RPS|HBA|HBB|IGK|IGL|IGH)", gene)) or gene in {"MALAT1", "NEAT1"}


def bh(values: pd.Series) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p)); result[order] = np.minimum(adjusted, 1)
    return result


def read_gmt(path: Path) -> dict[str, set[str]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                result[fields[0]] = {x.strip().upper() for x in fields[2:] if x.strip()}
    return result


def random_effects(effect1: np.ndarray, se1: np.ndarray, effect2: np.ndarray, se2: np.ndarray):
    effects = np.vstack([effect1, effect2])
    variances = np.vstack([se1 ** 2, se2 ** 2])
    weights = 1 / variances
    fixed = np.sum(weights * effects, axis=0) / np.sum(weights, axis=0)
    q = np.sum(weights * (effects - fixed) ** 2, axis=0)
    c = np.sum(weights, axis=0) - np.sum(weights ** 2, axis=0) / np.sum(weights, axis=0)
    tau2 = np.maximum(0, (q - 1) / c)
    random_weights = 1 / (variances + tau2)
    pooled = np.sum(random_weights * effects, axis=0) / np.sum(random_weights, axis=0)
    pooled_se = np.sqrt(1 / np.sum(random_weights, axis=0))
    z = pooled / pooled_se
    p = 2 * norm.sf(np.abs(z))
    i2 = np.maximum(0, (q - 1) / np.maximum(q, 1e-12)) * 100
    return pooled, pooled_se, p, tau2, i2


def load(cohort: str, threshold: int) -> pd.DataFrame:
    frame = pd.read_csv(IN_DIR / f"{cohort.lower()}_paired_state_voom_min{threshold}.csv")
    frame["gene"] = frame["gene"].astype(str).str.upper()
    return frame.set_index("gene")


primary = {c: load(c, 5) for c in ["GSE176078", "GSE161529"]}
sensitivity = {c: load(c, 10) for c in ["GSE176078", "GSE161529"]}
common = sorted(set(primary["GSE176078"].index) & set(primary["GSE161529"].index))
common = [g for g in common if not technical_gene(g)]

combined = pd.DataFrame(index=common)
for cohort, short in [("GSE176078", "gse176"), ("GSE161529", "gse161")]:
    for column in ["logFC", "SE", "P.Value", "adj.P.Val", "AveExpr"]:
        combined[f"{short}_{column}"] = primary[cohort].reindex(common)[column]
    sens = sensitivity[cohort].reindex(common)
    combined[f"{short}_sensitivity_logFC"] = sens["logFC"]
    combined[f"{short}_sensitivity_FDR"] = sens["adj.P.Val"]

meta = random_effects(
    combined["gse176_logFC"].to_numpy(), combined["gse176_SE"].to_numpy(),
    combined["gse161_logFC"].to_numpy(), combined["gse161_SE"].to_numpy(),
)
combined["meta_logFC"], combined["meta_SE"], combined["meta_p"], combined["tau2"], combined["I2"] = meta
combined["meta_FDR"] = bh(combined["meta_p"])
combined["same_positive_direction"] = (combined["gse176_logFC"] > 0) & (combined["gse161_logFC"] > 0)
combined["sensitivity_same_positive"] = (
    (combined["gse176_sensitivity_logFC"] > 0) & (combined["gse161_sensitivity_logFC"] > 0)
)
combined["tier_A"] = (
    (combined.index != "ARL11")
    & (combined["gse176_logFC"] >= 0.5) & (combined["gse161_logFC"] >= 0.5)
    & (combined["gse176_adj.P.Val"] < 0.05) & (combined["gse161_adj.P.Val"] < 0.05)
    & (combined["gse176_sensitivity_logFC"] >= 0.25) & (combined["gse161_sensitivity_logFC"] >= 0.25)
)
combined["tier_B"] = (
    (combined.index != "ARL11") & ~combined["tier_A"]
    & (combined["gse176_logFC"] >= 0.25) & (combined["gse161_logFC"] >= 0.25)
    & combined["sensitivity_same_positive"] & (combined["meta_FDR"] < 0.05) & (combined["I2"] < 75)
)
combined["min_cohort_logFC"] = combined[["gse176_logFC", "gse161_logFC"]].min(axis=1)
combined.index.name = "gene"
combined.reset_index().to_csv(IN_DIR / "cross_cohort_gene_evidence.csv.gz", index=False, compression="gzip")

tier_a = combined[combined["tier_A"]].sort_values(["min_cohort_logFC", "meta_FDR"], ascending=[False, True])
tier_b = combined[combined["tier_B"]].sort_values(["min_cohort_logFC", "meta_FDR"], ascending=[False, True])
if len(tier_a) >= 10:
    portable = tier_a[~tier_a.index.str.match(r"^(ENSG|LINC)")].copy()
    portable["worst_cohort_FDR"] = portable[["gse176_adj.P.Val", "gse161_adj.P.Val"]].max(axis=1)
    selected = portable.sort_values(
        ["worst_cohort_FDR", "min_cohort_logFC"], ascending=[True, False]
    ).head(30).copy()
    selection_rule = "top 30 portable Tier-A genes ranked by worst-cohort FDR, then smaller-cohort logFC"
else:
    selected = pd.concat([tier_a, tier_b]).head(30).copy()
    selection_rule = "Tier A plus Tier B, capped at 30 because Tier A contained fewer than 10 genes"
selected.reset_index().to_csv(IN_DIR / "consensus_macrophage_remodeling_signature.csv", index=False)

anchor_table = combined.reindex(ANCHORS + ["ARL11"]).reset_index()
anchor_table.to_csv(IN_DIR / "prespecified_anchor_evidence.csv", index=False)

# Functional ORA uses the genes tested in both cohorts as the universe.
universe = set(common)
foregrounds = {
    "tier_A_robust_core": set(tier_a.index),
    "consensus_top30": set(selected.index),
}
databases = {
    "GO_BP_2026": GENESETS / "GO_Biological_Process_2026.gmt",
    "Reactome_2024": GENESETS / "Reactome_Pathways_2024.gmt",
    "Hallmark_2020": GENESETS / "MSigDB_Hallmark_2020.gmt",
}
rows = []
for foreground_name, foreground in foregrounds.items():
    for database, path in databases.items():
        for term, all_genes in read_gmt(path).items():
            genes = all_genes & universe
            if not 10 <= len(genes) <= 500:
                continue
            overlap = foreground & genes
            a = len(overlap); b = len(foreground) - a; c = len(genes) - a
            d = len(universe) - a - b - c
            odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append({
                "foreground": foreground_name, "database": database, "term": term,
                "foreground_n": len(foreground), "background_n": len(universe),
                "term_background_n": len(genes), "overlap_n": len(overlap),
                "overlap_genes": ";".join(sorted(overlap)), "odds_ratio": odds, "pvalue": pvalue,
            })
ora = pd.DataFrame(rows)
ora["FDR"] = ora.groupby(["foreground", "database"])["pvalue"].transform(bh)
ora["fold_enrichment"] = (ora["overlap_n"] / ora["foreground_n"]) / (ora["term_background_n"] / ora["background_n"])
ora = ora.sort_values(["foreground", "FDR", "pvalue"])
ora.to_csv(IN_DIR / "consensus_signature_ora.csv.gz", index=False, compression="gzip")
focus_pattern = r"LIPID|LIPOPROTEIN|CHOLESTEROL|LYSOSOM|PHAGOSOM|VESICLE|MACROPHAGE|EFFEROCYT|EXTRACELLULAR MATRIX|T CELL"
focused = ora[ora["term"].str.contains(focus_pattern, case=False, regex=True)].copy()
focused.to_csv(IN_DIR / "consensus_signature_focused_ora.csv", index=False)

anchor_pass = anchor_table.set_index("gene").reindex(ANCHORS)["tier_A"].fillna(False)
focused_sig = focused[(focused["foreground"] == "tier_A_robust_core") & (focused["FDR"] < 0.05)]
functional_pass = focused_sig["term"].str.contains(r"LIPID|CHOLESTEROL|LYSOSOM|PHAGOSOM", case=False, regex=True).any()
rho = spearmanr(combined["gse176_logFC"], combined["gse161_logFC"], nan_policy="omit")
gate = len(tier_a) >= 10 and int(anchor_pass.sum()) >= 3 and bool(functional_pass)
summary = {
    "primary_paired_donors": {"GSE176078": 16, "GSE161529": 26},
    "sensitivity_paired_donors": {"GSE176078": 13, "GSE161529": 23},
    "common_tested_genes_n": len(common),
    "cross_cohort_logFC_spearman_rho": float(rho.statistic),
    "cross_cohort_logFC_spearman_p": float(rho.pvalue),
    "tier_A_genes_n": len(tier_a), "tier_B_genes_n": len(tier_b),
    "selected_signature_n": len(selected), "selection_rule": selection_rule,
    "selected_signature_genes": selected.index.tolist(),
    "tier_A_prespecified_anchors": [gene for gene in ANCHORS if bool(anchor_pass.get(gene, False))],
    "arl11_forced_into_signature": False,
    "arl11_tier_A": bool(combined.loc["ARL11", "tier_A"]) if "ARL11" in combined.index else False,
    "focused_functional_terms_fdr_0_05_n": int(len(focused_sig)),
    "spatial_projection_gate_pass": bool(gate),
}
(IN_DIR / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(combined["gse176_logFC"], combined["gse161_logFC"], s=5, alpha=0.25, color="#7f7f7f", rasterized=True)
ax.scatter(tier_a["gse176_logFC"], tier_a["gse161_logFC"], s=14, alpha=0.8, color="#D55E00", label=f"Tier A (n={len(tier_a)})")
for gene in ANCHORS:
    if gene in combined.index:
        ax.text(combined.loc[gene, "gse176_logFC"], combined.loc[gene, "gse161_logFC"], gene, fontsize=7)
ax.axhline(0, color="black", linewidth=0.6); ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("GSE176078 paired-pseudobulk logFC")
ax.set_ylabel("GSE161529 paired-pseudobulk logFC")
ax.set_title(f"Donor-aware macrophage-state concordance (rho={rho.statistic:.2f})")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "cross_cohort_state_logfc_concordance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

top_terms = ora[(ora["foreground"] == "tier_A_robust_core") & (ora["FDR"] < 0.05)].head(15).copy()
if not top_terms.empty:
    top_terms = top_terms.sort_values("FDR", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top_terms))))
    ax.barh(top_terms["term"].str.slice(0, 70), -np.log10(top_terms["FDR"].clip(lower=1e-300)), color="#0072B2")
    ax.set_xlabel("-log10(FDR)"); ax.set_title("Consensus state functional enrichment")
    fig.tight_layout()
    fig.savefig(FIG / "consensus_state_ora.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

lines = [
    "# WP23：donor-aware consensus macrophage remodeling signature", "",
    "## 设计", "",
    "- 两个队列分别将原始counts聚合为donor × ARL11-enriched cluster/other macrophages配对pseudobulk。",
    "- 主分析要求state ≥5 cells且other macrophages ≥20 cells；敏感性分析state ≥10 cells。",
    "- 使用paired donor fixed-effects limma-voom；ARL11明确不进入signature。",
    "- Tier A要求两个队列logFC≥0.5且FDR<0.05，并在≥10-cell敏感性分析中两个队列logFC≥0.25。", "",
    "## 结果", "",
    f"- Tier A robust genes：{len(tier_a)}；Tier B：{len(tier_b)}。",
    f"- 跨队列全基因logFC Spearman rho={rho.statistic:.3f}，P={rho.pvalue:.3g}。",
    f"- 预设anchors中进入Tier A：{', '.join(summary['tier_A_prespecified_anchors']) or '无'}。",
    f"- 固定signature（{len(selected)} genes）：{', '.join(selected.index)}。",
    f"- ARL11是否强制进入：否；ARL11是否满足Tier A：{'是' if summary['arl11_tier_A'] else '否'}。",
    f"- 空间投射gate：{'通过' if gate else '不通过'}。", "",
    "## 解释边界", "",
    "该signature代表ARL11所标记、但不由ARL11单基因定义的可复现macrophage remodeling state。它不是ARL11 regulatory network。",
]
(DOC / "wp23_donor_aware_consensus_state.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
