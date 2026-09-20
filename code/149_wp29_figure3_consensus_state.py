#!/usr/bin/env python
"""WP29 Figure 3 draft: donor-aware consensus macrophage state."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image


WP23 = ROOT / "output" / "tables" / "wp23_consensus_state"
OUT_TABLES = ROOT / "output" / "tables" / "wp29_manuscript_integration"
OUT_FIGURES = ROOT / "figures" / "wp29_main_figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

ORANGE = "#D96B16"
BLUE = "#2673B8"
RED = "#B2182B"
GREY = "#B8BDC2"
DARK_GREY = "#5E646B"
LIGHT_BLUE = "#E8F1F8"
LIGHT_ORANGE = "#FBE9D8"
ANCHORS = {"GPNMB", "LPL", "APOC1", "CYP27A1", "LIPA", "ACP5", "APOE"}


def crop_state_panel(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    # The source image contains three equal-width panels. Keep the right-most
    # state-location panel, including its title and axes.
    crop = image.crop((int(width * 0.665), 0, width, height))
    return np.asarray(crop)


def panel_label(ax, label: str):
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def draw_design(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        (0.02, 0.63, 0.20, 0.24, "Cluster cells\n(no ARL11 input)", LIGHT_BLUE),
        (0.27, 0.63, 0.20, 0.24, "Donor pseudobulk\nstate vs other TAMs", "#F1F3F5"),
        (0.52, 0.63, 0.20, 0.24, "Paired limma-voom\nper cohort", "#F1F3F5"),
        (0.77, 0.63, 0.20, 0.24, "Cross-cohort Tier A\nreplication", LIGHT_ORANGE),
    ]
    for x, y, w, h, text, color in steps:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014", facecolor=color, edgecolor=DARK_GREY, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)
    for x0, x1 in [(0.22, 0.27), (0.47, 0.52), (0.72, 0.77)]:
        ax.annotate("", xy=(x1, 0.75), xytext=(x0, 0.75), arrowprops={"arrowstyle": "->", "color": DARK_GREY, "lw": 1.4})

    final = patches.FancyBboxPatch((0.24, 0.13), 0.52, 0.25, boxstyle="round,pad=0.018", facecolor="#FFF4E8", edgecolor=ORANGE, linewidth=1.4)
    ax.add_patch(final)
    ax.text(0.50, 0.285, "Freeze portable 30-gene macrophage-state signature", ha="center", va="center", fontsize=8.7, fontweight="bold")
    ax.text(0.50, 0.205, "ARL11 not forced into the signature", ha="center", va="center", fontsize=9, color=RED)
    ax.annotate("", xy=(0.50, 0.39), xytext=(0.87, 0.63), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5})
    ax.text(0.02, 0.02, "Inferential unit: donor (not cell)", fontsize=8.5, color=DARK_GREY)
    ax.set_title("Donor-aware state construction", loc="left", fontsize=12, fontweight="bold")


def plot_concordance(ax, evidence: pd.DataFrame, signature_genes: set[str]):
    ax.scatter(evidence["gse176_logFC"], evidence["gse161_logFC"], s=7, color=GREY, alpha=0.35, linewidth=0)
    tier = evidence[evidence["tier_A"].astype(bool)]
    ax.scatter(tier["gse176_logFC"], tier["gse161_logFC"], s=18, color=ORANGE, alpha=0.72, linewidth=0, label="Tier A (n=190)")
    selected = evidence[evidence["gene"].isin(signature_genes)]
    ax.scatter(selected["gse176_logFC"], selected["gse161_logFC"], s=34, facecolor=BLUE, edgecolor="white", linewidth=0.4, zorder=4, label="Frozen 30 genes")
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.axvline(0, color="#333333", linewidth=0.7)
    label_offsets = {
        "LPL": (0.05, 0.08), "GPNMB": (0.05, 0.03), "CYP27A1": (0.05, 0.05),
        "APOC1": (0.04, -0.05), "LIPA": (-0.22, 0.08), "ACP5": (0.04, 0.02), "APOE": (-0.28, -0.05),
    }
    for _, row in evidence[evidence["gene"].isin(ANCHORS)].iterrows():
        dx, dy = label_offsets.get(row["gene"], (0.04, 0.04))
        ax.text(row["gse176_logFC"] + dx, row["gse161_logFC"] + dy, row["gene"], fontsize=7.2, color="#222222")
    ax.set_xlabel("GSE176078 paired-pseudobulk logFC")
    ax.set_ylabel("GSE161529 paired-pseudobulk logFC")
    ax.set_title("Cross-cohort gene-effect concordance", loc="left", fontsize=12, fontweight="bold")
    ax.text(0.98, 0.03, "Spearman rho=0.457\n10,341 common genes", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(color="#E3E6E8", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_signature(ax, signature: pd.DataFrame):
    frame = signature.copy().reset_index(drop=True)
    frame["rank"] = np.arange(len(frame))
    long = pd.concat(
        [
            frame[["gene", "rank", "gse176_logFC", "gse176_adj.P.Val"]].rename(columns={"gse176_logFC": "logFC", "gse176_adj.P.Val": "FDR"}).assign(cohort="GSE176078", xpos=0),
            frame[["gene", "rank", "gse161_logFC", "gse161_adj.P.Val"]].rename(columns={"gse161_logFC": "logFC", "gse161_adj.P.Val": "FDR"}).assign(cohort="GSE161529", xpos=1),
        ],
        ignore_index=True,
    )
    long["size"] = np.clip(-np.log10(long["FDR"].clip(lower=1e-300)), 2, 12) * 7.5
    norm = Normalize(vmin=0.5, vmax=max(3.0, float(long["logFC"].max())))
    scatter = ax.scatter(long["xpos"], long["rank"], s=long["size"], c=long["logFC"], cmap="YlOrRd", norm=norm, edgecolor="#555555", linewidth=0.25)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1], ["GSE176078", "GSE161529"])
    ax.set_yticks(frame["rank"], frame["gene"], fontsize=6.6)
    ax.invert_yaxis()
    for tick in ax.get_yticklabels():
        if tick.get_text() in ANCHORS:
            tick.set_fontweight("bold")
            tick.set_color(RED)
    ax.set_title("Frozen 30-gene state", loc="left", fontsize=12, fontweight="bold")
    ax.text(0.98, 1.01, "ARL11 excluded", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=RED)
    ax.grid(axis="y", color="#ECEEEF", linewidth=0.45)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("logFC", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def plot_enrichment(ax, ora: pd.DataFrame):
    selected = ora[(ora["foreground"] == "consensus_top30") & (ora["FDR"] < 0.05)].copy()
    selected = selected.sort_values("FDR", ascending=False)
    selected["label"] = selected["term"].str.replace(r" \(GO:\d+\)$", "", regex=True)
    selected["label"] = selected["label"].map(lambda x: "\n".join(textwrap.wrap(x, width=34)))
    selected["score"] = -np.log10(selected["FDR"])
    colors = [BLUE if db == "Hallmark_2020" else ORANGE for db in selected["database"]]
    ax.barh(np.arange(len(selected)), selected["score"], color=colors, alpha=0.88)
    ax.set_yticks(np.arange(len(selected)), selected["label"], fontsize=7.2)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title("Fixed-state functional enrichment", loc="left", fontsize=12, fontweight="bold")
    for y, (_, row) in enumerate(selected.iterrows()):
        ax.text(row["score"] + 0.03, y, f"{int(row['overlap_n'])} genes", va="center", fontsize=7)
    ax.axvline(-np.log10(0.05), color="#555555", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="#E3E6E8", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return selected


def main():
    evidence = pd.read_csv(WP23 / "cross_cohort_gene_evidence.csv.gz")
    signature = pd.read_csv(WP23 / "consensus_macrophage_remodeling_signature.csv")
    ora = pd.read_csv(WP23 / "consensus_signature_focused_ora.csv")
    signature_genes = set(signature["gene"])

    image1 = crop_state_panel(ROOT / "figures" / "macrophage_reclustering" / "gse176078_macrophage_reclustering.png")
    image2 = crop_state_panel(ROOT / "figures" / "macrophage_reclustering" / "gse161529_macrophage_reclustering.png")

    fig = plt.figure(figsize=(17.0, 11.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.86, 1.25], width_ratios=[1.05, 1.05, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    for ax, image, title, subtitle in [
        (ax_a, image1, "GSE176078 discovery state", "cluster 7 | 749 cells | 23 donors"),
        (ax_b, image2, "GSE161529 replication state", "cluster 6 | 2,384 cells | 30 donors"),
    ]:
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=8.5, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"})

    draw_design(ax_c)
    plot_concordance(ax_d, evidence, signature_genes)
    plot_signature(ax_e, signature)
    selected_ora = plot_enrichment(ax_f, ora)

    for label, ax in zip("ABCDEF", [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        panel_label(ax, label)
    fig.suptitle("Donor-aware analysis defines a reproducible ARL11-associated macrophage remodeling state", fontsize=16, fontweight="bold")
    fig.savefig(OUT_FIGURES / "figure3_consensus_state_draft.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / "figure3_consensus_state_draft.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {"metric": "primary_paired_donors_GSE176078", "value": 16},
            {"metric": "primary_paired_donors_GSE161529", "value": 26},
            {"metric": "common_tested_genes", "value": 10341},
            {"metric": "cross_cohort_logFC_spearman_rho", "value": 0.45728337499266103},
            {"metric": "tier_A_genes", "value": 190},
            {"metric": "frozen_signature_genes", "value": 30},
            {"metric": "ARL11_in_signature", "value": False},
            {"metric": "consensus_top30_FDR_significant_terms", "value": len(selected_ora)},
        ]
    )
    summary.to_csv(OUT_TABLES / "figure3_consensus_state_statistics.csv", index=False)


if __name__ == "__main__":
    main()
