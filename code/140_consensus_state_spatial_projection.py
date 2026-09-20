#!/usr/bin/env python
"""Project the frozen donor-aware consensus state into the existing Wu Visium cohort."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARL11_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "pipeline" / "python-library"))

import pandas as pd

source = ROOT / "scripts" / "122_wp19_spatial_interface_validation.py"
spec = importlib.util.spec_from_file_location("wp19_base", source)
wp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(wp)

TABLE = ROOT / "output" / "tables" / "wp24_consensus_spatial"
FIG = ROOT / "figures" / "wp24_consensus_spatial"
PROCESSED = ROOT / "data" / "processed" / "wp24_consensus_spatial"
DOC = ROOT / "docs"
for directory in (TABLE, FIG, PROCESSED, DOC):
    directory.mkdir(parents=True, exist_ok=True)
wp.TABLE_DIR = TABLE
wp.FIG_DIR = FIG
wp.PROCESSED_DIR = PROCESSED


signature_table = pd.read_csv(
    ROOT / "output" / "tables" / "wp23_consensus_state" / "consensus_macrophage_remodeling_signature.csv"
)
signature = signature_table["gene"].drop_duplicates().tolist()

spot_tables, block_tables, coverage_rows, association_tables = [], [], [], []
for sample, subtype in wp.SAMPLES.items():
    spots, blocks, coverage = wp.process_sample(sample, subtype, signature)
    spot_tables.append(spots); block_tables.append(blocks); coverage_rows.extend(coverage)
    association_tables.append(wp.patient_associations(sample, blocks))
spots = pd.concat(spot_tables, ignore_index=True)
blocks = pd.concat(block_tables, ignore_index=True)
associations = pd.concat(association_tables, ignore_index=True)
meta = wp.meta_analysis(associations)

sensitivity_rows = []
for grid_q in [3, 5]:
    grid_associations = []
    for sample, subtype in wp.SAMPLES.items():
        sample_spots = spots[spots["sample"].eq(sample)].copy()
        grid_blocks = wp.aggregate_blocks(sample_spots, grid_q)
        grid_blocks["sample"] = sample; grid_blocks["subtype"] = subtype
        current = wp.patient_associations(sample, grid_blocks)
        current["grid_q"] = grid_q
        grid_associations.append(current)
    current_meta = wp.meta_analysis(pd.concat(grid_associations, ignore_index=True))
    current_meta["grid_q"] = grid_q
    sensitivity_rows.append(current_meta)
sensitivity = pd.concat(sensitivity_rows, ignore_index=True)

spots.to_csv(PROCESSED / "wu_consensus_state_spot_scores.csv.gz", index=False, compression="gzip")
blocks.to_csv(PROCESSED / "wu_consensus_state_blocks.csv", index=False)
pd.DataFrame(coverage_rows).to_csv(TABLE / "gene_set_coverage.csv", index=False)
associations.to_csv(TABLE / "patient_block_associations.csv", index=False)
meta.to_csv(TABLE / "random_effects_meta.csv", index=False)
sensitivity.to_csv(TABLE / "block_grid_sensitivity_meta.csv", index=False)
wp.figures(spots, associations, meta)

expected = {
    "cancer_total": "positive", "local_cancer": "positive",
    "caf_total": "positive", "local_caf": "positive",
    "tumor_caf_coexistence": "positive", "boundary_proximity": "positive",
    "local_cd8": "negative", "distance_to_cd8_rich": "positive",
}
meta["expected_direction"] = meta["outcome"].map(expected)
meta["matches_expected_direction"] = (
    ((meta["expected_direction"] == "positive") & (meta["meta_rho"] > 0))
    | ((meta["expected_direction"] == "negative") & (meta["meta_rho"] < 0))
)
meta.to_csv(TABLE / "random_effects_meta_with_direction.csv", index=False)
positive = meta[(meta["fdr_bh"] < 0.05) & meta["direction_concordant"] & meta["matches_expected_direction"]]

coverage = pd.DataFrame(coverage_rows)
signature_coverage = coverage[coverage["gene_set"] == "frozen_ARL11_TAM_signature"]
summary = {
    "patients_n": len(wp.SAMPLES), "spots_n": int(len(spots)), "blocks_n": int(len(blocks)),
    "signature_genes_n": len(signature),
    "signature_genes_present_variable_per_sample": dict(zip(signature_coverage["sample"], signature_coverage["present_variable_n"])),
    "state_adjustment": ["generic_macro_score", "myeloid", "technical_QC_PC1"],
    "fdr_concordant_expected_outcomes": positive["outcome"].tolist(),
    "specific_spatial_niche_gate_pass": bool(len(positive) >= 1),
    "interpretation": "Visium is a multicellular spot assay; results concern spatial programs, not direct cell-cell distance.",
}
(TABLE / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# WP24：donor-aware consensus macrophage state的公共空间投射", "",
    "## 设计", "",
    "- Signature在任何空间结果查看前由两个scRNA队列的配对donor pseudobulk冻结。",
    "- 使用4例Wu et al. Visium；患者是重复单位，spots不作为独立患者。",
    "- State score调整generic macrophage score、Macrophage+Monocyte proportion及technical QC PC1。",
    "- 预设终点：tumor、CAF、tumor–CAF boundary、local CD8及distance to CD8-rich。",
    "- 4×4 blocks为主要分析，3×3和5×5为敏感性分析。", "",
    "## 主要结果", "",
]
for _, row in meta.iterrows():
    lines.append(
        f"- {row['outcome']}：meta rho={row['meta_rho']:.3f}（95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f}），"
        f"FDR={row['fdr_bh']:.3g}，I²={row['I2_percent']:.1f}%，direction concordant={'yes' if row['direction_concordant'] else 'no'}。"
    )
lines += ["", "## Gate", "", f"- 特定空间生态位升级：{'通过' if summary['specific_spatial_niche_gate_pass'] else '不通过'}。",
          "- 即使通过，Visium也不能证明ARL11+ macrophage与CD8/panCK/CAF的单细胞距离；该结论仍需single-cell spatial或multiplex IF。"]
(DOC / "wp24_consensus_state_spatial_projection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(meta.to_string(index=False))
