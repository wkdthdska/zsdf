# ARL11 breast cancer macrophage-state analysis

This archive contains analysis code and de-identified, study-level result tables supporting the manuscript submitted to *Breast Cancer Research and Treatment*. It does not include local tissue-microarray clinical records, original immunohistochemistry images, or patient-level data.

## Scope

The code documents the donor-aware derivation of an ARL11-associated macrophage-remodeling signature, independent validation, public spatial analysis, endocrine-treatment analyses, and survival meta-analysis. The 30-gene signature is fixed before independent validation and does not include ARL11.

## Spatial-analysis interpretation

The WP23 field `spatial_projection_gate_pass` is a signature-construction eligibility gate, not a spatial-result test. It is passed when the cross-cohort signature meets its prespecified donor-level robustness, anchor-recovery, and functional-enrichment criteria; WP23 does not read spatial data.

WP24 is the subsequent, patient-level Visium analysis of the frozen signature. Its field `specific_spatial_niche_gate_pass` is `false`: none of the prespecified tumor-nest, CAF, boundary, or CD8-related outcomes met both the false-discovery-rate and direction-concordance criteria. Thus, this archive supports eligibility for spatial projection but not a reproducible state-specific spatial niche in the evaluated Visium cohort.

## Public data sources

- GSE176078: single-cell and spatial breast-cancer data. Processed spatial data are also available at Zenodo DOI 10.5281/zenodo.4739739.
- GSE161529: independent breast single-cell dataset.
- GSE169246: independent pretreatment breast-cancer single-cell validation dataset.
- POETIC: GSE105777 and GSE126870. GSE126870 includes expanded POETIC processed data and samples from GSE105777.
- GSE59515: serial neoadjuvant letrozole dataset.
- TCGA-BRCA, METABRIC, and GSE96058: external bulk-transcriptomic survival analyses.

## Reproduction boundary

Set `ARL11_PROJECT_ROOT` to the top-level project directory before running code. The repository does not redistribute public source matrices or local human-participant data. Users must obtain public source data from their original repositories and reconstruct the project `data/processed` inputs as specified by the scripts. Results included here are gene-level, cohort-level, or meta-analytic summaries and contain no local patient-level records.

## Suggested run order

1. `137_consensus_state_pseudobulk.py`
2. `138_consensus_state_paired_voom.R`
3. `139_consensus_state_signature.py`
4. `142_wp26_validate_consensus_state.py`
5. `122_wp19_spatial_interface_validation.py` and `140_consensus_state_spatial_projection.py`
6. `144_prepare_wp27_consensus_prognosis.py`, `145_wp27_consensus_survival.R`, and `146_summarize_wp27_consensus_prognosis.py`
7. `147_wp28_fixed_state_treatment_remodeling.py`

The scripts retain the original analysis logic but use `ARL11_PROJECT_ROOT` instead of the authors' local filesystem path.
