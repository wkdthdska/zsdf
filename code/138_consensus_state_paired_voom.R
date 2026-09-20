#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(edgeR)
  library(limma)
})

root <- normalizePath(Sys.getenv("ARL11_PROJECT_ROOT", unset = "."), winslash = "/", mustWork = TRUE)
input_dir <- file.path(root, "data", "processed", "consensus_macrophage_state")
output_dir <- file.path(root, "output", "tables", "wp23_consensus_state")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

run_analysis <- function(cohort, state_min_cells) {
  prefix <- tolower(cohort)
  counts_samples_by_gene <- read.delim(
    gzfile(file.path(input_dir, paste0(prefix, "_state_pseudobulk_counts.tsv.gz"))),
    row.names = 1, check.names = FALSE
  )
  metadata <- read.csv(
    file.path(input_dir, paste0(prefix, "_state_pseudobulk_metadata.csv")),
    row.names = 1, check.names = FALSE
  )
  metadata <- metadata[rownames(counts_samples_by_gene), , drop = FALSE]
  cell_counts <- reshape(
    metadata[, c("patient_id", "state", "n_cells")], idvar = "patient_id",
    timevar = "state", direction = "wide"
  )
  state_col <- "n_cells.ARL11_marked_state"
  other_col <- "n_cells.other_macrophages"
  eligible_donors <- cell_counts$patient_id[
    !is.na(cell_counts[[state_col]]) & !is.na(cell_counts[[other_col]]) &
      cell_counts[[state_col]] >= state_min_cells & cell_counts[[other_col]] >= 20
  ]
  keep_samples <- metadata$patient_id %in% eligible_donors
  metadata <- metadata[keep_samples, , drop = FALSE]
  counts <- t(as.matrix(counts_samples_by_gene[keep_samples, , drop = FALSE]))
  storage.mode(counts) <- "integer"

  metadata$patient_id <- factor(metadata$patient_id)
  metadata$state <- factor(metadata$state, levels = c("other_macrophages", "ARL11_marked_state"))
  design <- model.matrix(~ patient_id + state, data = metadata)
  coefficient <- "stateARL11_marked_state"
  stopifnot(coefficient %in% colnames(design))

  y <- DGEList(counts = counts)
  keep_genes <- filterByExpr(y, design = design, min.count = 5, min.total.count = 15)
  y <- y[keep_genes, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y, method = "TMM")
  v <- voom(y, design, plot = FALSE)
  fit <- eBayes(lmFit(v, design), robust = TRUE, trend = FALSE)
  result <- topTable(fit, coef = coefficient, number = Inf, sort.by = "none")
  result$gene <- rownames(result)
  result$SE <- ifelse(abs(result$t) > 1e-12, abs(result$logFC / result$t), NA_real_)
  result$cohort <- cohort
  result$state_min_cells <- state_min_cells
  result$paired_donors_n <- length(eligible_donors)
  result <- result[, c("cohort", "state_min_cells", "paired_donors_n", "gene", "logFC", "SE",
                       "AveExpr", "t", "P.Value", "adj.P.Val", "B")]
  write.csv(
    result,
    file.path(output_dir, paste0(prefix, "_paired_state_voom_min", state_min_cells, ".csv")),
    row.names = FALSE
  )
  qc <- data.frame(
    cohort = cohort, state_min_cells = state_min_cells,
    paired_donors_n = length(eligible_donors), samples_n = nrow(metadata),
    genes_before_filter = nrow(counts), genes_after_filter = nrow(y),
    residual_df = fit$df.residual[1],
    min_library_size = min(colSums(counts)), median_library_size = median(colSums(counts))
  )
  write.csv(qc, file.path(output_dir, paste0(prefix, "_paired_state_voom_qc_min", state_min_cells, ".csv")), row.names = FALSE)
  qc
}

all_qc <- do.call(rbind, lapply(c("GSE176078", "GSE161529"), function(cohort) {
  do.call(rbind, lapply(c(5, 10), function(threshold) run_analysis(cohort, threshold)))
}))
write.csv(all_qc, file.path(output_dir, "paired_voom_qc_summary.csv"), row.names = FALSE)
print(all_qc)
