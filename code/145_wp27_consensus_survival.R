#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(survival)
  library(splines)
})

project <- normalizePath(Sys.getenv("ARL11_PROJECT_ROOT", unset = "."), winslash = "/", mustWork = TRUE)
input_dir <- file.path(project, "data", "processed", "wp27_consensus_prognosis")
output_dir <- file.path(project, "output", "tables", "wp27_consensus_prognosis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cohorts <- list(
  `TCGA-BRCA` = list(file = "tcga_brca_analysis_data.csv",
                     clinical = "age10_z + factor(stage) + strata(subtype)"),
  METABRIC = list(file = "metabric_analysis_data.csv",
                  clinical = "age10_z + factor(stage) + factor(grade) + strata(subtype)"),
  GSE96058 = list(file = "gse96058_analysis_data.csv",
                  clinical = "age10_z + tumor_size10_z + factor(grade) + factor(node) + strata(platform_subtype)")
)

z <- function(x) as.numeric(scale(as.numeric(x)))

fit_one <- function(data, cohort, score_type, model_type, clinical_terms) {
  if (score_type == "consensus") {
    raw_var <- "state_score_z"
    residual_var <- "state_residual_z"
  } else {
    raw_var <- "common_score_z"
    residual_var <- "common_residual_z"
  }
  if (model_type == "unadjusted") {
    predictor <- raw_var
    rhs <- raw_var
  } else if (model_type == "abundance_adjusted") {
    predictor <- residual_var
    rhs <- paste(residual_var, "+ generic_score_z")
  } else {
    predictor <- residual_var
    rhs <- paste(residual_var, "+ generic_score_z +", clinical_terms)
  }
  formula <- as.formula(paste("Surv(time, event) ~", rhs))
  variables <- all.vars(formula)
  analysis <- droplevels(data[complete.cases(data[, variables, drop = FALSE]), , drop = FALSE])
  analysis <- analysis[analysis$time > 0 & analysis$event %in% c(0, 1), , drop = FALSE]
  fit <- coxph(formula, data = analysis, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
  coefficient <- summary(fit)$coefficients[predictor, ]
  ci <- summary(fit)$conf.int[predictor, ]
  ph <- tryCatch(cox.zph(fit, transform = "km")$table, error = function(e) NULL)
  ph_chisq <- if (!is.null(ph) && predictor %in% rownames(ph)) ph[predictor, "chisq"] else NA_real_
  ph_p <- if (!is.null(ph) && predictor %in% rownames(ph)) ph[predictor, "p"] else NA_real_
  global_ph_p <- if (!is.null(ph) && "GLOBAL" %in% rownames(ph)) ph["GLOBAL", "p"] else NA_real_
  ph_detail <- if (!is.null(ph)) data.frame(
    cohort = cohort, score_type = score_type, model = model_type,
    term = rownames(ph), chisq = ph[, "chisq"], df = ph[, "df"], p_value = ph[, "p"],
    row.names = NULL, stringsAsFactors = FALSE
  ) else data.frame()
  events <- sum(analysis$event)
  parameters <- length(coef(fit))
  row <- data.frame(
    cohort = cohort, score_type = score_type, model = model_type,
    predictor = predictor, n = nrow(analysis), events = events,
    parameters = parameters, events_per_parameter = events / parameters,
    logHR = unname(coef(fit)[predictor]), se = coefficient["se(coef)"],
    HR = ci["exp(coef)"], CI_low = ci["lower .95"], CI_high = ci["upper .95"],
    p_value = coefficient["Pr(>|z|)"], concordance = summary(fit)$concordance[1],
    ph_chisq = ph_chisq, ph_p_value = ph_p, global_ph_p_value = global_ph_p,
    log_likelihood = fit$loglik[2], stringsAsFactors = FALSE
  )
  list(row = row, fit = fit, data = analysis, predictor = predictor, ph_detail = ph_detail)
}

cox_rows <- list()
increment_rows <- list()
nonlinear_rows <- list()
ph_rows <- list()
ph_robust_rows <- list()

fit_ph_robust <- function(data, cohort, score_type) {
  predictor <- if (score_type == "consensus") "state_residual_z" else "common_residual_z"
  if (cohort == "TCGA-BRCA") {
    formula <- as.formula(paste("Surv(time, event) ~", predictor,
                                "+ generic_score_z + age10_z + strata(stage) + strata(subtype)"))
    tt_function <- NULL
  } else if (cohort == "METABRIC") {
    formula <- as.formula(paste("Surv(time, event) ~", predictor,
                                "+ generic_score_z + age10_z + tt(age10_z) + strata(stage) + strata(grade) + strata(subtype)"))
    tt_function <- function(x, t, ...) x * log(pmax(t, 1e-6))
  } else {
    formula <- as.formula(paste("Surv(time, event) ~", predictor,
                                "+ generic_score_z + age10_z + tumor_size10_z + strata(grade) + strata(node) + strata(platform_subtype)"))
    tt_function <- NULL
  }
  variables <- all.vars(formula)
  analysis <- droplevels(data[complete.cases(data[, variables, drop = FALSE]), , drop = FALSE])
  analysis <- analysis[analysis$time > 0 & analysis$event %in% c(0, 1), , drop = FALSE]
  if (is.null(tt_function)) {
    fit <- coxph(formula, data = analysis, ties = "efron")
  } else {
    fit <- coxph(formula, data = analysis, ties = "efron", tt = tt_function)
  }
  coefficient <- summary(fit)$coefficients[predictor, ]
  ci <- summary(fit)$conf.int[predictor, ]
  data.frame(
    cohort = cohort, score_type = score_type, predictor = predictor,
    n = nrow(analysis), events = sum(analysis$event),
    logHR = unname(coef(fit)[predictor]), se = coefficient["se(coef)"],
    HR = ci["exp(coef)"], CI_low = ci["lower .95"], CI_high = ci["upper .95"],
    p_value = coefficient["Pr(>|z|)"], stringsAsFactors = FALSE
  )
}

for (cohort in names(cohorts)) {
  config <- cohorts[[cohort]]
  data <- read.csv(file.path(input_dir, config$file), stringsAsFactors = FALSE, na.strings = c("", "NA", "nan"))
  for (variable in c("state_score", "state_residual", "common_score", "common_residual", "generic_score", "age10", "tumor_size10")) {
    if (variable %in% names(data)) data[[paste0(variable, "_z")]] <- z(data[[variable]])
  }
  for (score_type in c("consensus", "common")) {
    for (model_type in c("unadjusted", "abundance_adjusted", "clinical")) {
      result <- fit_one(data, cohort, score_type, model_type, config$clinical)
      cox_rows[[length(cox_rows) + 1]] <- result$row
      ph_rows[[length(ph_rows) + 1]] <- result$ph_detail
      if (model_type == "clinical") {
        analysis <- result$data
        predictor <- result$predictor
        full_fit <- result$fit
        base_formula <- update(formula(full_fit), paste(". ~ . -", predictor))
        base_fit <- coxph(base_formula, data = analysis, ties = "efron", x = TRUE, y = TRUE)
        lr <- 2 * (full_fit$loglik[2] - base_fit$loglik[2])
        increment_rows[[length(increment_rows) + 1]] <- data.frame(
          cohort = cohort, score_type = score_type, n = nrow(analysis), events = sum(analysis$event),
          base_concordance = summary(base_fit)$concordance[1],
          full_concordance = summary(full_fit)$concordance[1],
          delta_concordance = summary(full_fit)$concordance[1] - summary(base_fit)$concordance[1],
          likelihood_ratio_chisq = lr, lr_p_value = pchisq(lr, df = 1, lower.tail = FALSE)
        )
        linear_formula <- formula(full_fit)
        spline_formula <- update(linear_formula, paste(". ~ . -", predictor, "+ ns(", predictor, ", df=3)"))
        spline_fit <- coxph(spline_formula, data = analysis, ties = "efron")
        nonlinear_lr <- 2 * (spline_fit$loglik[2] - full_fit$loglik[2])
        nonlinear_rows[[length(nonlinear_rows) + 1]] <- data.frame(
          cohort = cohort, score_type = score_type, n = nrow(analysis), events = sum(analysis$event),
          likelihood_ratio_chisq = max(0, nonlinear_lr), df = 2,
          p_value = pchisq(max(0, nonlinear_lr), df = 2, lower.tail = FALSE)
        )
      }
    }
    ph_robust_rows[[length(ph_robust_rows) + 1]] <- fit_ph_robust(data, cohort, score_type)
  }
}

cox_results <- do.call(rbind, cox_rows)
increment_results <- do.call(rbind, increment_rows)
nonlinear_results <- do.call(rbind, nonlinear_rows)
ph_results <- do.call(rbind, ph_rows)
ph_robust_results <- do.call(rbind, ph_robust_rows)
nonlinear_results$fdr_bh <- p.adjust(nonlinear_results$p_value, method = "BH")
write.csv(cox_results, file.path(output_dir, "cohort_cox_models.csv"), row.names = FALSE)
write.csv(increment_results, file.path(output_dir, "incremental_concordance.csv"), row.names = FALSE)
write.csv(nonlinear_results, file.path(output_dir, "nonlinearity_tests.csv"), row.names = FALSE)
write.csv(ph_results, file.path(output_dir, "cox_zph_all_terms.csv"), row.names = FALSE)
write.csv(ph_robust_results, file.path(output_dir, "ph_robust_sensitivity.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(output_dir, "R_session_info.txt"))
print(cox_results)
