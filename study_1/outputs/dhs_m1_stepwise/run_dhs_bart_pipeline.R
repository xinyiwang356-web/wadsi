#!/usr/bin/env Rscript

run_dhs_bart_pipeline <- function() {
  script_arg <- grep("^--file=", commandArgs(), value = TRUE)
  study_dir <- if (length(script_arg)) {
    dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
  } else {
    getwd()
  }
  config_path <- file.path(study_dir, "config_seed_asd_harmonization.yaml")
  if (!file.exists(config_path)) {
    stop("Run this script from study_1 or invoke it by its full path.")
  }

  required_packages <- c(
    "yaml", "readr", "dbarts", "ggplot2", "dplyr", "posterior"
  )
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages)) {
    stop(
      "Missing R packages: ", paste(missing_packages, collapse = ", "),
      ". Install them before running this pipeline."
    )
  }

  source(file.path(study_dir, "R", "compute_effects.R"), local = TRUE)
  config <- yaml::read_yaml(config_path)
  analysis <- config$analysis
  imputation <- config$imputation
  predictors <- analysis$predictors
  predictor_names <- vapply(predictors, `[[`, character(1), "name")
  covariate_names <- unlist(analysis$adjustment_covariates, use.names = FALSE)
  model_variables <- c(predictor_names, covariate_names)
  target_names <- unlist(imputation$target_variables, use.names = FALSE)
  overlap_names <- unlist(imputation$overlap_variables, use.names = FALSE)
  if (anyDuplicated(model_variables) ||
      anyDuplicated(c(overlap_names, target_names)) ||
      !setequal(model_variables, c(overlap_names, target_names))) {
    stop("Configured predictors and covariates must cover each overlap and target once.")
  }
  id_name <- analysis$identifier
  outcome_name <- analysis$synthetic_outcome$name
  m_count <- 5L
  input_dir <- file.path(study_dir, "outputs", "imputed_dhs")
  result_dir <- file.path(study_dir, analysis$output_directory)
  model_dir <- file.path(study_dir, analysis$model_directory)
  dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  log_connection <- file(file.path(result_dir, "dhs_bart_pipeline.log"), open = "wt")
  sink(log_connection, split = TRUE)
  sink(log_connection, type = "message")
  on.exit({
    sink(type = "message")
    sink()
    close(log_connection)
  }, add = TRUE)

  cat("DHS BART pipeline: synthetic demonstration only.\n")
  cat("DHS source data do not contain observed ASD outcomes.\n")
  cat("All reported outcome-based quantities are smoke-test outputs, not scientific estimates.\n")

  input_paths <- file.path(
    input_dir,
    sprintf("imputed_dhs_m%d.csv", seq_len(m_count))
  )
  missing_inputs <- input_paths[!file.exists(input_paths)]
  if (length(missing_inputs)) {
    stop("Missing imputed DHS inputs: ", paste(missing_inputs, collapse = ", "))
  }

  raw_datasets <- lapply(input_paths, function(path) {
    readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
  })
  names(raw_datasets) <- paste0("m", seq_len(m_count))

  required_columns <- unique(c(
    id_name, "is_dhs", model_variables, target_names
  ))
  for (m in seq_len(m_count)) {
    missing_columns <- setdiff(required_columns, names(raw_datasets[[m]]))
    if (length(missing_columns)) {
      stop(
        "Imputation m", m, " is missing required columns: ",
        paste(missing_columns, collapse = ", ")
      )
    }
    data <- raw_datasets[[m]]
    data <- data[!is.na(data$is_dhs) & data$is_dhs == 1, , drop = FALSE]
    if (!nrow(data)) {
      stop("Imputation m", m, " contains no DHS rows (is_dhs == 1).")
    }
    raw_datasets[[m]] <- data
  }

  ids <- as.character(raw_datasets[[1]][[id_name]])
  if (anyNA(ids) || anyDuplicated(ids)) {
    stop("The DHS identifier must be non-missing and unique.")
  }
  for (m in seq_len(m_count)) {
    current_ids <- as.character(raw_datasets[[m]][[id_name]])
    if (!identical(current_ids, ids)) {
      stop("DHS identifiers or row ordering differ in imputation m", m, ".")
    }
  }

  set.seed(as.integer(analysis$synthetic_outcome$seed))
  synthetic_y <- stats::rbinom(
    length(ids), size = 1L,
    prob = analysis$synthetic_outcome$prevalence
  )
  synthetic_missing <- stats::runif(length(ids)) <
    analysis$synthetic_outcome$missingness_rate
  synthetic_y[synthetic_missing] <- NA_integer_
  outcome_labels <- data.frame(
    RECORD_ID = ids,
    ASD_flag = synthetic_y
  )
  names(outcome_labels) <- c(id_name, outcome_name)
  readr::write_csv(
    outcome_labels,
    file.path(result_dir, "synthetic_outcome_labels.csv")
  )
  cat(
    "Generated one shared independent Bernoulli outcome and MCAR mask; ",
    sum(synthetic_y == 1L, na.rm = TRUE), " observed positives, ",
    sum(is.na(synthetic_y)), " missing outcomes among ", length(ids), " DHS rows.\n",
    sep = ""
  )

  variable_types <- list()
  for (name in names(config$variables)) {
    variable_types[[name]] <- config$variables[[name]]$type
  }
  for (name in names(imputation$variable_types)) {
    variable_types[[name]] <- imputation$variable_types[[name]]
  }
  for (predictor in predictors) {
    variable_types[[predictor$name]] <- predictor$type
  }
  categorical_names <- names(variable_types)[
    vapply(variable_types, identical, logical(1), "categorical")
  ]
  categorical_names <- intersect(categorical_names, model_variables)

  normalize_category <- function(x) {
    value <- trimws(as.character(x))
    numeric_value <- suppressWarnings(as.numeric(value))
    is_integer <- !is.na(numeric_value) &
      abs(numeric_value - round(numeric_value)) < 1e-8
    value[is_integer] <- as.character(as.integer(round(numeric_value[is_integer])))
    value
  }

  category_levels <- lapply(categorical_names, function(name) {
    configured <- imputation$categories[[name]]
    if (is.null(configured)) {
      configured <- config$variables[[name]]$recode_to_seed$seed_categories
      if (!is.null(configured)) {
        configured <- names(configured)
      }
    }
    explicit <- if (is.null(configured)) character() else
      normalize_category(unlist(configured, use.names = FALSE))
    observed <- unlist(lapply(raw_datasets, function(data) {
      normalize_category(unique(data[[name]]))
    }), use.names = FALSE)
    levels <- unique(c(explicit, observed))
    numeric_levels <- suppressWarnings(as.numeric(levels))
    if (length(levels) && !anyNA(numeric_levels)) {
      levels <- levels[order(numeric_levels)]
    } else {
      levels <- sort(levels)
    }
    if (!length(levels)) {
      stop("No categories found for ", name, ".")
    }
    levels
  })
  names(category_levels) <- categorical_names

  datasets <- lapply(seq_len(m_count), function(m) {
    data <- raw_datasets[[m]]
    for (name in setdiff(model_variables, categorical_names)) {
      data[[name]] <- suppressWarnings(as.numeric(data[[name]]))
    }
    for (name in categorical_names) {
      values <- normalize_category(data[[name]])
      data[[name]] <- factor(values, levels = category_levels[[name]])
      if (anyNA(data[[name]])) {
        stop("Unexpected or missing category in ", name, " for imputation m", m, ".")
      }
    }
    if (anyNA(data[, target_names, drop = FALSE])) {
      stop("Target variables contain missing values in imputation m", m, ".")
    }
    if (anyNA(data[, model_variables, drop = FALSE])) {
      stop("Model variables contain missing values in imputation m", m, ".")
    }
    data[[outcome_name]] <- outcome_labels[[outcome_name]]
    data
  })
  names(datasets) <- paste0("m", seq_len(m_count))

  analytic_datasets <- lapply(datasets, function(data) {
    data[!is.na(data[[outcome_name]]), c(id_name, model_variables, outcome_name),
         drop = FALSE]
  })
  if (!all(vapply(analytic_datasets, nrow, integer(1)) ==
           nrow(analytic_datasets[[1]]))) {
    stop("Outcome missingness is not aligned across imputations.")
  }
  observed_prevalence <- mean(analytic_datasets[[1]][[outcome_name]] == 1L)
  if (!is.finite(observed_prevalence) ||
      observed_prevalence <= 0 || observed_prevalence >= 1) {
    stop("Synthetic outcome must contain both observed outcome classes.")
  }
  cat("Observed synthetic outcome prevalence: ",
      sprintf("%.6f", observed_prevalence), ".\n", sep = "")

  effect_config <- list(
    outcome = list(name = outcome_name),
    predictors = predictors,
    population_priors = list(
      pY1 = analysis$population_prevalence,
      pY0 = 1 - analysis$population_prevalence
    )
  )
  nchain <- as.integer(analysis$bart$nchain)
  nskip <- as.integer(analysis$bart$nskip)
  ndpost <- as.integer(analysis$bart$ndpost)
  keepevery <- as.integer(analysis$bart$keepevery)
  if (keepevery < 1L || ndpost %% keepevery != 0L) {
    stop("BART ndpost must be divisible by a positive keepevery value.")
  }
  retained_per_chain <- ndpost %/% keepevery
  expected_draws <- nchain * retained_per_chain
  chunk_size <- 400L

  prediction_matrix <- function(fit, newdata) {
    predicted <- as.matrix(stats::predict(fit, newdata = newdata))
    n_people <- nrow(newdata)
    if (is.null(dim(predicted))) {
      if (n_people == 1L && length(predicted) == expected_draws) {
        predicted <- matrix(predicted, ncol = 1L)
      } else {
        stop("BART predict() returned an unexpected vector shape.")
      }
    } else if (ncol(predicted) == n_people) {
      # dbarts returns posterior draws by prediction row.
    } else if (nrow(predicted) == n_people) {
      predicted <- t(predicted)
    } else {
      stop("BART predict() returned an unexpected matrix shape.")
    }
    if (nrow(predicted) != expected_draws) {
      stop(
        "Expected ", expected_draws, " posterior predictions, got ",
        nrow(predicted), ". Check dbarts thinning settings."
      )
    }
    pmin(pmax(predicted, 1e-6), 1 - 1e-6)
  }

  set_feature_value <- function(data, name, value) {
    if (is.factor(data[[name]])) {
      data[[name]] <- factor(
        rep(as.character(value), nrow(data)),
        levels = levels(data[[name]])
      )
    } else {
      data[[name]] <- rep(as.numeric(value), nrow(data))
    }
    data
  }

  contrasts <- list()
  for (predictor in predictors) {
    name <- predictor$name
    if (identical(predictor$type, "continuous")) {
      all_values <- unlist(lapply(analytic_datasets, `[[`, name), use.names = FALSE)
      limits <- stats::quantile(all_values, c(0.1, 0.9), na.rm = TRUE, names = FALSE)
      contrasts[[length(contrasts) + 1L]] <- list(
        name = name, type = "continuous",
        alternative = limits[[2]], reference = limits[[1]]
      )
    } else {
      level_set <- category_levels[[name]]
      counts <- Reduce(`+`, lapply(analytic_datasets, function(data) {
        table(factor(as.character(data[[name]]), levels = level_set))
      }))
      reference <- names(counts)[which.max(counts)]
      alternatives <- setdiff(names(counts)[counts > 0], reference)
      for (alternative in alternatives) {
        observed_in_any_imputation <- any(vapply(
          analytic_datasets,
          function(data) any(as.character(data[[name]]) == alternative),
          logical(1)
        ))
        if (observed_in_any_imputation) {
          contrasts[[length(contrasts) + 1L]] <- list(
            name = name, type = "categorical",
            alternative = alternative, reference = reference
          )
        }
      }
    }
  }
  contrast_keys <- vapply(contrasts, function(x) {
    paste(x$name, x$alternative, "vs", x$reference, sep = ":")
  }, character(1))

  population_rr <- setNames(vector("list", length(contrasts)), contrast_keys)
  individual_rr <- setNames(vector("list", length(contrasts)), contrast_keys)
  profile_size <- min(
    as.integer(analysis$individual_profile_sample),
    nrow(analytic_datasets[[1]])
  )
  set.seed(as.integer(analysis$synthetic_outcome$seed) + 1L)
  profile_indices <- sort(sample(seq_len(nrow(analytic_datasets[[1]])), profile_size))
  profile_ids <- analytic_datasets[[1]][[id_name]][profile_indices]

  calculate_population_rr <- function(fit, data, contrast) {
    numerator <- denominator <- numeric(expected_draws)
    for (start in seq.int(1L, nrow(data), by = chunk_size)) {
      rows <- start:min(start + chunk_size - 1L, nrow(data))
      chunk <- data[rows, model_variables, drop = FALSE]
      alt <- set_feature_value(chunk, contrast$name, contrast$alternative)
      ref <- set_feature_value(chunk, contrast$name, contrast$reference)
      p_alt <- prediction_matrix(fit, alt)
      p_ref <- prediction_matrix(fit, ref)
      rr <- (p_alt / (1 - p_alt)) / (p_ref / (1 - p_ref))
      numerator_weight <- p_ref * analysis$population_prevalence /
        observed_prevalence
      denominator_weight <- numerator_weight +
        (1 - p_ref) * (1 - analysis$population_prevalence) /
        (1 - observed_prevalence)
      weights <- numerator_weight / denominator_weight
      numerator <- numerator + rowSums(rr * weights)
      denominator <- denominator + rowSums(weights)
    }
    numerator / denominator
  }

  calculate_individual_rr <- function(fit, data, contrast) {
    sample_data <- data[profile_indices, model_variables, drop = FALSE]
    alt <- set_feature_value(sample_data, contrast$name, contrast$alternative)
    ref <- set_feature_value(sample_data, contrast$name, contrast$reference)
    p_alt <- prediction_matrix(fit, alt)
    p_ref <- prediction_matrix(fit, ref)
    (p_alt / (1 - p_alt)) / (p_ref / (1 - p_ref))
  }

  for (m in seq_len(m_count)) {
    label <- paste0("m", m)
    cat("\nFitting DHS BART model ", label, " of ", m_count, ".\n", sep = "")
    data <- analytic_datasets[[m]]
    x_train <- data[, model_variables, drop = FALSE]
    y_train <- as.integer(data[[outcome_name]])
    model_path <- file.path(model_dir, paste0("dhs_bart_fit_", label, ".rds"))
    fit <- dbarts::bart(
      x.train = x_train,
      y.train = y_train,
      keeptrees = TRUE,
      verbose = FALSE,
      nchain = nchain,
      nthread = nchain,
      nskip = nskip,
      ndpost = ndpost,
      keepevery = keepevery,
      seed = as.integer(analysis$bart$seed) + m - 1L
    )
    if (!is.null(fit$fit$storeState)) {
      fit$fit$storeState()
    }
    saveRDS(fit, model_path)
    cat("Saved model: ", model_path, "\n", sep = "")

    latent <- fit$yhat.train
    if (is.null(dim(latent))) {
      stop("Fitted model does not expose posterior training predictions.")
    }
    if (nrow(latent) != expected_draws && ncol(latent) == expected_draws) {
      latent <- t(latent)
    }
    if (nrow(latent) != expected_draws || ncol(latent) != nrow(data)) {
      stop("Unexpected dbarts training prediction dimensions.")
    }
    draw_mean <- numeric(expected_draws)
    fitted_mean <- numeric(nrow(data))
    for (start in seq.int(1L, nrow(data), by = chunk_size)) {
      rows <- start:min(start + chunk_size - 1L, nrow(data))
      probability <- stats::pnorm(latent[, rows, drop = FALSE])
      draw_mean <- draw_mean + rowMeans(probability) *
        length(rows) / nrow(data)
      fitted_mean[rows] <- colMeans(probability)
    }
    trace_data <- do.call(rbind, lapply(seq_len(nchain), function(chain) {
      chain_rows <- ((chain - 1L) * retained_per_chain + 1L):
        (chain * retained_per_chain)
      data.frame(
        iteration = seq_len(retained_per_chain),
        mean_probability = draw_mean[chain_rows],
        chain = factor(paste("Chain", chain))
      )
    }))
    chain_draws <- matrix(
      draw_mean,
      nrow = retained_per_chain,
      ncol = nchain
    )
    convergence <- data.frame(
      quantity = "mean_fitted_probability",
      rhat = posterior::rhat(chain_draws),
      ess_bulk = posterior::ess_bulk(chain_draws),
      ess_tail = posterior::ess_tail(chain_draws),
      nchain = nchain,
      retained_draws_per_chain = retained_per_chain
    )
    readr::write_csv(
      convergence,
      file.path(result_dir, paste0("dhs_bart_convergence_", label, ".csv"))
    )
    cat(
      "Convergence diagnostics for ", label, ": R-hat=",
      signif(convergence$rhat[[1]], 5),
      ", bulk ESS=", round(convergence$ess_bulk[[1]]),
      ", tail ESS=", round(convergence$ess_tail[[1]]), ".\n",
      sep = ""
    )
    trace_plot <- ggplot2::ggplot(
      trace_data,
      ggplot2::aes(iteration, mean_probability, colour = chain)
    ) +
      ggplot2::geom_line(linewidth = 0.45) +
      ggplot2::facet_wrap(~chain, ncol = 2) +
      ggplot2::labs(
        title = paste("DHS synthetic BART trace -", label),
        x = "Retained iteration",
        y = "Mean fitted probability"
      ) +
      ggplot2::theme_minimal()
    ggplot2::ggsave(
      file.path(result_dir, paste0("dhs_bart_trace_", label, ".png")),
      trace_plot, width = 9, height = 6, dpi = 150
    )

    calibration_data <- data.frame(
      p_hat = fitted_mean,
      y = y_train
    )
    calibration_data$bin <- dplyr::ntile(calibration_data$p_hat, 10)
    calibration_summary <- dplyr::summarise(
      dplyr::group_by(calibration_data, bin),
      n = dplyr::n(),
      mean_predicted = mean(p_hat),
      observed_rate = mean(y),
      .groups = "drop"
    )
    calibration_plot <- ggplot2::ggplot(
      calibration_summary,
      ggplot2::aes(mean_predicted, observed_rate)
    ) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      ggplot2::geom_point(ggplot2::aes(size = n)) +
      ggplot2::coord_equal() +
      ggplot2::labs(
        title = paste("Synthetic-outcome calibration -", label),
        x = "Mean predicted probability",
        y = "Observed event rate"
      ) +
      ggplot2::theme_minimal()
    ggplot2::ggsave(
      file.path(result_dir, paste0("dhs_bart_calibration_", label, ".png")),
      calibration_plot, width = 7, height = 6, dpi = 150
    )
    readr::write_csv(
      calibration_summary,
      file.path(result_dir, paste0("dhs_bart_calibration_", label, ".csv"))
    )
    rm(latent, fitted_mean, calibration_data, calibration_plot, trace_plot)
    gc(verbose = FALSE)

    for (i in seq_along(contrasts)) {
      key <- contrast_keys[[i]]
      cat("Estimating posterior draws for ", key, " in ", label, ".\n", sep = "")
      population_rr[[key]][[m]] <- calculate_population_rr(
        fit, data, contrasts[[i]]
      )
      individual_rr[[key]][[m]] <- calculate_individual_rr(
        fit, data, contrasts[[i]]
      )
    }
    rm(fit)
    gc(verbose = FALSE)
  }

  summarize_draws <- function(draws) {
    c(
      median = stats::median(draws),
      lower = unname(stats::quantile(draws, 0.025)),
      upper = unname(stats::quantile(draws, 0.975))
    )
  }

  pooled_rr_rows <- lapply(seq_along(contrasts), function(i) {
    key <- contrast_keys[[i]]
    pooled <- unlist(population_rr[[key]], use.names = FALSE)
    summary <- summarize_draws(pooled)
    data.frame(
      predictor = contrasts[[i]]$name,
      alternative = as.character(contrasts[[i]]$alternative),
      reference = as.character(contrasts[[i]]$reference),
      pooled_draws = length(pooled),
      PRR_median = summary[["median"]],
      lower_95 = summary[["lower"]],
      upper_95 = summary[["upper"]]
    )
  })
  pooled_prr <- do.call(rbind, pooled_rr_rows)
  readr::write_csv(pooled_prr, file.path(result_dir, "pooled_prr_summary.csv"))

  individual_rr_rows <- lapply(seq_along(contrasts), function(i) {
    key <- contrast_keys[[i]]
    draws <- do.call(rbind, individual_rr[[key]])
    data.frame(
      profile_id = profile_ids,
      predictor = contrasts[[i]]$name,
      alternative = as.character(contrasts[[i]]$alternative),
      reference = as.character(contrasts[[i]]$reference),
      RR_median = apply(draws, 2, stats::median),
      lower_95 = apply(draws, 2, stats::quantile, probs = 0.025),
      upper_95 = apply(draws, 2, stats::quantile, probs = 0.975)
    )
  })
  individual_rr_summary <- do.call(rbind, individual_rr_rows)
  names(individual_rr_summary)[names(individual_rr_summary) == "profile_id"] <- id_name
  readr::write_csv(
    individual_rr_summary,
    file.path(result_dir, "pooled_individual_relative_risks.csv")
  )
  rr_plot <- ggplot2::ggplot(
    individual_rr_summary,
    ggplot2::aes(x = predictor, y = RR_median)
  ) +
    ggplot2::geom_boxplot(outlier.alpha = 0.2) +
    ggplot2::scale_y_log10() +
    ggplot2::labs(
      title = "Pooled individual conditional relative risks (sampled profiles)",
      x = NULL,
      y = "Posterior median relative risk (log scale)"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))
  ggplot2::ggsave(
    file.path(result_dir, "pooled_individual_relative_risks.png"),
    rr_plot, width = 9, height = 6, dpi = 150
  )

  reference_values <- setNames(vector("list", length(predictors)), predictor_names)
  for (predictor in predictors) {
    name <- predictor$name
    if (identical(predictor$type, "continuous")) {
      row <- pooled_prr[pooled_prr$predictor == name, , drop = FALSE]
      reference_values[[name]] <- if (row$PRR_median[[1]] >= 1) {
        as.numeric(row$reference[[1]])
      } else {
        as.numeric(row$alternative[[1]])
      }
    } else {
      candidates <- unique(c(
        pooled_prr$reference[pooled_prr$predictor == name],
        pooled_prr$alternative[pooled_prr$predictor == name]
      ))
      risk_scores <- setNames(rep(1, length(candidates)), candidates)
      rows <- pooled_prr[pooled_prr$predictor == name, , drop = FALSE]
      for (j in seq_len(nrow(rows))) {
        risk_scores[rows$alternative[[j]]] <- rows$PRR_median[[j]]
      }
      reference_values[[name]] <- names(which.min(risk_scores))
    }
  }
  readr::write_csv(
    data.frame(
      predictor = names(reference_values),
      reference_value = vapply(reference_values, as.character, character(1))
    ),
    file.path(result_dir, "attributable_fraction_reference_profile.csv")
  )
  cat("Reference profile selected from pooled PRR posterior medians.\n")

  caf_sample_size <- min(
    as.integer(analysis$caf_profile_sample),
    nrow(analytic_datasets[[1]])
  )
  set.seed(as.integer(analysis$synthetic_outcome$seed) + 2L)
  caf_indices <- sort(sample(seq_len(nrow(analytic_datasets[[1]])), caf_sample_size))
  caf_ids <- analytic_datasets[[1]][[id_name]][caf_indices]
  case_indices <- which(analytic_datasets[[1]][[outcome_name]] == 1L)
  caf_draws_by_imputation <- vector("list", m_count)
  paf_draws_by_imputation <- vector("list", m_count)
  shapley_results <- vector("list", m_count)
  shapley_case_count <- min(
    as.integer(analysis$shapley$n_individuals),
    length(case_indices)
  )
  if (shapley_case_count < 1L) {
    stop("The synthetic outcome produced no observed cases for Shapley analysis.")
  }
  set.seed(as.integer(analysis$synthetic_outcome$seed) + 3L)
  shapley_indices <- sort(sample(case_indices, shapley_case_count))

  for (m in seq_len(m_count)) {
    label <- paste0("m", m)
    cat("\nComputing AF and Shapley posterior draws for ", label, ".\n", sep = "")
    model_path <- file.path(model_dir, paste0("dhs_bart_fit_", label, ".rds"))
    fit <- readRDS(model_path)
    data <- analytic_datasets[[m]]
    reference_data <- data[, model_variables, drop = FALSE]
    for (name in predictor_names) {
      reference_data <- set_feature_value(
        reference_data, name, reference_values[[name]]
      )
    }

    caf_sample_draws <- matrix(
      NA_real_, nrow = expected_draws, ncol = caf_sample_size
    )
    paf_numerator <- numeric(expected_draws)
    for (start in seq.int(1L, nrow(data), by = chunk_size)) {
      rows <- start:min(start + chunk_size - 1L, nrow(data))
      observed_probability <- prediction_matrix(
        fit, data[rows, model_variables, drop = FALSE]
      )
      reference_probability <- prediction_matrix(
        fit, reference_data[rows, , drop = FALSE]
      )
      caf <- 1 - reference_probability / observed_probability
      sampled_positions <- which(caf_indices >= start & caf_indices <= max(rows))
      if (length(sampled_positions)) {
        local_rows <- caf_indices[sampled_positions] - start + 1L
        caf_sample_draws[, sampled_positions] <- caf[, local_rows, drop = FALSE]
      }
      case_positions <- which(data[[outcome_name]][rows] == 1L)
      if (length(case_positions)) {
        paf_numerator <- paf_numerator +
          rowSums(caf[, case_positions, drop = FALSE])
      }
    }
    if (anyNA(caf_sample_draws)) {
      stop("Failed to compute sampled CAF draws for ", label, ".")
    }
    caf_draws_by_imputation[[m]] <- caf_sample_draws
    paf_draws_by_imputation[[m]] <- paf_numerator / length(case_indices)

    reference_case_draws <- prediction_matrix(
      fit,
      reference_data[shapley_indices, , drop = FALSE]
    )
    set.seed(as.integer(analysis$bart$seed) + 100L + m)
    shapley_results[[m]] <- compute_shapley_af(
      bart_fit = fit,
      analytic_df = data[, c(model_variables, outcome_name), drop = FALSE],
      config = effect_config,
      p_ref = reference_case_draws,
      n_individuals = shapley_case_count,
      n_samples = as.integer(analysis$shapley$n_permutations),
      n_donors = as.integer(analysis$shapley$n_donors),
      n_cores = 1L,
      individual_indices = shapley_indices,
      p_ref_indices = shapley_indices
    )
    if (is.null(shapley_results[[m]]$population_shapley_draws)) {
      stop("Shapley calculation found no cases in ", label, ".")
    }
    rm(fit)
    gc(verbose = FALSE)
  }

  pooled_caf_draws <- do.call(rbind, caf_draws_by_imputation)
  caf_medians <- apply(pooled_caf_draws, 2, stats::median)
  caf_summary <- data.frame(
    profile_id = caf_ids,
    CAF_posterior_median = caf_medians
  )
  names(caf_summary)[names(caf_summary) == "profile_id"] <- id_name
  readr::write_csv(caf_summary, file.path(result_dir, "sampled_individual_caf.csv"))
  caf_plot <- ggplot2::ggplot(caf_summary, ggplot2::aes(CAF_posterior_median)) +
    ggplot2::geom_histogram(bins = 40, fill = "#0072B2", colour = "white") +
    ggplot2::labs(
      title = "Pooled posterior CAF (sampled DHS profiles)",
      x = "Pooled posterior median CAF",
      y = "Profiles"
    ) +
    ggplot2::theme_minimal()
  ggplot2::ggsave(
    file.path(result_dir, "pooled_caf_histogram.png"),
    caf_plot, width = 8, height = 6, dpi = 150
  )

  pooled_paf_draws <- unlist(paf_draws_by_imputation, use.names = FALSE)
  paf_summary <- summarize_draws(pooled_paf_draws)
  readr::write_csv(
    data.frame(
      estimand = "PAF",
      pooled_draws = length(pooled_paf_draws),
      posterior_median = paf_summary[["median"]],
      lower_95 = paf_summary[["lower"]],
      upper_95 = paf_summary[["upper"]],
      observed_cases_per_imputation = length(case_indices)
    ),
    file.path(result_dir, "pooled_paf_summary.csv")
  )

  pooled_population_shapley <- do.call(
    rbind,
    lapply(shapley_results, `[[`, "population_shapley_draws")
  )
  colnames(pooled_population_shapley) <- predictor_names
  shapley_population_summary <- do.call(rbind, lapply(seq_along(predictor_names), function(j) {
    summary <- summarize_draws(pooled_population_shapley[, j])
    data.frame(
      feature = predictor_names[[j]],
      pooled_draws = nrow(pooled_population_shapley),
      posterior_median = summary[["median"]],
      lower_95 = summary[["lower"]],
      upper_95 = summary[["upper"]],
      sampled_cases = shapley_case_count
    )
  }))
  readr::write_csv(
    shapley_population_summary,
    file.path(result_dir, "pooled_population_shapley_summary.csv")
  )
  shapley_draw_output <- data.frame(
    draw = rep(seq_len(nrow(pooled_population_shapley)), times = ncol(pooled_population_shapley)),
    feature = rep(predictor_names, each = nrow(pooled_population_shapley)),
    contribution = as.vector(pooled_population_shapley)
  )
  readr::write_csv(
    shapley_draw_output,
    file.path(result_dir, "pooled_population_shapley_draws.csv")
  )

  individual_shapley_rows <- lapply(seq_along(shapley_indices), function(k) {
    draws <- do.call(rbind, lapply(shapley_results, function(result) {
      result$individual_shapley_draws[[k]]
    }))
    do.call(rbind, lapply(seq_along(predictor_names), function(j) {
      summary <- summarize_draws(draws[, j])
      data.frame(
        profile_id = analytic_datasets[[1]][[id_name]][shapley_indices[[k]]],
        feature = predictor_names[[j]],
        posterior_median = summary[["median"]],
        lower_95 = summary[["lower"]],
        upper_95 = summary[["upper"]]
      )
    }))
  })
  individual_shapley <- do.call(rbind, individual_shapley_rows)
  names(individual_shapley)[names(individual_shapley) == "profile_id"] <- id_name
  readr::write_csv(
    individual_shapley,
    file.path(result_dir, "pooled_individual_shapley.csv")
  )
  shapley_plot <- ggplot2::ggplot(
    individual_shapley,
    ggplot2::aes(x = posterior_median, y = feature)
  ) +
    ggplot2::geom_point(
      position = ggplot2::position_jitter(height = 0.15, width = 0),
      alpha = 0.65
    ) +
    ggplot2::labs(
      title = "Pooled Shapley contributions for sampled synthetic cases",
      x = "Pooled posterior median contribution",
      y = NULL
    ) +
    ggplot2::theme_minimal()
  ggplot2::ggsave(
    file.path(result_dir, "pooled_shapley_beeswarm.png"),
    shapley_plot, width = 8, height = 6, dpi = 150
  )

  readr::write_csv(
    data.frame(
      target_variable = target_names,
      missing_values = vapply(target_names, function(name) {
        sum(vapply(datasets, function(data) sum(is.na(data[[name]])), integer(1)))
      }, integer(1))
    ),
    file.path(result_dir, "target_missingness_check.csv")
  )
  readr::write_csv(
    data.frame(
      synthetic_only = TRUE,
      outcome_prevalence_parameter = analysis$synthetic_outcome$prevalence,
      outcome_missingness_parameter = analysis$synthetic_outcome$missingness_rate,
      observed_outcome_prevalence = observed_prevalence,
      observed_outcome_missingness = mean(is.na(synthetic_y)),
      DHS_rows = length(ids),
      observed_outcome_rows = nrow(analytic_datasets[[1]]),
      imputation_datasets = m_count,
      posterior_draws_per_imputation = expected_draws,
      pooled_draws_per_estimand = m_count * expected_draws,
      shapley_cases = shapley_case_count,
      shapley_permutations = analysis$shapley$n_permutations,
      shapley_donors = analysis$shapley$n_donors
    ),
    file.path(result_dir, "dhs_analysis_metadata.csv")
  )
  cat("\nPipeline completed. All results are synthetic-outcome smoke tests only.\n")
  cat("Results directory: ", result_dir, "\n", sep = "")
}

run_dhs_bart_pipeline()
