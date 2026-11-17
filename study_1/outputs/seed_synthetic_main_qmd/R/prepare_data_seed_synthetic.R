coerce_variable_types <- function(raw_df, config) {
  vars_to_keep <- c(
    vapply(config$predictors, `[[`, character(1), "name"),
    vapply(config$covariates, `[[`, character(1), "name"),
    config$outcome$name
  )

  coerced_df <- raw_df[, vars_to_keep, drop = FALSE]
  type_map <- c(
    setNames(
      vapply(config$predictors, `[[`, character(1), "type"),
      vapply(config$predictors, `[[`, character(1), "name")
    ),
    setNames(
      vapply(config$covariates, `[[`, character(1), "type"),
      vapply(config$covariates, `[[`, character(1), "name")
    ),
    setNames(config$outcome$type, config$outcome$name)
  )

  for (variable in names(type_map)) {
    coerced_df[[variable]] <- switch(
      type_map[[variable]],
      continuous = as.numeric(coerced_df[[variable]]),
      categorical = as.factor(coerced_df[[variable]]),
      stop("Unsupported type for ", variable, ": ", type_map[[variable]])
    )
  }

  if (config$outcome$type == "categorical") {
    outcome_name <- config$outcome$name
    reference <- as.character(config$outcome$reference)
    outcome <- relevel(
      factor(as.character(coerced_df[[outcome_name]])),
      ref = reference
    )
    coerced_df[[outcome_name]] <- as.integer(outcome) - 1L
  }

  predictor_names <- vapply(
    config$predictors,
    function(predictor) predictor$name,
    character(1)
  )
  predictor_types <- vapply(
    config$predictors,
    function(predictor) predictor$type,
    character(1)
  )
  continuous_predictors <- predictor_names[predictor_types == "continuous"]

  predictor_stats_df <- if (length(continuous_predictors) > 0L) {
    do.call(rbind, lapply(continuous_predictors, function(variable) {
      data.frame(
        predictor = variable,
        mean = mean(coerced_df[[variable]], na.rm = TRUE),
        sd = stats::sd(coerced_df[[variable]], na.rm = TRUE),
        q10 = stats::quantile(
          coerced_df[[variable]], probs = 0.10, na.rm = TRUE
        ),
        q90 = stats::quantile(
          coerced_df[[variable]], probs = 0.90, na.rm = TRUE
        )
      )
    }))
  } else {
    data.frame(predictor = character(), mean = numeric(), sd = numeric())
  }
  rownames(predictor_stats_df) <- NULL

  list(
    coerced_df = coerced_df,
    predictor_stats_df = predictor_stats_df
  )
}

add_missingness_indicators <- function(coerced_df, config) {
  indicated_df <- coerced_df
  if (is.null(config$missingness_indicators) ||
      length(config$missingness_indicators) == 0L) {
    message("No missingness indicators specified in config.")
    return(list(indicated_df = indicated_df, config = config))
  }

  for (variable in config$missingness_indicators) {
    variable_type <- NULL
    variable_group <- NULL

    for (group in c("predictors", "covariates")) {
      variable_names <- vapply(
        config[[group]],
        `[[`,
        character(1),
        "name"
      )
      match_index <- which(variable_names == variable)
      if (length(match_index) > 0L) {
        variable_type <- config[[group]][[match_index[[1]]]]$type
        variable_group <- group
        break
      }
    }

    if (identical(variable_type, "continuous")) {
      missing_name <- paste0("miss_", variable)
      indicated_df[[missing_name]] <- factor(
        as.integer(is.na(indicated_df[[variable]])),
        levels = c(0, 1)
      )
      indicated_df[[variable]][is.na(indicated_df[[variable]])] <- 0
      config[[variable_group]] <- append(
        config[[variable_group]],
        list(list(name = missing_name, type = "categorical"))
      )
    } else if (identical(variable_type, "categorical")) {
      indicated_df[[variable]] <- forcats::fct_na_value_to_level(
        indicated_df[[variable]],
        level = "Missing"
      )
    } else {
      warning("Unsupported variable type for ", variable, ": ", variable_type)
    }
  }

  list(indicated_df = indicated_df, config = config)
}

impute_missing_data <- function(indicated_df, config) {
  imputed_df <- indicated_df
  outcome_name <- config$outcome$name
  outcome <- imputed_df[[outcome_name]]
  imputed_df[[outcome_name]] <- NULL

  if (!anyNA(imputed_df)) {
    imputed_df[[outcome_name]] <- outcome
    return(imputed_df)
  }

  mice_result <- mice::mice(imputed_df, m = 1, printFlag = FALSE)
  imputed_df <- mice::complete(mice_result, 1)
  imputed_df[[outcome_name]] <- outcome
  imputed_df
}

finalize_analytic_dataset <- function(imputed_df, config) {
  analytic_df <- imputed_df
  outcome_name <- config$outcome$name
  n_missing <- sum(is.na(analytic_df[[outcome_name]]))

  if (n_missing > 0L) {
    analytic_df <- analytic_df[!is.na(analytic_df[[outcome_name]]), ]
    cat(sprintf(
      "Dropped %d row%s with missing outcome (%s).\n",
      n_missing,
      ifelse(n_missing == 1L, "", "s"),
      outcome_name
    ))
  }

  analytic_df
}
