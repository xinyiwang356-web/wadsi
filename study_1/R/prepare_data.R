# -----------------------------------------------------------------------------
# coerce_variable_types.R
#
# Description:
#   This function prepares an analytic dataset from a raw data frame (`raw_df`)
#   and a configuration object (`config`). It:
#     1. Keeps only predictors, covariates, and the outcome.
#     2. Coerces variables to the correct type (numeric or factor).
#     3. Converts the outcome to 0/1 only if categorical (for modeling).
#     4. Computes mean, SD, and an interpercentile range for continuous predictors.
#     5. Returns both the processed dataset and summary statistics.
#
# Design choice:
#   Categorical predictors and covariates are kept as factors to preserve
#   labeling and ensure correct handling by R’s modeling functions. Only the
#   outcome is binarized to 0/1 when needed for logistic or binary models.
#
# Inputs:
#   - raw_df:   Original data frame containing all study variables.
#   - config:   A list with:
#       * predictors: list of lists with elements 'name' and 'type'
#       * covariates: list of lists with elements 'name' and 'type'
#       * outcome: list with elements 'name', 'type', and 'reference' (if categorical)
#   - lower_prob: lower percentile for interpercentile range (default 0.10)
#   - upper_prob: upper percentile for interpercentile range (default 0.90)
#
# Output:
#   A list containing:
#       * coerced_df: coerced analytic dataset
#       * predictor_stats_df: mean, SD, and interdecile range
#
# Example usage:
#   result <- coerce_variable_types(raw_df, config)
#   result$coerced_df
#   result$predictor_stats_df
#
# -----------------------------------------------------------------------------
coerce_variable_types <- function(raw_df, config) {

  # Variables used in the analysis
  vars_to_keep <- c(
    vapply(config$predictors, `[[`, character(1), "name"),
    vapply(config$covariates, `[[`, character(1), "name"),
    config$outcome$name
  )
  
  # Subset data
  coerced_df <- raw_df[, vars_to_keep, drop = FALSE]
  
  # Map variable names to types
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
  
  # Coerce variables
  for (v in names(type_map)) {
    coerced_df[[v]] <- switch(
      type_map[[v]],
      continuous  = as.numeric(coerced_df[[v]]),
      categorical = as.factor(coerced_df[[v]]),
      stop("Unsupported type for ", v, ": ", type_map[[v]])
    )
  }
  
  # Convert binary categorical outcome to 0/1
  if (config$outcome$type == "categorical") {
    outcome_name <- config$outcome$name
    ref <- as.character(config$outcome$reference)
    f <- relevel(factor(as.character(coerced_df[[outcome_name]])), ref = ref)
    coerced_df[[outcome_name]] <- as.integer(f) - 1
  }
  
  # Collect the continuous variable names
  continuous_predictors <- vapply(
    config$predictors,
    function(x) x$name,
    character(1)
  )
  types <- vapply(
    config$predictors,
    function(x) x$type,
    character(1)
  )
  continuous_predictors <- continuous_predictors[types == "continuous"]
  
  # Compute mean and SD for continuous predictors
  # Labels for percentile columns
  lower_label <- "10th_percentile"
  upper_label <- "90th_percentile"
  predictor_stats_df <- if (length(continuous_predictors) > 0) {
    do.call(rbind, lapply(continuous_predictors, function(v) {
      data.frame(
        predictor = v,
        mean = mean(coerced_df[[v]], na.rm = TRUE),
        sd = sd(coerced_df[[v]], na.rm = TRUE),
        q10 = quantile(coerced_df[[v]], probs = c(.10), na.rm = TRUE),
        q90 = quantile(coerced_df[[v]], probs = c(.90), na.rm = TRUE)
      )
    }))
  } else {
    data.frame(predictor = character(), mean = numeric(), sd = numeric())
  }
  rownames(predictor_stats_df) <- NULL
  
  # Return processed data and predictor statistics
  list(
    coerced_df = coerced_df,
    predictor_stats_df = predictor_stats_df
  )
}


# -----------------------------------------------------------------------------
# harmonize_dhs_to_seed.R
#
# Convert the nine shared DHS source fields to the Seed variable names and
# categories described in config_seed_asd_harmonization.yaml.
# Example from the study_1 directory:
#   dhs_seed_df <- harmonize_dhs_to_seed(
#     dhs_df,
#     "config_seed_asd_harmonization.yaml"
#   )
# -----------------------------------------------------------------------------
normalize_harmonization_source <- function(x) {
  values <- trimws(as.character(x))
  missing_values <- is.na(x) |
    is.na(values) |
    tolower(values) %in% c("", "na", "n/a", "null", "missing")
  values[missing_values] <- NA_character_
  values
}

recode_harmonization_factor <- function(source, mapping, categories, variable) {
  source_values <- normalize_harmonization_source(source)
  map_keys <- names(mapping)
  if (is.null(map_keys) || length(map_keys) == 0L) {
    stop("No source_to_seed mapping configured for ", variable, ".", call. = FALSE)
  }

  unknown_values <- unique(
    source_values[!is.na(source_values) & !source_values %in% map_keys]
  )
  if (length(unknown_values) > 0L) {
    stop(
      "Unmapped source value(s) for ", variable, ": ",
      paste(unknown_values, collapse = ", "),
      call. = FALSE
    )
  }

  recoded <- rep(NA_integer_, length(source_values))
  factor_levels <- as.integer(names(categories))
  for (key in map_keys) {
    selected <- !is.na(source_values) & source_values == key
    target <- mapping[[key]]
    if (length(target) > 0L && !is.null(target)) {
      target_code <- as.integer(target)
      if (length(target_code) != 1L || is.na(target_code) ||
          !target_code %in% factor_levels) {
        stop(
          "Invalid Seed category configured for source code ", key,
          " in ", variable, ".",
          call. = FALSE
        )
      }
      recoded[selected] <- target_code
    }
  }

  factor(
    recoded,
    levels = factor_levels,
    ordered = FALSE
  )
}

harmonize_dhs_to_seed <- function(
    dhs_df,
    config_path = "config_seed_asd_harmonization.yaml") {
  required_variables <- c(
    "dr_agebir_fx",
    "dr_agebir_mx",
    "dr_gawks",
    "DR_PREGHTN",
    "gest_dm_bin",
    "dr_biredu_mx_5c",
    "dr_prevlb_br_3c",
    "dr_race_mx_5c",
    "dr_sex_cx"
  )
  config <- yaml::read_yaml(config_path)
  variable_specs <- config$variables[required_variables]
  missing_specs <- required_variables[vapply(variable_specs, is.null, logical(1))]
  if (length(missing_specs) > 0L) {
    stop(
      "Harmonization config is missing variable specification(s): ",
      paste(missing_specs, collapse = ", "),
      call. = FALSE
    )
  }

  required_sources <- unique(unlist(
    lapply(variable_specs, `[[`, "dhs_source_columns"),
    use.names = FALSE
  ))
  missing_sources <- setdiff(required_sources, names(dhs_df))
  if (length(missing_sources) > 0L) {
    stop(
      "DHS data is missing required source column(s): ",
      paste(missing_sources, collapse = ", "),
      call. = FALSE
    )
  }

  harmonized <- data.frame(row.names = seq_len(nrow(dhs_df)))
  diabetes_precedence_rows <- integer()

  for (variable in required_variables) {
    spec <- variable_specs[[variable]]
    recode <- spec$recode_to_seed

    if (identical(spec$type, "continuous")) {
      source_column <- spec$dhs_source_columns[[1]]
      source_values <- normalize_harmonization_source(dhs_df[[source_column]])
      numeric_values <- suppressWarnings(as.numeric(source_values))
      invalid_values <- unique(
        source_values[!is.na(source_values) & is.na(numeric_values)]
      )
      if (length(invalid_values) > 0L) {
        stop(
          "Non-numeric source value(s) in ", source_column, ": ",
          paste(invalid_values, collapse = ", "),
          call. = FALSE
        )
      }
      harmonized[[variable]] <- numeric_values
    } else if (identical(recode$strategy, "diabetes_indicators")) {
      preexisting_column <- recode$preexisting_source_column
      pregnancy_column <- recode$during_pregnancy_source_column
      preexisting <- normalize_harmonization_source(dhs_df[[preexisting_column]])
      during_pregnancy <- normalize_harmonization_source(dhs_df[[pregnancy_column]])

      invalid_preexisting <- unique(
        preexisting[!is.na(preexisting) & !preexisting %in% c("0", "1")]
      )
      invalid_pregnancy <- unique(
        during_pregnancy[!is.na(during_pregnancy) &
          !during_pregnancy %in% c("0", "1")]
      )
      if (length(invalid_preexisting) > 0L || length(invalid_pregnancy) > 0L) {
        stop(
          "Expected 0/1 values in ", preexisting_column, " and ",
          pregnancy_column, ".",
          call. = FALSE
        )
      }

      both_known <- !is.na(preexisting) & !is.na(during_pregnancy)
      both_zero <- both_known & preexisting == "0" & during_pregnancy == "0"
      preexisting_positive <- !is.na(preexisting) & preexisting == "1"
      pregnancy_only <- both_known &
        preexisting == "0" & during_pregnancy == "1"
      both_positive <- both_known &
        preexisting == "1" & during_pregnancy == "1"

      recoded <- rep(NA_integer_, nrow(dhs_df))
      recoded[both_zero] <- 0L
      recoded[preexisting_positive] <- 1L
      recoded[pregnancy_only] <- 2L
      diabetes_precedence_rows <- which(both_positive)

      categories <- recode$seed_categories
      harmonized[[variable]] <- factor(
        recoded,
        levels = as.integer(names(categories))
      )
    } else {
      source_column <- recode$source_column
      if (is.null(source_column) || !source_column %in% names(dhs_df)) {
        stop(
          "No valid source_column configured for ", variable, ".",
          call. = FALSE
        )
      }
      harmonized[[variable]] <- recode_harmonization_factor(
        dhs_df[[source_column]],
        recode$source_to_seed,
        recode$seed_categories,
        variable
      )
    }
  }

  attr(harmonized, "harmonization_diagnostics") <- list(
    preexisting_priority_gest_dm_bin_rows = diabetes_precedence_rows,
    preexisting_priority_gest_dm_bin_count = length(diabetes_precedence_rows)
  )

  harmonized
}


# -----------------------------------------------------------------------------
# add_missingness_indicators.R
#
# Description:
#   This function handles missingness for selected variables listed in
#   config$missingness_indicators.
#
#   For continuous variables, it:
#     1. Creates a binary missingness indicator named miss_<var>
#     2. Replaces missing values with 0
#
#   For categorical variables, it:
#     1. Adds "Missing" as an explicit factor level
#
#   The configuration object is updated only for continuous variables,
#   since their new missingness indicators must be included in downstream
#   modeling.
#
# Inputs:
#   - coerced_df: data frame after type coercion
#   - config: configuration list containing predictors, covariates, and
#             missingness_indicators
#
# Output:
#   A list containing:
#     * indicated_df: updated data frame
#     * config: updated configuration
#
# -----------------------------------------------------------------------------
add_missingness_indicators <- function(coerced_df, config) {

  # Copy input data
  indicated_df <- coerced_df
  
  # Nothing to do
  if (is.null(config$missingness_indicators) || length(config$missingness_indicators) == 0) {
    message("No missingness indicators specified in config.")
    return(list(
      indicated_df = indicated_df,
      config = config
    ))
  }
  
  # Process each selected variable
  for (var in config$missingness_indicators) {
    
    # Determine whether the variable is a predictor or covariate, and get its type
    var_type <- NULL
    var_group <- NULL
    
    for (group in c("predictors", "covariates")) {
      var_names <- vapply(config[[group]], `[[`, character(1), "name")
      match_idx <- which(var_names == var)
      
      if (length(match_idx) > 0) {
        var_type <- config[[group]][[match_idx[1]]]$type
        var_group <- group
        break
      }
    }
    
    # Continuous variable: add indicator and replace missing values with 0
    if (var_type == "continuous") {
      
      # Create name for new missingness indicator
      miss_name <- paste0("miss_", var)
      
      # Add in missingness indicator
      indicated_df[[miss_name]] <- factor(
        as.integer(is.na(indicated_df[[var]])),
        levels = c(0, 1)
      )
      indicated_df[[var]][is.na(indicated_df[[var]])] <- 0
      
      # Add indicator to the same section of the config
      entry <- list(name = miss_name, type = "categorical")
      config[[var_group]] <- append(config[[var_group]], list(entry))
      
      # Categorical variable: add explicit Missing level
    } else if (var_type == "categorical") {
      indicated_df[[var]] <- forcats::fct_na_value_to_level(
        indicated_df[[var]],
        level = "Missing"
      )
      
    } else {
      warning("Unsupported variable type for ", var, ": ", var_type)
    }
  }
  
  # ---- Return updated dataset and config ----
  list(
    indicated_df = indicated_df,
    config = config
  )
}

# -----------------------------------------------------------------------------
# impute_missing_data.R
#
# Description:
#   Performs single imputation using the `mice` package. The outcome is removed
#   before imputation and added back afterward, so only predictors and
#   covariates are imputed.
#
#   Designed to be used after adding missingness indicators.
#
# Inputs:
#   - indicated_df: data frame after missingness handling
#   - config: configuration list containing at least:
#       * outcome: list with element 'name'
#
# Output:
#   - imputed data frame with outcome reattached
#
# Notes:
#   - Uses single imputation (m = 1).
#   - Relies on `mice` default methods for each variable type.
#
# -----------------------------------------------------------------------------

impute_missing_data <- function(indicated_df, config) {

  # Copy data
  imputed_df <- indicated_df
  
  # Grab outcome name
  outcome_name <- config$outcome$name
  
  # Remove outcome before imputation
  outcome <- imputed_df[[outcome_name]]
  imputed_df[[outcome_name]] <- NULL
  
  # If nothing is missing, return early
  if (!anyNA(imputed_df)) {
    imputed_df[[outcome_name]] <- outcome
    return(imputed_df)
  }
  
  # Perform single imputation
  mice_result <- mice::mice(imputed_df, m = 1, printFlag = FALSE)
  imputed_df <- mice::complete(mice_result, 1)
  
  # Re-attach outcome
  imputed_df[[outcome_name]] <- outcome
  
  # Return imputed data
  imputed_df
}

# -----------------------------------------------------------------------------
# finalize_analytic_dataset.R
#
# Description:
#   Prepares the final analytic dataset by removing rows with missing outcome
#   values.
#
# Inputs:
#   - imputed_df: data frame output from impute_missing_data()
#   - config: configuration list containing at least:
#       * outcome: list with element 'name'
#
# Output:
#   - analytic data frame with non-missing outcomes
#
# Example:
#   analytic_df <- finalize_analytic_dataset(imputed_df, config)
#   summary(analytic_df)
#
# -----------------------------------------------------------------------------
finalize_analytic_dataset <- function(imputed_df, config) {

  # Copy data
  analytic_df <- imputed_df
  
  # Drop rows with missing outcomes
  outcome_name <- config$outcome$name
  n_missing <- sum(is.na(analytic_df[[outcome_name]]))
  
  # Print number of rows with missing outcome
  if (n_missing > 0) {
    analytic_df <- analytic_df[!is.na(analytic_df[[outcome_name]]), ]
    cat(sprintf(
      "Dropped %d row%s with missing outcome (%s).\n",
      n_missing,
      ifelse(n_missing == 1, "", "s"),
      outcome_name
    ))
  }
  
  # ---- Return final dataset and updated config ----
  analytic_df
}