# Synthetic-only BART workflow example.
# The simulated outcome is for testing the pipeline, not for ASD inference.

project_candidates <- c(".", "study_1", file.path("wadsi", "study_1"))
existing_project <- project_candidates[vapply(
  project_candidates,
  function(path) file.exists(file.path(path, "R", "deployment.R")),
  logical(1)
)]
if (length(existing_project) == 0L) {
  stop("Run from the repository root or study_1 directory.")
}
project_dir <- normalizePath(existing_project[[1]], winslash = "/", mustWork = TRUE)
setwd(project_dir)
demo_dir <- file.path(project_dir, "synthetic_demo")
input_path <- file.path(demo_dir, "data", "synthetic_validation_data.csv")
if (!file.exists(input_path)) {
  stop("Synthetic input file not found: ", input_path)
}
input_path <- normalizePath(input_path, winslash = "/", mustWork = TRUE)

if (!requireNamespace("readr", quietly = TRUE) ||
    !requireNamespace("dbarts", quietly = TRUE)) {
  stop("Install the readr and dbarts packages before running this example.")
}
source("R/deployment.R")

output_dir <- Sys.getenv(
  "SYNTHETIC_ASD_OUTPUT_DIR",
  unset = file.path(demo_dir, "synthetic_asd_bart_demo")
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

synthetic_data <- readr::read_csv(
  input_path,
  na = c("", "NA", "N/A", "NULL"),
  col_types = readr::cols(.default = readr::col_guess()),
  show_col_types = FALSE
)
names(synthetic_data) <- tolower(names(synthetic_data))
if (anyDuplicated(names(synthetic_data))) {
  stop("Input has duplicate column names after case normalization.")
}
if ("asd_flag" %in% names(synthetic_data)) {
  stop("This demo expects unlabeled synthetic data and creates its own simulated outcome.")
}

synthetic_signature <- c(
  "avg_o3_prenatal_1yr", "avg_pm25_prenatal_1yr",
  "adi_natrank", "adi_staternk", "birthyear"
)
missing_signature <- setdiff(synthetic_signature, names(synthetic_data))
if (length(missing_signature) > 0L) {
  stop(
    "Input does not match the expected synthetic dataset schema; missing: ",
    paste(missing_signature, collapse = ", ")
  )
}

candidate_features <- c(
  "mother_pre_preg_bmi", "mother_age", "mother_hispanic_ethnicity",
  "mother_bridged_race_eth_cd", "mother_edcode", "marital_status_nm",
  "father_age", "father_edcode", "father_hispanic_ethnicity",
  "father_bridged_race_eth_cd", "sex", "mother_wt_gain_loss",
  "mother_cig_use_flag", "mother_cig_prev_cig", "mother_cig_first_cig",
  "mother_cig_second_cig", "mother_cig_last_cig", "mother_cig_second_hand",
  "prenatal_yesno", "mth_prenat_care_beg", "prenat_tot_visits",
  "live_births_living", "live_births_dead", "live_birth_order",
  "inter_preg_interval", "mr_none", "mr_diab", "mr_diab_gest",
  "mr_hypert_chronic", "mr_hypert_preg", "mr_eclampsia",
  "mr_infertility_drugs", "mr_infertility_tech", "mr_prev_cesarean_yesno",
  "mr_prev_ces_numb", "mr_unknown", "inf_any", "inf_none", "inf_syphilis",
  "inf_chlamydia", "inf_hepatitis_b", "inf_hepatitis_c", "inf_unknown",
  "ob_none", "ob_tocolysis", "ob_unknown", "cld_none", "cld_premature_rom",
  "cld_precip_labor", "cld_prolong_labor", "cld_unknown", "char_none",
  "char_induction", "char_non_vertex", "char_antibiotic", "char_chorioamnio",
  "char_meconium", "char_fetal_intolerance", "char_epidural", "char_unknown",
  "fetal_presentation", "final_route_method_deliv", "prev_ces_plus_method_route",
  "mm_none", "mm_transfusion", "mm_ruptured_uterus", "mm_unknown",
  "birth_weight_grams", "apgar_5", "obst_est_gest", "oe_gestation_weeks",
  "plurality", "ac_none", "ac_nicu", "ac_antibiotic_sepsis", "ac_unknown",
  "anom_none", "anom_downs", "anom_dow_karyo_confirm", "anom_dow_karyo_pending",
  "anom_chrom", "anom_chr_karyo_confirm", "anom_chr_karyo_pending", "anom_unknown"
)
feature_names <- intersect(candidate_features, names(synthetic_data))
if (length(feature_names) == 0L) {
  stop("No configured birth-record predictors were found in the synthetic input.")
}

required_simulation_features <- c("mother_age", "sex", "anom_downs")
missing_simulation_features <- setdiff(required_simulation_features, feature_names)
if (length(missing_simulation_features) > 0L) {
  stop(
    "Cannot create the demo-only simulated outcome; missing predictors: ",
    paste(missing_simulation_features, collapse = ", ")
  )
}

as_numeric <- function(values) {
  suppressWarnings(as.numeric(as.character(values)))
}
age <- as_numeric(synthetic_data$mother_age)
age_center <- mean(age, na.rm = TRUE)
age_scale <- stats::sd(age, na.rm = TRUE)
if (!is.finite(age_center) || !is.finite(age_scale) || age_scale == 0) {
  stop("mother_age must have at least two distinct observed numeric values.")
}
age_z <- (age - age_center) / age_scale
age_z[!is.finite(age_z)] <- 0
male <- as.integer(toupper(trimws(as.character(synthetic_data$sex))) %in%
                     c("M", "MALE"))
downs <- as_numeric(synthetic_data$anom_downs)
downs[!is.finite(downs)] <- 0
synthetic_probability <- stats::plogis(-3.6 + 0.45 * male + 0.2 * age_z + 0.8 * downs)
set.seed(20261007L)
synthetic_data$synthetic_asd_flag <- stats::rbinom(
  nrow(synthetic_data),
  size = 1L,
  prob = synthetic_probability
)

categorical_features <- intersect(
  feature_names,
  c(
    "mother_hispanic_ethnicity", "mother_bridged_race_eth_cd", "mother_edcode",
    "marital_status_nm", "father_edcode", "father_hispanic_ethnicity",
    "father_bridged_race_eth_cd", "sex", "mother_cig_use_flag",
    "mother_cig_prev_cig", "mother_cig_first_cig", "mother_cig_second_cig",
    "mother_cig_last_cig", "mother_cig_second_hand", "prenatal_yesno",
    "mth_prenat_care_beg", "fetal_presentation", "final_route_method_deliv",
    "prev_ces_plus_method_route", "plurality"
  )
)
categorical_features <- union(
  categorical_features,
  feature_names[grepl("^(mr|inf|ob|cld|char|mm|ac|anom)_", feature_names)]
)
unclassified_text <- feature_names[vapply(
  synthetic_data[feature_names],
  is.character,
  logical(1)
)]
unclassified_text <- setdiff(unclassified_text, categorical_features)
if (length(unclassified_text) > 0L) {
  stop(
    "Classify these text predictors as categorical before fitting: ",
    paste(unclassified_text, collapse = ", ")
  )
}

set.seed(20261008L)
demo_rows <- if (nrow(synthetic_data) > 10000L) {
  sample.int(nrow(synthetic_data), 10000L)
} else {
  seq_len(nrow(synthetic_data))
}
demo_data <- synthetic_data[demo_rows, , drop = FALSE]
outcome_name <- "synthetic_asd_flag"
class_rows <- split(seq_len(nrow(demo_data)), demo_data[[outcome_name]])
if (length(class_rows) != 2L || any(lengths(class_rows) < 2L)) {
  stop("The simulated outcome needs at least two examples per class.")
}
holdout_rows <- unlist(lapply(class_rows, function(rows) {
  sample(rows, size = max(1L, floor(length(rows) * 0.2)))
}), use.names = FALSE)
is_holdout <- seq_len(nrow(demo_data)) %in% holdout_rows

train_raw <- demo_data[!is_holdout, feature_names, drop = FALSE]
test_raw <- demo_data[is_holdout, feature_names, drop = FALSE]
train_y <- demo_data[[outcome_name]][!is_holdout]
test_y <- demo_data[[outcome_name]][is_holdout]
numeric_medians <- list()
factor_levels <- list()
prepare_features <- function(data, fit = FALSE) {
  prepared <- data[, feature_names, drop = FALSE]
  for (name in feature_names) {
    values <- prepared[[name]]
    if (name %in% categorical_features) {
      values <- as.character(values)
      values[is.na(values) | !nzchar(values)] <- "__MISSING__"
      if (fit) {
        observed_levels <- sort(setdiff(unique(values), c("__MISSING__", "__OTHER__")))
        levels_for_feature <- c(observed_levels, "__MISSING__", "__OTHER__")
        factor_levels[[name]] <<- levels_for_feature
      } else {
        levels_for_feature <- factor_levels[[name]]
        values[!values %in% levels_for_feature] <- "__OTHER__"
      }
      prepared[[name]] <- factor(values, levels = levels_for_feature)
    } else {
      numeric_values <- as_numeric(values)
      if (fit) {
        observed <- numeric_values[is.finite(numeric_values)]
        if (length(observed) == 0L) {
          stop("Numeric predictor has no observed training values: ", name)
        }
        numeric_medians[[name]] <<- stats::median(observed)
      }
      numeric_values[!is.finite(numeric_values)] <- numeric_medians[[name]]
      prepared[[name]] <- numeric_values
    }
  }
  prepared
}

x_train <- prepare_features(train_raw, fit = TRUE)
x_test <- prepare_features(test_raw)
x_all <- prepare_features(synthetic_data[feature_names], fit = FALSE)

cat("Synthetic source rows:", nrow(synthetic_data), "\n")
cat("Rows used for demo training/holdout:", nrow(demo_data), "\n")
cat("Training rows:", nrow(x_train), " Holdout rows:", nrow(x_test), "\n")
cat("Simulated positive prevalence:", mean(demo_data[[outcome_name]]), "\n")
cat("Predictors:", length(feature_names), "\n")

bart_fit <- dbarts::bart(
  x.train = x_train,
  y.train = train_y,
  keeptrees = TRUE,
  verbose = FALSE,
  nchain = 2L,
  nthread = 2L,
  nskip = 100L,
  ndpost = 100L,
  keepevery = 1L,
  seed = 20261009L
)
bart_fit$fit$storeState()

synthetic_config <- list(outcome = list(name = outcome_name))
deployment_model <- export_bart_model(bart_fit, x_train, synthetic_config)
deployment_model$numeric_medians <- numeric_medians
deployment_model$categorical_features <- categorical_features
deployment_model$raw_feature_names <- feature_names
holdout_predictions <- predict_bart_export(deployment_model, x_test)
all_predictions <- predict_bart_export(deployment_model, x_all)
metrics <- score_binary_predictions(test_y, holdout_predictions)
metrics$n_holdout <- length(test_y)
metrics$synthetic_asd_prevalence <- mean(test_y)
metrics$outcome <- outcome_name
metrics$interpretation <- "Synthetic demonstration only; not clinical validation."

saveRDS(deployment_model, file.path(output_dir, "synthetic_asd_bart_deployment.rds"))
readr::write_csv(metrics, file.path(output_dir, "synthetic_holdout_metrics.csv"))
readr::write_csv(
  data.frame(
    observed_synthetic_asd = test_y,
    predicted_synthetic_asd_probability = holdout_predictions
  ),
  file.path(output_dir, "synthetic_holdout_predictions.csv")
)
readr::write_csv(
  data.frame(predicted_synthetic_asd_probability = all_predictions),
  file.path(output_dir, "synthetic_dataset_predictions.csv")
)

print(metrics)
cat("Synthetic-only demo outputs saved under:", output_dir, "\n")
