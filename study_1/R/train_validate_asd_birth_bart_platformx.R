# Train and validate an ASD BART model from linked birth-record data.
# Run from study_1, or set ASD_BIRTH_DATA to another CSV or TSV path.

local_data_paths <- c(
  file.path("data", "DHS_vs_ASD_LABEL_merged_deid.csv"),
  file.path("..", "data", "DHS_vs_ASD_LABEL_merged_deid.csv"),
  file.path("wadsi", "data", "DHS_vs_ASD_LABEL_merged_deid.csv")
)
existing_local_path <- local_data_paths[file.exists(local_data_paths)]
default_input <- if (length(existing_local_path) > 0L) {
  existing_local_path[[1]]
} else {
  local_data_paths[[1]]
}

# The workspace-local de-identified extract is the default input.
input_path <- Sys.getenv(
  "ASD_BIRTH_DATA",
  unset = default_input
)
if (nzchar(input_path) && dir.exists(input_path)) {
  input_candidates <- list.files(
    input_path,
    pattern = "\\.(csv|tsv|txt)$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  if (length(input_candidates) != 1L) {
    stop("ASD_BIRTH_DATA directory must contain exactly one CSV/TSV/TXT file; set it to the intended file otherwise.")
  }
  input_path <- input_candidates[[1]]
}
if (!nzchar(input_path) || !file.exists(input_path) || dir.exists(input_path)) {
  stop("Set ASD_BIRTH_DATA to the merged ASD/birth-record CSV or TSV file.")
}
input_path <- normalizePath(input_path, winslash = "/", mustWork = TRUE)

project_candidates <- c(
  Sys.getenv("WADSI_STUDY_DIR", unset = ""),
  ".", "study_1", file.path("wadsi", "study_1")
)
project_candidates <- project_candidates[nzchar(project_candidates)]
existing_project <- project_candidates[vapply(
  project_candidates,
  function(path) file.exists(file.path(path, "R", "deployment.R")),
  logical(1)
)]
if (length(existing_project) == 0L) {
  stop("Run from study_1 or set WADSI_STUDY_DIR to the study_1 directory.")
}
project_dir <- existing_project[[1]]
setwd(project_dir)
source("R/deployment.R")

if (!requireNamespace("readr", quietly = TRUE) ||
    !requireNamespace("dbarts", quietly = TRUE)) {
  stop("Install the readr and dbarts packages before running this script.")
}

output_dir <- Sys.getenv("ASD_OUTPUT_DIR", unset = "outputs/asd_birth_model")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

file_extension <- tolower(tools::file_ext(input_path))
reader <- switch(
  file_extension,
  csv = readr::read_csv,
  tsv = readr::read_tsv,
  txt = readr::read_tsv,
  stop("Input must be a .csv, .tsv, or tab-delimited .txt file.")
)
birth_data <- reader(
  input_path,
  na = c("", "NA", "N/A", "NULL"),
  col_types = readr::cols(.default = readr::col_guess()),
  show_col_types = FALSE
)

outcome_name <- "asd_flag"
missing_columns <- setdiff(outcome_name, names(birth_data))
if (length(missing_columns) > 0) {
  stop("Required columns are missing: ", paste(missing_columns, collapse = ", "))
}

outcome_values <- as.character(birth_data[[outcome_name]])
outcome_values <- trimws(outcome_values)
unlabeled <- is.na(outcome_values) | !nzchar(outcome_values)
invalid_labels <- unique(outcome_values[!unlabeled & !outcome_values %in% c("0", "1")])
if (length(invalid_labels) > 0L) {
  stop("asd_flag must be coded only as 0 or 1; unexpected values: ",
       paste(invalid_labels, collapse = ", "))
}
if (all(unlabeled)) {
  stop("No rows have an observed asd_flag label.")
}
excluded_unlabeled <- sum(unlabeled)
birth_data <- birth_data[!unlabeled, , drop = FALSE]
outcome_values <- outcome_values[!unlabeled]
birth_data[[outcome_name]] <- as.integer(outcome_values)

candidate_features <- c(
  "mother_pre_preg_bmi", "mother_age", "mother_Hispanic_ethnicity",
  "mother_bridged_race_eth_cd", "mother_edcode", "marital_status_nm",
  "father_age", "father_edcode", "father_Hispanic_ethnicity",
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
feature_names <- intersect(candidate_features, names(birth_data))
if (length(feature_names) == 0) {
  stop("None of the configured birth-record predictors are present in the input file.")
}

categorical_features <- intersect(
  feature_names,
  c(
    "mother_Hispanic_ethnicity", "mother_bridged_race_eth_cd", "mother_edcode",
    "marital_status_nm", "father_edcode", "father_Hispanic_ethnicity",
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
  birth_data[feature_names],
  is.character,
  logical(1)
)]
unclassified_text <- setdiff(unclassified_text, categorical_features)
if (length(unclassified_text) > 0L) {
  stop("Classify these text predictors as categorical before fitting: ",
       paste(unclassified_text, collapse = ", "))
}

set.seed(20261002L)
class_rows <- split(seq_len(nrow(birth_data)), birth_data[[outcome_name]])
if (any(lengths(class_rows) < 2L)) {
  stop("At least two labeled people per ASD class are needed for a holdout split.")
}
holdout_rows <- unlist(lapply(class_rows, function(rows) {
  sample(rows, size = max(1L, floor(length(rows) * 0.2)))
}), use.names = FALSE)
is_holdout <- seq_len(nrow(birth_data)) %in% holdout_rows
train_raw <- birth_data[!is_holdout, feature_names, drop = FALSE]
test_raw <- birth_data[is_holdout, feature_names, drop = FALSE]
train_y <- birth_data[[outcome_name]][!is_holdout]
test_y <- birth_data[[outcome_name]][is_holdout]

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
      numeric_values <- suppressWarnings(as.numeric(values))
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
x_test <- prepare_features(test_raw, fit = FALSE)

nchain <- as.integer(Sys.getenv("ASD_BART_NCHAIN", unset = "4"))
nskip <- as.integer(Sys.getenv("ASD_BART_BURNIN", unset = "10000"))
retained_draws <- as.integer(Sys.getenv("ASD_BART_DRAWS", unset = "1000"))
thin <- as.integer(Sys.getenv("ASD_BART_THIN", unset = "10"))
if (anyNA(c(nchain, nskip, retained_draws, thin)) ||
    any(c(nchain, retained_draws, thin) < 1L) || nskip < 0L) {
  stop("BART settings must be valid positive integers (burn-in may be zero).")
}

cat("Training rows:", nrow(x_train), " Holdout rows:", nrow(x_test), "\n")
cat("Training ASD prevalence:", mean(train_y), " Holdout ASD prevalence:", mean(test_y), "\n")
cat("Eligible predictors:", length(feature_names), "\n")

bart_fit <- dbarts::bart(
  x.train = x_train,
  y.train = train_y,
  keeptrees = TRUE,
  verbose = FALSE,
  nchain = nchain,
  nthread = nchain,
  nskip = nskip,
  ndpost = retained_draws * thin,
  keepevery = thin,
  seed = 20261002L
)
bart_fit$fit$storeState()

asd_config <- list(outcome = list(name = outcome_name))
deployment_model <- export_bart_model(bart_fit, x_train, asd_config)
deployment_model$numeric_medians <- numeric_medians
deployment_model$categorical_features <- categorical_features
deployment_model$raw_feature_names <- feature_names
predicted <- predict_bart_export(deployment_model, x_test)
metrics <- score_binary_predictions(test_y, predicted)
metrics$n_holdout <- length(test_y)
metrics$asd_prevalence <- mean(test_y)
metrics$outcome <- outcome_name
metrics$n_unlabeled_excluded <- excluded_unlabeled

saveRDS(deployment_model, file.path(output_dir, "asd_birth_bart_deployment.rds"))
readr::write_csv(metrics, file.path(output_dir, "asd_holdout_metrics.csv"))
holdout_predictions <- data.frame(
  observed_asd = test_y,
  predicted_asd_probability = predicted
)
readr::write_csv(
  holdout_predictions,
  file.path(output_dir, "asd_holdout_predictions.csv")
)

print(metrics)
cat("Sanitized ASD model saved under:", output_dir, "\n")
cat("Holdout predictions saved under:", output_dir, "\n")