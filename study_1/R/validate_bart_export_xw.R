# validate_bart_export_xw.R

setwd("C:/Users/XXW306/Cochran_REPO/wadsi/study_1")

# Source helper functions and packages
source("R/imports.R")
source("R/deployment.R")

# --- Load the sanitized deployment export ---
deploy_path <- "outputs/models/reference_outcome_bart_deployment.rds"

if (!file.exists(deploy_path)) {
  stop("Saved deployment export not found: ", deploy_path)
}

deployment_model <- readRDS(deploy_path)

cat("Loaded sanitized export:", deploy_path, "\n")

# --- Example EHR-like validation data ---
# Replace this with your actual EHR dataset.
# It must have the same columns used in training, plus the outcome column.
config <- yaml::read_yaml("config_1.yaml")
data_path <- fs::path_abs(config$data_file)
ehr_data <- readr::read_csv(data_path, show_col_types = FALSE)

# Keep the outcome name and the predictor names used by the model
outcome_var <- config$outcome$name
model_input_names <- deployment_model$input_names

# Make sure the data contain the same columns expected by the exported model
missing_cols <- setdiff(model_input_names, names(ehr_data))
if (length(missing_cols) > 0) {
  stop("EHR-like data are missing required predictors: ",
       paste(missing_cols, collapse = ", "))
}

# If the EHR data contains the outcome, use it directly.
# If not, create a toy validation dataset for a smoke test.
if (!outcome_var %in% names(ehr_data)) {
  ehr_data[[outcome_var]] <- as.integer(stats::runif(nrow(ehr_data)) < 0.2)
}

# Reorder and coerce to the expected feature schema
ehr_data <- ehr_data[, c(model_input_names, outcome_var), drop = FALSE]

# Apply the same type coercion used in training
for (nm in model_input_names) {
  if (is.factor(ehr_data[[nm]])) {
    # Keep factor levels as-is if they already exist
    next
  }
  if (nm %in% c("smoker_current", "bp_treated")) {
    ehr_data[[nm]] <- as.factor(ehr_data[[nm]])
  } else {
    ehr_data[[nm]] <- suppressWarnings(as.numeric(ehr_data[[nm]]))
  }
}

# --- Validate the sanitized export ---
metrics <- validate_bart_export(
  model = deployment_model,
  ehr_data = ehr_data,
  outcome_var = outcome_var,
  output_path = "outputs/validation/ehr_validation_metrics.csv"
)

print(metrics)

# Optional: also print the predicted probabilities
predicted <- predict_bart_export(deployment_model, ehr_data[, model_input_names, drop = FALSE])
cat("\nFirst 10 predicted probabilities:\n")
print(head(predicted, 10))

# Optional: save a small validation summary
validation_summary <- data.frame(
  export_path = deploy_path,
  outcome_var = outcome_var,
  n_rows = nrow(ehr_data),
  brier_score = metrics$brier_score,
  negative_log_likelihood = metrics$negative_log_likelihood
)

readr::write_csv(validation_summary, "outputs/validation/ehr_validation_summary.csv")
cat("\nSaved validation summary to outputs/validation/ehr_validation_summary.csv\n")