# run_to_bart.R
# Run this from the repo root or set the working directory below.

# Set working directory to the study folder so the project-local source paths work.
setwd("C:/Users/XXW306/Cochran_REPO/wadsi/study_1")

# Source project packages and helper functions
source("R/imports.R")

# Load YAML config
config <- yaml::read_yaml("config_1.yaml")

# Read the configured dataset
data_path <- fs::path_abs(config$data_file)
if (!fs::file_exists(data_path)) {
  stop("Configured data file does not exist: ", data_path)
}

df <- readr::read_csv(data_path, show_col_types = FALSE)
raw_df <- df

# --- Data prep ---
result <- coerce_variable_types(raw_df, config)
coerced_df <- result$coerced_df
predictor_stats_df <- result$predictor_stats_df

result <- add_missingness_indicators(coerced_df, config)
indicated_df <- result$indicated_df
final_config <- result$config

imputed_df <- impute_missing_data(indicated_df, final_config)
analytic_df <- finalize_analytic_dataset(imputed_df, final_config)

# Quick check
print(summary(analytic_df))

# --- Build BART model ---
outcome_var <- final_config$outcome$name
predictor_vars <- vapply(final_config$predictors, `[[`, character(1), "name")
adjustment_vars <- setdiff(names(analytic_df), c(outcome_var, predictor_vars))

x_train <- analytic_df[, c(adjustment_vars, predictor_vars), drop = FALSE]
y_train <- analytic_df[[outcome_var]]

nchain <- 8L
nskip <- 30000L
ndpost <- 1000L

dir.create("outputs/models", recursive = TRUE, showWarnings = FALSE)

model_path <- "outputs/models/reference_outcome_bart.rds"

# Fit only if not already present
if (file.exists(model_path)) {
  bart_fit <- readRDS(model_path)
} else {
  bart_fit <- dbarts::bart(
    x.train   = x_train,
    y.train   = y_train,
    keeptrees = TRUE,
    verbose   = FALSE,
    nchain    = nchain,
    nthread   = 8L,
    nskip     = nskip,
    ndpost    = ndpost * 12L,
    keepevery = 12L,
    seed      = 20260322L
  )

  bart_fit$fit$storeState()
  saveRDS(bart_fit, model_path)
}

# --- Sanitize export for deployment ---
deployment_model <- export_bart_model(bart_fit, x_train, final_config)
saveRDS(deployment_model, "outputs/models/reference_outcome_bart_deployment.rds")

# Print a short summary for verification
cat("\nBART fit complete.\n")
cat("Train rows:", nrow(analytic_df), "\n")
cat("Predictors:", paste(predictor_vars, collapse = ", "), "\n")
cat("Model saved to:", model_path, "\n")
cat("Deployment model saved to: outputs/models/reference_outcome_bart_deployment.rds\n")