# Export and score a BART model without serializing the training data.

export_bart_model <- function(bart_fit, x_train, config) {
  if (!inherits(bart_fit, "bart")) {
    stop("bart_fit must be a dbarts::bart fit.")
  }
  if (is.null(bart_fit$fit)) {
    stop("The BART fit must be created with keeptrees = TRUE.")
  }

  tree_data <- dbarts::extract(bart_fit, type = "trees")
  tree_data$n <- NULL
  model_matrix <- dbarts::makeModelMatrixFromDataFrame(x_train)

  list(
    format_version = 1L,
    contains_training_data = FALSE,
    package = "dbarts",
    link = "probit",
    binary_offset = bart_fit$binaryOffset,
    input_names = names(x_train),
    factor_levels = lapply(x_train, function(x) if (is.factor(x)) levels(x) else NULL),
    model_matrix_drop = attr(model_matrix, "drop"),
    predictor_names = colnames(model_matrix),
    predictor_classes = vapply(x_train, function(x) class(x)[1], character(1)),
    outcome = config$outcome$name,
    trees = tree_data
  )
}

predict_bart_export <- function(model, newdata) {
  if (!is.data.frame(newdata)) {
    stop("newdata must be a data frame with the original predictor columns.")
  }
  if (!identical(names(newdata), model$input_names)) {
    stop(
      "newdata must have these predictor columns in this order: ",
      paste(model$input_names, collapse = ", ")
    )
  }

  for (name in model$input_names) {
    levels <- model$factor_levels[[name]]
    if (!is.null(levels)) {
      newdata[[name]] <- factor(newdata[[name]], levels = levels)
    } else {
      newdata[[name]] <- as.numeric(newdata[[name]])
    }
  }

  newdata <- dbarts::makeModelMatrixFromDataFrame(
    newdata,
    drop = model$model_matrix_drop
  )
  if (!identical(colnames(newdata), model$predictor_names)) {
    stop("The EHR data do not produce the model's expected predictor columns.")
  }

  trees <- model$trees
  draw_groups <- split(
    seq_len(nrow(trees)),
    interaction(trees$chain, trees$sample, drop = TRUE)
  )

  predict_tree <- function(tree, indices) {
    predictions <- numeric(nrow(newdata))

    visit <- function(rows, active) {
      if (tree$var[rows[1]] == -1L) {
        predictions[active] <<- tree$value[rows[1]]
        return(1L)
      }

      goes_left <- newdata[active, tree$var[rows[1]]] <= tree$value[rows[1]]
      left_nodes <- visit(rows[-1], active[goes_left])
      right_start <- 2L + left_nodes
      right_nodes <- visit(rows[right_start:length(rows)], active[!goes_left])
      1L + left_nodes + right_nodes
    }

    visit(seq_len(nrow(tree)), indices)
    predictions
  }

  latent_draws <- vapply(draw_groups, function(group) {
    draw_trees <- trees[group, , drop = FALSE]
    tree_groups <- split(seq_len(nrow(draw_trees)), draw_trees$tree)
    rowSums(vapply(tree_groups, function(tree_rows) {
      predict_tree(draw_trees[tree_rows, , drop = FALSE], seq_len(nrow(newdata)))
    }, numeric(nrow(newdata)))) + model$binary_offset
  }, numeric(nrow(newdata)))

  probability_draws <- pnorm(t(latent_draws))
  colMeans(probability_draws)
}

score_binary_predictions <- function(observed, predicted, output_path = NULL) {
  probability <- pmin(pmax(as.numeric(predicted), 1e-12), 1 - 1e-12)
  observed <- as.numeric(observed)
  metrics <- data.frame(
    brier_score = mean((probability - observed)^2),
    negative_log_likelihood = -mean(
      observed * log(probability) + (1 - observed) * log(1 - probability)
    )
  )

  if (!is.null(output_path)) {
    dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(metrics, output_path)
  }

  metrics
}

validate_bart_export <- function(model, ehr_data, outcome_var, output_path = NULL) {
  ehr_outcome <- ehr_data[[outcome_var]]
  ehr_predictors <- ehr_data[, setdiff(names(ehr_data), outcome_var), drop = FALSE]
  predicted <- predict_bart_export(model, ehr_predictors)
  score_binary_predictions(ehr_outcome, predicted, output_path)
}