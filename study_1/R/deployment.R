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
  binary_offset <- as.numeric(bart_fit$binaryOffset)
  if (length(binary_offset) == 0L || any(!is.finite(binary_offset))) {
    stop("The BART fit has an invalid binary offset.")
  }
  if (length(binary_offset) > 1L && any(binary_offset != binary_offset[1])) {
    stop("A row-specific binary offset cannot be exported for new-data scoring.")
  }

  list(
    format_version = 1L,
    contains_training_data = FALSE,
    package = "dbarts",
    link = "probit",
    binary_offset = binary_offset[1],
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
        if (any(active < 1L | active > length(predictions))) {
          stop("The BART tree walk produced an out-of-range row index.")
        }
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
    tree_predictions <- vapply(tree_groups, function(tree_rows) {
      predict_tree(draw_trees[tree_rows, , drop = FALSE], seq_len(nrow(newdata)))
    }, numeric(nrow(newdata)))
    rowSums(tree_predictions) + model$binary_offset
  }, numeric(nrow(newdata)))

  colMeans(stats::pnorm(t(latent_draws)))
}

score_binary_predictions <- function(observed, predicted) {
  probability <- pmin(pmax(as.numeric(predicted), 1e-12), 1 - 1e-12)
  observed <- as.numeric(observed)
  data.frame(
    brier_score = mean((probability - observed)^2),
    negative_log_likelihood = -mean(
      observed * log(probability) + (1 - observed) * log(1 - probability)
    )
  )
}
