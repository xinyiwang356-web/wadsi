# Synthetic BART demo

This is an end-to-end smoke test of the project's `dbarts` fit, export, and
prediction pipeline. It reads only
`study_1/synthetic_demo/data/synthetic_validation_data.csv`; it does not load
the SEED or other project datasets.

The input has no ASD outcome column. The script therefore generates a
reproducible `synthetic_asd_flag` from a simple, documented simulation using
mother age, sex, and `anom_downs`. The simulated probability is
`logit^-1(-3.6 + 0.45 * male + 0.2 * standardized_mother_age + 0.8 * anom_downs)`,
followed by a seeded Bernoulli draw. This simulated label is solely to exercise
supervised training and validation. It is not observed ASD status, a clinical
target, or a basis for evaluating real-world ASD prediction.

## Run

Install the R packages `readr` and `dbarts`, then from the repository root run:

```powershell
Rscript .\study_1\R\demo_train_validate_synthetic_asd_bart.R
```

For a shorter run, the example trains on a reproducible sample of up to 10,000
synthetic rows, reserves 20% for holdout evaluation, and scores all rows in the
synthetic input file. The reduced BART settings are intended for a quick
pipeline check, not model-quality comparisons.

## Outputs

By default, files are written to
`study_1/synthetic_demo/synthetic_asd_bart_demo`:

- `synthetic_asd_bart_deployment.rds`: exported model with no training rows
- `synthetic_holdout_metrics.csv`: synthetic holdout Brier score and log loss
- `synthetic_holdout_predictions.csv`: simulated holdout labels and probabilities
- `synthetic_dataset_predictions.csv`: one synthetic score per input row, with no
  record identifiers

Set `SYNTHETIC_ASD_OUTPUT_DIR` to override the output folder. This repository
includes the synthetic input and generated demo outputs so the complete example
is reproducible and inspectable; none of these files are real patient data.
