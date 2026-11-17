# Study 1 change log

This log records the synthetic-data and DHS imputation work, followed by the
DHS BART pipeline preparation. Synthetic records and outcomes described here
are for software development and pipeline testing; they are not observed
participant data or scientific estimates.

## 2026-11-09 — Full-size SEED-only imputation and DHS BART runner

### SEED codebook-based synthetic reference data

- Added [`generate_synthetic_seed_asd_data.py`](generate_synthetic_seed_asd_data.py)
  to create synthetic SEED-like records using aggregate distributions and
  missingness information in [`seed codebook.xlsx`](../data/seed%20codebook.xlsx).
- Saved the generated reference data as
  [`data/synthetic_seed_asd_data.csv`](../data/synthetic_seed_asd_data.csv).
- Categorical counts and group-specific missingness follow the codebook;
  continuous values are simulated from the reported group-specific means and
  standard deviations. The codebook does not provide joint distributions, so
  variables are generated independently within ASD/control groups. The
  generated records are not observed SEED participants.

### DHS harmonization and multiple imputation

- Added a DHS-to-SEED mapping and imputation specification in
  [`config_seed_asd_harmonization.yaml`](config_seed_asd_harmonization.yaml).
  It defines nine overlapping variables and fifteen SEED variables unavailable
  in DHS that are treated as imputation targets.
- Initially added [`mice_imputer.py`](mice_imputer.py) as the combined-source
  imputation workflow. Its legacy combined-data CLI and stacking helpers have
  since been retired in favor of the isolated fit/transform CLIs below.
- Fit five `IterativeImputer` models using SEED rows only (`max_iter=20`,
  posterior sampling, and deterministic per-imputation seeds based on 42).
  The DHS rows are transformed using those SEED-fitted models and do not
  contribute to model fitting. Categorical variables are encoded and decoded
  to valid configured categories; configured numerical bounds are applied.
- Generated
  [`outputs/imputed_dhs/imputed_dhs_m1.csv`](outputs/imputed_dhs/imputed_dhs_m1.csv)
  through `imputed_dhs_m5.csv`. Each contains 65,608 DHS rows; all fifteen
  target variables were checked for missing values, and category/range checks
  passed.
- The default output directory is `study_1/outputs/imputed_dhs/`. The full-size
  outputs were committed on the published `xw_synthetic_v1` branch.
- Added [`fit_imputer.py`](fit_imputer.py) and
  [`transform_imputer.py`](transform_imputer.py) for separate SEED-only model
  fitting and DHS-local transformation. The fit command exports fitted
  BayesianRidge/IterativeImputer state and a schema manifest without training
  rows, identifiers, or donor records. The manifest records column ordering,
  dtypes, category levels (including a reserved `Missing` level), bounds,
  missingness indicators, DHS source mappings, and artifact SHA-256 values.
- The transform command reads only the DHS input and the model artifacts,
  checks the manifest and input schema, applies DHS mappings locally, and
  writes five DHS datasets by default. Fully observed SEED variables are
  deterministically masked on a small training-only subset so their conditional
  models can impute variables that are entirely missing in DHS.
- Example commands:

  ```powershell
  python study_1/fit_imputer.py `
    --seed-csv data/synthetic_seed_asd_data.csv
  python study_1/transform_imputer.py `
    --dhs-csv study_1/synthetic_demo/data/synthetic_validation_data.csv
  ```

- **Artifact handling:** `joblib` bundles must be loaded only from trusted
  sources; SHA-256 checks detect mismatched or corrupted files but do not
  authenticate the publisher. Transfer the bundle and manifest together.
- Added focused isolation, schema, deterministic-transform, missingness, and
  artifact-integrity tests in [`test_imputer_isolation.py`](test_imputer_isolation.py).
- **Full-size isolated run completed 2026-11-11:** fit all five models on the
  5,210-row SEED reference dataset with `max_iter=20` and seeds 42–46, then
  transformed all 65,608 DHS rows. Outputs are in
  [`outputs/imputed_dhs_isolated/`](outputs/imputed_dhs_isolated/); each of the
  five files has zero missing values across all 15 target variables and passed
  configured category and bound checks. The matching bundle and manifest are
  in [`outputs/imputer_artifacts/`](outputs/imputer_artifacts/). The earlier
  `outputs/imputed_dhs/` files were left unchanged.

### DHS BART pipeline preparation

- Added [`run_dhs_bart_pipeline.R`](run_dhs_bart_pipeline.R) as a separate
  runner, leaving the existing NHANES pilot in `study_1.qmd` unchanged.
- Added DHS analysis settings to
  [`config_seed_asd_harmonization.yaml`](config_seed_asd_harmonization.yaml):
  the five existing predictors and the four overlap plus fifteen imputed
  target adjustment covariates.
- DHS data have no observed ASD outcome. For pipeline testing only, the runner
  creates one fixed-seed, independent synthetic Bernoulli outcome with a
  1.08% prevalence parameter and a 0.27% missingness mask, and reuses the same
  labels across the five imputations. This outcome is not inferred from the
  predictors and must not be used for scientific interpretation.
- Configured the requested eight chains, 30,000 burn-in iterations, 12,000
  post-burn iterations thinned by 12, yielding 1,000 retained draws per chain
  and 8,000 draws per fit. The runner is designed to save five model objects
  under `study_1/outputs/models/` and diagnostic plots, pooled summaries,
  individual/profile summaries, and metadata under
  `study_1/outputs/dhs_results/`.
- The runner pools posterior draws across imputations and exports CRR/PRR,
  CAF/PAF, and Shapley summaries. The Shapley helper in
  [`R/compute_effects.R`](R/compute_effects.R) was extended to accept selected
  row indices and return raw individual and population draws for pooling.
- **Execution pending:** the R runner has not been run in this environment
  because `Rscript` is unavailable. The five BART models, diagnostics, and
  results listed above have therefore not yet been generated or runtime
  verified. The runner's static structure and its YAML configuration were
  checked; the five imputed inputs were confirmed to contain complete target
  variables.

## 2026-11-10 — Step-by-step DHS M1 Quarto run

- Added [`study_1_dhs.qmd`](study_1_dhs.qmd), a DHS-specific companion to the
  NHANES pilot QMD. It reads the DHS harmonization/imputation configuration and
  M1 file, creates and reuses the configured synthetic-only outcome, checks
  model-variable completeness, and reports each data-preparation step.
- Rendered the document to
  [`outputs/dhs_m1_stepwise/study_1_dhs.html`](outputs/dhs_m1_stepwise/study_1_dhs.html).
  It records the 65,608 DHS input rows, 65,441 analytic rows after synthetic
  outcome missingness, and zero missing model values.
- A full-size M1 BART attempt failed with a native memory-allocation error
  (`std::bad_alloc`) on the available machine (about 3.3 GB free at the time).
  The QMD therefore defaults to an explicitly labeled smoke fit on a
  reproducible 5,000-row subset (the successful render used 3,000 rows). It
  saves an M1 smoke model, trace plot, convergence summary, and calibration
  outputs under `outputs/dhs_m1_stepwise/`. These are pipeline checks, not
  pooled results or scientific estimates.
- The full configured eight-chain fit remains available as `model_mode: full`
  but requires a machine with sufficient memory. The five-imputation pooled
  runner has not been run.

## Earlier synthetic BART smoke test

- Added a separate synthetic BART example using a synthetic validation file
  and simulated labels to exercise model training, export, and prediction.
  See [`SYNTHETIC_BART_DEMO.md`](SYNTHETIC_BART_DEMO.md).
- This smoke test is separate from the SEED-only MICE workflow and does not
  provide evidence about real-world ASD prediction.
