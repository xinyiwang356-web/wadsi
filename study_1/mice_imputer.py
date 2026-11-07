"""SEED-trained chained-equation imputation for harmonized SEED/DHS data.

The input must be the combined DataFrame produced by ``load_and_harmonize``:
one row per participant, an ``is_dhs`` column (0 for SEED, 1 for DHS), and
the configured overlap and target variables. Model fitting uses SEED rows
only; DHS rows are used only for conditional prediction.

Install dependencies with ``python -m pip install pandas numpy scikit-learn
PyYAML``.

Example:
    python study_1/mice_imputer.py \
        --input-csv harmonized_seed_dhs.csv \
        --config study_1/config_seed_asd_harmonization.yaml \
        --target-columns target_a target_b \
        --output-dir study_1/outputs/imputed

Variable types are read from ``variables.<name>.type`` in the YAML. Optional
``imputation`` settings can declare ``target_variables``,
``overlap_variables``, ``variable_types``, ``categories``, and ``bounds``.
Categorical values default to the categories observed in SEED; continuous
bounds default to the SEED minimum and maximum.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import BayesianRidge, LogisticRegression

LOGGER = logging.getLogger("mice_imputer")
IS_DHS_COLUMN = "is_dhs"
MAX_ITER = 20
N_IMPUTATIONS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: str
    categories: tuple[Any, ...] | None = None
    bounds: tuple[float, float] | None = None


class _ConstantModel:
    def __init__(self, value: Any) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value)


def _category_key(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return f"number:{float(value):.15g}"
    if isinstance(value, str):
        try:
            numeric_value = float(value)
        except ValueError:
            return f"text:{value}"
        if math.isfinite(numeric_value):
            return f"number:{numeric_value:.15g}"
    return f"text:{value}"


def _variable_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = config.get("variables", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("The YAML 'variables' entry must be a mapping.")
    return metadata


def _specs_for_columns(
    columns: Sequence[str],
    seed_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, VariableSpec]:
    metadata = _variable_metadata(config)
    imputation = config.get("imputation", {})
    imputation = imputation if isinstance(imputation, Mapping) else {}
    configured_types = imputation.get("variable_types", {})
    configured_types = configured_types if isinstance(configured_types, Mapping) else {}
    configured_categories = imputation.get("categories", {})
    configured_categories = configured_categories if isinstance(configured_categories, Mapping) else {}
    configured_bounds = imputation.get("bounds", {})
    configured_bounds = configured_bounds if isinstance(configured_bounds, Mapping) else {}

    specs: dict[str, VariableSpec] = {}
    for column in columns:
        variable_config = metadata.get(column, {})
        variable_config = variable_config if isinstance(variable_config, Mapping) else {}
        type_config = configured_types.get(column)
        if isinstance(type_config, Mapping):
            kind = type_config.get("type")
        else:
            kind = type_config
        kind = kind or variable_config.get("type")
        if kind not in {"continuous", "categorical", "ordinal", "binary"}:
            raise ValueError(
                f"No supported type is configured for {column!r}; use "
                "'continuous' or 'categorical' in the YAML."
            )
        kind = "categorical" if kind in {"categorical", "ordinal", "binary"} else kind

        observed_seed = seed_frame[column].dropna()
        if observed_seed.empty:
            raise ValueError(f"SEED has no observed values for {column!r}.")

        category_config = configured_categories.get(column)
        if category_config is None:
            category_config = variable_config.get("categories")
        if category_config is None:
            category_config = variable_config.get("seed_categories")
        if category_config is None:
            recode_config = variable_config.get("recode_to_seed", {})
            if isinstance(recode_config, Mapping):
                category_config = recode_config.get("seed_categories")
        if isinstance(category_config, Mapping):
            category_values = tuple(category_config.keys())
        elif isinstance(category_config, Sequence) and not isinstance(category_config, str):
            category_values = tuple(category_config)
        else:
            category_values = tuple(pd.unique(observed_seed)) if kind == "categorical" else None
        if kind == "categorical":
            assert category_values is not None
            configured_keys = [_category_key(value) for value in category_values]
            actual_seed_values = list(pd.unique(observed_seed))
            actual_seed_keys = {_category_key(value) for value in actual_seed_values}
            if not actual_seed_keys.issubset(set(configured_keys)):
                raise ValueError(
                    f"Observed SEED values for {column!r} include categories not "
                    "listed in its YAML specification."
                )
            # Use observed SEED representations so completed values retain the
            # same types as the supplied harmonized data.
            representatives = {_category_key(value): value for value in category_values}
            representatives.update(
                {_category_key(value): value for value in actual_seed_values}
            )
            ordered_keys = list(dict.fromkeys(configured_keys))
            category_values = tuple(representatives[key] for key in ordered_keys)

        bounds_config = configured_bounds.get(column)
        if bounds_config is None:
            bounds_config = variable_config.get("bounds")
        if isinstance(bounds_config, Mapping):
            bounds_config = (bounds_config.get("min"), bounds_config.get("max"))
        if bounds_config is None and (
            "min_value" in variable_config or "max_value" in variable_config
        ):
            bounds_config = (
                variable_config.get("min_value"),
                variable_config.get("max_value"),
            )
        if kind == "continuous":
            values = pd.to_numeric(observed_seed, errors="raise")
            if bounds_config is None:
                bounds = (float(values.min()), float(values.max()))
            else:
                if len(bounds_config) != 2 or bounds_config[0] is None or bounds_config[1] is None:
                    raise ValueError(
                        f"Bounds for {column!r} must contain both a minimum and maximum."
                    )
                bounds = (float(bounds_config[0]), float(bounds_config[1]))
            if not all(math.isfinite(bound) for bound in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid continuous bounds for {column!r}: {bounds}.")
        else:
            bounds = None

        specs[column] = VariableSpec(
            name=column,
            kind=kind,
            categories=category_values if kind == "categorical" else None,
            bounds=bounds,
        )
    return specs


def _category_maps(specs: Mapping[str, VariableSpec]) -> dict[str, dict[str, Any]]:
    return {
        name: {_category_key(value): value for value in spec.categories or ()}
        for name, spec in specs.items()
        if spec.kind == "categorical"
    }


def _encode_predictors(
    frame: pd.DataFrame,
    predictor_names: Sequence[str],
    specs: Mapping[str, VariableSpec],
    category_maps: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    for name in predictor_names:
        spec = specs[name]
        if spec.kind == "continuous":
            encoded.append(pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float))
        else:
            key_to_index = {
                key: index for index, key in enumerate(category_maps[name])
            }
            values = frame[name].map(_category_key)
            unknown = sorted(set(values) - key_to_index.keys())
            if unknown:
                raise ValueError(
                    f"Unexpected categories for {name!r}: {unknown[:5]}."
                )
            one_hot = np.zeros((len(frame), len(key_to_index)), dtype=float)
            indices = values.map(key_to_index).to_numpy(dtype=int)
            one_hot[np.arange(len(frame)), indices] = 1.0
            encoded.append(one_hot)
    if not encoded:
        raise ValueError("At least one predictor column is required.")
    return np.column_stack(encoded)


def _sample_initial_values(
    frame: pd.DataFrame,
    donor_frame: pd.DataFrame,
    original_missing: Mapping[str, np.ndarray],
    specs: Mapping[str, VariableSpec],
    rng: np.random.Generator,
) -> None:
    for name, spec in specs.items():
        missing = original_missing[name]
        count = int(missing.sum())
        if count == 0:
            continue
        observed = donor_frame[name].dropna().to_numpy()
        if spec.kind == "continuous":
            frame.loc[missing, name] = rng.choice(observed, size=count, replace=True)
        else:
            frame.loc[missing, name] = rng.choice(observed, size=count, replace=True)


def _fit_conditional_model(
    target: str,
    train_frame: pd.DataFrame,
    train_mask: np.ndarray,
    predictor_names: Sequence[str],
    specs: Mapping[str, VariableSpec],
    category_maps: Mapping[str, Mapping[str, Any]],
    random_state: int,
) -> tuple[Any, np.ndarray | None]:
    predictors = _encode_predictors(
        train_frame.loc[train_mask], predictor_names, specs, category_maps
    )
    target_spec = specs[target]
    target_values = train_frame.loc[train_mask, target].to_numpy()
    if target_spec.kind == "continuous":
        numeric_target = pd.to_numeric(
            pd.Series(target_values), errors="raise"
        ).to_numpy(dtype=float)
        if np.unique(numeric_target).size == 1:
            return _ConstantModel(float(numeric_target[0])), None
        model = BayesianRidge()
        model.fit(predictors, numeric_target)
        return model, None

    target_keys = np.array([_category_key(value) for value in target_values])
    classes = np.unique(target_keys)
    if classes.size == 1:
        return _ConstantModel(classes[0]), None
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(predictors, target_keys)
    return model, model.classes_


def _draw_predictions(
    model: Any,
    classes: np.ndarray | None,
    target: str,
    frame: pd.DataFrame,
    row_mask: np.ndarray,
    predictor_names: Sequence[str],
    specs: Mapping[str, VariableSpec],
    category_maps: Mapping[str, Mapping[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    predictor_values = _encode_predictors(
        frame.loc[row_mask], predictor_names, specs, category_maps
    )
    target_spec = specs[target]
    if target_spec.kind == "continuous":
        if isinstance(model, _ConstantModel):
            predictions = model.predict(predictor_values).astype(float)
        else:
            mean, standard_deviation = model.predict(
                predictor_values, return_std=True
            )
            predictions = rng.normal(mean, standard_deviation)
        assert target_spec.bounds is not None
        return np.clip(predictions, target_spec.bounds[0], target_spec.bounds[1])

    if isinstance(model, _ConstantModel):
        predicted_keys = model.predict(predictor_values)
    else:
        probabilities = model.predict_proba(predictor_values)
        model_classes = np.asarray(classes)
        predicted_keys = np.array(
            [
                rng.choice(model_classes, p=row_probabilities / row_probabilities.sum())
                for row_probabilities in probabilities
            ]
        )
    category_map = category_maps[target]
    return np.array([category_map[key] for key in predicted_keys], dtype=object)


def _run_one_imputation(
    seed_frame: pd.DataFrame,
    dhs_frame: pd.DataFrame,
    seed_missing: Mapping[str, np.ndarray],
    dhs_missing: Mapping[str, np.ndarray],
    specs: Mapping[str, VariableSpec],
    random_state: int,
    imputation_number: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    seed_state = seed_frame.copy()
    dhs_state = dhs_frame.copy()
    _sample_initial_values(seed_state, seed_state, seed_missing, specs, rng)
    _sample_initial_values(dhs_state, seed_state, dhs_missing, specs, rng)
    category_maps = _category_maps(specs)
    all_names = list(specs)

    LOGGER.info(
        "Starting imputation %d/%d (%d SEED rows; %d DHS rows)",
        imputation_number,
        N_IMPUTATIONS,
        len(seed_state),
        len(dhs_state),
    )
    for iteration in range(1, MAX_ITER + 1):
        updated_cells = 0
        changed_cells = 0
        max_scaled_numeric_change = 0.0
        for target in all_names:
            predictor_names = [name for name in all_names if name != target]
            observed_train = ~seed_missing[target]
            model, classes = _fit_conditional_model(
                target=target,
                train_frame=seed_state,
                train_mask=observed_train,
                predictor_names=predictor_names,
                specs=specs,
                category_maps=category_maps,
                random_state=random_state + imputation_number + iteration,
            )
            for state, missing_by_column in (
                (seed_state, seed_missing),
                (dhs_state, dhs_missing),
            ):
                missing = missing_by_column[target]
                count = int(missing.sum())
                if count == 0:
                    continue
                previous = state.loc[missing, target].to_numpy(copy=True)
                updated = _draw_predictions(
                    model=model,
                    classes=classes,
                    target=target,
                    frame=state,
                    row_mask=missing,
                    predictor_names=predictor_names,
                    specs=specs,
                    category_maps=category_maps,
                    rng=rng,
                )
                state.loc[missing, target] = updated
                updated_cells += count
                if specs[target].kind == "continuous":
                    previous_numeric = pd.to_numeric(
                        pd.Series(previous), errors="raise"
                    ).to_numpy(dtype=float)
                    updated_numeric = pd.to_numeric(
                        pd.Series(updated), errors="raise"
                    ).to_numpy(dtype=float)
                    changed = previous_numeric != updated_numeric
                    width = (specs[target].bounds or (0.0, 0.0))[1] - (
                        specs[target].bounds or (0.0, 0.0)
                    )[0]
                    if width > 0 and changed.any():
                        max_scaled_numeric_change = max(
                            max_scaled_numeric_change,
                            float(np.max(np.abs(updated_numeric - previous_numeric)) / width),
                        )
                else:
                    changed = np.array(
                        [
                            _category_key(before) != _category_key(after)
                            for before, after in zip(previous, updated)
                        ]
                    )
                changed_cells += int(changed.sum())
            LOGGER.info(
                "Imputation %d, iteration %d/%d, variable %s: "
                "%d SEED and %d DHS missing values updated",
                imputation_number,
                iteration,
                MAX_ITER,
                target,
                int(seed_missing[target].sum()),
                int(dhs_missing[target].sum()),
            )
        LOGGER.info(
            "Imputation %d, iteration %d/%d complete: %d cells updated; "
            "%d changed (%.1f%%); maximum scaled continuous change %.5f",
            imputation_number,
            iteration,
            MAX_ITER,
            updated_cells,
            changed_cells,
            100.0 * changed_cells / updated_cells if updated_cells else 0.0,
            max_scaled_numeric_change,
        )

    for name, spec in specs.items():
        missing = dhs_missing[name]
        if not missing.any():
            continue
        values = dhs_state.loc[missing, name]
        if values.isna().any():
            raise RuntimeError(f"Imputation left missing values in {name!r}.")
        if spec.kind == "categorical":
            valid = {_category_key(value) for value in spec.categories or ()}
            invalid = {_category_key(value) for value in values} - valid
            if invalid:
                raise RuntimeError(
                    f"Imputation produced invalid categories for {name!r}: "
                    f"{sorted(invalid)}."
                )
        else:
            lower, upper = spec.bounds or (-math.inf, math.inf)
            numeric_values = pd.to_numeric(values, errors="raise")
            if ((numeric_values < lower) | (numeric_values > upper)).any():
                raise RuntimeError(
                    f"Imputation produced values outside bounds for {name!r}."
                )
    return dhs_state


def impute_harmonized_data(
    data: pd.DataFrame,
    config: Mapping[str, Any],
    target_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> list[pd.DataFrame]:
    """Return five completed DHS datasets using conditional models fit on SEED.

    The input is not modified. The returned frames contain DHS rows only and
    retain all original columns.
    """
    if IS_DHS_COLUMN not in data:
        raise ValueError(f"Input data must contain an {IS_DHS_COLUMN!r} column.")
    if data[IS_DHS_COLUMN].isna().any() or not data[IS_DHS_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{IS_DHS_COLUMN!r} must contain only nonmissing 0/1 values.")
    imputation = config.get("imputation", {})
    imputation = imputation if isinstance(imputation, Mapping) else {}
    if target_columns is None:
        target_columns = imputation.get("target_variables")
    if not target_columns:
        raise ValueError(
            "Specify target_columns or add imputation.target_variables to the YAML."
        )
    if feature_columns is None:
        feature_columns = imputation.get("overlap_variables")
    if feature_columns is None:
        feature_columns = [
            name
            for name, spec in _variable_metadata(config).items()
            if isinstance(spec, Mapping)
            and spec.get("type") in {"continuous", "categorical", "ordinal", "binary"}
            and spec.get("status") != "unavailable"
            and name not in target_columns
        ]
    if not feature_columns:
        raise ValueError(
            "No overlap feature columns were configured; specify feature_columns "
            "or imputation.overlap_variables in the YAML."
        )

    variable_names = list(feature_columns) + list(target_columns)
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Feature and target columns must be distinct.")
    missing_columns = sorted(set(variable_names + [IS_DHS_COLUMN]) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Harmonized input is missing required columns: "
            + ", ".join(missing_columns)
        )
    seed_mask = data[IS_DHS_COLUMN].eq(0).to_numpy()
    dhs_mask = data[IS_DHS_COLUMN].eq(1).to_numpy()
    if not seed_mask.any() or not dhs_mask.any():
        raise ValueError("Input must contain at least one SEED and one DHS row.")

    seed_original = data.loc[seed_mask, variable_names].copy().reset_index(drop=True)
    dhs_original = data.loc[dhs_mask, variable_names].copy().reset_index(drop=True)
    seed_missing = {name: seed_original[name].isna().to_numpy() for name in variable_names}
    dhs_missing = {name: dhs_original[name].isna().to_numpy() for name in variable_names}
    specs = _specs_for_columns(variable_names, seed_original, config)

    for name, spec in specs.items():
        if spec.kind == "continuous":
            seed_original[name] = pd.to_numeric(seed_original[name], errors="raise")
            dhs_original[name] = pd.to_numeric(dhs_original[name], errors="raise")
        else:
            category_keys = {_category_key(value) for value in spec.categories or ()}
            for label, frame in (("SEED", seed_original), ("DHS", dhs_original)):
                observed = frame[name].dropna()
                unexpected = {
                    _category_key(value) for value in pd.unique(observed)
                } - category_keys
                if unexpected:
                    raise ValueError(
                        f"{label} values for {name!r} include categories not "
                        f"allowed by the YAML/SEED reference: {sorted(unexpected)}."
                    )
            seed_original[name] = seed_original[name].astype(object)
            dhs_original[name] = dhs_original[name].astype(object)

    results: list[pd.DataFrame] = []
    for number in range(1, N_IMPUTATIONS + 1):
        completed_variables = _run_one_imputation(
            seed_frame=seed_original,
            dhs_frame=dhs_original,
            seed_missing=seed_missing,
            dhs_missing=dhs_missing,
            specs=specs,
            random_state=RANDOM_STATE + number - 1,
            imputation_number=number,
        )
        completed = data.loc[dhs_mask].copy().reset_index(drop=True)
        for name in variable_names:
            completed[name] = completed_variables[name].to_numpy()
        missing_targets = completed[list(target_columns)].isna().sum()
        if missing_targets.any():
            raise RuntimeError(
                "Completed DHS data still contain missing target values: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in missing_targets.items()
                    if count
                )
            )
        results.append(completed)
    return results


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping):
        raise ValueError(f"YAML configuration {path} must contain a mapping.")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Combined harmonized CSV returned by load_and_harmonize().",
    )
    parser.add_argument("--config", required=True, type=Path, help="Harmonization YAML.")
    parser.add_argument(
        "--target-columns",
        nargs="+",
        help="Target SEED variables (or use imputation.target_variables in YAML).",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="Overlapping predictor variables (or use imputation.overlap_variables).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("imputed_dhs"),
        help="Directory for imputed_dhs_m1.csv through imputed_dhs_m5.csv.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data = pd.read_csv(args.input_csv)
    config = _load_yaml(args.config)
    imputations = impute_harmonized_data(
        data=data,
        config=config,
        target_columns=args.target_columns,
        feature_columns=args.feature_columns,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, completed in enumerate(imputations, start=1):
        output_path = args.output_dir / f"imputed_dhs_m{number}.csv"
        completed.to_csv(output_path, index=False)
        LOGGER.info("Saved completed DHS dataset: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
