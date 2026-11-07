"""SEED-only chained-equation imputation for harmonized SEED/DHS datasets.

By default, the script reads the synthetic source CSVs, maps the nine
overlapping DHS fields to SEED definitions, and creates the 15 unavailable DHS
targets as missing. Each IterativeImputer is fit only on SEED rows and then
used to transform DHS rows. Categorical variables are one-hot encoded before
imputation and decoded to valid categories afterward.

Dependencies: numpy, pandas, scikit-learn, and PyYAML.

Run with repository defaults:
    python study_1/mice_imputer.py

Or supply a previously harmonized combined CSV:
    python study_1/mice_imputer.py --input-csv harmonized_seed_dhs.csv
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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

LOGGER = logging.getLogger("mice_imputer")
IS_DHS_COLUMN = "is_dhs"
MAX_ITER = 20
N_IMPUTATIONS = 5
RANDOM_STATE = 42
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: str
    categories: tuple[Any, ...] | None = None
    bounds: tuple[float, float] | None = None


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


def _variable_specs(
    columns: Sequence[str],
    seed_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, VariableSpec]:
    metadata = _variable_metadata(config)
    imputation = config.get("imputation", {})
    imputation = imputation if isinstance(imputation, Mapping) else {}
    configured_types = imputation.get("variable_types", {})
    configured_categories = imputation.get("categories", {})
    configured_bounds = imputation.get("bounds", {})
    if not isinstance(configured_types, Mapping):
        configured_types = {}
    if not isinstance(configured_categories, Mapping):
        configured_categories = {}
    if not isinstance(configured_bounds, Mapping):
        configured_bounds = {}

    specs: dict[str, VariableSpec] = {}
    for name in columns:
        variable_config = metadata.get(name, {})
        variable_config = variable_config if isinstance(variable_config, Mapping) else {}
        kind = configured_types.get(name) or variable_config.get("type")
        if isinstance(kind, Mapping):
            kind = kind.get("type")
        if kind not in {"continuous", "categorical", "ordinal", "binary"}:
            raise ValueError(
                f"No supported type is configured for {name!r}; use continuous "
                "or categorical in the harmonization YAML."
            )
        kind = "continuous" if kind == "continuous" else "categorical"
        observed_seed = seed_frame[name].dropna()
        if observed_seed.empty:
            raise ValueError(f"SEED has no observed values for {name!r}.")

        categories: tuple[Any, ...] | None = None
        if kind == "categorical":
            category_config = configured_categories.get(name)
            if category_config is None:
                category_config = variable_config.get("categories")
            if category_config is None:
                category_config = variable_config.get("seed_categories")
            if category_config is None:
                recode = variable_config.get("recode_to_seed", {})
                if isinstance(recode, Mapping):
                    category_config = recode.get("seed_categories")
            if isinstance(category_config, Mapping):
                categories = tuple(category_config.keys())
            elif isinstance(category_config, Sequence) and not isinstance(category_config, str):
                categories = tuple(category_config)
            else:
                categories = tuple(pd.unique(observed_seed))

            keys = [_category_key(value) for value in categories]
            observed_keys = {_category_key(value) for value in pd.unique(observed_seed)}
            if not observed_keys.issubset(set(keys)):
                raise ValueError(
                    f"SEED values for {name!r} contain categories absent from "
                    "the YAML specification."
                )
            representatives = {
                _category_key(value): value for value in categories
            }
            representatives.update(
                {_category_key(value): value for value in pd.unique(observed_seed)}
            )
            categories = tuple(representatives[key] for key in dict.fromkeys(keys))

        bounds: tuple[float, float] | None = None
        if kind == "continuous":
            numeric_seed = pd.to_numeric(observed_seed, errors="raise")
            bounds_config = configured_bounds.get(name)
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
            if bounds_config is None:
                bounds = (float(numeric_seed.min()), float(numeric_seed.max()))
            else:
                if (
                    len(bounds_config) != 2
                    or bounds_config[0] is None
                    or bounds_config[1] is None
                ):
                    raise ValueError(
                        f"Both minimum and maximum bounds are required for {name!r}."
                    )
                bounds = (float(bounds_config[0]), float(bounds_config[1]))
            if (
                not all(math.isfinite(value) for value in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"Invalid continuous bounds for {name!r}: {bounds}.")

        specs[name] = VariableSpec(name, kind, categories, bounds)
    return specs


def _category_columns(
    specs: Mapping[str, VariableSpec],
) -> dict[str, tuple[tuple[str, Any], ...]]:
    columns: dict[str, tuple[tuple[str, Any], ...]] = {}
    for name, spec in specs.items():
        if spec.kind == "categorical":
            columns[name] = tuple(
                (f"{name}=={index}", value)
                for index, value in enumerate(spec.categories or ())
            )
    return columns


def _encode_frame(
    frame: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
    categorical_columns: Mapping[str, tuple[tuple[str, Any], ...]],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    encoded_columns: list[np.ndarray] = []
    encoded_names: list[str] = []
    minimums: list[float] = []
    maximums: list[float] = []
    for name, spec in specs.items():
        if spec.kind == "continuous":
            numeric = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
            encoded_columns.append(numeric)
            encoded_names.append(name)
            lower, upper = spec.bounds or (-math.inf, math.inf)
            minimums.append(lower)
            maximums.append(upper)
            continue

        category_keys = {
            _category_key(value): value for value in spec.categories or ()
        }
        observed_keys = {
            _category_key(value) for value in frame[name].dropna().unique()
        }
        unknown = observed_keys - category_keys.keys()
        if unknown:
            raise ValueError(
                f"Values for {name!r} contain categories not allowed by the "
                f"SEED reference: {sorted(unknown)}."
            )
        for encoded_name, category in categorical_columns[name]:
            category_key = _category_key(category)
            values = frame[name].map(
                lambda value: (
                    np.nan
                    if pd.isna(value)
                    else float(_category_key(value) == category_key)
                )
            )
            encoded_columns.append(values.to_numpy(dtype=float))
            encoded_names.append(encoded_name)
            minimums.append(0.0)
            maximums.append(1.0)
    if not encoded_columns:
        raise ValueError("No variables are available for imputation.")
    return (
        np.column_stack(encoded_columns),
        encoded_names,
        np.asarray(minimums, dtype=float),
        np.asarray(maximums, dtype=float),
    )


def _decode_frame(
    encoded: np.ndarray,
    original: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
    encoded_names: Sequence[str],
    categorical_columns: Mapping[str, tuple[tuple[str, Any], ...]],
) -> pd.DataFrame:
    name_to_index = {name: index for index, name in enumerate(encoded_names)}
    decoded = original.copy()
    for name, spec in specs.items():
        if spec.kind == "continuous":
            index = name_to_index[name]
            lower, upper = spec.bounds or (-math.inf, math.inf)
            decoded[name] = np.clip(encoded[:, index], lower, upper)
            continue

        category_columns = categorical_columns[name]
        category_indices = [
            name_to_index[encoded_name] for encoded_name, _ in category_columns
        ]
        category_values = [value for _, value in category_columns]
        category_scores = np.clip(encoded[:, category_indices], 0.0, 1.0)
        category_positions = np.argmax(category_scores, axis=1)
        decoded[name] = np.asarray(category_values, dtype=object)[category_positions]
    return decoded


def impute_harmonized_data(
    data: pd.DataFrame,
    config: Mapping[str, Any],
    target_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
) -> list[pd.DataFrame]:
    """Fit five SEED-only IterativeImputers and return completed DHS frames."""
    if IS_DHS_COLUMN not in data:
        raise ValueError(f"Input data must contain an {IS_DHS_COLUMN!r} column.")
    if (
        data[IS_DHS_COLUMN].isna().any()
        or not data[IS_DHS_COLUMN].isin([0, 1]).all()
    ):
        raise ValueError(f"{IS_DHS_COLUMN!r} must contain only nonmissing 0/1 values.")

    imputation = config.get("imputation", {})
    imputation = imputation if isinstance(imputation, Mapping) else {}
    if target_columns is None:
        target_columns = imputation.get("target_variables")
    if not target_columns:
        raise ValueError(
            "Specify target_columns or imputation.target_variables in the YAML."
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
            "Specify feature_columns or imputation.overlap_variables in the YAML."
        )
    if len(feature_columns) != 9 or len(target_columns) != 15:
        raise ValueError("The imputation pipeline requires 9 overlaps and 15 targets.")
    variable_names = list(feature_columns) + list(target_columns)
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Feature and target columns must be distinct.")
    missing_columns = set(variable_names + [IS_DHS_COLUMN]) - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Harmonized input is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    seed_mask = data[IS_DHS_COLUMN].eq(0)
    dhs_mask = data[IS_DHS_COLUMN].eq(1)
    if not seed_mask.any() or not dhs_mask.any():
        raise ValueError("Input must contain at least one SEED and one DHS row.")
    seed_frame = data.loc[seed_mask, variable_names].copy().reset_index(drop=True)
    dhs_frame = data.loc[dhs_mask, variable_names].copy().reset_index(drop=True)
    LOGGER.info(
        "Splitting data: %d SEED reference rows and %d DHS target rows.",
        len(seed_frame),
        len(dhs_frame),
    )
    specs = _variable_specs(variable_names, seed_frame, config)
    categorical_columns = _category_columns(specs)
    seed_encoded, encoded_names, minimums, maximums = _encode_frame(
        seed_frame, specs, categorical_columns
    )
    dhs_encoded, dhs_encoded_names, _, _ = _encode_frame(
        dhs_frame, specs, categorical_columns
    )
    if encoded_names != dhs_encoded_names:
        raise RuntimeError("SEED and DHS encoded variables are not aligned.")
    if np.isnan(seed_encoded).all(axis=0).any():
        raise ValueError("At least one encoded variable has no observed SEED values.")
    missing_targets = int(dhs_frame[list(target_columns)].isna().sum().sum())
    LOGGER.info(
        "DHS target missingness before imputation: %d / %d cells.",
        missing_targets,
        len(dhs_frame) * len(target_columns),
    )

    results: list[pd.DataFrame] = []
    for number in range(1, N_IMPUTATIONS + 1):
        LOGGER.info(
            "Imputation %d/%d: fitting IterativeImputer on SEED only "
            "(max_iter=%d; random_state=%d).",
            number,
            N_IMPUTATIONS,
            MAX_ITER,
            RANDOM_STATE,
        )
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=MAX_ITER,
            sample_posterior=True,
            random_state=RANDOM_STATE + number - 1,
            min_value=minimums,
            max_value=maximums,
            verbose=2,
        )
        imputer.fit(seed_encoded)
        LOGGER.info(
            "Imputation %d: SEED fit complete after %d iteration(s); "
            "%d conditional estimators learned.",
            number,
            imputer.n_iter_,
            len(imputer.imputation_sequence_),
        )
        completed_encoded = imputer.transform(dhs_encoded)
        completed_variables = _decode_frame(
            completed_encoded,
            dhs_frame,
            specs,
            encoded_names,
            categorical_columns,
        )
        missing = completed_variables[list(target_columns)].isna().sum()
        if missing.any():
            raise RuntimeError(
                "Imputation left missing target values: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in missing.items()
                    if count
                )
            )
        for name, spec in specs.items():
            if spec.kind == "categorical":
                valid_categories = {
                    _category_key(value) for value in spec.categories or ()
                }
                found_categories = {
                    _category_key(value)
                    for value in completed_variables[name].dropna().unique()
                }
                if found_categories - valid_categories:
                    raise RuntimeError(
                        f"Invalid category generated for {name!r}: "
                        f"{sorted(found_categories - valid_categories)}."
                    )
            else:
                lower, upper = spec.bounds or (-math.inf, math.inf)
                numeric = pd.to_numeric(completed_variables[name], errors="raise")
                if ((numeric < lower) | (numeric > upper)).any():
                    raise RuntimeError(
                        f"Imputed values for {name!r} violate bounds {lower}, {upper}."
                    )

        completed = data.loc[dhs_mask].copy().reset_index(drop=True)
        for name in variable_names:
            completed[name] = completed_variables[name].to_numpy()
        results.append(completed)
        LOGGER.info(
            "Imputation %d/%d complete: %d DHS rows; all 15 target columns "
            "contain zero missing values.",
            number,
            N_IMPUTATIONS,
            len(completed),
        )
    return results


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, Mapping):
        raise ValueError(f"YAML configuration {path} must contain a mapping.")
    return config


def _normalized_source(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.casefold() in {"", "na", "n/a", "null", "missing"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    return f"{numeric:.15g}" if math.isfinite(numeric) else text


def _recode_dhs_overlap(
    dhs_data: pd.DataFrame,
    variable: str,
    specification: Mapping[str, Any],
) -> pd.Series:
    source_columns = specification.get("dhs_source_columns", [])
    missing_sources = set(source_columns) - set(dhs_data.columns)
    if missing_sources:
        raise ValueError(
            f"DHS input is missing sources for {variable!r}: "
            + ", ".join(sorted(missing_sources))
        )
    recode = specification.get("recode_to_seed", {})
    recode = recode if isinstance(recode, Mapping) else {}
    if recode.get("strategy") == "diabetes_indicators":
        preexisting = dhs_data[recode["preexisting_source_column"]].map(
            _normalized_source
        )
        pregnancy = dhs_data[recode["during_pregnancy_source_column"]].map(
            _normalized_source
        )
        invalid = (
            set(preexisting.dropna().unique()) | set(pregnancy.dropna().unique())
        ) - {"0", "1"}
        if invalid:
            raise ValueError(
                f"Expected 0/1 source codes for {variable!r}: {sorted(invalid)}."
            )
        result = pd.Series(np.nan, index=dhs_data.index, dtype=float)
        result.loc[preexisting.eq("0") & pregnancy.eq("0")] = 0
        result.loc[preexisting.eq("1")] = 1
        result.loc[preexisting.eq("0") & pregnancy.eq("1")] = 2
        return result
    if specification.get("type") == "continuous":
        source = dhs_data[source_columns[0]]
        numeric = pd.to_numeric(source, errors="coerce")
        invalid = source.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"Non-numeric source values for {variable!r}: "
                f"{sorted(source.loc[invalid].astype(str).unique())[:10]}."
            )
        return numeric

    source_name = recode.get("source_column")
    mapping = recode.get("source_to_seed")
    if not source_name or not isinstance(mapping, Mapping):
        raise ValueError(f"Incomplete categorical mapping for {variable!r}.")
    canonical_mapping = {
        normalized: target
        for source, target in mapping.items()
        if (normalized := _normalized_source(source)) is not None
    }
    values = dhs_data[source_name].map(_normalized_source)
    unknown = set(values.dropna().unique()) - set(canonical_mapping)
    if unknown:
        raise ValueError(
            f"Unmapped DHS values for {variable!r}: {sorted(unknown)[:10]}."
        )
    return values.map(
        lambda value: np.nan if pd.isna(value) else canonical_mapping[str(value)]
    )


def load_and_harmonize(
    seed_csv: Path,
    dhs_csv: Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return combined source rows with 9 mapped overlaps and 15 target fields."""
    imputation = config.get("imputation")
    if not isinstance(imputation, Mapping):
        raise ValueError("The harmonization YAML must define an 'imputation' mapping.")
    overlap_variables = imputation.get("overlap_variables")
    target_variables = imputation.get("target_variables")
    if len(overlap_variables or []) != 9 or len(target_variables or []) != 15:
        raise ValueError("The YAML must define exactly 9 overlaps and 15 targets.")
    variable_columns = list(overlap_variables) + list(target_variables)

    seed_data = pd.read_csv(seed_csv)
    dhs_data = pd.read_csv(dhs_csv)
    missing_seed = set(variable_columns) - set(seed_data.columns)
    if missing_seed:
        raise ValueError(
            "SEED data lack required variables: " + ", ".join(sorted(missing_seed))
        )
    metadata = _variable_metadata(config)
    dhs_harmonized = pd.DataFrame(index=dhs_data.index)
    for name in overlap_variables:
        specification = metadata.get(name)
        if not isinstance(specification, Mapping):
            raise ValueError(f"No YAML mapping is defined for {name!r}.")
        dhs_harmonized[name] = _recode_dhs_overlap(
            dhs_data, name, specification
        )
    for name in target_variables:
        dhs_harmonized[name] = np.nan

    seed_harmonized = seed_data[variable_columns].copy()
    seed_harmonized[IS_DHS_COLUMN] = 0
    dhs_harmonized[IS_DHS_COLUMN] = 1
    if "synthetic_row_id" in seed_data:
        seed_harmonized.insert(
            0, "synthetic_row_id", seed_data["synthetic_row_id"].to_numpy()
        )
    if "RECORD_ID" in dhs_data:
        dhs_harmonized.insert(0, "RECORD_ID", dhs_data["RECORD_ID"].to_numpy())
    if "synthetic_asd_flag" in seed_data:
        seed_harmonized["synthetic_asd_flag"] = seed_data["synthetic_asd_flag"]
    combined = pd.concat([seed_harmonized, dhs_harmonized], ignore_index=True, sort=False)
    LOGGER.info(
        "Harmonized %d SEED and %d DHS rows: 9 overlaps and 15 targets.",
        len(seed_data),
        len(dhs_data),
    )
    LOGGER.info(
        "DHS starts with %d/%d target cells missing.",
        len(dhs_data) * len(target_variables),
        len(dhs_data) * len(target_variables),
    )
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Optional pre-harmonized combined CSV; defaults to loading source files.",
    )
    parser.add_argument(
        "--seed-csv",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synthetic_seed_asd_data.csv",
        help="SEED reference CSV used without --input-csv.",
    )
    parser.add_argument(
        "--dhs-csv",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "study_1"
            / "synthetic_demo"
            / "data"
            / "synthetic_validation_data.csv"
        ),
        help="DHS input CSV used without --input-csv.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "config_seed_asd_harmonization.yaml",
        help="YAML file with mappings, variable types, categories, and bounds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "study_1" / "outputs" / "imputed_dhs",
        help="Directory for imputed_dhs_m1.csv through imputed_dhs_m5.csv.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _load_yaml(args.config)
    if args.input_csv:
        combined = pd.read_csv(args.input_csv)
        LOGGER.info("Loaded pre-harmonized combined data from %s.", args.input_csv)
    else:
        combined = load_and_harmonize(args.seed_csv, args.dhs_csv, config)

    imputations = impute_harmonized_data(combined, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, completed in enumerate(imputations, start=1):
        output_path = args.output_dir / f"imputed_dhs_m{number}.csv"
        completed.to_csv(output_path, index=False)
        LOGGER.info("Saved completed DHS dataset: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
