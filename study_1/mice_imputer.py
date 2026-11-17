"""Shared helpers for isolated SEED model fitting and DHS-local imputation.

Use ``fit_imputer.py`` in the SEED environment and ``transform_imputer.py``
in the DHS environment. This module deliberately has no combined-data runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

LOGGER = logging.getLogger("mice_imputer")
IS_DHS_COLUMN = "is_dhs"
MAX_ITER = 20
N_IMPUTATIONS = 5
RANDOM_STATE = 42
SCHEMA_VERSION = "1.0"
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
            if not any(
                isinstance(value, str) and value == "Missing"
                for value in categories
            ):
                categories = (*categories, "Missing")
            keys = [_category_key(value) for value in categories]
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


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missingness_columns(
    frame: pd.DataFrame,
    variable_names: Sequence[str],
    indicator_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = list(indicator_names) if indicator_names is not None else [
        name for name in variable_names if frame[name].isna().any()
    ]
    indicators = np.column_stack(
        [frame[name].isna().to_numpy(dtype=float) for name in names]
    ) if names else np.empty((len(frame), 0), dtype=float)
    return indicators, [f"__missing__{name}" for name in names]


def _encode_with_missingness(
    frame: pd.DataFrame,
    specs: Mapping[str, VariableSpec],
    categorical_columns: Mapping[str, tuple[tuple[str, Any], ...]],
    indicator_variables: Sequence[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    encoded, names, minimums, maximums = _encode_frame(
        frame, specs, categorical_columns
    )
    indicators, indicator_names = _missingness_columns(
        frame, list(specs), indicator_variables
    )
    if indicators.shape[1]:
        encoded = np.column_stack((encoded, indicators))
        names.extend(indicator_names)
        minimums = np.concatenate((minimums, np.zeros(indicators.shape[1])))
        maximums = np.concatenate((maximums, np.ones(indicators.shape[1])))
    return encoded, names, minimums, maximums


def fit_seed_imputers(
    seed_data: pd.DataFrame,
    config: Mapping[str, Any],
    n_imputations: int = N_IMPUTATIONS,
    random_state: int = RANDOM_STATE,
    max_iter: int = MAX_ITER,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit portable imputation models using SEED rows only.

    The returned bundle contains fitted BayesianRidge estimator state and
    aggregate initialization statistics, but never the input dataframe.
    """
    if n_imputations < 1:
        raise ValueError("n_imputations must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if random_state < 0:
        raise ValueError("random_state must be nonnegative.")
    imputation = config.get("imputation", {})
    if not isinstance(imputation, Mapping):
        raise ValueError("The harmonization YAML must define an 'imputation' mapping.")
    feature_columns = list(imputation.get("overlap_variables", []))
    target_columns = list(imputation.get("target_variables", []))
    if len(feature_columns) != 9 or len(target_columns) != 15:
        raise ValueError("The imputation pipeline requires 9 overlaps and 15 targets.")
    variable_names = feature_columns + target_columns
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("Feature and target columns must be distinct.")
    if IS_DHS_COLUMN in seed_data and not seed_data[IS_DHS_COLUMN].eq(0).all():
        raise ValueError("The SEED fit input may contain only is_dhs == 0 rows.")
    missing_columns = set(variable_names) - set(seed_data.columns)
    if missing_columns:
        raise ValueError(
            "SEED data lack required variables: "
            + ", ".join(sorted(missing_columns))
        )
    if seed_data.empty:
        raise ValueError("SEED fit input contains no rows.")

    # Select only configured variables; identifiers and unrelated columns are
    # never passed into preprocessing or retained in the returned bundle.
    seed_frame = seed_data.loc[:, variable_names].copy()
    specs = _variable_specs(variable_names, seed_frame, config)
    categorical_columns = _category_columns(specs)
    indicator_variables = variable_names
    seed_encoded, encoded_names, minimums, maximums = _encode_with_missingness(
        seed_frame,
        specs,
        categorical_columns,
        indicator_variables,
    )
    if np.isnan(seed_encoded).all(axis=0).any():
        empty_columns = [
            encoded_names[index]
            for index in np.flatnonzero(np.isnan(seed_encoded).all(axis=0))
        ]
        raise ValueError(
            "SEED has no observed values for encoded feature(s): "
            + ", ".join(empty_columns)
        )

    # IterativeImputer only learns conditional models for columns missing at
    # fit time. A small deterministic SEED-only holdout gives fully observed
    # variables a conditional model when they are entirely missing in DHS.
    artificial_masking: dict[str, Any] = {
        "enabled": False,
        "fraction": 0.05,
        "variables": [],
        "random_state": random_state,
    }
    if len(seed_frame) > 1:
        seed_rng = np.random.RandomState(random_state)
        variable_encoded_columns: dict[str, list[int]] = {}
        encoded_index = {name: index for index, name in enumerate(encoded_names)}
        for name, spec in specs.items():
            if spec.kind == "continuous":
                variable_encoded_columns[name] = [encoded_index[name]]
            else:
                variable_encoded_columns[name] = [
                    encoded_index[encoded_name]
                    for encoded_name, _ in categorical_columns[name]
                ]
        masked_variables: list[str] = []
        mask_count = min(
            len(seed_frame) - 1,
            max(1, int(round(len(seed_frame) * artificial_masking["fraction"]))),
        )
        for name in variable_names:
            if seed_frame[name].isna().any():
                continue
            row_indices = seed_rng.choice(
                len(seed_frame), size=mask_count, replace=False
            )
            seed_encoded[np.ix_(row_indices, variable_encoded_columns[name])] = np.nan
            masked_variables.append(name)
        artificial_masking.update(
            enabled=bool(masked_variables),
            variables=masked_variables,
        )

    fitted_imputers: list[IterativeImputer] = []
    for number in range(n_imputations):
        seed = random_state + number
        LOGGER.info(
            "Imputation %d/%d: fitting on %d SEED rows only "
            "(max_iter=%d; random_state=%d).",
            number + 1,
            n_imputations,
            len(seed_frame),
            max_iter,
            seed,
        )
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            sample_posterior=True,
            random_state=seed,
            min_value=minimums,
            max_value=maximums,
            verbose=2,
        )
        imputer.fit(seed_encoded)
        fitted_imputers.append(imputer)
        LOGGER.info(
            "Imputation %d: SEED fit complete after %d iteration(s); "
            "%d conditional estimators learned.",
            number + 1,
            imputer.n_iter_,
            len(imputer.imputation_sequence_),
        )

    variable_specs = {
        name: {
            "type": spec.kind,
            "categories": [_json_value(value) for value in (spec.categories or ())],
            "bounds": list(spec.bounds) if spec.bounds is not None else None,
            "source_dtype": str(seed_frame[name].dtype),
            "missing_level": "Missing" if spec.kind == "categorical" else None,
        }
        for name, spec in specs.items()
    }
    categorical_columns_json = {
        name: [
            {"encoded_name": encoded_name, "value": _json_value(value)}
            for encoded_name, value in columns
        ]
        for name, columns in categorical_columns.items()
    }
    analysis = config.get("analysis", {})
    analysis = analysis if isinstance(analysis, Mapping) else {}
    variable_metadata = _variable_metadata(config)
    dhs_mapping = {
        name: variable_metadata[name]
        for name in feature_columns
        if name in variable_metadata
    }
    dhs_required_columns = list(dict.fromkeys(
        source
        for specification in dhs_mapping.values()
        for source in specification.get("dhs_source_columns", [])
    ))
    identifier = analysis.get("identifier")
    factor_levels = {
        name: [_json_value(value) for value in (spec.categories or ())]
        for name, spec in specs.items()
        if spec.kind == "categorical"
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "imputer": {
            "class": "sklearn.impute.IterativeImputer",
            "estimator": "sklearn.linear_model.BayesianRidge",
            "max_iter": max_iter,
            "n_imputations": n_imputations,
            "sample_posterior": True,
            "random_state_base": random_state,
            "random_states": [
                random_state + number for number in range(n_imputations)
            ],
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
            "python_version": platform.python_version(),
        },
        "input_schema": {
            "variable_order": variable_names,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "encoded_column_order": encoded_names,
            "variable_specs": variable_specs,
            "categorical_factor_levels": factor_levels,
            "categorical_encodings": categorical_columns_json,
            "missingness_indicator_variables": indicator_variables,
            "missingness_indicator_columns": [
                f"__missing__{name}" for name in indicator_variables
            ],
            "continuous_scaling": "none",
            "artificial_masking": artificial_masking,
            "dhs_source_columns": dhs_required_columns,
            "dhs_identifier_column": identifier,
            "dhs_overlap_mapping": dhs_mapping,
        },
        "privacy": {
            "training_rows_serialized": False,
            "identifiers_serialized": False,
            "donor_rows_serialized": False,
            "estimator_family": "parametric BayesianRidge; no PMM donor pool",
        },
    }
    schema_sha256 = _canonical_json_hash(manifest)
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha256,
        "imputers": fitted_imputers,
        "encoded_column_order": encoded_names,
        "variable_order": variable_names,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "variable_specs": variable_specs,
        "categorical_encodings": categorical_columns_json,
        "missingness_indicator_variables": indicator_variables,
        "minimums": minimums,
        "maximums": maximums,
    }
    return bundle, manifest


def save_imputer_artifacts(
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle_path: Path,
    manifest_path: Path,
) -> dict[str, str]:
    """Write a fitted artifact and a detached SHA-256-protected manifest."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), bundle_path, compress=3)
    final_manifest = copy.deepcopy(dict(manifest))
    final_manifest["integrity"] = {
        "bundle_sha256": _sha256_file(bundle_path),
        "schema_sha256": _canonical_json_hash(final_manifest),
        "bundle_file": bundle_path.name,
    }
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return final_manifest["integrity"]


def load_imputer_artifacts(
    bundle_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load artifacts only after validating their integrity and schema version."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}."
        )
    integrity = manifest.get("integrity", {})
    expected_bundle_hash = integrity.get("bundle_sha256")
    actual_bundle_hash = _sha256_file(bundle_path)
    if not expected_bundle_hash or actual_bundle_hash != expected_bundle_hash:
        raise ValueError("Imputer bundle SHA-256 verification failed.")
    manifest_payload = {key: value for key, value in manifest.items() if key != "integrity"}
    expected_schema_hash = integrity.get("schema_sha256")
    if (
        not expected_schema_hash
        or _canonical_json_hash(manifest_payload) != expected_schema_hash
    ):
        raise ValueError("Schema manifest SHA-256 verification failed.")
    fitted_sklearn_version = manifest.get("imputer", {}).get("sklearn_version")
    if fitted_sklearn_version != sklearn.__version__:
        raise ValueError(
            "Imputer bundle requires scikit-learn "
            f"{fitted_sklearn_version!r}; found {sklearn.__version__!r}."
        )
    bundle = joblib.load(bundle_path)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Bundle and manifest schema versions differ.")
    if bundle.get("schema_sha256") != expected_schema_hash:
        raise ValueError("Bundle does not match the schema manifest.")
    return bundle, manifest


def harmonize_dhs_only(
    dhs_data: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate and harmonize DHS locally, without loading SEED records."""
    if IS_DHS_COLUMN in dhs_data.columns and not dhs_data[IS_DHS_COLUMN].eq(1).all():
        raise ValueError("DHS transform input may contain only is_dhs == 1 rows.")
    schema = manifest["input_schema"]
    required = list(schema["dhs_source_columns"])
    identifier = schema.get("dhs_identifier_column")
    if identifier and identifier in dhs_data.columns:
        required.append(identifier)
    missing = set(required) - set(dhs_data.columns)
    if missing:
        raise ValueError(
            "DHS input is missing required columns: " + ", ".join(sorted(missing))
        )
    metadata = schema["dhs_overlap_mapping"]
    harmonized = pd.DataFrame(index=dhs_data.index)
    for name in schema["feature_columns"]:
        specification = metadata.get(name)
        if not isinstance(specification, Mapping):
            raise ValueError(f"No DHS harmonization mapping stored for {name!r}.")
        harmonized[name] = _recode_dhs_overlap(dhs_data, name, specification)
    for name in schema["target_columns"]:
        harmonized[name] = np.nan
    if identifier and identifier in dhs_data.columns:
        harmonized.insert(0, identifier, dhs_data[identifier].to_numpy())
    harmonized[IS_DHS_COLUMN] = 1
    return harmonized.reset_index(drop=True)


def transform_dhs_data(
    dhs_data: pd.DataFrame,
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list[pd.DataFrame]:
    """Apply frozen SEED models to DHS rows only."""
    schema = manifest["input_schema"]
    variable_names = list(bundle["variable_order"])
    missing_columns = set(variable_names) - set(dhs_data.columns)
    if missing_columns:
        raise ValueError(
            "Harmonized DHS input lacks required variables: "
            + ", ".join(sorted(missing_columns))
        )
    if dhs_data.empty:
        raise ValueError("DHS transform input contains no rows.")
    if IS_DHS_COLUMN in dhs_data.columns and not dhs_data[IS_DHS_COLUMN].eq(1).all():
        raise ValueError("DHS transform input may contain only is_dhs == 1 rows.")
    identifier = schema.get("dhs_identifier_column")
    identifier_values = None
    if identifier and identifier in dhs_data.columns:
        identifier_values = dhs_data[identifier].copy().reset_index(drop=True)

    frame = dhs_data.loc[:, variable_names].copy().reset_index(drop=True)
    specs = {
        name: VariableSpec(
            name=name,
            kind=specification["type"],
            categories=tuple(specification["categories"])
            if specification["type"] == "categorical"
            else None,
            bounds=tuple(specification["bounds"])
            if specification["bounds"] is not None
            else None,
        )
        for name, specification in bundle["variable_specs"].items()
    }
    categorical_columns = {
        name: tuple(
            (item["encoded_name"], item["value"])
            for item in columns
        )
        for name, columns in bundle["categorical_encodings"].items()
    }
    encoded, encoded_names, _, _ = _encode_with_missingness(
        frame,
        specs,
        categorical_columns,
        bundle["missingness_indicator_variables"],
    )
    if encoded_names != bundle["encoded_column_order"]:
        raise ValueError(
            "DHS feature ordering does not match the fitted schema manifest."
        )
    empty_predictors = [
        name for name in bundle["feature_columns"] if frame[name].isna().all()
    ]
    if empty_predictors:
        raise ValueError(
            "DHS has no observed values for required predictor(s): "
            + ", ".join(empty_predictors)
        )

    completed_sets: list[pd.DataFrame] = []
    random_states = manifest["imputer"]["random_states"]
    if len(random_states) != len(bundle["imputers"]):
        raise ValueError("Imputer count does not match manifest random-state list.")
    for number, (fitted_imputer, random_state) in enumerate(
        zip(bundle["imputers"], random_states, strict=True),
        start=1,
    ):
        # Reset a deep copy to a known RNG state so a fresh CLI process always
        # produces the same m-th completed dataset.
        local_imputer = copy.deepcopy(fitted_imputer)
        local_imputer.random_state = int(random_state)
        completed_encoded = local_imputer.transform(encoded)
        completed_variables = _decode_frame(
            completed_encoded,
            frame,
            specs,
            encoded_names,
            categorical_columns,
        )
        if completed_variables[list(bundle["target_columns"])].isna().any().any():
            raise RuntimeError(
                f"Imputation {number} left missing DHS target values."
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
                        f"Invalid category in {name!r}: "
                        f"{sorted(found_categories - valid_categories)}."
                    )
            else:
                lower, upper = spec.bounds or (-math.inf, math.inf)
                numeric = pd.to_numeric(completed_variables[name], errors="raise")
                if ((numeric < lower) | (numeric > upper)).any():
                    raise RuntimeError(
                        f"Values for {name!r} violate bounds {lower}, {upper}."
                    )

        completed = completed_variables.copy()
        if identifier_values is not None:
            completed.insert(0, identifier, identifier_values.to_numpy())
        completed[IS_DHS_COLUMN] = 1
        indicator_values, indicator_names = _missingness_columns(
            frame,
            variable_names,
            bundle["missingness_indicator_variables"],
        )
        for index, name in enumerate(indicator_names):
            completed[name] = indicator_values[:, index].astype(np.int8)
        completed_sets.append(completed)
        LOGGER.info(
            "Transform imputation %d/%d complete: %d DHS rows; "
            "all target values populated.",
            number,
            len(bundle["imputers"]),
            len(completed),
        )
    return completed_sets


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


if __name__ == "__main__":
    raise SystemExit(
        "mice_imputer.py is a shared library; run fit_imputer.py in the SEED "
        "environment and transform_imputer.py in the DHS environment."
    )
