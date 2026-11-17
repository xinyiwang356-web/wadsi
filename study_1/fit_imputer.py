"""Fit and export SEED-only DHS imputation models.

This command reads the SEED reference CSV only. The serialized bundle contains
fitted BayesianRidge/IterativeImputer state and schema metadata, not SEED rows,
identifiers, or donor records.

Example:
    python study_1/fit_imputer.py --seed-csv data/synthetic_seed_asd_data.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__:
    from .mice_imputer import (
        MAX_ITER,
        N_IMPUTATIONS,
        RANDOM_STATE,
        REPOSITORY_ROOT,
        _load_yaml,
        fit_seed_imputers,
        save_imputer_artifacts,
    )
else:
    from mice_imputer import (
        MAX_ITER,
        N_IMPUTATIONS,
        RANDOM_STATE,
        REPOSITORY_ROOT,
        _load_yaml,
        fit_seed_imputers,
        save_imputer_artifacts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit imputation models from SEED data only."
    )
    parser.add_argument(
        "--seed-csv",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synthetic_seed_asd_data.csv",
        help="SEED reference CSV; no DHS data is read by this command.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "config_seed_asd_harmonization.yaml",
        help="YAML containing variable schema and DHS source mappings.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "outputs"
        / "imputer_artifacts"
        / "imputer_bundle.joblib",
        help="Output path for the fitted imputer artifact.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "outputs"
        / "imputer_artifacts"
        / "schema_manifest.json",
        help="Output path for the SHA-256-protected JSON schema manifest.",
    )
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--n-imputations", type=int, default=N_IMPUTATIONS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_iter < 1:
        raise ValueError("--max-iter must be positive.")
    if args.n_imputations < 1:
        raise ValueError("--n-imputations must be positive.")
    if args.random_state < 0:
        raise ValueError("--random-state must be nonnegative.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _load_yaml(args.config)
    seed_data = pd.read_csv(args.seed_csv)
    if "is_dhs" in seed_data and not seed_data["is_dhs"].eq(0).all():
        raise ValueError("--seed-csv must contain only SEED rows (is_dhs == 0).")
    bundle, manifest = fit_seed_imputers(
        seed_data,
        config,
        n_imputations=args.n_imputations,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    integrity = save_imputer_artifacts(
        bundle,
        manifest,
        args.bundle,
        args.manifest,
    )
    logging.getLogger("fit_imputer").info(
        "Saved %d SEED-only imputer(s). Bundle SHA-256: %s. "
        "No training rows or identifiers are serialized.",
        len(bundle["imputers"]),
        integrity["bundle_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
