"""Apply a previously fitted SEED-only imputation bundle to DHS data locally.

The command reads DHS rows and the exported model/schema artifacts. It does
not load or require access to the SEED dataset.

Example:
    python study_1/transform_imputer.py --dhs-csv path/to/dhs.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__:
    from .mice_imputer import (
        REPOSITORY_ROOT,
        harmonize_dhs_only,
        load_imputer_artifacts,
        transform_dhs_data,
    )
else:
    from mice_imputer import (
        REPOSITORY_ROOT,
        harmonize_dhs_only,
        load_imputer_artifacts,
        transform_dhs_data,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Impute DHS rows using frozen SEED-derived models."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--dhs-csv",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "study_1"
            / "synthetic_demo"
            / "data"
            / "synthetic_validation_data.csv"
        ),
        help="Raw DHS/validation CSV; DHS is harmonized locally.",
    )
    input_group.add_argument(
        "--harmonized-csv",
        type=Path,
        help="Optional DHS-only CSV already harmonized to the SEED schema.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "outputs"
        / "imputer_artifacts"
        / "imputer_bundle.joblib",
        help="SEED-trained imputer bundle transferred to this environment.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "outputs"
        / "imputer_artifacts"
        / "schema_manifest.json",
        help="Schema and integrity manifest paired with the bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "study_1"
        / "outputs"
        / "imputed_dhs",
        help="Directory for imputed_dhs_m1.csv through m5.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bundle, manifest = load_imputer_artifacts(args.bundle, args.manifest)
    if args.harmonized_csv:
        dhs_data = pd.read_csv(args.harmonized_csv)
    else:
        raw_dhs = pd.read_csv(args.dhs_csv)
        dhs_data = harmonize_dhs_only(raw_dhs, manifest)
    completed_sets = transform_dhs_data(dhs_data, bundle, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("transform_imputer")
    for number, completed in enumerate(completed_sets, start=1):
        output_path = args.output_dir / f"imputed_dhs_m{number}.csv"
        completed.to_csv(output_path, index=False)
        logger.info("Saved completed DHS dataset: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
