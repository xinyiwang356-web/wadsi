"""Generate synthetic SEED-like rows from aggregate values in the SEED codebook.

The generated records are not observed participants. Categorical counts and
group-specific missingness counts are reproduced from the codebook; continuous
values are simulated from the reported group-specific means and SDs. Variables
are generated independently within ASD/control groups because the codebook
does not report their joint distributions. The codebook reports gestational
diabetes only as Yes/No, so synthetic Yes values use code 1 without claiming a
pre-existing-versus-gestational subtype.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEED = 20261104
IS_DHS_COLUMN = "is_dhs"
OVERLAP_VARIABLES = [
    "dr_gawks",
    "gest_dm_bin",
    "DR_PREGHTN",
    "dr_agebir_mx",
    "dr_agebir_fx",
    "dr_sex_cx",
    "dr_prevlb_br_3c",
    "dr_biredu_mx_5c",
    "dr_race_mx_5c",
]
TARGET_VARIABLES = [
    "DR_INF_FV_PG_MX",
    "dr_anx_mmh_mx",
    "dr_dep_mmh_mx",
    "dr_pg_incpovrat_up_4c",
    "DR_pgsASD_CX",
    *(f"DR_gwaPC{number}_CX" for number in range(1, 11)),
]
SYNTHETIC_ASD_COLUMN = "synthetic_asd_flag"


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value)) or str(value).strip() == ""


def _count(value: Any) -> int:
    if _is_missing(value):
        return 0
    match = re.match(r"\s*(\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse codebook count {value!r}.")
    return int(match.group(1))


def _continuous_summaries(codebook: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for row_index in range(1, 5):
        row = codebook.iloc[row_index]
        name = str(row.iloc[0]).split("(", maxsplit=1)[0].strip()
        if name == "DR_PREGHTN":
            continue
        summaries[name] = {
            "asd_mean": float(row.iloc[5]),
            "asd_sd": float(row.iloc[6]),
            "asd_missing": _count(row.iloc[7]),
            "control_mean": float(row.iloc[8]),
            "control_sd": float(row.iloc[9]),
            "control_missing": _count(row.iloc[10]),
        }

    for row_index in range(6, len(codebook)):
        row = codebook.iloc[row_index]
        name = str(row.iloc[0]).strip()
        if not name.startswith("DR_gwaPC"):
            continue
        summaries[name] = {
            "mean": float(row.iloc[2]),
            "sd": float(row.iloc[3]),
            "missing": _count(row.iloc[4]),
        }
    return summaries


def _categorical_summaries(
    codebook: pd.DataFrame,
) -> dict[str, list[tuple[str, int, int]]]:
    header_index = next(
        index
        for index, value in enumerate(codebook.iloc[:, 0])
        if str(value).strip() == "Categorical Variable Name"
    )
    summaries: dict[str, list[tuple[str, int, int]]] = {}
    current_name: str | None = None
    for _, row in codebook.iloc[header_index + 1 :].iterrows():
        if not _is_missing(row.iloc[0]):
            current_name = str(row.iloc[0]).strip()
            if current_name == "DR_PREG HTN":
                current_name = "DR_PREGHTN"
            summaries.setdefault(current_name, [])
        if current_name is None or _is_missing(row.iloc[1]):
            continue
        summaries[current_name].append(
            (str(row.iloc[1]).strip(), _count(row.iloc[4]), _count(row.iloc[6]))
        )
    return summaries


def _category_code(variable: str, label: str, category_index: int) -> float:
    normalized = label.casefold()
    if normalized == "missing":
        return np.nan
    if variable == "dr_sex_cx":
        return {"female": 2.0, "male": 1.0}[normalized]
    if variable == "dr_prevlb_br_3c":
        return {
            "first born": 1.0,
            "second born": 2.0,
            "third born or later": 3.0,
        }[normalized]
    if variable == "dr_biredu_mx_5c":
        return {
            "less than high school": 1.0,
            "high school graduate": 2.0,
            "some college": 3.0,
            "college graduate": 4.0,
            "advanced degree": 5.0,
        }[normalized]
    if variable == "dr_race_mx_5c":
        return {
            "white, non-hispanic": 1.0,
            "black, non-hispanic": 2.0,
            "asian/pacific islander, non-hispanic": 3.0,
            "other or multiracial, non-hispanic": 4.0,
            "hispanic": 5.0,
        }[normalized]
    if variable == "dr_pg_incpovrat_up_4c":
        return float(category_index + 1)
    if normalized == "no":
        return 0.0
    if normalized == "yes":
        return 1.0
    raise ValueError(f"Unexpected codebook category {label!r} for {variable!r}.")


def _make_group_labels(
    categorical: dict[str, list[tuple[str, int, int]]],
    rng: np.random.Generator,
) -> np.ndarray:
    sex_counts = categorical["dr_sex_cx"]
    n_asd = sum(asd_count for _, asd_count, _ in sex_counts)
    n_control = sum(control_count for _, _, control_count in sex_counts)
    labels = np.concatenate(
        [np.ones(n_asd, dtype=np.int8), np.zeros(n_control, dtype=np.int8)]
    )
    rng.shuffle(labels)
    return labels


def _simulate_continuous(
    variable: str,
    summaries: dict[str, dict[str, float | int]],
    asd_labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    summary = summaries[variable]
    values = np.empty(len(asd_labels), dtype=float)
    if "asd_mean" in summary:
        group_parameters = {
            1: (
                float(summary["asd_mean"]),
                float(summary["asd_sd"]),
                int(summary["asd_missing"]),
            ),
            0: (
                float(summary["control_mean"]),
                float(summary["control_sd"]),
                int(summary["control_missing"]),
            ),
        }
    else:
        group_parameters = {
            0: (float(summary["mean"]), float(summary["sd"]), int(summary["missing"])),
            1: (float(summary["mean"]), float(summary["sd"]), int(summary["missing"])),
        }

    plausible_bounds = {
        "dr_gawks": (20.0, 45.0),
        "dr_agebir_mx": (15.0, 55.0),
        "dr_agebir_fx": (15.0, 80.0),
    }
    for group, (mean, standard_deviation, missing_count) in group_parameters.items():
        indices = np.flatnonzero(asd_labels == group)
        values[indices] = rng.normal(mean, standard_deviation, len(indices))
        if variable in plausible_bounds:
            lower, upper = plausible_bounds[variable]
            values[indices] = np.clip(values[indices], lower, upper)
        if missing_count > len(indices):
            raise ValueError(
                f"Codebook missing count for {variable!r} exceeds group size."
            )
        missing_indices = rng.choice(indices, size=missing_count, replace=False)
        values[missing_indices] = np.nan
    return values


def _simulate_categorical(
    variable: str,
    rows: list[tuple[str, int, int]],
    asd_labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.full(len(asd_labels), np.nan)
    for group, count_column in ((1, 1), (0, 2)):
        group_indices = np.flatnonzero(asd_labels == group)
        generated: list[float] = []
        for category_index, (label, asd_count, control_count) in enumerate(rows):
            count = asd_count if count_column == 1 else control_count
            category_value = _category_code(variable, label, category_index)
            if not np.isnan(category_value):
                generated.extend([category_value] * count)
        missing_count = sum(
            asd_count if count_column == 1 else control_count
            for label, asd_count, control_count in rows
            if label.casefold() == "missing"
        )
        if len(generated) + missing_count != len(group_indices):
            raise ValueError(
                f"Codebook category counts for {variable!r} do not sum to "
                f"the group size ({len(group_indices)})."
            )
        generated_values = np.asarray(generated, dtype=float)
        rng.shuffle(generated_values)
        values[group_indices[: len(generated_values)]] = generated_values
    return values


def generate_dataset(codebook_path: Path, random_state: int = SEED) -> pd.DataFrame:
    codebook = pd.read_excel(codebook_path, sheet_name=0, header=None)
    continuous = _continuous_summaries(codebook)
    categorical = _categorical_summaries(codebook)
    required_categorical = {
        "gest_dm_bin",
        "DR_PREGHTN",
        "dr_sex_cx",
        "dr_prevlb_br_3c",
        "dr_biredu_mx_5c",
        "dr_race_mx_5c",
        "DR_INF_FV_PG_MX",
        "dr_anx_mmh_mx",
        "dr_dep_mmh_mx",
        "dr_pg_incpovrat_up_4c",
    }
    missing_specs = required_categorical - categorical.keys()
    if missing_specs:
        raise ValueError(
            "Codebook is missing categorical specifications for: "
            + ", ".join(sorted(missing_specs))
        )
    missing_continuous = {
        "dr_gawks",
        "dr_agebir_mx",
        "dr_agebir_fx",
        "DR_pgsASD_CX",
        *(f"DR_gwaPC{number}_CX" for number in range(1, 11)),
    } - continuous.keys()
    if missing_continuous:
        raise ValueError(
            "Codebook is missing continuous specifications for: "
            + ", ".join(sorted(missing_continuous))
        )

    rng = np.random.default_rng(random_state)
    asd_labels = _make_group_labels(categorical, rng)
    data = pd.DataFrame(
        {
            "synthetic_row_id": np.arange(1, len(asd_labels) + 1),
            SYNTHETIC_ASD_COLUMN: asd_labels,
            IS_DHS_COLUMN: np.zeros(len(asd_labels), dtype=np.int8),
        }
    )
    for variable in OVERLAP_VARIABLES + TARGET_VARIABLES:
        if variable in continuous:
            data[variable] = _simulate_continuous(
                variable, continuous, asd_labels, rng
            )
        else:
            data[variable] = _simulate_categorical(
                variable, categorical[variable], asd_labels, rng
            )
    return data[
        [
            "synthetic_row_id",
            *OVERLAP_VARIABLES,
            *TARGET_VARIABLES,
            SYNTHETIC_ASD_COLUMN,
            IS_DHS_COLUMN,
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "seed codebook.xlsx",
        help="SEED codebook XLSX (default: ../data/seed codebook.xlsx).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_seed_asd_data.csv",
        help="Output CSV path (default: ../data/synthetic_seed_asd_data.csv).",
    )
    parser.add_argument("--random-state", type=int, default=SEED)
    args = parser.parse_args()

    dataset = generate_dataset(args.codebook, args.random_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(
        f"Saved {len(dataset):,} synthetic SEED rows "
        f"({int(dataset[SYNTHETIC_ASD_COLUMN].sum()):,} ASD; "
        f"{int((dataset[SYNTHETIC_ASD_COLUMN] == 0).sum()):,} controls) "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
