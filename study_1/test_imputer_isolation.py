import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mice_imputer import (  # noqa: E402
    _load_yaml,
    fit_seed_imputers,
    harmonize_dhs_only,
    load_imputer_artifacts,
    save_imputer_artifacts,
    transform_dhs_data,
)


ROOT = Path(__file__).resolve().parents[1]
SEED_CSV = ROOT / "data" / "synthetic_seed_asd_data.csv"
DHS_CSV = (
    ROOT
    / "study_1"
    / "synthetic_demo"
    / "data"
    / "synthetic_validation_data.csv"
)
CONFIG = ROOT / "study_1" / "config_seed_asd_harmonization.yaml"


class ImputerIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _load_yaml(CONFIG)
        cls.seed = pd.read_csv(SEED_CSV).sample(n=600, random_state=9)
        cls.bundle, cls.manifest = fit_seed_imputers(
            cls.seed,
            cls.config,
            n_imputations=1,
            random_state=42,
            max_iter=2,
        )
        raw_dhs = pd.read_csv(DHS_CSV, nrows=32)
        cls.dhs = harmonize_dhs_only(raw_dhs, cls.manifest)

    def test_artifact_contains_model_state_but_no_row_data_or_ids(self) -> None:
        self.assertNotIn("RECORD_ID", self.bundle["variable_order"])
        self.assertFalse(any(isinstance(value, pd.DataFrame) for value in self.bundle.values()))
        self.assertFalse(any(isinstance(value, pd.Series) for value in self.bundle.values()))
        self.assertFalse(self.manifest["privacy"]["training_rows_serialized"])
        self.assertFalse(self.manifest["privacy"]["identifiers_serialized"])
        self.assertFalse(self.manifest["privacy"]["donor_rows_serialized"])

    def test_transform_is_deterministic_and_completes_targets(self) -> None:
        first = transform_dhs_data(self.dhs, self.bundle, self.manifest)[0]
        second = transform_dhs_data(self.dhs, self.bundle, self.manifest)[0]
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["is_dhs"].tolist(), [1] * len(first))
        targets = self.bundle["target_columns"]
        self.assertEqual(int(first[targets].isna().sum().sum()), 0)
        self.assertEqual(
            first["__missing__DR_gwaPC1_CX"].tolist(),
            [1] * len(first),
        )

    def test_unseen_factor_level_is_rejected(self) -> None:
        invalid = self.dhs.copy()
        invalid["dr_sex_cx"] = invalid["dr_sex_cx"].astype(object)
        invalid.loc[0, "dr_sex_cx"] = "unseen-level"
        with self.assertRaisesRegex(ValueError, "categories not allowed"):
            transform_dhs_data(invalid, self.bundle, self.manifest)

    def test_seed_rows_are_rejected_by_dhs_harmonizer(self) -> None:
        seed_marked_as_dhs = pd.read_csv(DHS_CSV, nrows=2)
        seed_marked_as_dhs["is_dhs"] = 0
        with self.assertRaisesRegex(ValueError, "is_dhs == 1"):
            harmonize_dhs_only(seed_marked_as_dhs, self.manifest)

    def test_bundle_integrity_is_checked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            bundle_path = Path(temporary_directory) / "bundle.joblib"
            manifest_path = Path(temporary_directory) / "schema_manifest.json"
            save_imputer_artifacts(
                self.bundle,
                self.manifest,
                bundle_path,
                manifest_path,
            )
            load_imputer_artifacts(bundle_path, manifest_path)
            with bundle_path.open("ab") as bundle_file:
                bundle_file.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "bundle SHA-256"):
                load_imputer_artifacts(bundle_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
