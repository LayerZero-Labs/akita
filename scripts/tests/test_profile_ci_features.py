import pathlib
import unittest

from scripts.profile_ci_features import (
    all_schedule_features,
    load_feature_graph,
    schedule_features,
    schedule_symbol,
)


class ProfileCiFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = pathlib.Path(__file__).resolve().parents[2]
        cls.graph = load_feature_graph(cls.repo)

    def test_recursive_profile_expands_direct_base_catalog(self) -> None:
        self.assertEqual(
            schedule_features(
                self.graph, "akita-pcs", "profile-ci-multi-group-recursive"
            ),
            {"fp128-onehot", "fp128-onehot-recursive"},
        )

    def test_recursive_multichunk_profile_expands_direct_base_catalog(self) -> None:
        self.assertEqual(
            schedule_features(
                self.graph,
                "akita-pcs",
                "profile-ci-multi-group-recursive-w8r2",
            ),
            {
                "fp128-onehot-multi-chunk",
                "fp128-onehot-recursive-multi-chunk-w8r2",
            },
        )

    def test_every_schedule_feature_has_a_linkage_symbol(self) -> None:
        symbols = {
            schedule_symbol(feature) for feature in all_schedule_features(self.graph)
        }
        self.assertIn("FP128_ONEHOT_RECURSIVE_SCHEDULES", symbols)
        self.assertIn("FP64_D256_ONEHOT_SCHEDULES", symbols)


if __name__ == "__main__":
    unittest.main()
