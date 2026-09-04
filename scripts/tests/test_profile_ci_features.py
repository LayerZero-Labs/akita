import pathlib
import unittest

from scripts.profile_ci_features import all_schedule_features, load_feature_graph, schedule_features


class ProfileCiFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = pathlib.Path(__file__).resolve().parents[2]
        cls.graph = load_feature_graph(cls.repo)

    def test_recursive_profile_enables_no_schedule_features(self) -> None:
        self.assertEqual(
            schedule_features(
                self.graph, "akita-pcs", "profile-ci-multi-group-recursive"
            ),
            set(),
        )

    def test_recursive_multichunk_profile_enables_no_schedule_features(self) -> None:
        self.assertEqual(
            schedule_features(
                self.graph,
                "akita-pcs",
                "profile-ci-multi-group-recursive-w8r2",
            ),
            set(),
        )

    def test_schedule_feature_surface_is_empty(self) -> None:
        self.assertEqual(all_schedule_features(self.graph), set())


if __name__ == "__main__":
    unittest.main()
