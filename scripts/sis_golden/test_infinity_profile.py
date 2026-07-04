#!/usr/bin/env python3
"""Pure-Python tests for infinity golden profile selection."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(GOLDEN_DIR))

from infinity_profile import (  # noqa: E402
    InfinityEstimatorProfile,
    _cli_profile_specified,
    default_metadata_path,
    default_output_path,
    profile_from_args,
    profile_from_metadata_with_overrides,
)


class InfinityProfileTests(unittest.TestCase):
    def test_default_slug_uses_committed_paths(self) -> None:
        profile = InfinityEstimatorProfile.default_optimizer()
        self.assertEqual(
            default_output_path(GOLDEN_DIR / "infinity_golden.csv", profile).name,
            "infinity_golden.csv",
        )
        self.assertEqual(
            default_metadata_path(GOLDEN_DIR / "infinity_golden.csv").name,
            "infinity_metadata.json",
        )

    def test_non_default_slug(self) -> None:
        profile = InfinityEstimatorProfile(red_shape_model="GSA")
        output = default_output_path(GOLDEN_DIR / "infinity_golden.csv", profile)
        self.assertEqual(output.name, "infinity_golden_adps16_gsa.csv")

    def test_metadata_round_trip(self) -> None:
        profile = InfinityEstimatorProfile.default_fixed()
        restored = InfinityEstimatorProfile.from_metadata(profile.to_metadata())
        self.assertEqual(restored, profile)

    def test_cli_override_beats_metadata(self) -> None:
        metadata = {"profile": InfinityEstimatorProfile.default_optimizer().to_metadata()}
        args = argparse.Namespace(
            red_cost_model="adps16",
            red_shape_model="gsa",
            adps16_mode="classical",
            nearest_neighbor="classical",
        )
        self.assertTrue(_cli_profile_specified(args))
        profile = profile_from_metadata_with_overrides(
            metadata,
            args,
            zeta="full optimizer",
        )
        self.assertEqual(profile.red_shape_model, "GSA")

    def test_metadata_used_when_cli_default(self) -> None:
        metadata = {
            "profile": {
                "norm": "infinity",
                "red_cost_model": "ADPS16",
                "red_shape_model": "LGSA",
                "zeta": "full optimizer",
            }
        }
        args = argparse.Namespace(
            red_cost_model="adps16",
            red_shape_model="lgsa",
            adps16_mode="classical",
            nearest_neighbor="classical",
        )
        self.assertFalse(_cli_profile_specified(args))
        profile = profile_from_metadata_with_overrides(
            metadata,
            args,
            zeta="full optimizer",
        )
        self.assertEqual(profile.adps16_mode, "classical")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
