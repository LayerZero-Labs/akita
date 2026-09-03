"""Consistency checks for the immutable lattice-estimator golden-source pin."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_estimator_pin import (  # noqa: E402
    PINNED_LATTICE_ESTIMATOR_RECORDED_DATE,
    PINNED_LATTICE_ESTIMATOR_REVISION_KIND,
    PINNED_LATTICE_ESTIMATOR_SHA,
    PINNED_LATTICE_ESTIMATOR_UPSTREAM_PR,
)


class LatticeEstimatorPinTests(unittest.TestCase):
    def test_pin_is_an_immutable_tested_revision_not_a_moving_head(self) -> None:
        self.assertEqual(PINNED_LATTICE_ESTIMATOR_REVISION_KIND, "tested-revision")
        self.assertEqual(PINNED_LATTICE_ESTIMATOR_RECORDED_DATE, "2026-06-27")
        self.assertEqual(
            PINNED_LATTICE_ESTIMATOR_UPSTREAM_PR,
            "https://github.com/malb/lattice-estimator/pull/217",
        )

    def test_all_golden_metadata_uses_the_declared_pin(self) -> None:
        metadata_files = sorted((ROOT / "scripts/sis_golden").glob("*metadata.json"))
        pinned_files = 0
        for path in metadata_files:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            actual = metadata.get("lattice_estimator_sha")
            if actual is None:
                continue
            pinned_files += 1
            self.assertEqual(actual, PINNED_LATTICE_ESTIMATOR_SHA, path)
            expected = metadata.get("expected_lattice_estimator_sha")
            if expected is not None:
                self.assertEqual(expected, PINNED_LATTICE_ESTIMATOR_SHA, path)
        self.assertGreater(pinned_files, 0)


if __name__ == "__main__":
    unittest.main()
