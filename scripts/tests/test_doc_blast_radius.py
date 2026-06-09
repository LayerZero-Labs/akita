#!/usr/bin/env python3
"""Tests for scripts/doc_blast_radius.py path matching."""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "doc_blast_radius.py"

spec = importlib.util.spec_from_file_location("doc_blast_radius", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class TestPathMatchesGlob(unittest.TestCase):
    def test_double_star_prefix(self) -> None:
        self.assertTrue(
            mod.path_matches_glob(
                "crates/akita-planner/src/lib.rs", "crates/akita-planner/**"
            )
        )
        self.assertFalse(
            mod.path_matches_glob("crates/akita-field/src/lib.rs", "crates/akita-planner/**")
        )

    def test_single_file(self) -> None:
        self.assertTrue(mod.path_matches_glob("Cargo.toml", "Cargo.toml"))
        self.assertFalse(mod.path_matches_glob("README.md", "Cargo.toml"))


class TestLoadMap(unittest.TestCase):
    def test_map_loads(self) -> None:
        regions = mod.load_map()
        self.assertGreater(len(regions), 5)
        self.assertTrue(all("id" in r for r in regions))


if __name__ == "__main__":
    unittest.main()
