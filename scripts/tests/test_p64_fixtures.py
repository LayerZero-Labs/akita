"""Cheap installed-fixture integrity checks; Rust CI separately admits every row."""

import hashlib
from pathlib import Path
import unittest


FIXTURES = Path(__file__).resolve().parents[2] / "crates/akita-pcs/tests/fixtures"
SHAPES = (
    "binary-claim",
    "prime-claim",
    "binary-and-prime-claims",
    "binary-and-prime-claims-plus-ordinary",
)
KEYS = {f"{log}-{shape}" for log in (22, 23) for shape in SHAPES}
DIRECTORIES = ("binary-source-p64", "binary-source-p64-dev")


class P64FixtureTests(unittest.TestCase):
    def test_complete_pin_rosters(self):
        for directory in DIRECTORIES:
            with self.subTest(mode=directory):
                lines = (FIXTURES / directory / "large-pins.txt").read_text().splitlines()
                self.assertEqual(len(lines), len(KEYS))
                seen = set()
                for line in lines:
                    fields = line.split()
                    self.assertEqual(len(fields), 3, line)
                    key, length, digest = fields
                    self.assertNotIn(key, seen, f"duplicate pin: {key}")
                    seen.add(key)
                    self.assertRegex(length, r"^[1-9][0-9]*$")
                    self.assertRegex(digest, r"^[0-9a-f]{64}$")
                self.assertEqual(seen, KEYS)

    def test_installed_sha256_manifest(self):
        expected = {
            f"{directory}/{name}"
            for directory in DIRECTORIES
            for name in [*(f"{key}.json" for key in KEYS), "large-pins.txt"]
        }
        lines = (FIXTURES / "P64-SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(lines), len(expected))
        seen = set()
        for line in lines:
            fields = line.split()
            self.assertEqual(len(fields), 2, line)
            digest, name = fields
            self.assertIn(name, expected)
            self.assertNotIn(name, seen, f"duplicate manifest entry: {name}")
            seen.add(name)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()
