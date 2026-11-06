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
    def test_complete_pin_roster(self):
        pins = FIXTURES.parents[3] / "scripts/binary-source-full-size-digests.tsv"
        rows = [
            line.split()
            for line in pins.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        rows = [row for row in rows if row[0].startswith("p64/")]
        self.assertEqual(len(rows), len(KEYS))
        seen = set()
        for fields in rows:
            self.assertEqual(len(fields), 5, fields)
            log, shape, length, digest, algorithm = fields
            self.assertIn(log, ("p64/22", "p64/23"))
            key = f"{log.removeprefix('p64/')}-{shape}"
            self.assertNotIn(key, seen, f"duplicate pin: {key}")
            seen.add(key)
            self.assertEqual(algorithm, "sha3-256")
            self.assertRegex(length, r"^[1-9][0-9]*$")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(seen, KEYS)

    def test_installed_sha256_manifest(self):
        expected = {
            f"{directory}/{name}"
            for directory in DIRECTORIES
            for name in (f"{key}.json" for key in KEYS)
        }
        lines = (FIXTURES / "P64-SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(lines), len(expected))
        seen = set()
        for line in lines:
            fields = line.split()
            self.assertEqual(len(fields), 2, line)
            digest, name = fields
            name = name.removeprefix("./")
            self.assertIn(name, expected)
            self.assertNotIn(name, seen, f"duplicate manifest entry: {name}")
            seen.add(name)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()
