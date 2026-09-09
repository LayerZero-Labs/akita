"""Keep the checked-in numerical regressions tied to an independent oracle."""

import sys
import unittest
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parents[1] / "sis_golden"
sys.path.insert(0, str(ORACLE_DIR))

from probability_oracle import fixtures  # noqa: E402


class SisProbabilityOracleTests(unittest.TestCase):
    def test_fixtures_match_independent_decimal_series(self):
        for name, expected in fixtures().items():
            with self.subTest(name=name):
                self.assertEqual((ORACLE_DIR / name).read_text(), expected)


if __name__ == "__main__":
    unittest.main()
