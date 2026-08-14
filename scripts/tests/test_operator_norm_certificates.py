"""Run the bundled exact operator-norm accepted-support certificates."""

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE_ROOT = REPO_ROOT / "scripts" / "operator_norm"


class OperatorNormCertificateTests(unittest.TestCase):
    def run_checker(self, dimension: str, certificate: str) -> None:
        directory = CERTIFICATE_ROOT / dimension
        result = subprocess.run(
            [sys.executable, "check_cert.py", certificate],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("ALL CHECKS PASS", result.stdout)

    def test_d64_certificate(self) -> None:
        self.run_checker("d64", "cert_d64_a31_b11_gamma18.json")

    def test_d128_certificate(self) -> None:
        self.run_checker("d128", "cert_d128_w31_gamma13.json")


if __name__ == "__main__":
    unittest.main()
