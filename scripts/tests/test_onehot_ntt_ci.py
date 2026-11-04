import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class OnehotNttCiTests(unittest.TestCase):
    def run_comparison(self, *, fallback=False, corrupt=False, missing_verify=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sections = []
            times = [100, 100, 10, 1, 9, 2, 11, 3]
            for trial, elapsed in enumerate(times):
                mode = "ntt-panel5@8x48x0" if trial % 2 else "accumulation"
                marker = "one hot commit schedule"
                if trial % 2 and not fallback:
                    marker = "[onehot_ntt] primes=5 blocks=8 tile=48 binary=false safe_chunk=16384"
                sections.append(f"[experiment] trial={trial} onehot_commit={mode} requested={mode}\n{marker}")
                sections.append(f"[onehot_fp128_k16] commit: {elapsed:.6f}s")
                sections.append("[onehot_fp128_k16] prove: 1.000000s")
                sections.append("[onehot_fp128_k16] verify multi threaded OK: 0.001s")
                if not (missing_verify and trial == 7):
                    sections.append("[onehot_fp128_k16] verify single threaded OK: 0.001s")
                for extension in ["proof", "commitment"]:
                    payload = b"different" if corrupt and trial == 7 else b"identical"
                    (root / f"nv38-{trial}-{mode}.{extension}").write_bytes(payload)
            (root / "nv38.log").write_text("\n".join(sections) + "\n INFO Done\n")
            script = Path(__file__).resolve().parents[2] / "profile/onehot-ntt/analyze.py"
            result = subprocess.run(
                ["python3", str(script), "ci-verify", str(root), "38"],
                capture_output=True, text=True, check=False,
            )
            summary = root / "nv38-summary.json"
            report = root / "ci-report.md"
            return result, json.loads(summary.read_text()) if summary.exists() else None, report.read_text() if report.exists() else None

    def test_reports_medians_and_excludes_both_warmups(self):
        result, summary, report = self.run_comparison()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["accumulation_median_s"], 10)
        self.assertEqual(summary["ntt_median_s"], 2)
        self.assertEqual(summary["speedup"], 5)
        self.assertEqual(sum(t["warmup"] for t in summary["trials"]), 2)
        self.assertIn("80.0%", report)
        self.assertIn("separately from the merge-base table", report)

    def test_rejects_ntt_fallback(self):
        result, summary, report = self.run_comparison(fallback=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("five-prime NTT kernel was not observed", result.stderr)
        self.assertIsNone(report)

    def test_rejects_different_serialized_bytes(self):
        result, summary, report = self.run_comparison(corrupt=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)

    def test_rejects_unverified_trial(self):
        result, summary, report = self.run_comparison(missing_verify=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)
