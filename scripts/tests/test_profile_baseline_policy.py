import argparse
import csv
import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import profile_bench_merge_base_policy as policy
from scripts import profile_bench_report as report


class BaselinePolicyTests(unittest.TestCase):
    def catalog(self, keys):
        out = io.StringIO()
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["family", "logical_key", "lookup_key_digest"])
        for family, key in keys:
            writer.writerow([family, key, "a" * 64])
        return out.getvalue()

    def resolve(self, keys, cases, modes=None):
        with mock.patch.object(policy, "merge_base_profile_modes", return_value=modes or {"dense_fp32", "onehot_fp32"}), mock.patch.object(policy, "git_show", return_value=self.catalog(keys)):
            return policy.resolve_baseline("main", cases)

    def test_resized_onehot_does_not_disable_dense(self):
        result = self.resolve(
            [("fp32_dense", "final=30:1;precommitted="), ("fp32_onehot", "final=34:1;precommitted=")],
            ["dense_fp32:30:1", "onehot_fp32:31:1"],
        )
        dense, onehot = result["cases"]
        self.assertTrue(dense["supported"])
        self.assertFalse(onehot["supported"])
        self.assertIn("no row", onehot["reason"])

    def test_future_baseline_automatically_supports_resized_onehot(self):
        result = self.resolve(
            [("fp32_onehot", "final=31:1;precommitted=")], ["onehot_fp32:31:1"]
        )
        self.assertTrue(result["cases"][0]["supported"])

    def test_direct_and_recursive_are_distinct_cases(self):
        result = self.resolve(
            [("fp128_onehot", "final=33:1;precommitted=")],
            ["onehot_fp128:33:1", "onehot_fp128:33:1:recursive"], {"onehot_fp128"},
        )
        self.assertEqual([x["supported"] for x in result["cases"]], [True, False])

    def test_grouped_lookup_checks_precommits_and_setup_mode(self):
        mode = "onehot_fp128_multi_group_recursive"
        result = self.resolve(
            [("fp128_onehot_recursive", "final=31:2;precommitted=16:1,16:1")],
            [f"{mode}:31:4:recursive", f"{mode}:29:4:recursive"], {mode},
        )
        self.assertEqual([x["supported"] for x in result["cases"]], [True, False])

    def test_missing_mode_is_explicitly_unsupported(self):
        result = self.resolve([], ["onehot_fp32:31:1"], {"dense_fp32"})
        self.assertIn("does not define mode", result["cases"][0]["reason"])

    def test_metadata_errors_are_fatal(self):
        with mock.patch.object(policy, "merge_base_profile_modes", return_value={"dense_fp32"}), mock.patch.object(policy, "git_show", return_value="bad header\n"):
            with self.assertRaisesRegex(ValueError, "catalog header"):
                policy.resolve_baseline("main", ["dense_fp32:30:1"])
        with mock.patch.object(policy, "merge_base_profile_modes", side_effect=subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(subprocess.CalledProcessError):
                policy.resolve_baseline("main", ["dense_fp32:30:1"])


class MixedBaselineRunTests(unittest.TestCase):
    def run_cases(self, root, failure=None, baseline_binary="base", all_unsupported=False):
        path = root / "policy.json"
        path.write_text(json.dumps({"cases": [
            {"case": "dense_fp32:30:1", "supported": not all_unsupported,
             "reason": "unsupported catalog key" if all_unsupported else ""},
            {"case": "onehot_fp32:31:1", "supported": False, "reason": "unsupported catalog key"},
        ]}))
        args = argparse.Namespace(
            binary="head", output_dir=str(root / "head"), baseline_binary=baseline_binary,
            baseline_output_dir=str(root / "base") if baseline_binary else "",
            baseline_policy=str(path), case=["dense_fp32:30:1", "onehot_fp32:31:1"],
            mode="onehot_fp128", setup_mode="direct", runs=2, warmups=1, benchmark_shard="test",
        )
        calls = []

        def run(binary, run_dir, case, **kwargs):
            calls.append((binary, case.mode))
            code = 101 if failure == (binary, case.mode) else 0
            return {
                "mode": case.mode, "num_vars": case.num_vars, "num_polys": case.num_polys,
                "setup_contribution_mode": case.setup_mode, "case_id": case.case_id,
                "exit_code": code, "error": "proof failed" if code else "",
            }, code

        with mock.patch.object(report, "run_benchmark_case", side_effect=run):
            status = report.run_benchmark(args)
        return status, calls

    def test_dense_pairs_interleave_and_onehot_has_no_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            status, calls = self.run_cases(root)
            self.assertEqual(status, 0)
            self.assertEqual(calls, [(b, "dense_fp32") for b in ["head", "base"] * 3] + [("head", "onehot_fp32")] * 3)
            head = json.loads((root / "head/summary.json").read_text())["cases"]
            base = json.loads((root / "base/summary.json").read_text())["cases"]
            self.assertEqual(len(base), 1)
            self.assertEqual(base[0]["mode"], "dense_fp32")
            self.assertEqual(head[1]["baseline_status"], "unsupported")
            self.assertEqual(head[1]["baseline_reason"], "unsupported catalog key")

    def test_all_unsupported_cases_run_without_a_baseline_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            status, calls = self.run_cases(root, baseline_binary="", all_unsupported=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(binary == "head" for binary, _ in calls))
            self.assertFalse((root / "base").exists())

    def test_supported_baseline_crash_fails_and_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            status, calls = self.run_cases(root, ("base", "dense_fp32"))
            self.assertEqual(status, 101)
            self.assertIn(("head", "onehot_fp32"), calls)
            head = json.loads((root / "head/summary.json").read_text())["cases"]
            self.assertEqual(head[0]["exit_code"], 101)
            self.assertIn("paired binary failed", head[0]["error"])

    def test_unsupported_baseline_does_not_hide_head_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, _ = self.run_cases(pathlib.Path(tmp), ("head", "onehot_fp32"))
            self.assertEqual(status, 101)

    def test_unsupported_case_cannot_reuse_a_stale_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            case = {"mode": "onehot_fp32", "num_vars": 31, "num_polys": 1,
                    "exit_code": 0, "commit_s": 2.0, "runs": 1,
                    "baseline_status": "unsupported", "baseline_reason": "missing catalog row"}
            (root / "head.json").write_text(json.dumps({"cases": [case]}))
            base = root / "base"
            base.mkdir()
            (base / "summary.json").write_text(json.dumps({"cases": [{**case, "commit_s": 1.0}]}))
            args = argparse.Namespace(summary=str(root / "head.json"), main_baseline_dir=str(base),
                                      previous_baseline_dir="", compact=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(report.render_report(args), 0)
            self.assertIn("comparisons are available for `0` of `1`", output.getvalue())
            self.assertIn("Unsupported merge-base cases", output.getvalue())
            self.assertIn("missing catalog row", output.getvalue())
            self.assertNotIn("+100.0%", output.getvalue())

    def test_missing_baseline_binary_is_fatal_for_supported_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "require --baseline-binary"):
                self.run_cases(pathlib.Path(tmp), baseline_binary="")


if __name__ == "__main__":
    unittest.main()
