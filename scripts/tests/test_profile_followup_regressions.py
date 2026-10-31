import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import profile_bench_report as report
from scripts.profile_bench_collect import collect, read_shard_summary


class SampleFailureTests(unittest.TestCase):
    def run_cases(self, root, baseline=False, warmups=0, fail_on=2):
        specs = ["onehot_fp32:31:1", "dense_fp32:30:1"]
        args = argparse.Namespace(
            binary="head", baseline_binary="base" if baseline else "",
            output_dir=str(root / "head"),
            baseline_output_dir=str(root / "base") if baseline else "",
            baseline_policy="", case=specs, mode="onehot_fp128", setup_mode="direct",
            runs=2, warmups=warmups, benchmark_shard="test",
        )
        calls = []
        counts = {}

        def run(binary, directory, case, **kwargs):
            key = binary, case.mode
            counts[key] = counts.get(key, 0) + 1
            calls.append(key)
            target = "base" if baseline else "head"
            failed = binary == target and case.mode == "onehot_fp32" and counts[key] == fail_on
            raw = {
                "mode": case.mode, "num_vars": case.num_vars, "num_polys": case.num_polys,
                "exit_code": 101 if failed else 0,
            }
            if case.mode == "onehot_fp32" and (not failed or baseline):
                raw["onehot_k"] = 256 if binary == "base" else 16
            if failed:
                raw.update(error="process failed", failure_phase="startup")
            else:
                raw["commit_s"] = 1.0
            return report.normalize_case_summary(raw), raw["exit_code"]

        with mock.patch.object(report, "run_benchmark_case", side_effect=run):
            code = report.run_benchmark(args)
        head = json.loads((root / "head/summary.json").read_text())["cases"]
        base = json.loads((root / "base/summary.json").read_text())["cases"] if baseline else []
        return code, calls, head, base

    def test_startup_failure_after_success_preserves_neighbor_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, head, _ = self.run_cases(Path(tmp))
        hot, dense = head
        self.assertEqual(code, 101)
        self.assertEqual(len(calls), 4)
        self.assertEqual(hot["exit_code"], 101)
        self.assertEqual(hot["onehot_k"], 16)
        self.assertEqual([s["exit_code"] for s in hot["samples"]], [0, 101])
        self.assertEqual(hot["failure_phase"], "startup")
        self.assertEqual(dense["exit_code"], 0)
        self.assertEqual(dense["runs"], 2)
        self.assertEqual(dense["commit_s"], 1.0)

    def test_cross_density_sibling_failure_only_propagates_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, calls, head, base = self.run_cases(Path(tmp), baseline=True)
        self.assertEqual(code, 101)
        self.assertEqual(len(calls), 8)
        self.assertEqual(head[0]["onehot_k"], 16)
        self.assertEqual(base[0]["onehot_k"], 256)
        self.assertEqual(head[0]["exit_code"], 101)
        self.assertIn("paired binary failed", head[0]["error"])
        self.assertEqual([s["exit_code"] for s in head[0]["samples"]], [0, 0])
        self.assertEqual(head[0]["runs"], 2)
        self.assertEqual(head[1]["exit_code"], 0)
        self.assertEqual(base[1]["exit_code"], 0)

    def test_cancelled_binary_without_measurements_has_unknown_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, head, base = self.run_cases(Path(tmp), baseline=True, warmups=1, fail_on=1)
        self.assertEqual(code, 101)
        self.assertEqual(head[0]["exit_code"], 101)
        self.assertIsNone(head[0].get("onehot_k"))
        self.assertEqual(head[0]["samples"], [])
        self.assertEqual(head[0]["runs"], 0)
        self.assertEqual(base[0]["onehot_k"], 256)
        self.assertEqual(head[1]["exit_code"], 0)

    def test_successful_density_mismatch_is_still_fatal(self):
        samples = [{"exit_code": 0, "onehot_k": k} for k in (16, 256)]
        with self.assertRaisesRegex(ValueError, "successful benchmark samples"):
            report.combine_case_run_summaries(samples)


class EnvelopeValidationTests(unittest.TestCase):
    def malformed_payloads(self):
        case = {"mode": "dense_fp32", "num_vars": 30, "exit_code": 0}
        return [None, [], 3, "not an envelope", {"cases": None}, {"cases": {}},
                {"cases": [None]}, {"cases": [[]]}, {"cases": [3]},
                {"cases": [dict(case, num_vars="30")]},
                {"cases": [dict(case, exit_code=True)]},
                {"cases": [dict(case, num_polys=0)]},
                {"cases": [dict(case, mode=123)]},
                {"cases": [dict(case, commit_s=[])]},
                {"cases": [dict(case, commit_s=float("nan"))]},
                {"cases": [dict(case, commit_s=10 ** 400)]},
                {"cases": [dict(case, schema_version=None)]},
                {"cases": [dict(case, baseline_status="maybe")]},
                {"cases": [dict(case, baseline_status="unsupported")]},
                {"cases": [dict(case, env=[])]},
                {"cases": [dict(case, env=None)]},
                {"cases": [dict(case, warnings={})]},
                {"cases": [dict(case, planned_levels=[None])]},
                {"cases": [], "warmups": "1"},
                {"cases": [], "warmups": False},
                {"cases": [], "warmups": -1},
                {"cases": [], "schema_version": []}]

    def test_malformed_head_envelopes_render_all_cases_as_failures(self):
        for payload in self.malformed_payloads():
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                part = root / "parts/profile-bench-data-1-fp32-base/pr"
                part.mkdir(parents=True)
                (part / "summary.json").write_text(json.dumps(payload))
                collect(root / "parts", root / "head", root / "base",
                        Path(".github/workflows/profile-bench.yml"))
                cases = json.loads((root / "head/summary.json").read_text())["cases"]
                self.assertEqual(len(cases), 13)
                self.assertTrue(all(case["exit_code"] == 1 for case in cases))
                self.assertEqual(cases[0]["failure_phase"], "benchmark orchestration")
                output = io.StringIO()
                args = argparse.Namespace(summary=str(root / "head/summary.json"),
                                          main_baseline_dir="", previous_baseline_dir="", compact=True)
                with contextlib.redirect_stdout(output):
                    self.assertEqual(report.render_report(args), 0)
                self.assertIn("0 of 13 profiles passed", output.getvalue())

    def test_malformed_baseline_envelopes_preserve_unpaired_neighbor(self):
        for payload in self.malformed_payloads():
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                part = root / "parts/profile-bench-data-1-fp32-base"
                (part / "pr").mkdir(parents=True)
                (part / "base").mkdir()
                cases = [
                    {"mode": "dense_fp32", "num_vars": 30, "exit_code": 0,
                     "baseline_status": "available"},
                    {"mode": "onehot_fp32", "num_vars": 31, "exit_code": 0,
                     "baseline_status": "unsupported", "baseline_reason": "missing catalog key",
                     "onehot_k": 16},
                ]
                (part / "pr/summary.json").write_text(json.dumps({"warmups": 2, "cases": cases}))
                (part / "base/summary.json").write_text(json.dumps(payload))
                collect(root / "parts", root / "head", root / "base",
                        Path(".github/workflows/profile-bench.yml"))
                summary = json.loads((root / "head/summary.json").read_text())
                self.assertEqual(summary["warmups"], 2)
                self.assertEqual(summary["cases"][0]["exit_code"], 1)
                self.assertEqual(summary["cases"][0]["failure_phase"], "baseline aggregation")
                self.assertEqual(summary["cases"][1]["exit_code"], 0)

    def test_signal_exit_status_does_not_discard_a_passing_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root / "parts/profile-bench-data-1-fp32-base/pr"
            part.mkdir(parents=True)
            cases = [
                {"mode": "dense_fp32", "num_vars": 30, "exit_code": 0},
                {"mode": "onehot_fp32", "num_vars": 31, "exit_code": -9,
                 "failure_phase": "startup", "error": "process killed"},
            ]
            (part / "summary.json").write_text(json.dumps({"cases": cases}))
            collect(root / "parts", root / "head", root / "base",
                    Path(".github/workflows/profile-bench.yml"))
            combined = json.loads((root / "head/summary.json").read_text())["cases"]
            self.assertEqual(combined[0]["exit_code"], 0)
            self.assertEqual(combined[1]["exit_code"], -9)
            self.assertEqual(combined[1]["failure_phase"], "startup")

    def test_shared_parser_rejects_duplicate_and_unexpected_records(self):
        case = {"mode": "dense_fp32", "num_vars": 30, "exit_code": 0}
        expected = {report.case_id("dense_fp32", 30, 1)}
        for cases in ([case, case], [dict(case, num_vars=31)]):
            with self.subTest(cases=cases), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "summary.json"
                path.write_text(json.dumps({"cases": cases}))
                with self.assertRaises(ValueError):
                    read_shard_summary(path, expected)
