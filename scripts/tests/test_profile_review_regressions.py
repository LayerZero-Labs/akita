import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import profile_bench_merge_base_policy as policy
from scripts import profile_bench_report as report
from scripts.profile_bench_collect import collect


class RuntimeContractTests(unittest.TestCase):
    def snapshot(self, name):
        path = Path("scripts/profile_workload_legacy") / f"{name}.json"
        return json.loads(path.read_text())

    def resolve(self, name, cases):
        rows = self.snapshot(name)["workloads"]
        keys = [("fp32_dense", "final=30:1;precommitted="),
                ("fp64_dense", "final=29:1;precommitted="),
                ("fp128_dense", "final=28:1;precommitted=")]
        # Both frozen catalogs retain grouped keys the corresponding driver
        # rejects. Model membership independently of accepted runtime shapes.
        for nv in ([32, 34] if name == "main" else [31, 34]):
            keys.append(("fp128_onehot", f"final={nv}:2;precommitted=16:1,16:1"))
        catalog = "family\tlogical_key\tlookup_key_digest\n" + "".join(
            f"{family}\t{key}\t{'a' * 64}\n" for family, key in keys
        )
        with (
            mock.patch.object(policy, "load_workloads", return_value=rows),
            mock.patch.object(policy, "merge_base_profile_modes",
                              return_value={row["mode"] for row in rows}),
            mock.patch.object(policy, "git_show", return_value=catalog),
        ):
            return policy.resolve_baseline("fixture", cases)

    def test_retained_catalog_row_is_not_a_runtime_capability(self):
        result = self.resolve("main", ["onehot_fp128_multi_group:32:4",
                                      "onehot_fp128_multi_group:34:4"])
        self.assertEqual([case["supported"] for case in result["cases"]], [False, True])
        self.assertIn("driver does not accept", result["cases"][0]["reason"])
        result = self.resolve("k16", ["onehot_fp128_multi_group:34:4",
                                     "onehot_fp128_multi_group:31:4"])
        self.assertEqual([case["supported"] for case in result["cases"]], [False, True])

    def test_synced_main_keeps_only_the_three_dense_baselines(self):
        cases = [spec for _, _, spec in policy.read_matrix_cases(
            Path(".github/workflows/profile-bench.yml")
        )]
        result = self.resolve("main", cases)
        self.assertEqual(sum(case["supported"] for case in result["cases"]), 3)
        self.assertTrue(all(case["case"].startswith("dense_")
                           for case in result["cases"] if case["supported"]))

    def test_legacy_snapshot_lookup_does_not_need_historical_git_objects(self):
        snapshot = self.snapshot("main")
        revision = "1160b0a8d574f7090bf47adfa026658a5099787a"
        self.assertIn(revision, snapshot["revisions"])
        with mock.patch.object(policy.subprocess, "check_output", side_effect=["", revision]):
            self.assertEqual(policy.load_workloads(revision), snapshot["workloads"])

    def test_setup_mode_is_part_of_runtime_contract(self):
        result = self.resolve("main", ["onehot_fp128_multi_group:34:4:recursive"])
        self.assertFalse(result["cases"][0]["supported"])


class DensityTests(unittest.TestCase):
    def case(self, k):
        return report.normalize_case_summary({
            "mode": "onehot_fp128", "num_vars": 36, "num_polys": 1,
            "workload_label": f"1-of-{k} one-hot", "commit_s": 2.0, "exit_code": 0,
        })

    def test_extract_uses_the_binary_density(self):
        case = report.extract_summary(
            "=== onehot_fp128 (1-of-256) ===", "onehot_fp128", 36, 1
        )
        self.assertEqual(case["onehot_k"], 256)
        self.assertEqual(case["workload_label"], "1-of-256 one-hot")

    def test_normalization_preserves_historical_density(self):
        case = self.case(256)
        self.assertEqual(report.normalize_case_summary(case)["onehot_k"], 256)
        self.assertNotEqual(case["comparison_id"], self.case(16)["comparison_id"])
        self.assertIsNone(report.compatible_baseline(self.case(16), case))

    def test_unknown_density_is_not_comparable(self):
        case = report.normalize_case_summary({"mode": "onehot_fp128", "num_vars": 36})
        self.assertIsNone(case["onehot_k"])
        self.assertIsNone(report.compatible_baseline(case, case))

    def test_report_identifies_both_densities_and_omits_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "head.json").write_text(json.dumps({"cases": [self.case(16)]}))
            (root / "base").mkdir()
            (root / "base/summary.json").write_text(json.dumps({"cases": [self.case(256)]}))
            args = argparse.Namespace(summary=str(root / "head.json"),
                                      main_baseline_dir=str(root / "base"),
                                      previous_baseline_dir="", compact=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(report.render_report(args), 0)
            text = output.getvalue()
            self.assertIn("Incompatible source workloads", text)
            self.assertIn("1\\-of\\-16", text)
            self.assertIn("1\\-of\\-256", text)
            self.assertNotIn("+0.0%", text)


class OrchestrationFailureTests(unittest.TestCase):
    def test_all_missing_shards_still_report_all_thirteen_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect(root / "parts", root / "head", root / "base",
                    Path(".github/workflows/profile-bench.yml"))
            payload = json.loads((root / "head/summary.json").read_text())
            self.assertEqual(len(payload["cases"]), 13)
            self.assertTrue(all(case["exit_code"] == 1 for case in payload["cases"]))
            output = io.StringIO()
            args = argparse.Namespace(summary=str(root / "head/summary.json"),
                                      main_baseline_dir="", previous_baseline_dir="", compact=True)
            with contextlib.redirect_stdout(output):
                self.assertEqual(report.render_report(args), 0)
            self.assertIn("0 of 13 profiles passed", output.getvalue())
            self.assertIn("Failed cases", output.getvalue())

    def test_missing_case_preserves_its_passing_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root / "parts/profile-bench-data-1-fp32-base/pr"
            part.mkdir(parents=True)
            case = {"mode": "dense_fp32", "num_vars": 30, "num_polys": 1, "exit_code": 0}
            (part / "summary.json").write_text(json.dumps({"cases": [case]}))
            collect(root / "parts", root / "head", root / "base",
                    Path(".github/workflows/profile-bench.yml"))
            cases = json.loads((root / "head/summary.json").read_text())["cases"]
            self.assertEqual(len(cases), 13)
            self.assertEqual(sum(case["exit_code"] == 0 for case in cases), 1)

    def test_missing_supported_baseline_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root / "parts/profile-bench-data-1-fp32-base/pr"
            part.mkdir(parents=True)
            case = {"mode": "dense_fp32", "num_vars": 30, "num_polys": 1,
                    "exit_code": 0, "baseline_status": "available"}
            neighbor = {"mode": "onehot_fp32", "num_vars": 31, "num_polys": 1,
                        "exit_code": 0, "onehot_k": 16, "baseline_status": "unsupported",
                        "baseline_reason": "missing catalog key"}
            (part / "summary.json").write_text(json.dumps({"cases": [case, neighbor]}))
            collect(root / "parts", root / "head", root / "base",
                    Path(".github/workflows/profile-bench.yml"))
            cases = json.loads((root / "head/summary.json").read_text())["cases"]
            self.assertEqual(cases[0]["exit_code"], 1)
            self.assertEqual(cases[0]["failure_phase"], "baseline aggregation")
            self.assertEqual(cases[1]["exit_code"], 0)

    def test_failure_handler_preserves_passing_cases_and_fills_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = report.parse_case_spec("dense_fp32:30:1", "onehot_fp128")
            passed = {"case_id": case.case_id, "mode": case.mode, "exit_code": 0}
            (root / "summary.json").write_text(json.dumps({"cases": [passed]}))
            args = argparse.Namespace(output_dir=tmp, only_missing=True,
                                      case=["dense_fp32:30:1", "onehot_fp32:31:1"],
                                      mode="onehot_fp128", setup_mode="direct",
                                      failure_phase="baseline_metadata", error="Git lookup failed",
                                      benchmark_shard="1-fp32-base")
            report.write_failure_summary(args)
            cases = json.loads((root / "summary.json").read_text())["cases"]
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]["exit_code"], 0)
            self.assertEqual(cases[1]["failure_phase"], "baseline_metadata")
