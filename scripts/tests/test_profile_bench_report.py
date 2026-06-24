import json
import pathlib
import tempfile
import unittest


class ProfileBenchReportTests(unittest.TestCase):
    def test_plan_case_runs_orders_warmups_then_measured(self) -> None:
        from scripts.profile_bench_report import BenchmarkCaseSpec, ScheduledRun, plan_case_runs

        case = BenchmarkCaseSpec(mode="onehot_fp128_d64", num_vars=24, num_polys=1)
        summary_dir = pathlib.Path("/tmp/bench-root")
        schedule = plan_case_runs("/bin/profile", summary_dir, case, runs=2, warmups=1)

        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0].kind, "warmup")
        self.assertEqual(schedule[1].kind, "measured")
        self.assertEqual(schedule[2].kind, "measured")
        self.assertEqual(schedule[1].run_index, 1)
        self.assertEqual(schedule[2].run_index, 2)
        self.assertEqual(schedule[0].run_dir, summary_dir / case.case_id / "warmup-1")
        self.assertEqual(schedule[1].run_dir, summary_dir / case.case_id / "run-1")
        self.assertEqual(schedule[2].run_dir, summary_dir / case.case_id / "run-2")

    def test_interleaved_schedule_alternates_binaries(self) -> None:
        from scripts.profile_bench_report import BenchmarkCaseSpec, plan_case_runs

        case = BenchmarkCaseSpec(mode="onehot_fp128_d64", num_vars=24, num_polys=1)
        binaries = [
            ("/bin/pr", pathlib.Path("/tmp/pr")),
            ("/bin/base", pathlib.Path("/tmp/base")),
        ]
        plans = [
            plan_case_runs(binary, summary_dir, case, runs=2, warmups=1)
            for binary, summary_dir in binaries
        ]
        self.assertEqual(len({len(plan) for plan in plans}), 1)
        schedule = [run for slot in zip(*plans) for run in slot]

        self.assertEqual(
            [run.binary for run in schedule],
            [
                "/bin/pr",
                "/bin/base",
                "/bin/pr",
                "/bin/base",
                "/bin/pr",
                "/bin/base",
            ],
        )

    def test_configured_cases_rejects_duplicate_case_ids(self) -> None:
        from scripts.profile_bench_report import configured_cases

        args = type(
            "Args",
            (),
            {
                "case": ["onehot_fp128_d64:24:1", "onehot_fp128_d64:24:1"],
                "mode": "onehot_fp128_d64",
                "num_vars": 24,
                "num_polys": 1,
            },
        )()
        with self.assertRaisesRegex(ValueError, "duplicate benchmark case ids"):
            configured_cases(args)

    def test_write_aggregate_summaries_propagates_sibling_failure(self) -> None:
        from scripts.profile_bench_report import (
            BenchmarkCaseSpec,
            ScheduledRun,
            case_status,
            write_aggregate_summaries,
        )

        case = BenchmarkCaseSpec(mode="onehot_fp128_d64", num_vars=24, num_polys=1)
        pr_dir = pathlib.Path("pr-root")
        base_dir = pathlib.Path("base-root")
        ok_summary = {
            "case_id": case.case_id,
            "exit_code": 0,
            "run_index": 1,
            "setup_s": 1.0,
            "commit_s": 2.0,
            "prove_total_s": 3.0,
            "verify_total_s": 4.0,
            "max_rss_kib": 100,
            "proof_size_bytes": 10,
        }
        failed_summary = {
            "case_id": case.case_id,
            "exit_code": 1,
            "run_index": 1,
            "failure_phase": "prove",
            "error": "boom",
            "setup_s": 1.0,
            "commit_s": 2.0,
            "prove_total_s": 3.0,
            "verify_total_s": 4.0,
            "max_rss_kib": 100,
            "proof_size_bytes": 10,
        }
        results = [
            (
                ScheduledRun(
                    "/bin/pr",
                    pr_dir,
                    pr_dir / case.case_id / "run-1",
                    case,
                    "measured",
                    1,
                ),
                ok_summary,
            ),
            (
                ScheduledRun(
                    "/bin/base",
                    base_dir,
                    base_dir / case.case_id / "run-1",
                    case,
                    "measured",
                    1,
                ),
                failed_summary,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            pr_path = pathlib.Path(tmp) / "pr"
            base_path = pathlib.Path(tmp) / "base"
            remapped = []
            for run, summary in results:
                summary_dir = pr_path if run.summary_dir == pr_dir else base_path
                run_dir = summary_dir / run.run_dir.relative_to(run.summary_dir)
                remapped.append(
                    (
                        ScheduledRun(
                            run.binary, summary_dir, run_dir, run.case, run.kind, run.run_index
                        ),
                        summary,
                    )
                )
            write_aggregate_summaries([pr_path, base_path], [case], remapped, warmups=1)

            pr_summary = json.loads((pr_path / "summary.json").read_text(encoding="utf-8"))
            base_summary = json.loads((base_path / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(pr_summary["cases"]), 1)
            self.assertEqual(len(base_summary["cases"]), 1)
            self.assertEqual(case_status(pr_summary["cases"][0]), "fail")
            self.assertEqual(case_status(base_summary["cases"][0]), "fail")
            self.assertIn("paired binary failed", pr_summary["cases"][0]["error"])


if __name__ == "__main__":
    unittest.main()
