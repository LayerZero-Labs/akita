import json
import pathlib
import tempfile
import unittest


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


class CiTestTimingReportTests(unittest.TestCase):
    def test_parse_junit_dedup_suffix_and_sort(self) -> None:
        from scripts import ci_test_timing_report as report

        junit_path = FIXTURES_DIR / "sample-non-zk.xml"
        tests = report.parse_junit(junit_path)
        self.assertGreaterEqual(len(tests), 3)
        self.assertGreaterEqual(tests[0].duration_s, tests[-1].duration_s)
        ids = [t.id for t in tests]
        self.assertIn("bin_a::t1", ids)
        self.assertIn("bin_a::t1#2", ids)

    def test_merge_and_render_single_pass(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "out"
            args = type(
                "Args",
                (),
                {
                    "output_dir": str(out_dir),
                    "source_sha": "3735fc4f00000000000000000000000000000000",
                    "source_branch": "quang/ci-test-timing",
                    "workflow_run_id": 123,
                    "passes": ["ci"],
                    "profiles": ["ci"],
                    "junits": [str(FIXTURES_DIR / "sample-non-zk.xml")],
                    "timings": [],
                    "started_ats": ["100"],
                    "finished_ats": ["110"],
                    "exit_codes": ["0"],
                    "passes_sharded": True,
                    "shard_count": 2,
                },
            )()
            report.merge_command(args)
            summary_path = out_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["pass_layout"], "single")
            self.assertEqual(summary["passes"]["ci"]["wall_s"], 10.0)

            render_args = type(
                "Args",
                (),
                {
                    "summary": str(summary_path),
                    "output_dir": str(out_dir),
                    "main_baseline_dir": "",
                    "previous_baseline_dir": "",
                    "compact": True,
                },
            )()
            report.render_command(render_args)
            comment = (out_dir / "comment.md").read_text(encoding="utf-8")
            self.assertIn(report.MARKER, comment)
            self.assertIn("Run summary", comment)
            self.assertNotIn("all-features", comment)
            self.assertNotIn("Critical path", comment)

    def test_merge_accepts_timing_file(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            timing = pathlib.Path(tmp) / "timing.json"
            timing.write_text(
                '{"started_at_epoch":100,"finished_at_epoch":150,"exit_code":0}\n',
                encoding="utf-8",
            )
            out_dir = pathlib.Path(tmp) / "out"
            args = type(
                "Args",
                (),
                {
                    "output_dir": str(out_dir),
                    "source_sha": "abc",
                    "source_branch": "main",
                    "workflow_run_id": 1,
                    "passes": [],
                    "profiles": [],
                    "junits": [str(FIXTURES_DIR / "sample-non-zk.xml")],
                    "timings": [str(timing)],
                    "started_ats": [],
                    "finished_ats": [],
                    "exit_codes": [],
                    "passes_sharded": False,
                    "shard_count": 0,
                },
            )()
            report.merge_command(args)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["passes"]["ci"]["wall_s"], 50.0)

    def test_render_v2_current_against_v1_main_baseline(self) -> None:
        from scripts import ci_test_timing_report as report

        current = {
            "schema_version": 2,
            "pass_layout": "single",
            "pass_order": ["ci"],
            "passes_sharded": True,
            "shard_count": 2,
            "passes": {
                "ci": {
                    "wall_s": 500.0,
                    "exit_code": 0,
                    "test_count": 1,
                    "skipped": 0,
                    "failed": 0,
                    "tests": [
                        {
                            "id": "bin_a::t1",
                            "binary": "bin_a",
                            "test": "t1",
                            "classname": "bin_a",
                            "duration_s": 40.0,
                        }
                    ],
                }
            },
        }
        main = {
            "schema_version": 1,
            "passes_parallel": True,
            "passes": {
                "non-zk": {
                    "wall_s": 600.0,
                    "tests": [
                        {
                            "id": "bin_a::t1",
                            "binary": "bin_a",
                            "test": "t1",
                            "classname": "bin_a",
                            "duration_s": 10.0,
                        }
                    ],
                },
                "all-features": {"wall_s": 200.0, "tests": []},
            },
        }
        comment, _ = report.render_report(current, main, None, compact=False)
        self.assertIn("Baseline layout mismatch", comment)
        self.assertIn("600.0", comment)
        self.assertIn("Regressions vs main", comment)

    def test_prepare_shards_combines_shard_artifacts(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = pathlib.Path(tmp) / "shards"
            shard_dir.mkdir()
            (shard_dir / "junit-shard-1.xml").write_text(
                (FIXTURES_DIR / "sample-non-zk.xml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (shard_dir / "junit-shard-2.xml").write_text(
                (FIXTURES_DIR / "sample-all-features.xml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (shard_dir / "timing-shard-1.json").write_text(
                '{"started_at_epoch":100,"finished_at_epoch":120,"exit_code":0,"shard_index":1,"shard_total":2}\n',
                encoding="utf-8",
            )
            (shard_dir / "timing-shard-2.json").write_text(
                '{"started_at_epoch":110,"finished_at_epoch":150,"exit_code":1,"shard_index":2,"shard_total":2}\n',
                encoding="utf-8",
            )

            merged_junit = pathlib.Path(tmp) / "merged.xml"
            merged_timing = pathlib.Path(tmp) / "merged-timing.json"
            args = type(
                "Args",
                (),
                {
                    "input_dir": str(shard_dir),
                    "junit_glob": "junit-shard-*.xml",
                    "timing_glob": "timing-shard-*.json",
                    "output_junit": str(merged_junit),
                    "output_timing": str(merged_timing),
                    "expected_shard_count": 0,
                },
            )()
            report.prepare_shards_command(args)

            tests = report.parse_junit(merged_junit)
            self.assertGreater(len(tests), 3)
            timing = json.loads(merged_timing.read_text(encoding="utf-8"))
            self.assertEqual(timing["started_at_epoch"], 100)
            self.assertEqual(timing["finished_at_epoch"], 150)
            self.assertEqual(timing["exit_code"], 1)

    def test_read_timing_command(self) -> None:
        from scripts import ci_test_timing_report as report
        import io
        import contextlib

        with tempfile.TemporaryDirectory() as tmp:
            timing = pathlib.Path(tmp) / "timing.json"
            timing.write_text(
                '{"started_at_epoch":100,"finished_at_epoch":150,"exit_code":0}\n',
                encoding="utf-8",
            )
            args = type("Args", (), {"timing": str(timing)})()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                report.read_timing_command(args)
            self.assertEqual(buf.getvalue().strip(), "100 150 0")

    def test_prepare_shards_finds_nested_shard_artifacts(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            nested = pathlib.Path(tmp) / "shards" / "ci-test-pass-shard-1"
            nested.mkdir(parents=True)
            (nested / "junit-shard-1.xml").write_text(
                (FIXTURES_DIR / "sample-non-zk.xml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (nested / "timing-shard-1.json").write_text(
                '{"started_at_epoch":100,"finished_at_epoch":120,"exit_code":0,"shard_index":1,"shard_total":1}\n',
                encoding="utf-8",
            )

            args = type(
                "Args",
                (),
                {
                    "input_dir": str(pathlib.Path(tmp) / "shards"),
                    "junit_glob": "junit-shard-*.xml",
                    "timing_glob": "timing-shard-*.json",
                    "output_junit": str(pathlib.Path(tmp) / "merged.xml"),
                    "output_timing": str(pathlib.Path(tmp) / "merged-timing.json"),
                    "expected_shard_count": 0,
                },
            )()
            status = report.prepare_shards_command(args)
            self.assertEqual(status, 0)

    def test_prepare_shards_requires_expected_shard_count(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = pathlib.Path(tmp) / "shards"
            shard_dir.mkdir()
            (shard_dir / "junit-shard-1.xml").write_text(
                (FIXTURES_DIR / "sample-non-zk.xml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (shard_dir / "timing-shard-1.json").write_text(
                '{"started_at_epoch":100,"finished_at_epoch":120,"exit_code":0,"shard_index":1,"shard_total":2}\n',
                encoding="utf-8",
            )

            args = type(
                "Args",
                (),
                {
                    "input_dir": str(shard_dir),
                    "junit_glob": "junit-shard-*.xml",
                    "timing_glob": "timing-shard-*.json",
                    "output_junit": str(pathlib.Path(tmp) / "merged.xml"),
                    "output_timing": str(pathlib.Path(tmp) / "merged-timing.json"),
                    "expected_shard_count": 0,
                },
            )()
            status = report.prepare_shards_command(args)
            self.assertEqual(status, 1)
            timing = json.loads((pathlib.Path(tmp) / "merged-timing.json").read_text(encoding="utf-8"))
            self.assertTrue(timing.get("missing_shards"))


if __name__ == "__main__":
    unittest.main()
