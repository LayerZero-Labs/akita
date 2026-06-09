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
        # Sorted by duration descending.
        self.assertGreaterEqual(tests[0].duration_s, tests[-1].duration_s)
        ids = [t.id for t in tests]
        self.assertIn("bin_a::t1", ids)
        self.assertIn("bin_a::t1#2", ids)

    def test_merge_and_render_smoke(self) -> None:
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
                    "passes": ["non-zk", "all-features"],
                    "junits": [
                        str(FIXTURES_DIR / "sample-non-zk.xml"),
                        str(FIXTURES_DIR / "sample-all-features.xml"),
                    ],
                    "started_ats": ["100", "200"],
                    "finished_ats": ["110", "260"],
                    "exit_codes": ["0", "1"],
                    "passes_parallel": True,
                    "passes_sharded": False,
                    "shard_count": 0,
                },
            )()
            report.merge_command(args)
            summary_path = out_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("passes", summary)
            self.assertIn("non-zk", summary["passes"])
            self.assertEqual(summary["passes"]["non-zk"]["wall_s"], 10.0)

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
            comment_path = out_dir / "comment.md"
            self.assertTrue(comment_path.exists())
            comment = comment_path.read_text(encoding="utf-8")
            self.assertIn(report.MARKER, comment)
            self.assertIn("CI test timing", comment)
            self.assertIn("Critical path", comment)
            self.assertTrue(summary.get("passes_parallel"))

    def test_prepare_pass_combines_shard_artifacts(self) -> None:
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
            report.prepare_pass_command(args)

            tests = report.parse_junit(merged_junit)
            self.assertGreater(len(tests), 3)
            timing = json.loads(merged_timing.read_text(encoding="utf-8"))
            self.assertEqual(timing["started_at_epoch"], 100)
            self.assertEqual(timing["finished_at_epoch"], 150)
            self.assertEqual(timing["exit_code"], 1)

    def test_read_timing_command(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            timing = pathlib.Path(tmp) / "timing.json"
            timing.write_text(
                '{"started_at_epoch":100,"finished_at_epoch":150,"exit_code":0}\n',
                encoding="utf-8",
            )
            args = type("Args", (), {"timing": str(timing)})()
            report.read_timing_command(args)

    def test_prepare_pass_finds_nested_shard_artifacts(self) -> None:
        from scripts import ci_test_timing_report as report

        with tempfile.TemporaryDirectory() as tmp:
            nested = pathlib.Path(tmp) / "shards" / "ci-test-pass-non-zk-shard-1"
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
            status = report.prepare_pass_command(args)
            self.assertEqual(status, 0)

    def test_prepare_pass_requires_expected_shard_count(self) -> None:
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
            status = report.prepare_pass_command(args)
            self.assertEqual(status, 1)
            timing = json.loads((pathlib.Path(tmp) / "merged-timing.json").read_text(encoding="utf-8"))
            self.assertTrue(timing.get("missing_shards"))


if __name__ == "__main__":
    unittest.main()

