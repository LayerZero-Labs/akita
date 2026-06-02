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


if __name__ == "__main__":
    unittest.main()

