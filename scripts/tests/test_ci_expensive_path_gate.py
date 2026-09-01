import subprocess
import unittest

from scripts.ci_expensive_path_gate import (
    GateError,
    author_is_approved,
    changed_paths,
    parse_labels,
    requires_expensive_ci,
)


class ExpensivePathGateTests(unittest.TestCase):
    BASE_SHA = "a" * 40
    HEAD_SHA = "b" * 40

    def test_gate_covers_every_expensive_schedule_input(self) -> None:
        for path in (
            "crates/akita-planner/src/lib.rs",
            "scripts/generate-schedule-tables.sh",
            "specs/evidence/subring-coefficient-packing/base.tsv",
            "specs/evidence/subring-coefficient-packing/head.tsv",
            "specs/evidence/subring-coefficient-packing/comparison.tsv",
        ):
            with self.subTest(path=path):
                self.assertTrue(requires_expensive_ci((path,)))
        self.assertFalse(requires_expensive_ci(("docs/ci-test-timing.md",)))

    def test_git_diff_failure_is_a_gate_error(self) -> None:
        def fail(*_args, **_kwargs):
            return subprocess.CompletedProcess(
                args=["git", "diff"],
                returncode=128,
                stdout="",
                stderr="fatal: bad revision",
            )

        with self.assertRaisesRegex(GateError, "fatal: bad revision"):
            changed_paths(self.BASE_SHA, self.HEAD_SHA, run=fail)

    def test_changed_paths_use_the_exact_three_dot_range(self) -> None:
        calls = []

        def succeed(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="docs/readme.md\ncrates/akita-pcs/src/lib.rs\n",
                stderr="",
            )

        paths = changed_paths(self.BASE_SHA, self.HEAD_SHA, run=succeed)
        self.assertEqual(paths, ("docs/readme.md", "crates/akita-pcs/src/lib.rs"))
        self.assertEqual(
            calls[0][0],
            ["git", "diff", "--name-only", f"{self.BASE_SHA}...{self.HEAD_SHA}"],
        )
        self.assertFalse(calls[0][1]["check"])

    def test_author_approval_is_independent_from_path_selection(self) -> None:
        self.assertTrue(author_is_approved("MEMBER", "base/repo", "fork/repo", []))
        self.assertTrue(
            author_is_approved("NONE", "base/repo", "base/repo", [])
        )
        self.assertTrue(
            author_is_approved("NONE", "base/repo", "fork/repo", ["ci-approved"])
        )
        self.assertFalse(author_is_approved("NONE", "base/repo", "fork/repo", []))

    def test_malformed_labels_fail_closed(self) -> None:
        self.assertEqual(parse_labels('["ci-approved"]'), ["ci-approved"])
        for raw in ("not json", "{}", '["ci-approved", 7]'):
            with self.subTest(raw=raw), self.assertRaises(GateError):
                parse_labels(raw)


if __name__ == "__main__":
    unittest.main()
