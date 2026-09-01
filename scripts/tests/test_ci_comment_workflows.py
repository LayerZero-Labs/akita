import pathlib
import unittest


class CiCommentWorkflowPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        cls.workflows = {
            "profile benchmark": (
                repo / ".github/workflows/profile-bench-comment.yml"
            ).read_text(encoding="utf-8"),
            "test timing": (
                repo / ".github/workflows/test-timing-comment.yml"
            ).read_text(encoding="utf-8"),
        }

    def test_cross_fork_fallback_uses_exact_immutable_head_identity(self) -> None:
        required_fragments = (
            "workflowRun.pull_requests || []",
            "workflowRun.head_repository?.owner?.login",
            "workflowRun.head_repository?.full_name",
            "workflowRun.head_branch",
            "workflowRun.head_sha",
            "github.rest.pulls.list",
            "state: 'open'",
            "head: `${headOwner}:${headBranch}`",
            "candidate.head?.sha === headSha",
            "candidate.head?.repo?.full_name?.toLowerCase() === expectedHeadRepository",
            "candidate.base?.repo?.full_name?.toLowerCase() === expectedBaseRepository",
            "prs.length !== 1",
        )
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                for fragment in required_fragments:
                    self.assertIn(fragment, workflow)

    def test_cross_fork_fallback_does_not_gate_on_author_association(self) -> None:
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertNotIn("author_association", workflow)
                self.assertNotIn("authorAssociation", workflow)

    def test_write_token_rejects_unmarked_or_oversized_comment_bodies(self) -> None:
        expected_markers = {
            "profile benchmark": "<!-- akita-profile-bench-report -->",
            "test timing": "<!-- akita-ci-test-timing -->",
        }
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertIn(expected_markers[name], workflow)
                self.assertIn("body.startsWith(marker)", workflow)
                self.assertIn("Buffer.byteLength(body, 'utf8')", workflow)
                self.assertIn("const maxBodyBytes = 60_000", workflow)

    def test_reporters_run_only_after_non_cancelled_pull_request_workflows(self) -> None:
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertIn("types: [completed]", workflow)
                self.assertIn(
                    "github.event.workflow_run.event == 'pull_request'", workflow
                )
                self.assertIn(
                    "github.event.workflow_run.conclusion != 'cancelled'", workflow
                )


if __name__ == "__main__":
    unittest.main()
