import pathlib
import tempfile
import unittest

from scripts.ci_comment_workflow import (
    PolicyError,
    ResolvedPullRequest,
    artifact_for_run,
    resolve_workflow_run_pr,
    run_matches_pr_identity,
    validate_comment_file,
    validate_pr_head,
)


REPOSITORY = "LayerZero-Labs/akita"
HEAD_REPOSITORY = "external/akita"
HEAD_SHA = "a" * 40


def workflow_run(*, native=None, **overrides):
    payload = {
        "pull_requests": [] if native is None else native,
        "head_repository": {
            "full_name": HEAD_REPOSITORY,
            "owner": {"login": "external"},
        },
        "head_branch": "feature",
        "head_sha": HEAD_SHA,
    }
    payload.update(overrides)
    return payload


def candidate(
    number=459,
    *,
    sha=HEAD_SHA,
    head_repository=HEAD_REPOSITORY,
    base_repository=REPOSITORY,
):
    return {
        "number": number,
        "head": {
            "sha": sha,
            "ref": "feature",
            "repo": {"full_name": head_repository},
        },
        "base": {"repo": {"full_name": base_repository}},
    }


class ResolverTests(unittest.TestCase):
    def test_native_association_does_not_query_fallback(self) -> None:
        queries = []
        resolved = resolve_workflow_run_pr(
            workflow_run(native=[{"number": 459}]),
            REPOSITORY,
            lambda head: queries.append(head) or [],
        )
        self.assertEqual(resolved.number, 459)
        self.assertEqual(resolved.head_sha, HEAD_SHA)
        self.assertEqual(queries, [])

    def test_exact_fork_match(self) -> None:
        resolved = resolve_workflow_run_pr(
            workflow_run(), REPOSITORY, lambda head: [candidate()]
        )
        self.assertEqual(resolved.number, 459)
        self.assertEqual(resolved.head_repository, HEAD_REPOSITORY)

    def test_missing_metadata_fails_closed(self) -> None:
        for key in ("head_repository", "head_branch", "head_sha"):
            with self.subTest(key=key), self.assertRaises(PolicyError):
                resolve_workflow_run_pr(
                    workflow_run(**{key: None}), REPOSITORY, lambda head: [candidate()]
                )

    def test_wrong_sha_repo_or_base_yields_no_match(self) -> None:
        variants = (
            candidate(sha="b" * 40),
            candidate(head_repository="other/akita"),
            candidate(base_repository="other/base"),
        )
        for wrong in variants:
            with self.subTest(candidate=wrong), self.assertRaisesRegex(
                PolicyError, "found 0"
            ):
                resolve_workflow_run_pr(workflow_run(), REPOSITORY, lambda head: [wrong])

    def test_zero_or_multiple_matches_fail_closed(self) -> None:
        for candidates, expected in (([], "found 0"), ([candidate(), candidate(460)], "found 2")):
            with self.subTest(count=len(candidates)), self.assertRaisesRegex(
                PolicyError, expected
            ):
                resolve_workflow_run_pr(
                    workflow_run(), REPOSITORY, lambda head, values=candidates: values
                )

    def test_api_failure_is_not_converted_to_a_match(self) -> None:
        def fail(_head):
            raise RuntimeError("API unavailable")

        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            resolve_workflow_run_pr(workflow_run(), REPOSITORY, fail)


class CommentBoundaryTests(unittest.TestCase):
    MARKER = "<!-- marker -->"

    def write_bytes(self, raw: bytes) -> pathlib.Path:
        temp = tempfile.NamedTemporaryFile(delete=False)
        self.addCleanup(pathlib.Path(temp.name).unlink, missing_ok=True)
        temp.write(raw)
        temp.close()
        return pathlib.Path(temp.name)

    def exact_utf8_body(self, size: int) -> bytes:
        prefix = self.MARKER.encode("utf-8")
        remaining = size - len(prefix)
        multibyte_count = remaining // 2
        ascii_tail = b"x" if remaining % 2 else b""
        return prefix + "é".encode("utf-8") * multibyte_count + ascii_tail

    def test_accepts_exact_utf8_byte_limit(self) -> None:
        path = self.write_bytes(self.exact_utf8_body(60_000))
        body = validate_comment_file(path, self.MARKER, 60_000)
        self.assertTrue(body.startswith(self.MARKER))
        self.assertEqual(len(body.encode("utf-8")), 60_000)

    def test_rejects_one_byte_over_utf8_limit_before_read(self) -> None:
        path = self.write_bytes(self.exact_utf8_body(60_001))
        with self.assertRaisesRegex(PolicyError, "limit is 60000"):
            validate_comment_file(path, self.MARKER, 60_000)

    def test_rejects_invalid_utf8_and_wrong_marker(self) -> None:
        with self.assertRaisesRegex(PolicyError, "valid UTF-8"):
            validate_comment_file(
                self.write_bytes(self.MARKER.encode() + b"\xff"), self.MARKER, 60_000
            )
        with self.assertRaisesRegex(PolicyError, "required report marker"):
            validate_comment_file(self.write_bytes(b"wrong"), self.MARKER, 60_000)


class ArtifactBoundaryTests(unittest.TestCase):
    class Client:
        def __init__(self, artifacts):
            self.artifacts = artifacts

        def list_run_artifacts(self, _owner, _repo, _run_id):
            return self.artifacts

    def artifact(self, *, size=5_000_000, sha=HEAD_SHA):
        return {
            "id": 123,
            "name": "report-data",
            "size_in_bytes": size,
            "expired": False,
            "workflow_run": {"head_sha": sha},
        }

    def test_accepts_exact_archive_limit(self) -> None:
        artifact = artifact_for_run(
            self.Client([self.artifact()]),
            "LayerZero-Labs",
            "akita",
            1,
            "report-data",
            5_000_000,
            HEAD_SHA,
        )
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.id, 123)

    def test_rejects_oversized_or_wrong_head_artifact(self) -> None:
        with self.assertRaisesRegex(PolicyError, "limit is 5000000"):
            artifact_for_run(
                self.Client([self.artifact(size=5_000_001)]),
                "LayerZero-Labs",
                "akita",
                1,
                "report-data",
                5_000_000,
                HEAD_SHA,
            )
        with self.assertRaisesRegex(PolicyError, "expected head"):
            artifact_for_run(
                self.Client([self.artifact(sha="b" * 40)]),
                "LayerZero-Labs",
                "akita",
                1,
                "report-data",
                5_000_000,
                HEAD_SHA,
            )


class IdentityTests(unittest.TestCase):
    EXPECTED = ResolvedPullRequest(
        number=459,
        head_sha=HEAD_SHA,
        head_branch="feature",
        head_repository=HEAD_REPOSITORY,
        base_repository=REPOSITORY,
    )

    def test_final_pr_revalidation_accepts_only_original_open_head(self) -> None:
        validate_pr_head({"state": "open", **candidate()}, self.EXPECTED)
        variants = (
            {"state": "closed", **candidate()},
            {"state": "open", **candidate(sha="b" * 40)},
            {"state": "open", **candidate(head_repository="other/akita")},
            {"state": "open", **candidate(base_repository="other/base")},
        )
        for pull_request in variants:
            with self.subTest(pull_request=pull_request), self.assertRaises(PolicyError):
                validate_pr_head(pull_request, self.EXPECTED)

    def test_previous_run_requires_head_repository_and_pr_when_associated(self) -> None:
        base = {
            "head_repository": {"full_name": HEAD_REPOSITORY},
            "pull_requests": [],
        }
        self.assertTrue(run_matches_pr_identity(base, HEAD_REPOSITORY, 459))
        self.assertTrue(
            run_matches_pr_identity(
                {**base, "pull_requests": [{"number": 459}]}, HEAD_REPOSITORY, 459
            )
        )
        self.assertFalse(
            run_matches_pr_identity(
                {**base, "pull_requests": [{"number": 460}]}, HEAD_REPOSITORY, 459
            )
        )
        self.assertFalse(
            run_matches_pr_identity(
                {**base, "head_repository": {"full_name": "other/akita"}},
                HEAD_REPOSITORY,
                459,
            )
        )


class WorkflowWiringTests(unittest.TestCase):
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
        cls.profile_parent = (repo / ".github/workflows/profile-bench.yml").read_text(
            encoding="utf-8"
        )
        cls.ci_parent = (repo / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    def test_reporters_call_canonical_helper_and_use_least_privilege(self) -> None:
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertIn("scripts/ci_comment_workflow.py resolve-pr", workflow)
                self.assertIn("scripts/ci_comment_workflow.py upsert-comment", workflow)
                self.assertIn("pull-requests: read", workflow)
                self.assertNotIn("pull-requests: write", workflow)
                self.assertNotIn("--head-branch '${{", workflow)
                self.assertNotIn("--head-repository '${{", workflow)

    def test_artifact_metadata_guard_precedes_download(self) -> None:
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertLess(
                    workflow.index("ci_comment_workflow.py resolve-artifact"),
                    workflow.index("actions/download-artifact@"),
                )

    def test_reporters_run_only_after_non_cancelled_pull_request_workflows(self) -> None:
        for name, workflow in self.workflows.items():
            with self.subTest(workflow=name):
                self.assertIn("types: [completed]", workflow)
                self.assertIn("github.event.workflow_run.event == 'pull_request'", workflow)
                self.assertIn("github.event.workflow_run.conclusion != 'cancelled'", workflow)

    def test_profile_parent_packages_structured_data_without_write_scope(self) -> None:
        self.assertIn("main-baseline/summary.json", self.profile_parent)
        self.assertIn("previous-baseline/summary.json", self.profile_parent)
        self.assertIn("run.head_repository?.full_name?.toLowerCase()", self.profile_parent)
        self.assertNotIn("issues: write", self.profile_parent)
        self.assertNotIn("pull-requests: write", self.profile_parent)

    def test_expensive_jobs_have_path_timeout_and_fanout_limits(self) -> None:
        self.assertIn("paths:", self.profile_parent)
        self.assertIn("max-parallel: 2", self.profile_parent)
        self.assertIn("timeout-minutes: 30", self.profile_parent)
        self.assertIn("Expensive test path gate", self.ci_parent)
        self.assertIn("needs.expensive-paths.outputs.run == 'true'", self.ci_parent)
        self.assertIn("ci-approved", self.profile_parent)
        self.assertIn("ci-approved", self.ci_parent)
        self.assertIn("max-parallel: 2", self.ci_parent)
        self.assertIn("timeout-minutes: 30", self.ci_parent)


if __name__ == "__main__":
    unittest.main()
