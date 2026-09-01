import json
import pathlib
import tempfile
import unittest

from scripts.ci_comment_workflow import (
    PolicyError,
    ResolvedPullRequest,
    artifact_for_run,
    profile_baselines_for_comment,
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
        "base": {
            "ref": "main",
            "sha": "d" * 40,
            "repo": {"full_name": base_repository},
        },
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

    def test_rejects_non_scalar_artifact_numbers_as_policy_input(self) -> None:
        malformed = self.artifact()
        malformed["id"] = []
        malformed["size_in_bytes"] = {}
        with self.assertRaisesRegex(PolicyError, "invalid numeric metadata"):
            artifact_for_run(
                self.Client([malformed]),
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
        base_branch="main",
        base_sha="d" * 40,
    )

    def test_final_pr_revalidation_accepts_only_original_open_head(self) -> None:
        validate_pr_head({"state": "open", **candidate()}, self.EXPECTED)
        variants = (
            {"state": "closed", **candidate()},
            {"state": "open", **candidate(sha="b" * 40)},
            {"state": "open", **candidate(head_repository="other/akita")},
            {"state": "open", **candidate(base_repository="other/base")},
            {
                "state": "open",
                **{**candidate(), "base": {**candidate()["base"], "ref": "other"}},
            },
            {
                "state": "open",
                **{**candidate(), "base": {**candidate()["base"], "sha": "e" * 40}},
            },
        )
        for pull_request in variants:
            with self.subTest(pull_request=pull_request), self.assertRaises(PolicyError):
                validate_pr_head(pull_request, self.EXPECTED)

    def test_previous_run_requires_head_repository_and_pr_when_associated(self) -> None:
        base = {
            "head_repository": {"full_name": HEAD_REPOSITORY},
            "head_branch": "feature",
            "pull_requests": [],
        }
        self.assertTrue(
            run_matches_pr_identity(base, HEAD_REPOSITORY, "feature", 459)
        )
        self.assertTrue(
            run_matches_pr_identity(
                {**base, "pull_requests": [{"number": 459}]},
                HEAD_REPOSITORY,
                "feature",
                459,
            )
        )
        self.assertFalse(
            run_matches_pr_identity(
                {**base, "pull_requests": [{"number": 460}]},
                HEAD_REPOSITORY,
                "feature",
                459,
            )
        )
        self.assertFalse(
            run_matches_pr_identity(
                {**base, "head_repository": {"full_name": "other/akita"}},
                HEAD_REPOSITORY,
                "feature",
                459,
            )
        )
        self.assertFalse(
            run_matches_pr_identity(
                {**base, "head_branch": "other"}, HEAD_REPOSITORY, "feature", 459
            )
        )


class ProfileBaselineIdentityTests(unittest.TestCase):
    MAIN_SHA = "b" * 40
    PREVIOUS_SHA = "c" * 40

    class Client:
        def __init__(
            self,
            *,
            main_sha,
            previous_repository=HEAD_REPOSITORY,
            previous_branch="feature",
            artifact_size=5_000_000,
        ):
            self.main_sha = main_sha
            self.previous_repository = previous_repository
            self.previous_branch = previous_branch
            self.artifact_size = artifact_size

        def compare_commits(self, _owner, _repo, _base_sha, _head_sha):
            return {"merge_base_commit": {"sha": self.main_sha}}

        def get_workflow_run(self, _owner, _repo, run_id):
            if run_id == 200:
                return {
                    "id": 200,
                    "run_number": 20,
                    "workflow_id": 1,
                    "name": "Akita Profile Benchmarks",
                    "event": "pull_request",
                    "head_sha": HEAD_SHA,
                    "head_repository": {"full_name": HEAD_REPOSITORY},
                    "head_branch": "feature",
                    "pull_requests": [{"number": 459}],
                }
            return {
                "id": 100,
                "run_number": 10,
                "workflow_id": 1,
                "name": "Akita Profile Benchmarks",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "head_sha": ProfileBaselineIdentityTests.PREVIOUS_SHA,
                "head_repository": {"full_name": self.previous_repository},
                "head_branch": self.previous_branch,
                "pull_requests": [{"number": 459}],
            }

        def list_run_artifacts(self, _owner, _repo, _run_id):
            return [
                {
                    "id": 123,
                    "name": "profile-bench-data",
                    "size_in_bytes": self.artifact_size,
                    "expired": False,
                    "workflow_run": {
                        "head_sha": ProfileBaselineIdentityTests.PREVIOUS_SHA
                    },
                }
            ]

    def paths(self, metadata):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = pathlib.Path(temp.name)
        metadata_path = root / "baseline-metadata.json"
        main_summary = root / "main.json"
        previous_summary = root / "previous.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        main_summary.write_text("{}", encoding="utf-8")
        previous_summary.write_text("{}", encoding="utf-8")
        return metadata_path, main_summary, previous_summary

    def resolve(self, client, metadata):
        metadata_path, main_summary, previous_summary = self.paths(metadata)
        return profile_baselines_for_comment(
            client,
            "LayerZero-Labs",
            "akita",
            metadata_path=metadata_path,
            main_summary_path=main_summary,
            previous_summary_path=previous_summary,
            current_run_id=200,
            workflow_name="Akita Profile Benchmarks",
            artifact_name="profile-bench-data",
            head_sha=HEAD_SHA,
            head_repository=HEAD_REPOSITORY,
            head_branch="feature",
            pr_number=459,
            base_sha="d" * 40,
            max_input_bytes=2_000_000,
            max_artifact_bytes=5_000_000,
        )

    def metadata(self, **overrides):
        value = {
            "schema_version": 1,
            "main_baseline_sha": self.MAIN_SHA,
            "previous_baseline_sha": self.PREVIOUS_SHA,
            "previous_run_id": "100",
        }
        value.update(overrides)
        return value

    def test_authenticates_both_baseline_commit_identities(self) -> None:
        baselines = self.resolve(self.Client(main_sha=self.MAIN_SHA), self.metadata())
        self.assertEqual(baselines.main_sha, self.MAIN_SHA)
        self.assertEqual(baselines.previous_sha, self.PREVIOUS_SHA)

    def test_rejects_each_bad_baseline_without_dropping_the_other(self) -> None:
        forged_main = self.resolve(
            self.Client(main_sha=self.MAIN_SHA),
            self.metadata(main_baseline_sha="e" * 40),
        )
        self.assertEqual(forged_main.main_sha, "")
        self.assertEqual(forged_main.previous_sha, self.PREVIOUS_SHA)
        self.assertRegex(forged_main.warnings[0], "wrong merge-base")

        wrong_fork = self.resolve(
            self.Client(main_sha=self.MAIN_SHA, previous_repository="other/akita"),
            self.metadata(),
        )
        self.assertEqual(wrong_fork.main_sha, self.MAIN_SHA)
        self.assertEqual(wrong_fork.previous_sha, "")
        self.assertRegex(wrong_fork.warnings[0], "another fork or PR")

        wrong_branch = self.resolve(
            self.Client(main_sha=self.MAIN_SHA, previous_branch="other"),
            self.metadata(),
        )
        self.assertEqual(wrong_branch.main_sha, self.MAIN_SHA)
        self.assertEqual(wrong_branch.previous_sha, "")
        self.assertRegex(wrong_branch.warnings[0], "another fork or PR")

        malformed_run_id = self.resolve(
            self.Client(main_sha=self.MAIN_SHA),
            self.metadata(previous_run_id=[]),
        )
        self.assertEqual(malformed_run_id.main_sha, self.MAIN_SHA)
        self.assertEqual(malformed_run_id.previous_sha, "")
        self.assertTrue(malformed_run_id.warnings)

    def test_previous_archive_uses_archive_not_input_limit(self) -> None:
        baselines = self.resolve(
            self.Client(main_sha=self.MAIN_SHA, artifact_size=5_000_000),
            self.metadata(),
        )
        self.assertEqual(baselines.previous_sha, self.PREVIOUS_SHA)
        oversized = self.resolve(
            self.Client(main_sha=self.MAIN_SHA, artifact_size=5_000_001),
            self.metadata(),
        )
        self.assertEqual(oversized.main_sha, self.MAIN_SHA)
        self.assertEqual(oversized.previous_sha, "")
        self.assertRegex(oversized.warnings[0], "limit is 5000000")


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
        self.assertIn("baseline-metadata.json", self.profile_parent)
        self.assertIn("run.head_repository?.full_name?.toLowerCase()", self.profile_parent)
        self.assertNotIn("issues: write", self.profile_parent)
        self.assertNotIn("pull-requests: write", self.profile_parent)

    def test_profile_reporter_authenticates_and_renders_baseline_identities(self) -> None:
        profile = self.workflows["profile benchmark"]
        self.assertIn("resolve-profile-baselines", profile)
        self.assertIn("steps.baselines.outputs.main-sha", profile)
        self.assertIn("steps.baselines.outputs.previous-sha", profile)
        self.assertIn("- Prior PR run:", profile)
        self.assertIn('--max-input-bytes "$AKITA_REPORT_MAX_INPUT_BYTES"', profile)
        self.assertIn('--max-artifact-bytes "$AKITA_REPORT_MAX_ARTIFACT_BYTES"', profile)
        self.assertIn('main_baseline_dir=""', profile)
        self.assertIn('previous_baseline_dir=""', profile)
        self.assertIn('if [ -n "$AKITA_BENCH_MAIN_BASELINE_SHA" ]', profile)
        self.assertIn('if [ -n "$AKITA_BENCH_PREVIOUS_BASELINE_SHA" ]', profile)
        self.assertLess(
            profile.index("resolve-profile-baselines"),
            profile.index("scripts/profile_bench_report.py render"),
        )

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
