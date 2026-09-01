#!/usr/bin/env python3
"""Trusted helpers for workflow_run PR comment reporters.

The pull-request workflows that produce artifacts are untrusted.  The two
workflow_run reporters checkout this module from the default branch and use it
to resolve the destination PR, bound artifact and file sizes, revalidate the PR
head immediately before writing, and upsert the final comment.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable


DEFAULT_ARTIFACT_LIMIT = 5_000_000
DEFAULT_BODY_LIMIT = 60_000
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")


class PolicyError(ValueError):
    """An untrusted input failed a fail-closed reporter policy."""


@dataclass(frozen=True)
class ResolvedPullRequest:
    number: int
    head_sha: str
    head_branch: str
    head_repository: str
    base_repository: str
    base_branch: str = ""
    base_sha: str = ""


@dataclass(frozen=True)
class Artifact:
    id: int
    size_in_bytes: int


@dataclass(frozen=True)
class ProfileBaselines:
    main_sha: str
    previous_sha: str
    warnings: tuple[str, ...] = ()


def repository_parts(repository: str) -> tuple[str, str]:
    if REPOSITORY_RE.fullmatch(repository) is None:
        raise PolicyError(f"invalid repository identity: {repository!r}")
    parts = repository.split("/", 1)
    return parts[0], parts[1]


def _repository_name(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("full_name") or "")


def _head_owner(workflow_run: dict[str, object]) -> str:
    repository = workflow_run.get("head_repository")
    if not isinstance(repository, dict):
        return ""
    owner = repository.get("owner")
    if not isinstance(owner, dict):
        return ""
    return str(owner.get("login") or "")


def resolve_workflow_run_pr(
    workflow_run: dict[str, object],
    repository: str,
    list_open_pulls: Callable[[str], list[dict[str, object]]],
) -> ResolvedPullRequest:
    """Resolve one workflow run to one exact open PR.

    GitHub's native association is preferred.  When it is empty for a fork,
    the fallback query is filtered by immutable head SHA plus exact head/base
    repository identity.  Ambiguous or incomplete input is rejected.
    """

    native = workflow_run.get("pull_requests") or []
    if not isinstance(native, list):
        raise PolicyError("workflow_run.pull_requests is not a list")
    if len(native) > 1:
        raise PolicyError(f"expected at most one native PR association; found {len(native)}")

    repository_parts(repository)
    head_repository = _repository_name(workflow_run.get("head_repository"))
    head_owner = _head_owner(workflow_run)
    head_branch = str(workflow_run.get("head_branch") or "")
    head_sha = str(workflow_run.get("head_sha") or "")
    if not all((head_owner, head_repository, head_branch, head_sha)):
        raise PolicyError("workflow run is missing immutable head identity metadata")
    repository_parts(head_repository)
    if head_repository.split("/", 1)[0].lower() != head_owner.lower():
        raise PolicyError("workflow run head owner and repository disagree")
    if "\n" in head_branch or "\r" in head_branch:
        raise PolicyError("workflow run head branch contains a line break")
    if SHA_RE.fullmatch(head_sha) is None:
        raise PolicyError("workflow run head SHA is not a full hexadecimal commit ID")

    if len(native) == 1:
        raw_number = native[0].get("number") if isinstance(native[0], dict) else None
        try:
            number = int(raw_number)
        except (TypeError, ValueError) as error:
            raise PolicyError("native PR association has no valid number") from error
    else:
        candidates = list_open_pulls(f"{head_owner}:{head_branch}")
        expected_head = head_repository.lower()
        expected_base = repository.lower()
        exact = []
        for candidate in candidates:
            head = candidate.get("head")
            base = candidate.get("base")
            if not isinstance(head, dict) or not isinstance(base, dict):
                continue
            if str(head.get("sha") or "") != head_sha:
                continue
            if _repository_name(head.get("repo")).lower() != expected_head:
                continue
            if _repository_name(base.get("repo")).lower() != expected_base:
                continue
            exact.append(candidate)
        if len(exact) != 1:
            raise PolicyError(
                "expected exactly one open PR for "
                f"{head_repository}:{head_branch}@{head_sha}; found {len(exact)}"
            )
        try:
            number = int(exact[0].get("number"))
        except (TypeError, ValueError) as error:
            raise PolicyError("resolved PR has no valid number") from error
    if number <= 0:
        raise PolicyError("resolved PR number must be positive")

    return ResolvedPullRequest(
        number=number,
        head_sha=head_sha,
        head_branch=head_branch,
        head_repository=head_repository,
        base_repository=repository,
    )


def validate_comment_file(path: pathlib.Path, marker: str, max_bytes: int) -> str:
    """Validate a rendered comment without reading a known-oversized file."""

    if not marker:
        raise PolicyError("comment marker is empty")
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise PolicyError(f"comment file does not exist: {path}") from error
    if size > max_bytes:
        raise PolicyError(f"comment file is {size} bytes; limit is {max_bytes}")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise PolicyError(f"comment grew beyond the {max_bytes}-byte limit while reading")
    try:
        body = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PolicyError("comment is not valid UTF-8") from error
    if not body.startswith(marker):
        raise PolicyError("comment is missing the required report marker")
    return body


def validate_input_files(paths: Iterable[pathlib.Path], max_bytes: int) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise PolicyError(f"report input is not a regular file: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise PolicyError(f"report input {path} is {size} bytes; limit is {max_bytes}")


def validate_pr_head(
    pull_request: dict[str, object], expected: ResolvedPullRequest
) -> None:
    if str(pull_request.get("state") or "") != "open":
        raise PolicyError(f"PR #{expected.number} is no longer open")
    head = pull_request.get("head")
    base = pull_request.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise PolicyError(f"PR #{expected.number} is missing head/base identity")
    if str(head.get("sha") or "") != expected.head_sha:
        raise PolicyError(f"PR #{expected.number} head moved after workflow resolution")
    if str(head.get("ref") or "") != expected.head_branch:
        raise PolicyError(f"PR #{expected.number} head branch changed")
    if _repository_name(head.get("repo")).lower() != expected.head_repository.lower():
        raise PolicyError(f"PR #{expected.number} head repository changed")
    if _repository_name(base.get("repo")).lower() != expected.base_repository.lower():
        raise PolicyError(f"PR #{expected.number} base repository changed")
    if expected.base_branch and str(base.get("ref") or "") != expected.base_branch:
        raise PolicyError(f"PR #{expected.number} base branch changed")
    if expected.base_sha and str(base.get("sha") or "") != expected.base_sha:
        raise PolicyError(f"PR #{expected.number} base commit changed")


def run_matches_pr_identity(
    run: dict[str, object], head_repository: str, head_branch: str, pr_number: int
) -> bool:
    if _repository_name(run.get("head_repository")).lower() != head_repository.lower():
        return False
    if str(run.get("head_branch") or "") != head_branch:
        return False
    associations = run.get("pull_requests") or []
    if not isinstance(associations, list):
        return False
    if associations:
        return any(
            isinstance(candidate, dict) and candidate.get("number") == pr_number
            for candidate in associations
        )
    return True


def profile_baselines_for_comment(
    client: "GitHubApi",
    owner: str,
    repo: str,
    *,
    metadata_path: pathlib.Path,
    main_summary_path: pathlib.Path,
    previous_summary_path: pathlib.Path,
    current_run_id: int,
    workflow_name: str,
    artifact_name: str,
    head_sha: str,
    head_repository: str,
    head_branch: str,
    pr_number: int,
    base_sha: str,
    max_input_bytes: int,
    max_artifact_bytes: int,
) -> ProfileBaselines:
    """Authenticate fork-produced baseline identity metadata before rendering."""

    validate_input_files((metadata_path,), max_input_bytes)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PolicyError("benchmark baseline metadata is missing or invalid") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise PolicyError("benchmark baseline metadata has an unsupported schema")
    if SHA_RE.fullmatch(head_sha) is None or SHA_RE.fullmatch(base_sha) is None:
        raise PolicyError("benchmark baseline resolution requires full commit IDs")

    warnings: list[str] = []
    main_sha = ""
    if main_summary_path.exists():
        try:
            validate_input_files((main_summary_path,), max_input_bytes)
            claimed_main_sha = str(metadata.get("main_baseline_sha") or "")
            comparison = client.compare_commits(owner, repo, base_sha, head_sha)
            merge_base = comparison.get("merge_base_commit")
            trusted_main_sha = ""
            if isinstance(merge_base, dict):
                trusted_main_sha = str(merge_base.get("sha") or "")
            if SHA_RE.fullmatch(trusted_main_sha) is None:
                raise PolicyError(
                    "GitHub comparison did not return a full merge-base commit ID"
                )
            if claimed_main_sha != trusted_main_sha:
                raise PolicyError("fork artifact claimed the wrong merge-base commit")
            main_sha = trusted_main_sha
        except (PolicyError, RuntimeError, OSError, ValueError) as error:
            warnings.append(f"merge-base benchmark rejected: {error}")

    previous_sha = ""
    if previous_summary_path.exists():
        try:
            validate_input_files((previous_summary_path,), max_input_bytes)
            previous_run_id = int(metadata.get("previous_run_id") or 0)
            claimed_previous_sha = str(metadata.get("previous_baseline_sha") or "")
            if previous_run_id <= 0 or SHA_RE.fullmatch(claimed_previous_sha) is None:
                raise PolicyError("previous benchmark identity is incomplete")
            if claimed_previous_sha == head_sha:
                raise PolicyError("previous benchmark commit equals the current PR head")

            current_run = client.get_workflow_run(owner, repo, current_run_id)
            previous_run = client.get_workflow_run(owner, repo, previous_run_id)
            current_number = int(current_run.get("run_number") or 0)
            previous_number = int(previous_run.get("run_number") or 0)
            current_workflow_id = int(current_run.get("workflow_id") or 0)
            previous_workflow_id = int(previous_run.get("workflow_id") or 0)
            if (
                current_run.get("name") != workflow_name
                or current_run.get("event") != "pull_request"
                or str(current_run.get("head_sha") or "") != head_sha
                or not run_matches_pr_identity(
                    current_run, head_repository, head_branch, pr_number
                )
            ):
                raise PolicyError("current benchmark run does not match the resolved PR")
            if (
                current_number <= 0
                or previous_number <= 0
                or previous_number >= current_number
                or current_workflow_id <= 0
                or previous_workflow_id != current_workflow_id
            ):
                raise PolicyError("benchmark baseline is not an earlier workflow run")
            if previous_run.get("name") != workflow_name:
                raise PolicyError("previous benchmark run belongs to another workflow")
            if previous_run.get("event") != "pull_request":
                raise PolicyError(
                    "previous benchmark run was not triggered by a pull request"
                )
            if (
                previous_run.get("status") != "completed"
                or previous_run.get("conclusion") not in {"success", "failure"}
            ):
                raise PolicyError(
                    "previous benchmark run is not a completed reportable run"
                )
            if str(previous_run.get("head_sha") or "") != claimed_previous_sha:
                raise PolicyError("fork artifact claimed the wrong previous-run commit")
            if not run_matches_pr_identity(
                previous_run, head_repository, head_branch, pr_number
            ):
                raise PolicyError("previous benchmark run belongs to another fork or PR")
            if artifact_for_run(
                client,
                owner,
                repo,
                previous_run_id,
                artifact_name,
                max_artifact_bytes,
                claimed_previous_sha,
            ) is None:
                raise PolicyError(
                    "previous benchmark run has no matching bounded artifact"
                )
            previous_sha = claimed_previous_sha
        except (PolicyError, RuntimeError, OSError, TypeError, ValueError) as error:
            warnings.append(f"previous PR benchmark rejected: {error}")

    return ProfileBaselines(
        main_sha=main_sha,
        previous_sha=previous_sha,
        warnings=tuple(warnings),
    )


class GitHubApi:
    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            raise PolicyError("GITHUB_TOKEN is empty")
        self.token = token
        self.api_url = api_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, object] | None = None,
        payload: dict[str, object] | None = None,
    ) -> object:
        url = f"{self.api_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "akita-ci-comment-reporter",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {error.code} {detail}") from error
        return json.loads(raw) if raw else {}

    def paginate(
        self, path: str, *, query: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for page in range(1, 11):
            page_query = dict(query or {})
            page_query.update({"per_page": 100, "page": page})
            payload = self.request("GET", path, query=page_query)
            if not isinstance(payload, list):
                raise RuntimeError(f"GitHub API {path} did not return a list")
            results.extend(candidate for candidate in payload if isinstance(candidate, dict))
            if len(payload) < 100:
                return results
        raise RuntimeError(f"GitHub API pagination exceeded 1000 items for {path}")

    def list_open_pulls(self, owner: str, repo: str, head: str) -> list[dict[str, object]]:
        return self.paginate(
            f"/repos/{owner}/{repo}/pulls",
            query={"state": "open", "head": head},
        )

    def get_pull(self, owner: str, repo: str, number: int) -> dict[str, object]:
        result = self.request("GET", f"/repos/{owner}/{repo}/pulls/{number}")
        if not isinstance(result, dict):
            raise RuntimeError("GitHub pull request response was not an object")
        return result

    def list_issue_comments(
        self, owner: str, repo: str, number: int
    ) -> list[dict[str, object]]:
        return self.paginate(f"/repos/{owner}/{repo}/issues/{number}/comments")

    def create_comment(self, owner: str, repo: str, number: int, body: str) -> dict[str, object]:
        result = self.request(
            "POST", f"/repos/{owner}/{repo}/issues/{number}/comments", payload={"body": body}
        )
        return result if isinstance(result, dict) else {}

    def update_comment(self, owner: str, repo: str, comment_id: int, body: str) -> dict[str, object]:
        result = self.request(
            "PATCH", f"/repos/{owner}/{repo}/issues/comments/{comment_id}", payload={"body": body}
        )
        return result if isinstance(result, dict) else {}

    def list_run_artifacts(
        self, owner: str, repo: str, run_id: int
    ) -> list[dict[str, object]]:
        result = self.request(
            "GET",
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts",
            query={"per_page": 100},
        )
        if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list):
            raise RuntimeError("GitHub artifact response was not an artifact list")
        return [value for value in result["artifacts"] if isinstance(value, dict)]

    def list_workflow_runs(
        self,
        owner: str,
        repo: str,
        *,
        event: str,
        branch: str,
        status: str = "completed",
    ) -> list[dict[str, object]]:
        result = self.request(
            "GET",
            f"/repos/{owner}/{repo}/actions/runs",
            query={
                "event": event,
                "branch": branch,
                "status": status,
                "per_page": 100,
            },
        )
        if not isinstance(result, dict) or not isinstance(result.get("workflow_runs"), list):
            raise RuntimeError("GitHub workflow-runs response was not a run list")
        return [value for value in result["workflow_runs"] if isinstance(value, dict)]

    def get_workflow_run(
        self, owner: str, repo: str, run_id: int
    ) -> dict[str, object]:
        result = self.request("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}")
        if not isinstance(result, dict):
            raise RuntimeError("GitHub workflow-run response was not an object")
        return result

    def compare_commits(
        self, owner: str, repo: str, base_sha: str, head_sha: str
    ) -> dict[str, object]:
        if SHA_RE.fullmatch(base_sha) is None or SHA_RE.fullmatch(head_sha) is None:
            raise PolicyError("commit comparison requires full commit IDs")
        result = self.request(
            "GET", f"/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
        )
        if not isinstance(result, dict):
            raise RuntimeError("GitHub comparison response was not an object")
        return result


def artifact_for_run(
    client: GitHubApi,
    owner: str,
    repo: str,
    run_id: int,
    name: str,
    max_bytes: int,
    expected_head_sha: str = "",
) -> Artifact | None:
    matches = [
        candidate
        for candidate in client.list_run_artifacts(owner, repo, run_id)
        if candidate.get("name") == name and not candidate.get("expired")
    ]
    if len(matches) != 1:
        return None
    candidate = matches[0]
    workflow_run = candidate.get("workflow_run")
    if expected_head_sha and (
        not isinstance(workflow_run, dict)
        or str(workflow_run.get("head_sha") or "") != expected_head_sha
    ):
        raise PolicyError(f"artifact {name!r} does not belong to expected head {expected_head_sha}")
    try:
        size = int(candidate.get("size_in_bytes") or 0)
        artifact_id = int(candidate["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise PolicyError(f"artifact {name!r} has invalid numeric metadata") from error
    if size < 0 or size > max_bytes:
        raise PolicyError(f"artifact {name!r} is {size} bytes; limit is {max_bytes}")
    if artifact_id <= 0:
        raise PolicyError(f"artifact {name!r} has a non-positive ID")
    return Artifact(id=artifact_id, size_in_bytes=size)


def _write_outputs(values: dict[str, object]) -> None:
    output_path = pathlib.Path(os.environ["GITHUB_OUTPUT"])
    with output_path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def _warning(message: str) -> None:
    print(f"::warning::{message}")


def _event_workflow_run() -> dict[str, object]:
    payload = json.loads(pathlib.Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    workflow_run = payload.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise PolicyError("event payload has no workflow_run object")
    return workflow_run


def _github_client() -> GitHubApi:
    return GitHubApi(
        os.environ.get("GITHUB_TOKEN", ""),
        os.environ.get("GITHUB_API_URL") or "https://api.github.com",
    )


def resolve_pr_command(_args: argparse.Namespace) -> int:
    empty = {
        "issue-number": "",
        "head-sha": "",
        "head-branch": "",
        "head-repository": "",
        "base-repository": "",
        "base-branch": "",
        "base-sha": "",
    }
    try:
        repository = os.environ["GITHUB_REPOSITORY"]
        owner, repo = repository_parts(repository)
        client = _github_client()
        resolved = resolve_workflow_run_pr(
            _event_workflow_run(),
            repository,
            lambda head: client.list_open_pulls(owner, repo, head),
        )
        pull_request = client.get_pull(owner, repo, resolved.number)
        validate_pr_head(pull_request, resolved)
        base = pull_request.get("base")
        base_branch = str(base.get("ref") or "") if isinstance(base, dict) else ""
        base_sha = str(base.get("sha") or "") if isinstance(base, dict) else ""
        if not base_branch or "\n" in base_branch or "\r" in base_branch:
            raise PolicyError("resolved PR base branch is invalid")
        if SHA_RE.fullmatch(base_sha) is None:
            raise PolicyError("resolved PR base commit is not a full commit ID")
        resolved = ResolvedPullRequest(
            number=resolved.number,
            head_sha=resolved.head_sha,
            head_branch=resolved.head_branch,
            head_repository=resolved.head_repository,
            base_repository=resolved.base_repository,
            base_branch=base_branch,
            base_sha=base_sha,
        )
    except (KeyError, PolicyError, RuntimeError, OSError, json.JSONDecodeError) as error:
        _warning(f"Could not resolve workflow run to one PR: {error}")
        _write_outputs(empty)
        return 0
    _write_outputs(
        {
            "issue-number": resolved.number,
            "head-sha": resolved.head_sha,
            "head-branch": resolved.head_branch,
            "head-repository": resolved.head_repository,
            "base-repository": resolved.base_repository,
            "base-branch": resolved.base_branch,
            "base-sha": resolved.base_sha,
        }
    )
    print(f"Resolved workflow run to PR #{resolved.number} at {resolved.head_sha}.")
    return 0


def resolve_artifact_command(args: argparse.Namespace) -> int:
    _write_outputs({"artifact-id": ""})
    try:
        owner, repo = repository_parts(os.environ["GITHUB_REPOSITORY"])
        client = _github_client()
        artifact = artifact_for_run(
            client,
            owner,
            repo,
            args.run_id,
            args.name,
            args.max_bytes,
            args.expected_head_sha,
        )
        if artifact is None:
            raise PolicyError(f"expected exactly one unexpired {args.name!r} artifact")
    except (KeyError, PolicyError, RuntimeError, OSError, TypeError, ValueError) as error:
        _warning(f"Artifact rejected before download: {error}")
        return 0
    _write_outputs({"artifact-id": artifact.id})
    return 0


def select_timing_baselines_command(args: argparse.Namespace) -> int:
    outputs = {
        "previous-artifact-id": "",
        "previous-run-id": "",
        "previous-sha": "",
        "previous-label": "",
        "main-artifact-id": "",
        "main-run-id": "",
        "main-sha": "",
        "main-label": "",
    }
    try:
        owner, repo = repository_parts(os.environ["GITHUB_REPOSITORY"])
        client = _github_client()

        def first_artifact(
            runs: list[dict[str, object]],
            allowed_conclusions: set[str],
            *,
            expected_head_repository: str = "",
            expected_pr_number: int = 0,
        ) -> tuple[dict[str, object], Artifact] | None:
            for run in runs:
                if int(run.get("id") or 0) == args.current_run_id:
                    continue
                if run.get("name") != args.workflow_name:
                    continue
                if run.get("conclusion") not in allowed_conclusions:
                    continue
                if expected_head_repository and not run_matches_pr_identity(
                    run,
                    expected_head_repository,
                    args.head_branch,
                    expected_pr_number,
                ):
                    continue
                artifact = artifact_for_run(
                    client,
                    owner,
                    repo,
                    int(run["id"]),
                    args.artifact_name,
                    args.max_bytes,
                    str(run.get("head_sha") or ""),
                )
                if artifact is not None:
                    return run, artifact
            return None

        previous_runs = client.list_workflow_runs(
            owner, repo, event="pull_request", branch=args.head_branch
        )
        previous = first_artifact(
            previous_runs,
            {"success", "failure"},
            expected_head_repository=args.head_repository,
            expected_pr_number=args.pr_number,
        )
        if previous:
            run, artifact = previous
            outputs.update(
                {
                    "previous-artifact-id": artifact.id,
                    "previous-run-id": int(run["id"]),
                    "previous-sha": str(run.get("head_sha") or ""),
                    "previous-label": "the previous update of this fork PR with a timing artifact",
                }
            )

        main_runs = client.list_workflow_runs(owner, repo, event="push", branch=args.base_ref)
        main = first_artifact(main_runs, {"success"})
        if main:
            run, artifact = main
            outputs.update(
                {
                    "main-artifact-id": artifact.id,
                    "main-run-id": int(run["id"]),
                    "main-sha": str(run.get("head_sha") or ""),
                    "main-label": f"the latest successful `{args.base_ref}` run",
                }
            )
    except (KeyError, PolicyError, RuntimeError, OSError, TypeError, ValueError) as error:
        _warning(f"Could not determine bounded timing baselines: {error}")
    _write_outputs(outputs)
    return 0


def check_files_command(args: argparse.Namespace) -> int:
    try:
        validate_input_files((pathlib.Path(raw) for raw in args.paths), args.max_bytes)
    except PolicyError as error:
        _warning(str(error))
        return 1
    return 0


def resolve_profile_baselines_command(args: argparse.Namespace) -> int:
    outputs = {"main-sha": "", "previous-sha": ""}
    try:
        owner, repo = repository_parts(os.environ["GITHUB_REPOSITORY"])
        baselines = profile_baselines_for_comment(
            _github_client(),
            owner,
            repo,
            metadata_path=pathlib.Path(args.metadata),
            main_summary_path=pathlib.Path(args.main_summary),
            previous_summary_path=pathlib.Path(args.previous_summary),
            current_run_id=args.current_run_id,
            workflow_name=args.workflow_name,
            artifact_name=args.artifact_name,
            head_sha=args.head_sha,
            head_repository=args.head_repository,
            head_branch=args.head_branch,
            pr_number=args.pr_number,
            base_sha=args.base_sha,
            max_input_bytes=args.max_input_bytes,
            max_artifact_bytes=args.max_artifact_bytes,
        )
        outputs = {
            "main-sha": baselines.main_sha,
            "previous-sha": baselines.previous_sha,
        }
        for warning in baselines.warnings:
            _warning(warning)
    except (
        KeyError,
        PolicyError,
        RuntimeError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        _warning(f"Benchmark baseline identities rejected: {error}")
    _write_outputs(outputs)
    return 0


def upsert_comment_command(args: argparse.Namespace) -> int:
    expected = ResolvedPullRequest(
        number=args.pr_number,
        head_sha=args.head_sha,
        head_branch=args.head_branch,
        head_repository=args.head_repository,
        base_repository=args.base_repository,
        base_branch=args.base_branch,
        base_sha=args.base_sha,
    )
    try:
        body = validate_comment_file(pathlib.Path(args.comment), args.marker, args.max_bytes)
        owner, repo = repository_parts(os.environ["GITHUB_REPOSITORY"])
        client = _github_client()
        validate_pr_head(client.get_pull(owner, repo, args.pr_number), expected)
        comments = client.list_issue_comments(owner, repo, args.pr_number)
        existing = next(
            (
                comment
                for comment in comments
                if isinstance(comment.get("user"), dict)
                and comment["user"].get("login") == "github-actions[bot]"
                and args.marker in str(comment.get("body") or "")
            ),
            None,
        )
        if existing is not None:
            response = client.update_comment(owner, repo, int(existing["id"]), body)
            action = "Updated"
        else:
            response = client.create_comment(owner, repo, args.pr_number, body)
            action = "Created"
        print(f"{action} PR comment: {response.get('html_url', '(URL unavailable)')}")
    except PolicyError as error:
        _warning(f"Comment write skipped: {error}")
        return 0
    except (KeyError, RuntimeError, OSError, TypeError, ValueError) as error:
        _warning(f"Comment write failed: {error}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("resolve-pr")

    artifact = subparsers.add_parser("resolve-artifact")
    artifact.add_argument("--run-id", type=int, required=True)
    artifact.add_argument("--name", required=True)
    artifact.add_argument("--max-bytes", type=int, default=DEFAULT_ARTIFACT_LIMIT)
    artifact.add_argument("--expected-head-sha", default="")

    baselines = subparsers.add_parser("select-timing-baselines")
    baselines.add_argument("--current-run-id", type=int, required=True)
    baselines.add_argument("--workflow-name", required=True)
    baselines.add_argument("--artifact-name", required=True)
    baselines.add_argument("--head-branch", required=True)
    baselines.add_argument("--head-repository", required=True)
    baselines.add_argument("--pr-number", type=int, required=True)
    baselines.add_argument("--base-ref", required=True)
    baselines.add_argument("--max-bytes", type=int, default=DEFAULT_ARTIFACT_LIMIT)

    files = subparsers.add_parser("check-files")
    files.add_argument("--max-bytes", type=int, required=True)
    files.add_argument("paths", nargs="+")

    profile_baselines = subparsers.add_parser("resolve-profile-baselines")
    profile_baselines.add_argument("--metadata", required=True)
    profile_baselines.add_argument("--main-summary", required=True)
    profile_baselines.add_argument("--previous-summary", required=True)
    profile_baselines.add_argument("--current-run-id", type=int, required=True)
    profile_baselines.add_argument("--workflow-name", required=True)
    profile_baselines.add_argument("--artifact-name", required=True)
    profile_baselines.add_argument("--head-sha", required=True)
    profile_baselines.add_argument("--head-repository", required=True)
    profile_baselines.add_argument("--head-branch", required=True)
    profile_baselines.add_argument("--pr-number", type=int, required=True)
    profile_baselines.add_argument("--base-sha", required=True)
    profile_baselines.add_argument("--max-input-bytes", type=int, required=True)
    profile_baselines.add_argument("--max-artifact-bytes", type=int, required=True)

    upsert = subparsers.add_parser("upsert-comment")
    upsert.add_argument("--pr-number", type=int, required=True)
    upsert.add_argument("--head-sha", required=True)
    upsert.add_argument("--head-branch", required=True)
    upsert.add_argument("--head-repository", required=True)
    upsert.add_argument("--base-repository", required=True)
    upsert.add_argument("--base-branch", required=True)
    upsert.add_argument("--base-sha", required=True)
    upsert.add_argument("--comment", required=True)
    upsert.add_argument("--marker", required=True)
    upsert.add_argument("--max-bytes", type=int, default=DEFAULT_BODY_LIMIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands = {
        "resolve-pr": resolve_pr_command,
        "resolve-artifact": resolve_artifact_command,
        "select-timing-baselines": select_timing_baselines_command,
        "check-files": check_files_command,
        "resolve-profile-baselines": resolve_profile_baselines_command,
        "upsert-comment": upsert_comment_command,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
