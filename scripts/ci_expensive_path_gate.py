#!/usr/bin/env python3
"""Fail-closed path and author gate for expensive pull-request CI jobs."""

from __future__ import annotations

import fnmatch
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence


SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
EXPENSIVE_PATH_PATTERNS = (
    ".cargo/*",
    ".config/nextest.toml",
    "Cargo.lock",
    "Cargo.toml",
    "*/Cargo.toml",
    "crates/*",
    "fixtures/*",
    "rust-toolchain.toml",
    "scripts/generate-schedule-tables.sh",
    "specs/evidence/subring-coefficient-packing/*",
    "third_party/*",
)


class GateError(ValueError):
    """The expensive-CI gate could not establish a safe decision."""


def changed_paths(
    base_sha: str,
    head_sha: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, ...]:
    """Return the PR path list, failing closed when Git cannot compute it."""

    if SHA_RE.fullmatch(base_sha) is None or SHA_RE.fullmatch(head_sha) is None:
        raise GateError("path gate requires full base and head commit IDs")
    result = run(
        ["git", "diff", "--name-only", f"{base_sha}...{head_sha}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "no diagnostic"
        raise GateError(f"could not determine changed paths: {detail}")
    return tuple(path for path in result.stdout.splitlines() if path)


def requires_expensive_ci(paths: Iterable[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        for path in paths
        for pattern in EXPENSIVE_PATH_PATTERNS
    )


def author_is_approved(
    association: str,
    base_repository: str,
    head_repository: str,
    labels: Sequence[str],
) -> bool:
    return (
        association in TRUSTED_ASSOCIATIONS
        or head_repository.lower() == base_repository.lower()
        or "ci-approved" in labels
    )


def parse_labels(raw: str) -> list[str]:
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError as error:
        raise GateError("LABELS_JSON is not valid JSON") from error
    if not isinstance(labels, list):
        raise GateError("LABELS_JSON is not an array")
    if not all(isinstance(label, str) for label in labels):
        raise GateError("LABELS_JSON contains a non-string label")
    return labels


def write_outputs(output_path: pathlib.Path, *, run: bool, approved: bool) -> None:
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"run={str(run).lower()}\n")
        output.write(f"approved={str(approved).lower()}\n")


def main() -> int:
    try:
        output_path = pathlib.Path(os.environ["GITHUB_OUTPUT"])
        if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
            write_outputs(output_path, run=True, approved=True)
            return 0

        paths = changed_paths(
            os.environ.get("BASE_SHA", ""), os.environ.get("HEAD_SHA", "")
        )
        approved = author_is_approved(
            os.environ.get("AUTHOR_ASSOCIATION", ""),
            os.environ.get("BASE_REPOSITORY", ""),
            os.environ.get("HEAD_REPOSITORY", ""),
            parse_labels(os.environ.get("LABELS_JSON", "")),
        )
        write_outputs(
            output_path,
            run=requires_expensive_ci(paths),
            approved=approved,
        )
    except (GateError, KeyError, OSError) as error:
        print(f"::error::Expensive CI path gate failed closed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
