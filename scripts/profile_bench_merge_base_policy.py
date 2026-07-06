#!/usr/bin/env python3
"""Merge-base baseline policy for profile-bench CI.

On pull requests, each matrix group may benchmark PR head interleaved with the
merge-base profile binary. When merge-base does not yet define every profile mode
in the group's cases, we skip building and running merge-base for that group so
PR comments show Main=n/a instead of failing the job.

This script is the single source of truth for that decision.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MODES_RS = "crates/akita-pcs/examples/profile/modes.rs"
PCS_TOML = "crates/akita-pcs/Cargo.toml"
PROFILE_CI_MARKER = "const PROFILE_CI_MODES"
PROFILE_ALL_MARKER = "const PROFILE_ALL_MODES"
PROFILE_MODES_MARKER = "const PROFILE_MODES"
NAME_RE = re.compile(r'name:\s*"([^"]+)"')


def parse_case_mode(case_spec: str) -> str:
    mode, _, _ = case_spec.partition(":")
    mode = mode.strip()
    if not mode:
        raise ValueError(f"invalid bench case spec: {case_spec!r}")
    return mode


def git_show(ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"],
        text=True,
    )


def merge_base_has_profile_ci(ref: str) -> bool:
    text = git_show(ref, PCS_TOML)
    match = re.search(r"^profile-ci\s*=\s*\[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL)
    return match is not None


def modes_from_block(text: str, marker: str) -> set[str] | None:
    start = text.find(marker)
    if start < 0:
        return None
    block = text[start:]
    end = block.find("];")
    if end < 0:
        return None
    return set(NAME_RE.findall(block[: end + 2]))


def profile_modes_from_modes_rs(text: str, *, profile_ci: bool) -> set[str]:
    markers = (
        (PROFILE_CI_MARKER, PROFILE_MODES_MARKER, PROFILE_ALL_MARKER)
        if profile_ci
        else (PROFILE_ALL_MARKER, PROFILE_MODES_MARKER, PROFILE_CI_MARKER)
    )
    for marker in markers:
        modes = modes_from_block(text, marker)
        if modes:
            return modes
    raise ValueError("no profile mode table found in modes.rs")


def merge_base_profile_modes(ref: str) -> set[str]:
    text = git_show(ref, MODES_RS)
    profile_ci = merge_base_has_profile_ci(ref)
    return profile_modes_from_modes_rs(text, profile_ci=profile_ci)


def baseline_available(ref: str, case_specs: list[str]) -> tuple[bool, list[str]]:
    required = [parse_case_mode(spec) for spec in case_specs]
    available_modes = merge_base_profile_modes(ref)
    missing = sorted({mode for mode in required if mode not in available_modes})
    return not missing, missing


def append_github_env(path: Path, key: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def cmd_check(args: argparse.Namespace) -> int:
    if not args.merge_base_ref:
        print("merge-base ref is required", file=sys.stderr)
        return 2
    if not args.case:
        print("at least one --case is required", file=sys.stderr)
        return 2

    try:
        available, missing = baseline_available(args.merge_base_ref, args.case)
    except (subprocess.CalledProcessError, ValueError) as error:
        print(f"::warning::Could not evaluate merge-base baseline policy: {error}")
        available = False
        missing = []

    if args.github_env:
        env_path = Path(args.github_env)
        append_github_env(env_path, "AKITA_BENCH_BASELINE_AVAILABLE", "1" if available else "0")
        missing_modes = ",".join(missing)
        with env_path.open("a", encoding="utf-8") as handle:
            handle.write("AKITA_BENCH_BASELINE_MISSING_MODES<<EOF\n")
            handle.write(f"{missing_modes}\n")
            handle.write("EOF\n")

    if available:
        print(
            f"Merge-base {args.merge_base_ref} defines all profile modes "
            f"for this group; benchmarking interleaved with merge-base."
        )
    elif missing:
        print(
            "::notice::Skipping merge-base baseline for this group: merge-base "
            f"does not define profile mode(s) {', '.join(missing)}. "
            "Benchmarking PR head only until those modes land on the base branch."
        )
    else:
        print(
            "::notice::Skipping merge-base baseline for this group; "
            "could not confirm merge-base profile modes."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Decide whether merge-base baseline applies.")
    check.add_argument(
        "--merge-base-ref",
        required=True,
        help="Git ref for merge-base (commit SHA).",
    )
    check.add_argument(
        "--case",
        action="append",
        default=[],
        help="Bench case as MODE:NUM_VARS:NUM_POLYS[:SETUP_MODE]. Repeatable.",
    )
    check.add_argument(
        "--github-env",
        default="",
        help="Append AKITA_BENCH_BASELINE_AVAILABLE to this file (GITHUB_ENV).",
    )
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
