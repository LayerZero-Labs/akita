#!/usr/bin/env python3
"""Merge-base baseline policy for profile-bench CI.

Single source of truth for whether a PR matrix group benchmarks interleaved
with the merge-base profile binary.

Skip merge-base (PR head only, Main=n/a in the report) when either:

1. Merge-base does not define every profile mode in the group's cases.
2. Merge-base defines the modes but its profile binary cannot complete a smoke
   run for every case (broken schedules, missing Cargo features, panic, etc.).

Otherwise build merge-base and interleave. Smoke failures are not CI failures.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MODES_RS = "crates/akita-pcs/examples/profile/modes.rs"
PCS_TOML = "crates/akita-pcs/Cargo.toml"
PROFILE_CI_MARKER = "const PROFILE_CI_MODES"
PROFILE_ALL_MARKER = "const PROFILE_ALL_MODES"
PROFILE_MODES_MARKER = "const PROFILE_MODES"
NAME_RE = re.compile(r'name:\s*"([^"]+)"')

SKIP_UNSET_MERGE_BASE = "unset_merge_base"
SKIP_MISSING_MODES = "missing_modes"
SKIP_SMOKE_FAILED = "smoke_failed"
SKIP_POLICY_ERROR = "policy_error"


@dataclass(frozen=True)
class BenchCase:
    mode: str
    num_vars: int
    num_polys: int
    setup_mode: str = "direct"

    @property
    def spec(self) -> str:
        if self.setup_mode == "direct":
            return f"{self.mode}:{self.num_vars}:{self.num_polys}"
        return f"{self.mode}:{self.num_vars}:{self.num_polys}:{self.setup_mode}"


@dataclass(frozen=True)
class BaselineDecision:
    available: bool
    skip_reason: str
    missing_modes: tuple[str, ...]
    smoke_failures: tuple[str, ...]

    @property
    def skip_message(self) -> str:
        if self.available:
            return (
                "Benchmarking PR head interleaved with merge-base; "
                "all group modes are defined and smoke runs succeeded."
            )
        if self.skip_reason == SKIP_UNSET_MERGE_BASE:
            return "Skipping merge-base baseline: merge-base SHA is unset."
        if self.skip_reason == SKIP_MISSING_MODES:
            modes = ", ".join(self.missing_modes)
            return (
                "Skipping merge-base baseline: merge-base does not define "
                f"profile mode(s) {modes}. Benchmarking PR head only until "
                "those modes land on the base branch."
            )
        if self.skip_reason == SKIP_SMOKE_FAILED:
            cases = ", ".join(self.smoke_failures)
            return (
                "Skipping merge-base baseline: merge-base profile binary failed "
                f"smoke run(s) for {cases}. Benchmarking PR head only."
            )
        return (
            "Skipping merge-base baseline: could not evaluate merge-base "
            "profile capability for this group."
        )


def parse_case_mode(case_spec: str) -> str:
    mode, _, _ = case_spec.partition(":")
    mode = mode.strip()
    if not mode:
        raise ValueError(f"invalid bench case spec: {case_spec!r}")
    return mode


def normalize_setup_mode(setup_mode: str) -> str:
    if setup_mode not in {"direct", "recursive"}:
        raise ValueError(f"unsupported setup mode: {setup_mode!r}")
    return setup_mode


def parse_case_spec(case_spec: str) -> BenchCase:
    parts = case_spec.split(":")
    if len(parts) == 3:
        mode, num_vars_str, num_polys_str = parts
        setup_mode = "direct"
    elif len(parts) == 4:
        mode, num_vars_str, num_polys_str, setup_mode = parts
        setup_mode = normalize_setup_mode(setup_mode)
    else:
        raise ValueError(
            f"invalid case spec {case_spec!r}; expected "
            "MODE:NUM_VARS:NUM_POLYS or MODE:NUM_VARS:NUM_POLYS:SETUP_MODE"
        )
    num_vars = int(num_vars_str)
    num_polys = int(num_polys_str)
    if num_vars <= 0 or num_polys <= 0:
        raise ValueError(f"invalid case spec {case_spec!r}; NUM_VARS and NUM_POLYS must be positive")
    return BenchCase(
        mode=mode.strip(),
        num_vars=num_vars,
        num_polys=num_polys,
        setup_mode=setup_mode,
    )


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


def missing_modes_for_cases(ref: str, cases: list[BenchCase]) -> list[str]:
    available_modes = merge_base_profile_modes(ref)
    required = {case.mode for case in cases}
    return sorted(mode for mode in required if mode not in available_modes)


def smoke_profile_case(binary: Path, case: BenchCase) -> tuple[bool, str]:
    env = os.environ.copy()
    env["AKITA_MODE"] = case.mode
    env["AKITA_NUM_VARS"] = str(case.num_vars)
    env["AKITA_NUM_POLYS"] = str(case.num_polys)
    env["AKITA_SETUP_MODE"] = case.setup_mode
    env["AKITA_PROFILE_TRACE"] = "0"
    env["AKITA_PROFILE_SPAN_CLOSES"] = "0"
    env["AKITA_PROFILE_LOG"] = "info"
    env["AKITA_PROFILE_ANSI"] = "0"

    completed = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode == 0:
        return True, ""
    combined = completed.stdout + completed.stderr
    tail = combined[-800:].strip()
    if tail:
        return False, f"exit {completed.returncode}: {tail}"
    return False, f"exit {completed.returncode}"


def smoke_profile_cases(binary: Path, cases: list[BenchCase]) -> list[str]:
  failures: list[str] = []
  with tempfile.TemporaryDirectory(prefix="profile-bench-smoke-") as tmp:
      tmp_path = Path(tmp)
      for case in cases:
          ok, error = smoke_profile_case(binary, case)
          if ok:
              continue
          log_path = tmp_path / f"{case.spec.replace(':', '_')}.log"
          log_path.write_text(error + "\n", encoding="utf-8")
          failures.append(case.spec)
  return failures


def resolve_baseline(
    merge_base_ref: str | None,
    case_specs: list[str],
    *,
    smoke_binary: Path | None = None,
) -> BaselineDecision:
    if not merge_base_ref:
        return BaselineDecision(
            available=False,
            skip_reason=SKIP_UNSET_MERGE_BASE,
            missing_modes=(),
            smoke_failures=(),
        )
    if not case_specs:
        return BaselineDecision(
            available=False,
            skip_reason=SKIP_POLICY_ERROR,
            missing_modes=(),
            smoke_failures=(),
        )

    try:
        cases = [parse_case_spec(spec) for spec in case_specs]
        missing = missing_modes_for_cases(merge_base_ref, cases)
        if missing:
            return BaselineDecision(
                available=False,
                skip_reason=SKIP_MISSING_MODES,
                missing_modes=tuple(missing),
                smoke_failures=(),
            )

        if smoke_binary is None:
            return BaselineDecision(
                available=True,
                skip_reason="",
                missing_modes=(),
                smoke_failures=(),
            )

        if not smoke_binary.is_file():
            return BaselineDecision(
                available=False,
                skip_reason=SKIP_POLICY_ERROR,
                missing_modes=(),
                smoke_failures=(),
            )
        smoke_failures = smoke_profile_cases(smoke_binary, cases)
        if smoke_failures:
            return BaselineDecision(
                available=False,
                skip_reason=SKIP_SMOKE_FAILED,
                missing_modes=(),
                smoke_failures=tuple(smoke_failures),
            )

        return BaselineDecision(
            available=True,
            skip_reason="",
            missing_modes=(),
            smoke_failures=(),
        )
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        print(f"::warning::Merge-base baseline policy error: {error}", file=sys.stderr)
        return BaselineDecision(
            available=False,
            skip_reason=SKIP_POLICY_ERROR,
            missing_modes=(),
            smoke_failures=(),
        )


def modes_defined_on_merge_base(decision: BaselineDecision) -> bool:
    return decision.skip_reason not in {
        SKIP_UNSET_MERGE_BASE,
        SKIP_MISSING_MODES,
        SKIP_POLICY_ERROR,
    }


def write_github_env(path: Path, decision: BaselineDecision) -> None:
    modes_ok = modes_defined_on_merge_base(decision)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"AKITA_BENCH_BASELINE_MODES_OK={'1' if modes_ok else '0'}\n")
        handle.write(f"AKITA_BENCH_BASELINE_AVAILABLE={'1' if decision.available else '0'}\n")
        handle.write(f"AKITA_BENCH_BASELINE_SKIP_REASON={decision.skip_reason}\n")
        handle.write("AKITA_BENCH_BASELINE_MISSING_MODES<<EOF\n")
        handle.write(",".join(decision.missing_modes) + "\n")
        handle.write("EOF\n")
        handle.write("AKITA_BENCH_BASELINE_SMOKE_FAILURES<<EOF\n")
        handle.write(",".join(decision.smoke_failures) + "\n")
        handle.write("EOF\n")


def emit_decision(
    decision: BaselineDecision,
    *,
    merge_base_ref: str | None,
    smoke_ran: bool,
) -> None:
    if decision.available:
        if smoke_ran:
            print(
                "Merge-base profile binary passed smoke runs for every case; "
                "benchmarking PR head interleaved with merge-base."
            )
        else:
            print(
                "Merge-base defines all profile modes for this group; "
                "will build merge-base and run smoke checks."
            )
        if merge_base_ref:
            print(f"Merge-base ref: {merge_base_ref}")
    else:
        print(f"::notice::{decision.skip_message}")


def cmd_resolve(args: argparse.Namespace) -> int:
    smoke_binary = Path(args.smoke_binary) if args.smoke_binary else None
    decision = resolve_baseline(
        args.merge_base_ref or None,
        args.case,
        smoke_binary=smoke_binary,
    )
    if args.github_env:
        write_github_env(Path(args.github_env), decision)
    emit_decision(
        decision,
        merge_base_ref=args.merge_base_ref or None,
        smoke_ran=smoke_binary is not None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser(
        "resolve",
        help="Decide whether merge-base baseline applies for this group's cases.",
    )
    resolve.add_argument(
        "--merge-base-ref",
        default="",
        help="Git ref for merge-base (commit SHA). Empty skips baseline.",
    )
    resolve.add_argument(
        "--case",
        action="append",
        default=[],
        help="Bench case as MODE:NUM_VARS:NUM_POLYS[:SETUP_MODE]. Repeatable.",
    )
    resolve.add_argument(
        "--smoke-binary",
        default="",
        help=(
            "Optional merge-base profile binary. When set, each case is smoke-run "
            "on merge-base after the mode-name check passes."
        ),
    )
    resolve.add_argument(
        "--github-env",
        default="",
        help="Append baseline decision variables to this file (GITHUB_ENV).",
    )
    resolve.set_defaults(func=cmd_resolve)

    check = subparsers.add_parser(
        "check",
        help="Alias for resolve without --smoke-binary (mode-name check only).",
    )
    check.add_argument("--merge-base-ref", required=True)
    check.add_argument("--case", action="append", default=[])
    check.add_argument("--github-env", default="")
    check.set_defaults(func=cmd_resolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
