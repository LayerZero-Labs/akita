#!/usr/bin/env python3
"""Determine merge-base benchmark support per case from its shipped catalogs.

Unsupported modes or catalog keys run at the head only. Invalid metadata,
build failures, and failures while benchmarking supported cases are errors.
"""
from __future__ import annotations

import argparse
import csv
import os
import io
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MODES_RS = "crates/akita-pcs/examples/profile/modes.rs"
PCS_TOML = "crates/akita-pcs/Cargo.toml"
CATALOG = "artifacts/schedule-catalog.tsv"
PROFILE_CI_MARKER = "const PROFILE_CI_MODES"
PROFILE_SELECTED_MARKER = "const PROFILE_SELECTED_MODES"
PROFILE_ALL_MARKER = "const PROFILE_ALL_MODES"
PROFILE_MODES_MARKER = "const PROFILE_MODES"
NAME_RE = re.compile(r'name:\s*"([^"]+)"')

WORKLOADS = "crates/akita-pcs/examples/profile/workloads.json"


def read_matrix_cases(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text()
    pattern = re.compile(
        r"^\s+- name:\s*(\S+)\s*\n"
        r"\s+profile_feature:\s*(\S+)\s*\n"
        r"\s+cases:\s*\|\s*\n", re.MULTILINE,
    )
    cases = []
    for group in pattern.finditer(text):
        for line in text[group.end():].splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_]+:\d+:\d+(?::[a-z]+)?", line.strip()):
                break
            cases.append((*group.group(1, 2), line.strip()))
    if not cases:
        raise ValueError("no benchmark matrix cases")
    return cases


def load_workloads(ref: str) -> list[dict]:
    present = subprocess.check_output(
        ["git", "ls-tree", "--name-only", ref, "--", WORKLOADS], text=True
    ).strip()
    if present:
        rows = json.loads(git_show(ref, WORKLOADS))
    else:
        # Frozen adapters for pre-contract drivers. An unknown old driver is
        # unsupported; never infer runtime support from its catalog alone.
        revision = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        rows = []
        for path in sorted((Path(__file__).parent / "profile_workload_legacy").glob("*.json")):
            snapshot = json.loads(path.read_text())
            if revision in snapshot["revisions"]:
                rows = snapshot["workloads"]
                break
    seen = set()
    for row in rows:
        key = (row["mode"], row["setup_mode"])
        if key in seen or not row["family"]:
            raise ValueError("invalid or duplicate workload declaration")
        seen.add(key)
        for field in ("num_vars", "num_polys"):
            values = row[field]
            if values is not None and (not values or any(type(v) is not int or v <= 0 for v in values)):
                raise ValueError("invalid workload shape")
        pre = row["precommitted"]
        if any(len(g) != 2 or any(type(v) is not int or v <= 0 for v in g) for g in pre):
            raise ValueError("invalid precommitted topology")
        if pre and row["num_polys"] != [sum(g[1] for g in pre) + row["final_polys"]]:
            raise ValueError("inconsistent workload topology")
    return rows


@dataclass(frozen=True)
class BenchCase:
    mode: str
    num_vars: int
    num_polys: int
    setup_mode: str = "direct"

    @property
    def spec(self) -> str:
        suffix = "" if self.setup_mode == "direct" else f":{self.setup_mode}"
        return f"{self.mode}:{self.num_vars}:{self.num_polys}{suffix}"


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
    if not mode.strip() or num_vars <= 0 or num_polys <= 0:
        raise ValueError(
            f"invalid case spec {case_spec!r}; MODE must be nonempty and sizes positive"
        )
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
        (
            PROFILE_SELECTED_MARKER,
            PROFILE_CI_MARKER,
            PROFILE_MODES_MARKER,
            PROFILE_ALL_MARKER,
        )
        if profile_ci
        else (
            PROFILE_ALL_MARKER,
            PROFILE_MODES_MARKER,
            PROFILE_SELECTED_MARKER,
            PROFILE_CI_MARKER,
        )
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


def resolve_baseline(merge_base_ref: str, case_specs: list[str]) -> dict[str, object]:
    if not merge_base_ref or not case_specs:
        raise ValueError("merge-base ref and benchmark cases are required")
    cases = [parse_case_spec(spec) for spec in case_specs]
    if len({case.spec for case in cases}) != len(cases):
        raise ValueError("duplicate benchmark cases")
    modes = merge_base_profile_modes(merge_base_ref)
    workloads = load_workloads(merge_base_ref)
    table = csv.DictReader(io.StringIO(git_show(merge_base_ref, CATALOG)), delimiter="\t")
    if not {"family", "logical_key", "lookup_key_digest"}.issubset(table.fieldnames or []):
        raise ValueError("invalid merge-base catalog header")
    keys = set()
    for row in table:
        if (
            not row["family"]
            or not row["logical_key"].startswith("final=")
            or ";precommitted=" not in row["logical_key"]
            or not re.fullmatch(r"[0-9a-f]{64}", row["lookup_key_digest"])
        ):
            raise ValueError("invalid merge-base catalog row")
        keys.add((row["family"], row["logical_key"]))
    decisions = []
    for case in cases:
        reason = ""
        if case.mode not in modes:
            reason = f"merge-base does not define mode {case.mode}"
        else:
            row = next((row for row in workloads if row["mode"] == case.mode
                        and row["setup_mode"] == case.setup_mode), None)
            if row is None:
                reason = "merge-base driver has no workload contract for this mode/setup"
            elif any(row[key] is not None and value not in row[key] for key, value in
                     (("num_vars", case.num_vars), ("num_polys", case.num_polys))):
                reason = "merge-base driver does not accept this workload shape"
            else:
                family = row["family"]
                final_np = row["final_polys"] if row["precommitted"] else case.num_polys
                pre = ",".join(f"{nv}:{np}" for nv, np in row["precommitted"])
                logical_key = f"final={case.num_vars}:{final_np};precommitted={pre}"
                if (family, logical_key) not in keys:
                    reason = f"merge-base catalog {family} has no row for {logical_key}"
        decisions.append({"case": case.spec, "supported": not reason, "reason": reason})
    return {"merge_base_ref": merge_base_ref, "cases": decisions}


def cmd_resolve(args: argparse.Namespace) -> int:
    policy = resolve_baseline(args.merge_base_ref, args.case)
    Path(args.output).write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    available = any(item["supported"] for item in policy["cases"])
    if args.github_env:
        with Path(args.github_env).open("a", encoding="utf-8") as handle:
            handle.write(f"AKITA_BENCH_BASELINE_AVAILABLE={int(available)}\n")
    for item in policy["cases"]:
        status = "interleaved" if item["supported"] else item["reason"]
        print(f"Baseline {item['case']}: {status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--merge-base-ref")
    selection.add_argument("--binary", help="Attest each case using the compiled head driver.")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--github-env", default="")
    args = parser.parse_args(argv)
    if not args.case:
        parser.error("at least one --case is required")
    if args.binary:
        for spec in args.case:
            case = parse_case_spec(spec)
            env = dict(os.environ, AKITA_MODE=case.mode, AKITA_NUM_VARS=str(case.num_vars),
                       AKITA_NUM_POLYS=str(case.num_polys), AKITA_SETUP_MODE=case.setup_mode,
                       AKITA_PROFILE_CONTRACT_ONLY="1", AKITA_PROFILE_TRACE="0",
                       AKITA_PROFILE_ANSI="0")
            subprocess.run([args.binary], env=env, check=True)
        return 0
    if not args.output:
        parser.error("--merge-base-ref requires --output")
    return cmd_resolve(args)


if __name__ == "__main__":
    raise SystemExit(main())
