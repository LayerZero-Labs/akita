#!/usr/bin/env python3
"""Determine merge-base benchmark support per case from its shipped catalogs.

Unsupported modes or catalog keys run at the head only. Invalid metadata,
build failures, and failures while benchmarking supported cases are errors.
"""
from __future__ import annotations

import argparse
import csv
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

# Named profile workloads: catalog family and whether two nv16 precommits
# precede the two-polynomial final group.
MODE_CATALOGS = {
    "dense_fp32": ("fp32_dense", False),
    "dense_fp64": ("fp64_dense", False),
    "dense_fp128": ("fp128_dense", False),
    "onehot_fp32": ("fp32_onehot", False),
    "onehot_fp64": ("fp64_onehot", False),
    "onehot_fp128": ("fp128_onehot", False),
    "onehot_fp128_multi_group": ("fp128_onehot", True),
    "onehot_fp128_multi_group_recursive": ("fp128_onehot_recursive", True),
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": ("fp128_onehot_recursive_multi_chunk_w8r2", True),
    "onehot_fp128_multi_chunk_w2r2": ("fp128_onehot_multi_chunk_w2r2", False),
    "onehot_fp128_multi_chunk_w4r2": ("fp128_onehot_multi_chunk_w4r2", False),
    "onehot_fp128_multi_chunk_w8r2": ("fp128_onehot_multi_chunk", False),
}

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
    table = csv.DictReader(io.StringIO(git_show(merge_base_ref, CATALOG)), delimiter="\t")
    if not {"family", "logical_key", "lookup_key_digest"}.issubset(table.fieldnames or []):
        raise ValueError("invalid merge-base catalog header")
    keys = set()
    for row in table:
        if (not row["family"] or not row["logical_key"].startswith("final=")
                or ";precommitted=" not in row["logical_key"]
                or not re.fullmatch(r"[0-9a-f]{64}", row["lookup_key_digest"])):
            raise ValueError("invalid merge-base catalog row")
        keys.add((row["family"], row["logical_key"]))
    decisions = []
    for case in cases:
        reason = ""
        if case.mode not in modes:
            reason = f"merge-base does not define mode {case.mode}"
        else:
            if case.mode not in MODE_CATALOGS:
                raise ValueError(f"no catalog mapping for profile mode {case.mode}")
            family, grouped = MODE_CATALOGS[case.mode]
            if case.setup_mode == "recursive" and not family.endswith(("_recursive", "_recursive_multi_chunk_w8r2")):
                family += "_recursive"
            if grouped:
                if case.num_polys != 4:
                    raise ValueError("multi-group profiles require four polynomials")
                logical_key = f"final={case.num_vars}:2;precommitted=16:1,16:1"
            else:
                logical_key = f"final={case.num_vars}:{case.num_polys};precommitted="
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
    parser.add_argument("command", choices=["resolve"])
    parser.add_argument("--merge-base-ref", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--github-env", default="")
    args = parser.parse_args(argv)
    return cmd_resolve(args)


if __name__ == "__main__":
    raise SystemExit(main())
