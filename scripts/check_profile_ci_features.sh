#!/usr/bin/env bash
# Hard gate: every CI profile bench mode must be covered by akita-pcs profile-ci.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

repo = Path(".")
workflow = repo / ".github/workflows/profile-bench.yml"
pcs = repo / "crates/akita-pcs/Cargo.toml"
modes_rs = repo / "crates/akita-pcs/examples/profile/modes.rs"
linkage = repo / "scripts/check_profile_ci_linkage.sh"

MODE_FEATURE = {
    "dense_fp32": "schedules-fp32-dense",
    "onehot_fp32": "schedules-fp32-onehot",
    "dense_fp64": "schedules-fp64-dense",
    "onehot_fp64": "schedules-fp64-onehot",
    "dense_fp128": "schedules-fp128-dense",
    "onehot_fp128": "schedules-fp128-onehot",
    "onehot_fp128_multi_group": "schedules-fp128-onehot",
    "onehot_fp128_multi_group_recursive": "schedules-fp128-onehot-recursive",
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": "schedules-fp128-onehot-recursive-multi-chunk-w8r2",
    "onehot_fp128_multi_chunk_w8r2": "schedules-fp128-onehot-multi-chunk",
    "onehot_fp128_multi_chunk_w2r2": "schedules-fp128-onehot-multi-chunk-w2r2",
    "onehot_fp128_multi_chunk_w4r2": "schedules-fp128-onehot-multi-chunk-w4r2",
}
MODE_NUM_POLYS = {mode: {1, 4} for mode in MODE_FEATURE}
FEATURE_SYMBOL = {
    "schedules-fp32-dense": "FP32_DENSE_SCHEDULES",
    "schedules-fp32-onehot": "FP32_ONEHOT_SCHEDULES",
    "schedules-fp64-dense": "FP64_DENSE_SCHEDULES",
    "schedules-fp64-onehot": "FP64_ONEHOT_SCHEDULES",
    "schedules-fp128-dense": "FP128_DENSE_SCHEDULES",
    "schedules-fp128-onehot": "FP128_ONEHOT_SCHEDULES",
    "schedules-fp128-onehot-recursive": "FP128_ONEHOT_RECURSIVE_SCHEDULES",
    "schedules-fp128-onehot-recursive-multi-chunk-w8r2": "FP128_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES",
    "schedules-fp128-onehot-multi-chunk": "FP128_ONEHOT_MULTI_CHUNK_SCHEDULES",
    "schedules-fp128-onehot-multi-chunk-w2r2": "FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES",
    "schedules-fp128-onehot-multi-chunk-w4r2": "FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES",
}

text = pcs.read_text(encoding="utf-8")
match = re.search(r"^profile-ci\s*=\s*\[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL)
if not match:
    print("profile-ci feature not found in akita-pcs/Cargo.toml", file=sys.stderr)
    raise SystemExit(1)

profile_ci: set[str] = set()
for line in match.group(1).splitlines():
    line = line.strip().rstrip(",")
    if not line or line.startswith("#"):
        continue
    if "/" in line:
        line = line.split("/", 1)[1]
    profile_ci.add(line.strip('"'))

modes_text = modes_rs.read_text(encoding="utf-8")
profile_modes_match = re.search(
    r"const PROFILE_CI_MODES:.*?=\s*&\[(.*?)\n\];",
    modes_text,
    flags=re.DOTALL,
)
if not profile_modes_match:
    print("PROFILE_CI_MODES not found in profile example", file=sys.stderr)
    raise SystemExit(1)
profile_ci_modes = set(
    re.findall(r'name:\s*"([^"]+)"', profile_modes_match.group(1))
)

linkage_text = linkage.read_text(encoding="utf-8")
forbidden_match = re.search(r"forbidden=\(\n(.*?)\n\)", linkage_text, flags=re.DOTALL)
if not forbidden_match:
    print("forbidden symbol list not found in profile linkage guard", file=sys.stderr)
    raise SystemExit(1)
forbidden_symbols = set(
    re.findall(r"^\s*([A-Z0-9_]+)\s*$", forbidden_match.group(1), flags=re.MULTILINE)
)

wf = workflow.read_text(encoding="utf-8")
case_line = re.compile(r"^([^:]+:\d+:\d+(?::[^:\s]+)?)\s*$")

def cases_after_pipe(start: int) -> list[str]:
    cases: list[str] = []
    for line in wf[start:].splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            break
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = case_line.match(stripped)
        if m:
            cases.append(m.group(1))
        else:
            break
    return cases

bench_cases: list[str] = []
for anchor in re.finditer(r"^\s+cases:\s*\|\s*\n", wf, flags=re.MULTILINE):
    bench_cases.extend(cases_after_pipe(anchor.end()))

if not bench_cases:
    print("No matrix bench cases found in profile-bench.yml", file=sys.stderr)
    raise SystemExit(1)
failed = False
for case_spec in bench_cases:
    mode, num_vars, num_polys_s, *setup_mode = case_spec.split(":")
    num_polys = int(num_polys_s)
    if setup_mode and setup_mode[0] not in {"direct", "recursive"}:
        print(
            f"bench case '{case_spec}' uses unsupported setup contribution mode '{setup_mode[0]}'",
            file=sys.stderr,
        )
        failed = True
    if mode not in MODE_FEATURE:
        print(f"bench case mode '{mode}' is missing from MODE_FEATURE table", file=sys.stderr)
        failed = True
        continue
    if mode not in profile_ci_modes:
        print(
            f"bench case mode '{mode}' is not registered in PROFILE_CI_MODES",
            file=sys.stderr,
        )
        failed = True
    required = MODE_FEATURE[mode]
    if required not in profile_ci:
        print(
            f"profile-ci does not enable required feature '{required}' for bench mode '{mode}'",
            file=sys.stderr,
        )
        failed = True
    linked_symbol = FEATURE_SYMBOL.get(required)
    if linked_symbol is None:
        print(
            f"profile-ci feature '{required}' has no linkage symbol mapping",
            file=sys.stderr,
        )
        failed = True
    elif linked_symbol in forbidden_symbols:
        print(
            f"profile-ci bench mode '{mode}' requires symbol '{linked_symbol}' but the linkage guard forbids it",
            file=sys.stderr,
        )
        failed = True
    if num_polys not in MODE_NUM_POLYS[mode]:
        print(
            f"bench case '{case_spec}' uses num_polys={num_polys} outside generated keys [1, 4]",
            file=sys.stderr,
        )
        failed = True

if failed:
    raise SystemExit(1)

print("profile-ci feature coverage check passed.")
PY
