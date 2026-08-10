#!/usr/bin/env bash
# Hard gate: every profile bench mode must be covered by both its narrow matrix
# feature and the akita-pcs profile-ci compatibility union.
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
    "onehot_fp32": "schedules-fp32-onehot",
    "dense_fp32": "schedules-fp32-dense",
    "onehot_fp64": "schedules-fp64-onehot",
    "dense_fp64": "schedules-fp64-dense",
    "dense_fp128": "schedules-fp128-dense",
    "onehot_fp128": "schedules-fp128-onehot",
    "onehot_fp128_multi_group": "schedules-fp128-onehot",
    "onehot_fp128_multi_group_recursive": "schedules-fp128-onehot-recursive",
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": "schedules-fp128-onehot-recursive-multi-chunk-w8r2",
    "onehot_fp128_multi_chunk_w8r2": "schedules-fp128-onehot-multi-chunk",
    "onehot_fp128_multi_chunk_w2r2": "schedules-fp128-onehot-multi-chunk-w2r2",
    "onehot_fp128_multi_chunk_w4r2": "schedules-fp128-onehot-multi-chunk-w4r2",
}
MODE_NUM_POLYS = {
    "onehot_fp32": {1},
    "dense_fp32": {1},
    "onehot_fp64": {1},
    "dense_fp64": {1},
    "dense_fp128": {1},
    "onehot_fp128": {1},
    "onehot_fp128_multi_group": {4},
    "onehot_fp128_multi_group_recursive": {4},
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": {4},
    "onehot_fp128_multi_chunk_w8r2": {1},
    "onehot_fp128_multi_chunk_w2r2": {1},
    "onehot_fp128_multi_chunk_w4r2": {1},
}
MODE_NUM_VARS = {
    "onehot_fp32": {30},
    "dense_fp32": {26},
    "onehot_fp64": {30},
    "dense_fp64": {26},
    "dense_fp128": {26},
    "onehot_fp128": {32},
    "onehot_fp128_multi_group": {32},
    "onehot_fp128_multi_group_recursive": {32},
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": {32},
    "onehot_fp128_multi_chunk_w8r2": {32},
    "onehot_fp128_multi_chunk_w2r2": {32},
    "onehot_fp128_multi_chunk_w4r2": {32},
}
MODE_SETUP = {mode: "direct" for mode in MODE_FEATURE}
MODE_SETUP["onehot_fp128_multi_group_recursive"] = "recursive"
MODE_SETUP["onehot_fp128_multi_group_recursive_multi_chunk_w8r2"] = "recursive"
PROFILE_BENCH_MARKER = "profile-bench-selected"

text = pcs.read_text(encoding="utf-8")


def feature_members(feature: str) -> set[str]:
    match = re.search(
        rf"^{re.escape(feature)}\s*=\s*\[(.*?)\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        print(f"{feature} feature not found in akita-pcs/Cargo.toml", file=sys.stderr)
        raise SystemExit(1)

    members: set[str] = set()
    for line in match.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        if "/" in line:
            line = line.split("/", 1)[1]
        members.add(line.strip('"'))
    return members


profile_ci = feature_members("profile-ci")

modes_text = modes_rs.read_text(encoding="utf-8")
selected_match = re.search(
    r"const PROFILE_SELECTED_MODES:.*?=\s*&\[(.*?)\n\];",
    modes_text,
    flags=re.DOTALL,
)
if not selected_match:
    print("PROFILE_SELECTED_MODES not found in profile example", file=sys.stderr)
    raise SystemExit(1)
selected_modes = set(re.findall(r'name:\s*"([^"]+)"', selected_match.group(1)))

linkage_text = linkage.read_text(encoding="utf-8")
forbidden_match = re.search(r"forbidden=\(\n(.*?)\n\)", linkage_text, flags=re.DOTALL)
if not forbidden_match:
    print("forbidden symbol list not found in profile linkage guard", file=sys.stderr)
    raise SystemExit(1)
forbidden_symbols = set(
    re.findall(r"^\s*([A-Z0-9_]+)\s*$", forbidden_match.group(1), flags=re.MULTILINE)
)
feature_symbols = {
    "schedules-fp32-onehot": "FP32_ONEHOT_SCHEDULES",
    "schedules-fp32-dense": "FP32_DENSE_SCHEDULES",
    "schedules-fp64-onehot": "FP64_ONEHOT_SCHEDULES",
    "schedules-fp64-dense": "FP64_DENSE_SCHEDULES",
    "schedules-fp128-dense": "FP128_DENSE_SCHEDULES",
    "schedules-fp128-onehot": "FP128_ONEHOT_SCHEDULES",
    "schedules-fp128-onehot-recursive": "FP128_ONEHOT_RECURSIVE_SCHEDULES",
    "schedules-fp128-onehot-recursive-multi-chunk-w8r2": "FP128_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES",
    "schedules-fp128-onehot-multi-chunk": "FP128_ONEHOT_MULTI_CHUNK_SCHEDULES",
    "schedules-fp128-onehot-multi-chunk-w2r2": "FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES",
    "schedules-fp128-onehot-multi-chunk-w4r2": "FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES",
}

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

case_anchors = list(re.finditer(r"^\s+cases:\s*\|\s*\n", wf, flags=re.MULTILINE))
group_pattern = re.compile(
    r"^\s+- name:\s*(\S+)\s*\n"
    r"\s+profile_feature:\s*(\S+)\s*\n"
    r"\s+cases:\s*\|\s*\n",
    flags=re.MULTILINE,
)
groups = list(group_pattern.finditer(wf))
if len(groups) != len(case_anchors):
    print("every benchmark matrix group must declare profile_feature before cases", file=sys.stderr)
    raise SystemExit(1)

bench_cases: list[tuple[str, str, str]] = []
for group in groups:
    group_name, profile_feature = group.group(1, 2)
    for case_spec in cases_after_pipe(group.end()):
        bench_cases.append((group_name, profile_feature, case_spec))

if not bench_cases:
    print("No matrix bench cases found in profile-bench.yml", file=sys.stderr)
    raise SystemExit(1)
failed = False
matrix_features: dict[str, set[str]] = {}
group_requirements: dict[tuple[str, str], set[str]] = {}
for group_name, profile_feature, case_spec in bench_cases:
    mode, num_vars_s, num_polys_s, *setup_mode = case_spec.split(":")
    num_vars = int(num_vars_s)
    num_polys = int(num_polys_s)
    actual_setup_mode = setup_mode[0] if setup_mode else "direct"
    if actual_setup_mode not in {"direct", "recursive"}:
        print(
            f"bench case '{case_spec}' uses unsupported setup contribution mode "
            f"'{actual_setup_mode}'",
            file=sys.stderr,
        )
        failed = True
    if mode not in MODE_FEATURE:
        print(f"bench case mode '{mode}' is missing from MODE_FEATURE table", file=sys.stderr)
        failed = True
        continue
    required = MODE_FEATURE[mode]
    if mode not in selected_modes:
        print(
            f"bench case mode '{mode}' is not registered in PROFILE_SELECTED_MODES",
            file=sys.stderr,
        )
        failed = True
    if profile_feature not in matrix_features:
        matrix_features[profile_feature] = feature_members(profile_feature)
    selected = matrix_features[profile_feature]
    group_requirements.setdefault((group_name, profile_feature), set()).add(required)
    if PROFILE_BENCH_MARKER not in selected:
        print(
            f"matrix group '{group_name}' feature '{profile_feature}' does not enable "
            f"the '{PROFILE_BENCH_MARKER}' registry marker",
            file=sys.stderr,
        )
        failed = True
    if required not in selected:
        print(
            f"matrix group '{group_name}' feature '{profile_feature}' does not enable "
            f"required feature '{required}' for bench mode '{mode}'",
            file=sys.stderr,
        )
        failed = True
    if required not in profile_ci:
        print(
            f"profile-ci does not enable required feature '{required}' for bench mode '{mode}'",
            file=sys.stderr,
        )
        failed = True
    linked_symbol = feature_symbols.get(required)
    if linked_symbol is None:
        print(f"schedule feature '{required}' has no linkage symbol mapping", file=sys.stderr)
        failed = True
    elif linked_symbol in forbidden_symbols:
        print(
            f"bench mode '{mode}' requires symbol '{linked_symbol}' but the linkage guard forbids it",
            file=sys.stderr,
        )
        failed = True
    if num_polys not in MODE_NUM_POLYS[mode]:
        expected = ", ".join(str(value) for value in sorted(MODE_NUM_POLYS[mode]))
        print(
            f"bench case '{case_spec}' uses num_polys={num_polys}; expected one of [{expected}]",
            file=sys.stderr,
        )
        failed = True
    if num_vars not in MODE_NUM_VARS[mode]:
        expected = ", ".join(str(value) for value in sorted(MODE_NUM_VARS[mode]))
        print(
            f"bench case '{case_spec}' uses num_vars={num_vars}; expected one of [{expected}]",
            file=sys.stderr,
        )
        failed = True
    if actual_setup_mode != MODE_SETUP[mode]:
        print(
            f"bench case '{case_spec}' uses setup mode '{actual_setup_mode}'; "
            f"expected '{MODE_SETUP[mode]}'",
            file=sys.stderr,
        )
        failed = True

for (group_name, profile_feature), required in group_requirements.items():
    selected = matrix_features[profile_feature]
    selected_schedules = {member for member in selected if member.startswith("schedules-")}
    if selected_schedules != required:
        print(
            f"matrix group '{group_name}' feature '{profile_feature}' enables schedule "
            f"features {sorted(selected_schedules)}, expected exactly {sorted(required)}",
            file=sys.stderr,
        )
        failed = True

if failed:
    raise SystemExit(1)

print("profile benchmark feature coverage check passed.")
PY
