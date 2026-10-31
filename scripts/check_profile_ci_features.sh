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

from scripts.profile_bench_merge_base_policy import read_matrix_cases, resolve_baseline
from scripts.profile_ci_features import (
    load_feature_graph,
    resolve_feature,
)

repo = Path(".")
workflow = repo / ".github/workflows/profile-bench.yml"
profile_main = repo / "crates/akita-pcs/examples/profile/main.rs"
profile_modes = repo / "crates/akita-pcs/examples/profile/modes.rs"
modes_rs = profile_modes

MODES = {
    "onehot_fp32", "dense_fp32", "onehot_fp64", "dense_fp64",
    "dense_fp128", "onehot_fp128", "onehot_fp128_multi_group",
    "onehot_fp128_multi_group_recursive",
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2",
    "onehot_fp128_multi_chunk_w8r2", "onehot_fp128_multi_chunk_w2r2",
    "onehot_fp128_multi_chunk_w4r2",
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
    "onehot_fp32": {31},
    "dense_fp32": {30},
    "onehot_fp64": {32},
    "dense_fp64": {29},
    "dense_fp128": {28},
    "onehot_fp128": {33},
    "onehot_fp128_multi_group": {31},
    "onehot_fp128_multi_group_recursive": {31},
    "onehot_fp128_multi_group_recursive_multi_chunk_w8r2": {29},
    "onehot_fp128_multi_chunk_w8r2": {29},
    "onehot_fp128_multi_chunk_w2r2": {29},
    "onehot_fp128_multi_chunk_w4r2": {29},
}
MODE_SETUP = {mode: {"direct"} for mode in MODES}
MODE_SETUP["onehot_fp128"] = {"direct", "recursive"}
MODE_SETUP["onehot_fp128_multi_group_recursive"] = {"recursive"}
MODE_SETUP["onehot_fp128_multi_group_recursive_multi_chunk_w8r2"] = {"recursive"}
PROFILE_BENCH_MARKER = "profile-bench-selected"

feature_graph = load_feature_graph(repo)
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

bench_cases = read_matrix_cases(workflow)
case_specs = [case_spec for _, _, case_spec in bench_cases]
head_policy = resolve_baseline("HEAD", case_specs)
unsupported = [item for item in head_policy["cases"] if not item["supported"]]
if unsupported:
    for item in unsupported:
        print(f"head benchmark case {item['case']} is unsupported: {item['reason']}", file=sys.stderr)
    raise SystemExit(1)
fp128_direct = "onehot_fp128:33:1:direct"
fp128_recursive = "onehot_fp128:33:1:recursive"
if fp128_direct not in case_specs or case_specs.index(fp128_direct) + 1 >= len(case_specs):
    print("fp128 nv33 direct benchmark case is missing", file=sys.stderr)
    raise SystemExit(1)
if case_specs[case_specs.index(fp128_direct) + 1] != fp128_recursive:
    print(
        "fp128 nv33 recursive benchmark must immediately follow the direct case",
        file=sys.stderr,
    )
    raise SystemExit(1)
failed = False
matrix_features: dict[str, set[str]] = {}
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
    if mode not in MODES:
        print(f"bench case mode '{mode}' is missing from MODES", file=sys.stderr)
        failed = True
        continue
    if mode not in selected_modes:
        print(
            f"bench case mode '{mode}' is not registered in PROFILE_SELECTED_MODES",
            file=sys.stderr,
        )
        failed = True
    if profile_feature not in matrix_features:
        matrix_features[profile_feature] = resolve_feature(
            feature_graph, "akita-pcs", profile_feature
        )
    if ("akita-pcs", PROFILE_BENCH_MARKER) not in resolve_feature(
        feature_graph, "akita-pcs", profile_feature
    ):
        print(
            f"matrix group '{group_name}' feature '{profile_feature}' does not enable "
            f"the '{PROFILE_BENCH_MARKER}' registry marker",
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
    if actual_setup_mode not in MODE_SETUP[mode]:
        expected_setup = ", ".join(sorted(MODE_SETUP[mode]))
        print(
            f"bench case '{case_spec}' uses setup mode '{actual_setup_mode}'; "
            f"expected one of [{expected_setup}]",
            file=sys.stderr,
        )
        failed = True

matrix_profile_features = set(matrix_features)
feature_pattern = re.compile(r'feature\s*=\s*"(profile-ci-[^"]+)"')
for source in (profile_main, profile_modes):
    declared = set(feature_pattern.findall(source.read_text(encoding="utf-8")))
    if declared != matrix_profile_features:
        print(
            f"{source} declares profile group features {sorted(declared)}, expected "
            f"exactly {sorted(matrix_profile_features)}",
            file=sys.stderr,
        )
        failed = True

if failed:
    raise SystemExit(1)

print("profile benchmark feature coverage check passed.")
PY
