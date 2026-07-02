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

MODE_FEATURE = {
    "onehot_fp32_d128": "schedules-fp32-d128-onehot",
    "onehot_fp64_d128": "schedules-fp64-d128-onehot",
    "dense_fp128_d64": "schedules-fp128-d64-full",
    "onehot_fp128_d64": "schedules-fp128-d64-onehot",
    "onehot_fp128_d64_tensor": "schedules-fp128-d64-onehot-tensor",
}
MODE_NUM_POLYS = {mode: {1, 4} for mode in MODE_FEATURE}

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
    required = MODE_FEATURE[mode]
    if required not in profile_ci:
        print(
            f"profile-ci does not enable required feature '{required}' for bench mode '{mode}'",
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
