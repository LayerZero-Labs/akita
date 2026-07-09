#!/usr/bin/env bash
# Hard gate: every CI profile bench mode must be covered by akita-pcs profile-ci,
# and every matrix group's own bench cases must be covered by that same
# group's narrower pcs_mode_features list (what CI actually builds with).
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
    "onehot_fp32_d128": "mode-onehot-fp32-d128",
    "onehot_fp64_d128": "mode-onehot-fp64-d128",
    "dense_fp128_d64": "mode-dense-fp128-d64",
    "onehot_fp128_d64": "mode-onehot-fp128-d64",
    "onehot_fp128_d64_tensor": "mode-onehot-fp128-d64-tensor",
    "onehot_fp128_d64_multi_chunk_w8r2": "mode-onehot-fp128-d64-multi-chunk-w8r2",
    "onehot_fp128_d64_multi_chunk_w2r2": "mode-onehot-fp128-d64-multi-chunk-w2r2",
    "onehot_fp128_d64_multi_chunk_w4r2": "mode-onehot-fp128-d64-multi-chunk-w4r2",
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

def cases_after_pipe(text: str, start: int) -> list[str]:
    cases: list[str] = []
    for line in text[start:].splitlines():
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
    bench_cases.extend(cases_after_pipe(wf, anchor.end()))

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

# Additional, per-group check: CI no longer builds each matrix job from the
# umbrella `profile-ci` feature — it builds from that job's own narrower
# `pcs_mode_features` list (see profile-bench.yml). A mode covered by the
# umbrella is not necessarily covered by its own group's feature list, and a
# narrow build missing a mode fails at runtime with an "unknown mode" error
# rather than at this gate. Find each matrix group's `- name: ...` entry,
# associate it with that same entry's `pcs_mode_features:` line and `cases:`
# block, and assert every case's mode is covered by that group's own feature
# list.
group_anchor = re.search(r"^([ \t]*)group:[ \t]*\n", wf, flags=re.MULTILINE)
if not group_anchor:
    print("matrix 'group:' key not found in profile-bench.yml", file=sys.stderr)
    raise SystemExit(1)

item_probe = re.search(r"\n([ \t]+)- name:\s*(\S+)", wf[group_anchor.end() :])
if not item_probe:
    print("no matrix group entries found under 'group:'", file=sys.stderr)
    raise SystemExit(1)
item_indent = item_probe.group(1)

item_re = re.compile(rf"^{re.escape(item_indent)}- name:\s*(\S+)", flags=re.MULTILINE)
group_matches = [m for m in item_re.finditer(wf) if m.start() >= group_anchor.end()]
if not group_matches:
    print("no matrix group entries found under 'group:'", file=sys.stderr)
    raise SystemExit(1)

pcs_mode_features_re = re.compile(r"^\s*pcs_mode_features:\s*(\S+)\s*$", flags=re.MULTILINE)

for idx, gm in enumerate(group_matches):
    group_name = gm.group(1)
    block_start = gm.start()
    block_end = group_matches[idx + 1].start() if idx + 1 < len(group_matches) else len(wf)
    block = wf[block_start:block_end]

    group_cases: list[str] = []
    for anchor in re.finditer(r"^\s+cases:\s*\|\s*\n", block, flags=re.MULTILINE):
        group_cases.extend(cases_after_pipe(block, anchor.end()))
    if not group_cases:
        print(f"matrix group '{group_name}' has no bench cases", file=sys.stderr)
        failed = True
        continue

    features_match = pcs_mode_features_re.search(block)
    if not features_match:
        print(
            f"matrix group '{group_name}' has bench cases but no 'pcs_mode_features' key",
            file=sys.stderr,
        )
        failed = True
        continue
    group_features = {f for f in features_match.group(1).split(",") if f}

    for case_spec in group_cases:
        mode = case_spec.split(":", 1)[0]
        if mode not in MODE_FEATURE:
            # Already reported by the umbrella coverage loop above.
            continue
        required = MODE_FEATURE[mode]
        if required not in group_features:
            print(
                f"matrix group '{group_name}' bench case '{case_spec}' needs feature "
                f"'{required}' but that group's pcs_mode_features "
                f"({', '.join(sorted(group_features)) or '<empty>'}) does not include it",
                file=sys.stderr,
            )
            failed = True

if failed:
    raise SystemExit(1)

print("profile-ci feature coverage check passed.")
print("per-group pcs_mode_features coverage check passed.")
PY
