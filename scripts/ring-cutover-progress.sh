#!/usr/bin/env bash
# Runtime ring-dimension cutover: progress report and merge gate.
# See specs/runtime-ring-cutover.md, "Kernel dispatch" invariants.
#
# Usage:
#   scripts/ring-cutover-progress.sh                # progress report (always exit 0)
#   scripts/ring-cutover-progress.sh --merge-gate   # exit 1 unless the cutover is complete
#
# What it checks:
#   1. `const D` count in the prover orchestration spine. These files are
#      orchestration by definition (they read the schedule); the count must
#      decrease monotonically across slices and be zero at merge. Kernels
#      belong in compute/, backend/, or dedicated kernel modules, not here.
#   2. Banned #227 bridge/facade names anywhere in crates/. These reintroduce
#      the typed round-trip the cutover exists to delete.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

mode="report"
if [[ "${1:-}" == "--merge-gate" ]]; then
  mode="gate"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--merge-gate]" >&2
  exit 2
fi

# Prover orchestration spine: zero `const D` here at merge time.
# (The verifier spine is transitionally per-level monomorphized; it gets the
# same treatment when per-role dims land. See the spec's acceptance section.)
spine=(
  crates/akita-prover/src/protocol/core.rs
  crates/akita-prover/src/protocol/core/prove.rs
  crates/akita-prover/src/protocol/core/fold.rs
  crates/akita-prover/src/protocol/core/root_fold.rs
  crates/akita-prover/src/protocol/core/suffix.rs
)

# Names that reintroduce the #227 facade/bridge architecture, plus wrapper
# names already built and reverted once on this branch.
banned_patterns=(
  'into_typed'
  'try_to_ring_commitment'
  'append_as_ring_commitment'
  'TypedCommitmentProver'
  'TypedCommitmentVerifier'
  'prove_fold_at_ring_d'
  'prove_suffix_fold_at_ring_d'
  'batched_prove_at_ring_dim'
)
banned="$(IFS='|'; echo "${banned_patterns[*]}")"

# Count const-generic `D` parameters (`const D: usize` followed by `>`/`,`/`)`),
# excluding comment lines and test-style pinned constants
# (`const D: usize = 4;`), which are allowed in #[cfg(test)] modules.
spine_const_d() {
  local f="$1"
  grep -n 'const D' "$f" \
    | grep -vE '^[0-9]+:\s*//' \
    | grep -vE 'const D: usize = [0-9]' \
    || true
}

total=0
echo "== const D in prover orchestration spine, non-test (target: 0) =="
for f in "${spine[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  (missing) $f"
    continue
  fi
  count="$(spine_const_d "$f" | wc -l | tr -d ' ')"
  total=$((total + count))
  printf '  %-60s %s\n' "$f" "$count"
done
echo "  TOTAL: $total"
if [[ "$total" -gt 0 && "$mode" == "report" ]]; then
  echo
  echo "-- remaining const D sites (the burn-down list) --"
  for f in "${spine[@]}"; do
    [[ -f "$f" ]] || continue
    spine_const_d "$f" | sed "s|^|$f:|"
  done
fi

echo
echo "== banned bridge/facade names in crates/, non-comment (target: none) =="
banned_hits="$(rg -n "$banned" crates/ -g '*.rs' 2>/dev/null | grep -vE '^[^:]+:[0-9]+:\s*//' || true)"
if [[ -n "$banned_hits" ]]; then
  echo "$banned_hits"
else
  echo "  none"
fi

if [[ "$mode" == "gate" ]]; then
  fail=0
  if [[ "$total" -gt 0 ]]; then
    echo
    echo "MERGE GATE FAIL: $total const D site(s) remain in the orchestration spine." >&2
    fail=1
  fi
  if [[ -n "$banned_hits" ]]; then
    echo
    echo "MERGE GATE FAIL: banned bridge/facade names present." >&2
    fail=1
  fi
  if [[ "$fail" -eq 1 ]]; then
    exit 1
  fi
  echo
  echo "Merge gate passed."
fi
