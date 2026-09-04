#!/usr/bin/env bash
# Schedule rows are external artifacts. No legacy generated-row symbol may be
# linked into a profile binary, regardless of the selected benchmark mode.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

binary="${1:-target/release/examples/profile}"
profile_feature="${2:-profile-ci}"
if [[ ! -f "$binary" ]]; then
  echo "profile binary not found: $binary" >&2
  exit 1
fi

if command -v llvm-nm >/dev/null 2>&1; then
  nm_cmd=(llvm-nm)
elif command -v nm >/dev/null 2>&1; then
  nm_cmd=(nm)
else
  echo "neither llvm-nm nor nm found" >&2
  exit 1
fi

if ! symbols=$("${nm_cmd[@]}" "$binary" 2>&1); then
  echo "failed to inspect profile binary with ${nm_cmd[0]}:" >&2
  echo "$symbols" >&2
  exit 1
fi

if grep -Eq 'FP(32|64|128)_[A-Z0-9_]+_SCHEDULES' <<< "$symbols"; then
  echo "legacy generated schedule rows linked in CI profile binary: $profile_feature" >&2
  exit 1
fi

artifact_magic='{"magic":[65,75,83,67,72,68,48,49]'
if grep -aFq "$artifact_magic" "$binary"; then
  echo "external schedule artifact payload linked in CI profile binary: $profile_feature" >&2
  exit 1
fi

echo "CI profile contains no compiled schedule rows or artifact payloads for $profile_feature."
