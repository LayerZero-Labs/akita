#!/usr/bin/env bash
# Flag specs in specs/ (excluding specs/archive/) that reference symbols, crates,
# or modules that no longer exist in the codebase. A non-archived spec that names
# a dead symbol is a staleness signal (see specs/PRUNING.md).
#
# This is a heuristic guard, not a correctness check: a hit means "review this
# spec for staleness", not "the build is broken". Run it in the monthly index
# pass and optionally in CI as a non-blocking informational job.
#
# Usage: scripts/check-spec-references.sh
# Exit:  0 if no dead references found, 1 otherwise.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

# Symbols / crates / modules that have been removed from the codebase. Keep this
# list in sync as renames and cutovers land. Word-ish boundaries avoid matching
# live names (e.g. "akita-config" must not match the dead "akita-cfg").
dead_patterns=(
  'akita-scheme'
  'akita-cfg'
  'akita-derive'
  'ScheduleProvider'
  'PlannerConfig'
  'WCommitmentConfig'
  'PlanPolicy\b'
  'sis_offline'
  'sis_policy\.rs'
  'schedule_policy\.rs'
  '_with_policy'
  '\bFp16\b'
  '\bQ16\b'
  'NttSlotCache'
  'MultiDNttCaches'
)

pattern="$(IFS='|'; echo "${dead_patterns[*]}")"

# Search specs/ but not the archive (archived specs are allowed to be stale).
# Pass an explicit path (specs) so rg never blocks reading stdin in CI.
matches="$(rg -n --glob '!specs/archive/**' --glob '!specs/PRUNING.md' "$pattern" specs || true)"

if [[ -n "$matches" ]]; then
  echo "Stale spec references found (dead symbols). Review these specs against specs/PRUNING.md:" >&2
  echo >&2
  echo "$matches" >&2
  echo >&2
  echo "If the spec is shipped/superseded, update its Status and archive it." >&2
  exit 1
fi

echo "No dead spec references found."
