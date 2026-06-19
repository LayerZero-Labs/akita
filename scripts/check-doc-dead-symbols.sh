#!/usr/bin/env bash
# Flag dead removed symbols in non-historical docs/*.md.
# README.md and AGENTS.md may cite removed names when describing the cutover;
# they are covered by review and the blast-radius comment instead.
# Historical snapshots (banner in first 8 lines) are skipped.
# See docs/documentation.md and scripts/check-spec-references.sh.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

dead_patterns=(
  'akita-scheme'
  'akita-cfg'
  'akita-derive'
  'ScheduleProvider'
  'PlannerConfig'
  'WCommitmentConfig'
  'sis_offline'
  'sis_policy\.rs'
  'schedule_policy\.rs'
  '_with_policy'
)

pattern="$(IFS='|'; echo "${dead_patterns[*]}")"

scan_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    return 0
  fi
  if head -n 8 "$f" | grep -qi 'historical snapshot'; then
    return 0
  fi
  rg -n "$pattern" "$f" 2>/dev/null || true
}

# Meta / intentionally descriptive docs (cite removed names on purpose).
skip_docs=(documentation.md crate-graph.md)

matches=""
for f in docs/*.md; do
  base="$(basename "$f")"
  for skip in "${skip_docs[@]}"; do
    if [[ "$base" == "$skip" ]]; then
      continue 2
    fi
  done
  hit="$(scan_file "$f")"
  if [[ -n "$hit" ]]; then
    matches+="$hit"$'\n'
  fi
done

if [[ -n "$matches" ]]; then
  echo "Dead symbol references in docs/ (non-historical). Review:" >&2
  echo >&2
  echo "$matches" >&2
  exit 1
fi

echo "No dead symbol references in docs/."
