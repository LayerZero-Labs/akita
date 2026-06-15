#!/usr/bin/env bash
# Run all blocking documentation guardrail checks. See docs/documentation.md.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "==> check-spec-references.sh (live specs)"
"$repo_root/scripts/check-spec-references.sh"

echo "==> check-doc-dead-symbols.sh"
"$repo_root/scripts/check-doc-dead-symbols.sh"

echo "==> check-book-chapter-paths.sh"
"$repo_root/scripts/check-book-chapter-paths.sh"

echo "==> check-book-source-paths.sh"
"$repo_root/scripts/check-book-source-paths.sh"

if command -v mdbook >/dev/null 2>&1; then
  echo "==> mdbook build"
  (cd book && mdbook build)
else
  echo "skip: mdbook not installed (CI installs it; optional locally)"
fi

echo "All documentation guardrails passed."
