#!/usr/bin/env bash
# Verify Book-chapter: headers in specs/ point at existing book pages.
# Exit 0 if all paths exist, 1 otherwise.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

fail=0

while IFS= read -r line; do
  file="${line%%:*}"
  if [[ "$(basename "$file")" == "TEMPLATE.md" ]]; then
    continue
  fi
  path="$(echo "$line" | sed -n 's/.*| Book-chapter[[:space:]]*|[[:space:]]*\([^|]*\).*/\1/p' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if [[ -z "$path" ]] || [[ "$path" == *"e.g."* ]]; then
    continue
  fi
  # Allow book/src/... or bare relative paths under book/src.
  if [[ "$path" != book/* ]]; then
    path="${path#src/}"
    path="book/src/${path#book/src/}"
  fi
  if [[ ! -f "$path" ]]; then
    echo "error: $file: Book-chapter path does not exist: $path" >&2
    fail=1
  fi
done < <(rg -n '^\| Book-chapter' specs --glob '!specs/archive/**' || true)

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "All Book-chapter paths exist."
