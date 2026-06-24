#!/usr/bin/env bash
# Verify repository paths cited in book chapter source lists exist.
# Exit 0 if all paths exist, 1 otherwise.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "error: ripgrep (rg) is required" >&2
  exit 2
fi

fail=0

while IFS= read -r raw; do
  path="${raw#\`}"
  path="${path%\`}"
  # Strip optional line-range suffix (e.g. foo.rs:12-34 or foo.md:49-68).
  file="${path%%:*}"
  if [[ ! -e "$file" ]]; then
    echo "error: book cites missing path: $file (from \`$path\`)" >&2
    fail=1
  fi
done < <(
  rg -o '`(?:crates|specs|docs)/[^`]+`' book/src \
    | sed 's/^`//;s/`$//' \
    | sort -u
)

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "All book source paths exist."
