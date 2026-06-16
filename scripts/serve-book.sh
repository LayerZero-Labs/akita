#!/usr/bin/env bash
# Serve the Akita Book locally with live reload.
#
# Requires mdbook 0.4.x plus the katex and mermaid preprocessors (see
# book/README.md for the exact pins; the 0.5.x mdbook line is not yet supported
# by the published preprocessors).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/book"

if ! command -v mdbook >/dev/null 2>&1; then
  echo "error: mdbook not found. Install it with: cargo install mdbook --version '^0.4'" >&2
  exit 1
fi

exec mdbook serve "$@"
