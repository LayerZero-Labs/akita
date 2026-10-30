#!/usr/bin/env bash
# Reject run-time construction of a trust anchor in production code.
#
# A `TrustAnchor` must be a compile-time constant built from a `'static`
# list: an anchor read from storage voids every guarantee the release loaders
# give. Every production `TrustAnchor::new(` must therefore open a `const` or
# `static` item. Test code is exempt.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

status=0
while IFS= read -r match; do
  file="${match%%:*}"
  rest="${match#*:}"
  line="${rest%%:*}"
  text="${rest#*:}"
  case "$file" in
    */tests/* | */tests.rs | *_tests.rs | */benches/* | */examples/*) continue ;;
  esac
  if [[ "$file" == crates/akita-types/src/preprocessing/anchor.rs ]]; then
    continue
  fi
  if ! grep -qE '^[[:space:]]*(pub(\([a-z]+\))?[[:space:]]+)?(const|static)[[:space:]]+[A-Z0-9_]+[[:space:]]*:[[:space:]]*TrustAnchor[[:space:]]*=[[:space:]]*TrustAnchor::new\(' <<<"$text"; then
    echo "$file:$line: TrustAnchor must be a const or static item" >&2
    status=1
  fi
done < <(git ls-files '*.rs' | xargs grep -n 'TrustAnchor::new(' || true)

if [ "$status" -eq 0 ]; then
  echo "trust anchor construction check passed"
fi
exit "$status"
