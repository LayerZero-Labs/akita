#!/usr/bin/env bash
set -euo pipefail

tree="$(cargo tree --workspace --edges normal,build --prefix none)"

if grep -q '^akita-field v' <<<"$tree"; then
  echo "error: integrated dependency graph still contains akita-field" >&2
  exit 1
fi

identities="$(
  grep '^jolt-field v' <<<"$tree" \
    | sed 's/ (\*)$//' \
    | sort -u
)"
count="$(grep -c '^jolt-field v' <<<"$identities" || true)"

if [[ "$count" -ne 1 ]]; then
  echo "error: expected exactly one jolt-field package identity, found $count" >&2
  printf '%s\n' "$identities" >&2
  exit 1
fi

printf 'shared field identity: %s\n' "$identities"
