#!/usr/bin/env bash
set -euo pipefail

# The integrated dependency graph must contain no akita-field and exactly one
# jolt-field package identity.
#
# Structural check over `cargo metadata` package IDs: immune to `cargo tree`
# rendering (CARGO_TERM_COLOR=always colorizes the `(*)` dedup marker, which
# broke the previous text parse). The worst-case color environment is forced
# below as a permanent regression guard.
export CARGO_TERM_COLOR=always

check_workspace() {
  local label="$1"
  local manifest="$2"
  local metadata
  local akita_identities
  local identities
  local count

  metadata="$(cargo metadata --format-version 1 --locked --manifest-path "$manifest")"

  akita_identities="$(jq -r '.packages[] | select(.name == "akita-field") | .id' <<<"$metadata" | sort -u)"
  if [[ -n "$akita_identities" ]]; then
    echo "error: $label dependency graph still contains akita-field" >&2
    printf '%s\n' "$akita_identities" >&2
    exit 1
  fi

  identities="$(jq -r '.packages[] | select(.name == "jolt-field") | .id' <<<"$metadata" | sort -u)"
  count="$(grep -c . <<<"$identities" || true)"

  if [[ "$count" -ne 1 ]]; then
    echo "error: expected exactly one jolt-field package identity in $label, found $count" >&2
    printf '%s\n' "$identities" >&2
    exit 1
  fi

  printf 'shared field identity (%s): %s\n' "$label" "$identities"
}

check_workspace root Cargo.toml
check_workspace fuzz fuzz/Cargo.toml
check_workspace recursion-profile profile/akita-recursion/Cargo.toml
