#!/usr/bin/env bash
set -euo pipefail

# The integrated dependency graph must contain no akita-field, exactly one
# jolt-field and jolt-poly package identity, and one shared Jolt Git commit.
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
  local poly_identities
  local count
  local poly_count
  local field_source
  local poly_source

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

  poly_identities="$(jq -r '.packages[] | select(.name == "jolt-poly") | .id' <<<"$metadata" | sort -u)"
  poly_count="$(grep -c . <<<"$poly_identities" || true)"
  if [[ "$poly_count" -ne 1 ]]; then
    echo "error: expected exactly one jolt-poly package identity in $label, found $poly_count" >&2
    printf '%s\n' "$poly_identities" >&2
    exit 1
  fi

  field_source="$(jq -r '.packages[] | select(.name == "jolt-field") | .source' <<<"$metadata" | sort -u)"
  poly_source="$(jq -r '.packages[] | select(.name == "jolt-poly") | .source' <<<"$metadata" | sort -u)"
  if [[ "$field_source" != "$poly_source" || "$field_source" != git+https://github.com/a16z/jolt* ]]; then
    echo "error: jolt-field and jolt-poly do not use the same Jolt Git source in $label" >&2
    printf '%s\n%s\n' "$identities" "$poly_identities" >&2
    exit 1
  fi

  printf 'shared field identity (%s): %s\n' "$label" "$identities"
  printf 'shared polynomial identity (%s): %s\n' "$label" "$poly_identities"
}

check_workspace root Cargo.toml
check_workspace fuzz fuzz/Cargo.toml
check_workspace recursion-profile profile/akita-recursion/Cargo.toml
