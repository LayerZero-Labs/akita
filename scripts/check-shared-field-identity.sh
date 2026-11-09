#!/usr/bin/env bash
set -euo pipefail

# The integrated dependency graph must contain no akita-field, exactly one
# jolt-algebra and jolt-poly package identity, at most one jolt-field identity
# (the re-export of jolt-algebra that Jolt's own crates may still name), and
# one shared Jolt Git commit.
#
# Structural check over `cargo metadata` package IDs: immune to `cargo tree`
# rendering (CARGO_TERM_COLOR=always colorizes the `(*)` dedup marker, which
# broke the previous text parse). The worst-case color environment is forced
# below as a permanent regression guard.
export CARGO_TERM_COLOR=always

package_ids() {
  jq -r --arg name "$2" '.packages[] | select(.name == $name) | .id' <<<"$1" | sort -u
}

package_sources() {
  jq -r --arg name "$2" '.packages[] | select(.name == $name) | .source' <<<"$1" | sort -u
}

check_workspace() {
  local label="$1"
  local manifest="$2"
  local metadata
  local package
  local identities
  local count
  local source
  local shared_source=""

  metadata="$(cargo metadata --format-version 1 --locked --manifest-path "$manifest")"

  identities="$(package_ids "$metadata" akita-field)"
  if [[ -n "$identities" ]]; then
    echo "error: $label dependency graph still contains akita-field" >&2
    printf '%s\n' "$identities" >&2
    exit 1
  fi

  for package in jolt-algebra jolt-poly jolt-field; do
    identities="$(package_ids "$metadata" "$package")"
    count="$(grep -c . <<<"$identities" || true)"
    if [[ "$package" == jolt-field && "$count" -eq 0 ]]; then
      continue
    fi
    if [[ "$count" -ne 1 ]]; then
      echo "error: expected exactly one $package package identity in $label, found $count" >&2
      printf '%s\n' "$identities" >&2
      exit 1
    fi
    source="$(package_sources "$metadata" "$package")"
    if [[ "$source" != git+https://github.com/a16z/jolt* ]]; then
      echo "error: $package does not come from the Jolt Git repository in $label: $source" >&2
      exit 1
    fi
    if [[ -z "$shared_source" ]]; then
      shared_source="$source"
    elif [[ "$source" != "$shared_source" ]]; then
      echo "error: $package does not use the same Jolt Git source as jolt-algebra in $label" >&2
      printf '%s\n%s\n' "$shared_source" "$source" >&2
      exit 1
    fi
    printf 'shared %s identity (%s): %s\n' "$package" "$label" "$identities"
  done
}

check_workspace root Cargo.toml
check_workspace fuzz fuzz/Cargo.toml
check_workspace recursion-profile profile/akita-recursion/Cargo.toml
