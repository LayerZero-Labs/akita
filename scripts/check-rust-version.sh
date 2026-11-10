#!/usr/bin/env bash
# Check that each Cargo workspace declares the toolchain it is built with as its
# minimum Rust version, and that member crates inherit that declaration.
#
# CI builds only with the channel in `rust-toolchain.toml`, so a lower
# `rust-version` would advertise a compiler that nothing tests.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

status=0

check_workspace() {
  local dir="$1"
  local manifest="${dir}/Cargo.toml"
  local toolchain="${dir}/rust-toolchain.toml"
  local declared channel

  declared="$(awk '
    /^\[/ { in_pkg = ($0 == "[workspace.package]") }
    in_pkg && /^rust-version *=/ { gsub(/.*= *"|".*/, ""); print; exit }
  ' "${manifest}")"
  channel="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "${toolchain}")"

  if [ -z "${declared}" ]; then
    echo "${manifest}: missing [workspace.package] rust-version" >&2
    status=1
  elif [ "${declared}" != "${channel}" ]; then
    echo "${manifest}: rust-version ${declared} does not match ${toolchain} channel ${channel}" >&2
    status=1
  fi
}

check_workspace .
check_workspace profile/akita-recursion

# Member crates must inherit the workspace value instead of pinning their own.
while IFS= read -r manifest; do
  case "${manifest}" in
    Cargo.toml | profile/akita-recursion/Cargo.toml | fuzz/*) continue ;;
  esac
  if grep -Eq '^rust-version *= *"' "${manifest}"; then
    echo "${manifest}: use rust-version.workspace = true" >&2
    status=1
  fi
done < <(git ls-files '*Cargo.toml')

if [ "${status}" -eq 0 ]; then
  echo "rust-version declarations match the pinned toolchains."
fi
exit "${status}"
