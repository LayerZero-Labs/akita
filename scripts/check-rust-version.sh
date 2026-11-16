#!/usr/bin/env bash
# Check that each Cargo workspace declares the toolchain it is built with as its
# minimum Rust version, and that every member package inherits that declaration
# with `rust-version.workspace = true`.
#
# CI builds only with the channel in `rust-toolchain.toml`, so a lower
# `rust-version` would advertise a compiler that nothing tests. A member that
# omits the line advertises no minimum at all, because Cargo applies
# `[workspace.package]` values only to members that opt in.
#
# Usage: scripts/check-rust-version.sh [repository-root]
set -euo pipefail

cd "${1:-$(git rev-parse --show-toplevel)}"

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

# Every member package must inherit the workspace value: an explicit version
# can drift from it, and an omitted one declares no minimum. The fuzz crates
# form their own workspace, which declares no minimum.
while IFS= read -r manifest; do
  case "${manifest}" in
    Cargo.toml | profile/akita-recursion/Cargo.toml | fuzz/*) continue ;;
  esac
  grep -q '^\[package\]' "${manifest}" || continue
  if grep -Eq '^rust-version *= *"' "${manifest}"; then
    echo "${manifest}: replace the explicit rust-version with rust-version.workspace = true" >&2
    status=1
  elif ! grep -Eq '^rust-version(\.workspace *= *true| *= *\{ *workspace *= *true *\})' "${manifest}"; then
    echo "${manifest}: missing rust-version.workspace = true" >&2
    status=1
  fi
done < <(git ls-files '*Cargo.toml')

if [ "${status}" -eq 0 ]; then
  echo "rust-version declarations match the pinned toolchains."
fi
exit "${status}"
