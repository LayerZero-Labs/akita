#!/usr/bin/env bash
# Exercise scripts/check-rust-version.sh on small throwaway repositories: one
# that should pass and one for each failure the guard must report.
set -euo pipefail

guard="$(cd "$(dirname "$0")" && pwd)/check-rust-version.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
failures=0

# Build a minimal repository with both workspaces and one member in each.
make_repo() {
  local root="$1"
  mkdir -p "${root}/crates/a/src" "${root}/profile/akita-recursion/b/src"
  for ws in "${root}" "${root}/profile/akita-recursion"; do
    printf '[workspace.package]\nrust-version = "1.99.1"\n' > "${ws}/Cargo.toml"
    printf '[toolchain]\nchannel = "1.99.1"\n' > "${ws}/rust-toolchain.toml"
  done
  for member in crates/a profile/akita-recursion/b; do
    printf '[package]\nname = "m"\nrust-version.workspace = true\n' > "${root}/${member}/Cargo.toml"
  done
}

# Run the guard after `mutate` edits a fresh repository; compare pass or fail.
expect() {
  local name="$1" want="$2" mutate="$3" root="${tmp}/$1"
  make_repo "${root}"
  (cd "${root}" && eval "${mutate}" && git init -q && git add -A)
  if "${guard}" "${root}" > /dev/null 2>&1; then got=pass; else got=fail; fi
  if [ "${got}" != "${want}" ]; then
    echo "FAIL ${name}: expected ${want}, got ${got}" >&2
    failures=1
  else
    echo "ok   ${name}: ${got}"
  fi
}

expect inherited pass ':'
expect missing_inheritance fail 'sed -i.bak "/rust-version/d" profile/akita-recursion/b/Cargo.toml && rm profile/akita-recursion/b/Cargo.toml.bak'
expect explicit_override fail 'sed -i.bak "s/rust-version.workspace = true/rust-version = \"1.95\"/" crates/a/Cargo.toml && rm crates/a/Cargo.toml.bak'
expect workspace_mismatch fail 'sed -i.bak "s/1.99.1/1.95/" Cargo.toml && rm Cargo.toml.bak'
expect missing_workspace_value fail 'printf "[workspace]\n" > profile/akita-recursion/Cargo.toml'

exit "${failures}"
