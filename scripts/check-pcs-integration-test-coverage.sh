#!/usr/bin/env bash
set -euo pipefail

# akita-pcs's integration tests are consolidated into a single binary
# (tests/integration_test_suite.rs) that pulls in each file under
# tests/integration_test/ via an explicit `#[path = "..."] mod ...;` line.
# Cargo's target auto-discovery does not look inside tests/integration_test/
# (it only scans direct children of tests/), so a file dropped there without
# a corresponding line in the suite silently never compiles or runs. This
# script is the guard against that: it fails if any top-level .rs file under
# tests/integration_test/ isn't referenced by the suite.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suite_file="${repo_root}/crates/akita-pcs/tests/integration_test_suite.rs"
test_dir="${repo_root}/crates/akita-pcs/tests/integration_test"

missing=()
while IFS= read -r -d '' file; do
  name="$(basename "${file}" .rs)"
  if ! grep -q "path = \"integration_test/${name}\.rs\"" "${suite_file}"; then
    missing+=("${name}.rs")
  fi
done < <(find "${test_dir}" -maxdepth 1 -name '*.rs' -type f -print0)

if [ "${#missing[@]}" -gt 0 ]; then
  echo "error: the following files under tests/integration_test/ are not declared as" >&2
  echo "a 'mod' in tests/integration_test_suite.rs, so they never compile or run:" >&2
  for f in "${missing[@]}"; do
    echo "  - ${f}" >&2
  done
  echo >&2
  echo "add a '#[path = \"integration_test/<name>.rs\"] mod <name>;' line for each" >&2
  echo "to tests/integration_test_suite.rs." >&2
  exit 1
fi

echo "ok: all $(find "${test_dir}" -maxdepth 1 -name '*.rs' -type f | wc -l | tr -d ' ') top-level files under tests/integration_test/ are wired into the suite"
