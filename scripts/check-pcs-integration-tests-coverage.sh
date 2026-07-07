#!/usr/bin/env bash
set -euo pipefail

# akita-pcs's integration tests are consolidated into a single binary
# (tests/integration_tests_suite.rs) that pulls in each file under
# tests/integration_tests/ via an explicit `#[path = "..."] mod ...;` line.
# Cargo's target auto-discovery does not look inside tests/integration_tests/
# (it only scans direct children of tests/), so a file or module directory
# dropped there without a corresponding line in the suite silently never
# compiles or runs. This script is the guard against that: it fails if any
# top-level .rs file, or top-level directory with its own mod.rs (e.g. a
# module like `algebra/`), under tests/integration_tests/ isn't referenced
# by the suite.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
suite_file="${repo_root}/crates/akita-pcs/tests/integration_tests_suite.rs"
test_dir="${repo_root}/crates/akita-pcs/tests/integration_tests"

missing=()
while IFS= read -r -d '' file; do
  name="$(basename "${file}" .rs)"
  if ! grep -q "path = \"integration_tests/${name}\.rs\"" "${suite_file}"; then
    missing+=("${name}.rs")
  fi
done < <(find "${test_dir}" -maxdepth 1 -name '*.rs' -type f -print0)

while IFS= read -r -d '' dir; do
  name="$(basename "${dir}")"
  if [ -f "${dir}/mod.rs" ] && ! grep -q "path = \"integration_tests/${name}/mod\.rs\"" "${suite_file}"; then
    missing+=("${name}/mod.rs")
  fi
done < <(find "${test_dir}" -maxdepth 1 -type d -print0)

if [ "${#missing[@]}" -gt 0 ]; then
  echo "error: the following files under tests/integration_tests/ are not declared as" >&2
  echo "a 'mod' in tests/integration_tests_suite.rs, so they never compile or run:" >&2
  for f in "${missing[@]}"; do
    echo "  - ${f}" >&2
  done
  echo >&2
  echo "add a '#[path = \"integration_tests/<name>.rs\"] mod <name>;' line (or, for a" >&2
  echo "module directory, '#[path = \"integration_tests/<name>/mod.rs\"] mod <name>;')" >&2
  echo "for each to tests/integration_tests_suite.rs." >&2
  exit 1
fi

file_count="$(find "${test_dir}" -maxdepth 1 -name '*.rs' -type f | wc -l | tr -d ' ')"
dir_count="$(find "${test_dir}" -maxdepth 1 -type d -exec test -e '{}/mod.rs' \; -print | wc -l | tr -d ' ')"
echo "ok: all ${file_count} top-level files and ${dir_count} top-level module directories under tests/integration_tests/ are wired into the suite"
