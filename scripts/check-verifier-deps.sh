#!/usr/bin/env bash
set -euo pipefail

pkg="${1:-akita-verifier}"

if ! cargo metadata --format-version 1 --no-deps | grep -q "\"name\":\"${pkg}\""; then
  echo "${pkg} not present yet; skipping verifier dependency hygiene check"
  exit 0
fi

tree="$(cargo tree -p "${pkg}" --edges normal)"
for forbidden in akita-prover akita-planner akita-pcs; do
  if grep -qE "(^|[[:space:]])${forbidden}([[:space:]]|$)" <<<"${tree}"; then
    echo "forbidden dependency found in ${pkg}: ${forbidden}" >&2
    exit 1
  fi
done

echo "${pkg} dependency hygiene check passed"
