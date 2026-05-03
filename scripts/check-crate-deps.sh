#!/usr/bin/env bash
set -euo pipefail

pkg="${1:?usage: scripts/check-crate-deps.sh <package> [forbidden-package ...]}"
shift

if ! cargo metadata --format-version 1 --no-deps | grep -q "\"name\":\"${pkg}\""; then
  echo "${pkg} not present yet; skipping dependency hygiene check"
  exit 0
fi

if [ "$#" -gt 0 ]; then
  forbidden=("$@")
else
  case "${pkg}" in
    akita-verifier)
      forbidden=(akita-prover hachi-pcs akita-planner)
      ;;
    akita-prover)
      forbidden=(akita-verifier hachi-pcs akita-planner)
      ;;
    *)
      echo "no default forbidden dependency set for ${pkg}; pass forbidden packages explicitly" >&2
      exit 2
      ;;
  esac
fi

tree="$(cargo tree -p "${pkg}" --edges normal)"
for candidate in "${forbidden[@]}"; do
  if grep -qE "(^|[[:space:]])${candidate}([[:space:]]|$)" <<<"${tree}"; then
    echo "forbidden dependency found in ${pkg}: ${candidate}" >&2
    exit 1
  fi
done

echo "${pkg} dependency hygiene check passed"
