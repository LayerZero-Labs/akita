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
  # Note: `akita-planner` is intentionally NOT forbidden here. The DP search now
  # sits *below* `akita-config` (`akita-config → akita-planner`) and serves as a
  # verifier-reachable runtime fallback, so it legitimately appears in the
  # dependency tree of every config-consuming crate.
  case "${pkg}" in
    akita-verifier)
      forbidden=(akita-prover akita-pcs)
      ;;
    akita-prover)
      forbidden=(akita-verifier akita-pcs)
      ;;
    akita-config)
      forbidden=(akita-prover akita-verifier akita-pcs)
      ;;
    akita-setup)
      forbidden=(akita-verifier akita-pcs)
      ;;
    akita-scheme)
      forbidden=(akita-pcs)
      ;;
    *)
      echo "no default forbidden dependency set for ${pkg}; pass forbidden packages explicitly" >&2
      exit 2
      ;;
  esac
fi

# Walk both the default-feature graph and the all-features graph so an
# opt-in feature can't sneak a forbidden crate into a downstream build.
default_tree="$(cargo tree -p "${pkg}" --edges normal)"
all_features_tree="$(cargo tree -p "${pkg}" --edges normal --all-features)"

for label in default all-features; do
  case "${label}" in
    default)      tree="${default_tree}" ;;
    all-features) tree="${all_features_tree}" ;;
  esac
  for candidate in "${forbidden[@]}"; do
    if grep -qE "(^|[[:space:]])${candidate}([[:space:]]|$)" <<<"${tree}"; then
      echo "forbidden dependency found in ${pkg} (${label}): ${candidate}" >&2
      exit 1
    fi
  done
done

echo "${pkg} dependency hygiene check passed"
