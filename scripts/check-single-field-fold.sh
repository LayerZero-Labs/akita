#!/usr/bin/env bash
# Fail if single-field fold modules mention forbidden EOR / tensor-projection symbols.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATTERN='extension_opening_reduction|tensor_root_projection|RootTensorProjectionPoly|prove_extension_opening_reduction|prove_grouped_extension_opening_reduction'

FILES=(
  "$ROOT/crates/akita-prover/src/protocol/core/fold/single_field.rs"
  "$ROOT/crates/akita-verifier/src/protocol/core/fold/single_field.rs"
)

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "missing single-field fold module: $file" >&2
    exit 1
  fi
done

if rg -n "$PATTERN" "${FILES[@]}"; then
  echo "single-field fold modules must not reference EOR or root tensor projection symbols" >&2
  exit 1
fi

echo "single-field fold symbol check passed"
