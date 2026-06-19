#!/usr/bin/env bash
# Conservative smoke check: profile-ci binary must not link obvious non-profile tables.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

binary="${1:-target/release/examples/profile}"
if [[ ! -f "$binary" ]]; then
  echo "profile binary not found: $binary" >&2
  exit 1
fi

if command -v llvm-nm >/dev/null 2>&1; then
  nm_cmd=(llvm-nm)
elif command -v nm >/dev/null 2>&1; then
  nm_cmd=(nm)
else
  echo "neither llvm-nm nor nm found" >&2
  exit 1
fi

if ! symbols=$("${nm_cmd[@]}" "$binary" 2>&1); then
  echo "failed to inspect profile binary with ${nm_cmd[0]}:" >&2
  echo "$symbols" >&2
  exit 1
fi

# Families outside the profile-ci union; presence indicates accidental full-table linkage.
forbidden=(
  FP128_D128_FULL_SCHEDULES
  FP128_D128_ONEHOT_SCHEDULES
  FP128_D64_ONEHOT_TENSOR_SCHEDULES
  FP128_D64_ONEHOT_TIERED_SCHEDULES
  FP32_D256_ONEHOT_SCHEDULES
  FP64_D128_SCHEDULES
  FP64_D256_ONEHOT_SCHEDULES
)

failed=0
for sym in "${forbidden[@]}"; do
  if grep -q "$sym" <<< "$symbols"; then
    echo "forbidden schedule symbol linked in profile-ci binary: $sym" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  exit 1
fi

echo "profile-ci linkage smoke check passed."
