#!/usr/bin/env bash
# A CI profile binary may link only the schedule families selected by its
# narrow matrix feature. The compatibility union is available for local use.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

binary="${1:-target/release/examples/profile}"
profile_feature="${2:-profile-ci}"
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

profile_symbols=(
  FP128_DENSE_SCHEDULES
  FP128_ONEHOT_SCHEDULES
  FP128_ONEHOT_RECURSIVE_SCHEDULES
  FP128_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES
  FP128_ONEHOT_MULTI_CHUNK_SCHEDULES
  FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES
  FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES
  FP32_D128_ONEHOT_SCHEDULES
  FP64_D128_ONEHOT_SCHEDULES
)
outside_profile_symbols=(
  FP128_D128_DENSE_SCHEDULES
  FP128_D128_ONEHOT_SCHEDULES
  FP128_D64_ONEHOT_TIERED_SCHEDULES
  FP32_D256_ONEHOT_SCHEDULES
  FP64_D128_DENSE_SCHEDULES
  FP64_D256_ONEHOT_SCHEDULES
)

case "$profile_feature" in
  profile-ci-fp128-dense)
    allowed=(FP128_DENSE_SCHEDULES)
    ;;
  profile-ci-flat-onehot)
    allowed=(FP128_ONEHOT_SCHEDULES FP32_D128_ONEHOT_SCHEDULES FP64_D128_ONEHOT_SCHEDULES)
    ;;
  profile-ci-multi-group-direct)
    allowed=(FP128_ONEHOT_SCHEDULES)
    ;;
  profile-ci-multi-group-recursive)
    allowed=(FP128_ONEHOT_RECURSIVE_SCHEDULES)
    ;;
  profile-ci-multi-group-recursive-w8r2)
    allowed=(FP128_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES)
    ;;
  profile-ci-distributed)
    allowed=(FP128_ONEHOT_MULTI_CHUNK_SCHEDULES FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES)
    ;;
  profile-ci)
    allowed=("${profile_symbols[@]}")
    ;;
  *)
    echo "unknown profile feature for linkage check: $profile_feature" >&2
    exit 1
    ;;
esac

is_allowed() {
  local candidate="$1"
  local allowed_symbol
  for allowed_symbol in "${allowed[@]}"; do
    if [[ "$candidate" == "$allowed_symbol" ]]; then
      return 0
    fi
  done
  return 1
}

failed=0
for sym in "${profile_symbols[@]}" "${outside_profile_symbols[@]}"; do
  if is_allowed "$sym"; then
    continue
  fi
  if grep -q "$sym" <<< "$symbols"; then
    echo "schedule symbol outside $profile_feature linked in CI profile binary: $sym" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  exit 1
fi

echo "CI profile linkage check passed for $profile_feature."
