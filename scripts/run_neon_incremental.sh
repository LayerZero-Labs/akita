#!/usr/bin/env bash
# Run field_arith NEON capture in small Criterion filters so a killed job only loses one chunk.
#
# Usage:
#   ./scripts/run_neon_incremental.sh              # all base|ext4|ext5 chunks
#   ./scripts/run_neon_incremental.sh --from 3     # resume at chunk index 3
#   ./scripts/run_neon_incremental.sh --only ext4  # Akita fp4 (rs/tw/pw short labels)
#   ./scripts/run_neon_incremental.sh --only p3    # Plonky3 rows only (~216 cases)
#   ./scripts/run_neon_incremental.sh --only square # packed_square* only (akita+p3)
#
# Logs: bench-logs/neon/chunk_*.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "${HOME}/.cargo/env" 2>/dev/null || true
export RUSTFLAGS="-Ctarget-cpu=native"

FROM=0
MODE="full"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM="$2"
      shift 2
      ;;
    --only)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  full)
    CHUNKS=(
      'field_arith/base/latency_chain'
      'field_arith/base/throughput_stream'
      # ext4: ring_subfield (rs) first, then tower (tw), power (pw); short labels fit Criterion 64-char dirs
      'field_arith/ext4/latency_chain/.*_rs_fp4'
      'field_arith/ext4/throughput_stream/.*_rs_fp4'
      'field_arith/ext4/latency_chain/.*_tw_fp4'
      'field_arith/ext4/throughput_stream/.*_tw_fp4'
      'field_arith/ext4/latency_chain/.*_pw_fp4'
      'field_arith/ext4/throughput_stream/.*_pw_fp4'
      'field_arith/ext5/latency_chain'
      'field_arith/ext5/throughput_stream'
    )
    ;;
  ext4)
    CHUNKS=(
      'field_arith/ext4/latency_chain/.*_rs_fp4'
      'field_arith/ext4/throughput_stream/.*_rs_fp4'
      'field_arith/ext4/latency_chain/.*_tw_fp4'
      'field_arith/ext4/throughput_stream/.*_tw_fp4'
      'field_arith/ext4/latency_chain/.*_pw_fp4'
      'field_arith/ext4/throughput_stream/.*_pw_fp4'
    )
    ;;
  p3)
    CHUNKS=(
      'field_arith/base/.*p3'
      'field_arith/ext4/.*p3'
      'field_arith/ext5/.*p3'
    )
    ;;
  square)
    CHUNKS=('packed_square')
    ;;
  *)
    echo "unknown --only mode: $MODE (full|ext4|p3|square)" >&2
    exit 2
    ;;
esac

LOG_DIR="${ROOT}/bench-logs/neon"
mkdir -p "$LOG_DIR"
FAILURES="${LOG_DIR}/failures.txt"
: >"$FAILURES"

run_chunk() {
  local idx=$1
  local filter=$2
  local slug
  slug=$(echo "$filter" | tr '/.*' '__' | tr -cd '[:alnum:]_-')
  local log="${LOG_DIR}/chunk_${idx}_${slug}.log"
  echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') chunk ${idx} filter=${filter} ===" | tee "$log"
  if caffeinate -i cargo +1.95 bench -p akita-pcs --bench field_arith -- \
    --save-baseline neon "$filter" >>"$log" 2>&1; then
    echo "=== chunk ${idx} OK ===" | tee -a "$log"
    return 0
  fi
  echo "=== chunk ${idx} FAILED (exit $?) ===" | tee -a "$log"
  echo "${idx} ${filter}" >>"$FAILURES"
  return 1
}

failed=0
for i in "${!CHUNKS[@]}"; do
  if (( i < FROM )); then
    continue
  fi
  run_chunk "$i" "${CHUNKS[$i]}" || failed=$((failed + 1))
done

if (( failed > 0 )); then
  echo "${failed} chunk(s) failed; see ${FAILURES} and ${LOG_DIR}/chunk_*.log" >&2
  exit 1
fi
echo "=== all ${#CHUNKS[@]} chunk(s) done; collect with scripts/field_microbench_collect.py ==="
