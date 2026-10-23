#!/usr/bin/env bash
# Reproduce the final matched-thread point; optional first argument overrides thread count.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export RUSTFLAGS='-C target-cpu=native' RUST_MIN_STACK=67108864
cargo build -p akita-pcs --example profile --release --no-default-features --features parallel,profile-onehot-fp128,transcript-blake2b
akita_repro_threads="${1:-16}"
export RAYON_NUM_THREADS="$akita_repro_threads" AKITA_PROFILE_PROVE_THREADS="$akita_repro_threads" AKITA_PROFILE_VERIFY_THREADS="$akita_repro_threads"
export AKITA_MODE=onehot_fp128 AKITA_NUM_VARS=38 AKITA_SETUP_MODE=recursive AKITA_PROFILE_K16=1
export AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=info AKITA_PROFILE_SPAN_CLOSES=0 AKITA_PROFILE_ANSI=0
export AKITA_PROFILE_COMPARE_ONEHOT=1 AKITA_PROFILE_COMPARE_ORDER='accumulation,ntt-exact,best,accumulation,best'
unset AKITA_PROFILE_ONEHOT_FILE AKITA_SCALAR_NTT AKITA_ONEHOT_NTT_PREWARM_EXACT
unset AKITA_AVX512_NTT
akita_repro_stamp=$(date -u +%Y%m%dT%H%M%SZ)
akita_repro_results="$PWD/experiment-results/reproduce-mac-$akita_repro_stamp"
mkdir -p "$akita_repro_results"
cp profile/onehot-ntt/analyze.py "$akita_repro_results/analyze.py"
export AKITA_PROFILE_SAVE_BYTES="$akita_repro_results/nv38"
target/release/examples/profile 2>&1 | tee "$akita_repro_results/nv38.log"
python3 "$akita_repro_results/analyze.py" verify "$akita_repro_results" 38
printf 'Results: %s\n' "$akita_repro_results"
