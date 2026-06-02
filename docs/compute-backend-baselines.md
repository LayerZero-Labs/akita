# Compute Backend Baselines

Short-run CPU baselines for the compute-backend cutover.
They are meant to catch large regressions while the backend operation boundary is being introduced.
Rerun with default Criterion settings before treating a sub-2% delta as meaningful.

The canonical x86 capture below is from `leopard` (AMD Ryzen 9 9950X, AVX-512 via `target-cpu=native`).
An older Apple M4 Max capture from 2026-05-24 is kept in the historical appendix for rough comparison only.

## Commands

Run from a clean `cargo +1.95` build on the branch under test.
Pin ISA with `RUSTFLAGS` so numbers stay comparable across machines.

```bash
export RUSTFLAGS='-C target-cpu=native'   # leopard AVX-512; use -C target-cpu=x86-64-v3 for AVX2-only

cargo +1.95 bench -p akita-pcs --bench root_kernels -- --sample-size 10 --warm-up-time 1 --measurement-time 3
cargo +1.95 bench -p akita-pcs --bench ring_ntt -- --sample-size 10 --warm-up-time 1 --measurement-time 2
cargo +1.95 bench -p akita-pcs --bench onehot_root_projection_commit -- --sample-size 10 --warm-up-time 1 --measurement-time 10
cargo +1.95 bench -p akita-pcs --bench extension_opening_reduction -- --sample-size 10 --warm-up-time 1 --measurement-time 10

AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=error AKITA_PROFILE_ANSI=0 \
  AKITA_MODE=onehot_fp128_d32 AKITA_NUM_VARS=32 \
  cargo +1.95 run --release -p akita-pcs --example profile
```

Criterion reports "Gnuplot not found, using plotters backend" on leopard.
`onehot_root_projection_commit` and `extension_opening_reduction` use internal sample/measurement times; the command-line `--measurement-time` flag does not override them.

The removed benches `onehot_batched_commit` and `onehot_batched_opening` are replaced by `onehot_root_projection_commit` and `extension_opening_reduction` respectively.
The profile example no longer accepts bare `AKITA_MODE=onehot`; use an explicit preset such as `onehot_fp128_d32`.

## Leopard (x86_64, AVX-512)

### Environment

- Date captured: 2026-06-02 UTC
- Commit: `e1e00acf95bf7088fc5b72e7460c6c6b84faadc1`
- Host: `leopard` (Linux 6.8.0-111-generic x86_64)
- CPU: AMD Ryzen 9 9950X 16-Core Processor (32 logical CPUs)
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- `RUSTFLAGS`: `-C target-cpu=native`
- Features: workspace defaults
- Raw logs: `/tmp/akita-leopard-compute-baselines-20260602T063659Z/` on leopard

### Root Kernels

Low-level CPU-kernel baselines.
These build CPU NTT state directly and do not measure the backend operation boundary.

| Benchmark | Time interval |
| --- | ---: |
| `root_kernels/dense_root_matvec_full_nv25_d32` | `[1.0100 s 1.0282 s 1.0448 s]` |
| `root_kernels/dense_root_matvec_full_nv25_d32_single_row_subkernel` | `[862.00 ms 871.23 ms 881.72 ms]` |
| `root_kernels/dense_root_predecomp_digit_matvec_full_nv25_d32` | `[1.4906 s 1.4938 s 1.4973 s]` |

### Ring NTT

| Benchmark | Time interval |
| --- | ---: |
| `ring_schoolbook_mul_d64` | `[2.7429 µs 2.7446 µs 2.7471 µs]` |
| `ntt_single_prime_forward_inverse_d64` | `[401.01 ns 401.07 ns 401.19 ns]` |
| `ring_ntt_crt_round_trip_d64_q32_2xi32` | `[1.6695 µs 1.6701 µs 1.6711 µs]` |
| `ring_schoolbook_mul_d32_q128m159` | `[2.8554 µs 2.8558 µs 2.8563 µs]` |
| `ring_partial_split_mul_d32_q128m159` | `[1.2571 µs 1.2573 µs 1.2574 µs]` |
| `ring_crt_ntt_mul_d32_q128m159_k5` | `[3.7692 µs 3.7705 µs 3.7729 µs]` |
| `ring_partial_split_mul_i8_rhs_d32_q128m159` | `[1.2817 µs 1.2820 µs 1.2825 µs]` |
| `ring_crt_ntt_mul_i8_rhs_d32_q128m159_k5` | `[3.2225 µs 3.2249 µs 3.2275 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_scalar/8` | `[4.0464 µs 4.0474 µs 4.0486 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_packed/8` | `[4.2776 µs 4.2782 µs 4.2788 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/crt_simd/8` | `[16.490 µs 16.491 µs 16.493 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_scalar/32` | `[16.005 µs 16.008 µs 16.012 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_packed/32` | `[17.012 µs 17.013 µs 17.014 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/crt_simd/32` | `[65.921 µs 65.942 µs 65.953 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_scalar/128` | `[63.890 µs 63.906 µs 63.917 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_packed/128` | `[67.919 µs 67.925 µs 67.939 µs]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/crt_simd/128` | `[263.91 µs 264.42 µs 265.48 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_scalar/8` | `[4.0531 µs 4.0544 µs 4.0557 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_packed/8` | `[4.2558 µs 4.2564 µs 4.2574 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/crt_simd/8` | `[16.493 µs 16.497 µs 16.509 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_scalar/32` | `[16.035 µs 16.040 µs 16.044 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_packed/32` | `[17.022 µs 17.056 µs 17.084 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/crt_simd/32` | `[65.883 µs 65.893 µs 65.899 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_scalar/128` | `[63.890 µs 63.906 µs 63.917 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_packed/128` | `[67.919 µs 67.925 µs 67.939 µs]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/crt_simd/128` | `[263.63 µs 263.65 µs 263.67 µs]` |
| `ring_partial_split_cyclic_mul_d32_q128m159` | `[1.0196 µs 1.0197 µs 1.0198 µs]` |
| `ring_crt_ntt_cyclic_mul_d32_q128m159_k5` | `[3.7107 µs 3.7117 µs 3.7126 µs]` |
| `ring_partial_split_quotient_d32_q128m159` | `[2.3593 µs 2.3595 µs 2.3597 µs]` |
| `ring_crt_ntt_quotient_d32_q128m159_k5` | `[7.3490 µs 7.3500 µs 7.3512 µs]` |
| `ring_partial_split_cached_matvec_d32_q128m159` | `[26.373 µs 26.378 µs 26.385 µs]` |
| `ring_partial_split_packed_cached_matvec_d32_q128m159` | `[30.594 µs 30.598 µs 30.603 µs]` |
| `ring_crt_ntt_simd_cached_matvec_d32_q128m159_k5` | `[18.338 µs 18.339 µs 18.340 µs]` |
| `ring_partial_split_cached_matvec_i8_rhs_d32_q128m159` | `[26.382 µs 26.384 µs 26.386 µs]` |
| `ring_partial_split_packed_cached_matvec_i8_rhs_d32_q128m159` | `[30.462 µs 30.464 µs 30.468 µs]` |
| `ring_crt_ntt_simd_cached_matvec_i8_rhs_d32_q128m159_k5` | `[18.338 µs 18.341 µs 18.344 µs]` |

Batch-scaling IDs use the total lane count (`8`, `32`, `128`), not the old `/2` labels from the 2026-05-24 capture.

### One-Hot Root Projection Commit

`nv26`, `np4` (bench defaults).

| Benchmark | Time interval |
| --- | ---: |
| `onehot_root_projection_commit/fp32_d64/nv26_np4/project_roots_uncached` | `[14.178 ms 14.194 ms 14.209 ms]` |
| `onehot_root_projection_commit/fp32_d64/nv26_np4/commit_transformed_roots` | `[8.3138 ms 8.3683 ms 8.4161 ms]` |
| `onehot_root_projection_commit/fp32_d64/nv26_np4/scheme_commit_uncached_projection` | `[25.573 ms 25.662 ms 25.753 ms]` |
| `onehot_root_projection_commit/fp64_d32/nv26_np4/project_roots_uncached` | `[13.085 ms 13.090 ms 13.097 ms]` |
| `onehot_root_projection_commit/fp64_d32/nv26_np4/commit_transformed_roots` | `[9.1917 ms 9.2301 ms 9.2703 ms]` |
| `onehot_root_projection_commit/fp64_d32/nv26_np4/scheme_commit_uncached_projection` | `[24.334 ms 24.397 ms 24.472 ms]` |

### Extension Opening Reduction

Sparse tensor-factor sumcheck for one-hot openings (`nv26`, `np4`).

| Benchmark | Time interval |
| --- | ---: |
| `extension_opening_reduction/fp32_d64/onehot_nv26_np4/prove_sumcheck` | `[68.347 ms 68.738 ms 69.296 ms]` |
| `extension_opening_reduction/fp64_d32/onehot_nv26_np4/prove_sumcheck` | `[31.481 ms 31.645 ms 31.860 ms]` |

### Profile Example: `onehot_fp128_d32` nv32

| Metric | Result |
| --- | ---: |
| Setup expand | `1.854626 s` |
| Backend prepare | `0.057278 s` |
| Setup (aggregate) | `1.911906 s` |
| Commit | `0.213751 s` |
| Prove | `0.746483 s` |
| Verify | `0.019012 s` |
| Proof total | `66,064 bytes` |
| Fold bytes | `27,280 bytes` |
| Tail bytes | `38,784 bytes` |
| Levels | `6` |

## Notes For Comparison

- Compare setup expansion, backend preparation, and repeated commit/prove separately when those timing lines are available.
- Treat large one-hot projection or extension-opening changes as high signal.
- Treat small sub-2% Criterion deltas as suspect until rerun with longer default settings.
- If hardware load, Rust version, feature flags, or `RAYON_NUM_THREADS` change, record a new baseline instead of comparing directly.

## Historical Appendix: Apple M4 Max (2026-05-24)

Captured before the bench-target rename and profile-mode cleanup.
Do not compare directly to leopard rows without accounting for architecture and code drift.

### Environment

- Date captured: 2026-05-24 local time
- Commit: `324d14b731d624abfb60fbb8010f4df907f3501f`
- Host: `Darwin Quangs-MacBook-Pro.local 25.5.0 ... RELEASE_ARM64_T6041 arm64`
- CPU: Apple M4 Max (16 logical CPUs)
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Raw local logs: `/tmp/akita-metal-baselines/`

### Root Kernels (M4)

| Benchmark | Time interval |
| --- | ---: |
| `root_kernels/dense_root_matvec_full_nv25_d32` | `[1.9827 s 2.0213 s 2.0570 s]` |
| `root_kernels/dense_root_matvec_full_nv25_d32_single_row_subkernel` | `[1.6988 s 1.7103 s 1.7227 s]` |
| `root_kernels/dense_root_predecomp_digit_matvec_full_nv25_d32` | `[2.2886 s 2.3411 s 2.4018 s]` |

### Ring NTT (M4, selected rows)

| Benchmark | Time interval |
| --- | ---: |
| `ring_schoolbook_mul_d64` | `[4.6332 µs 4.6436 µs 4.6544 µs]` |
| `ntt_single_prime_forward_inverse_d64` | `[274.45 ns 275.65 ns 276.54 ns]` |
| `ring_ntt_crt_round_trip_d64_k6` | `[5.6906 µs 5.7043 µs 5.7348 µs]` |
| `ring_crt_ntt_simd_cached_matvec_d32_q128m159_k5` | `[21.057 µs 21.166 µs 21.286 µs]` |

The M4 capture used `ring_ntt_crt_round_trip_d64_k6` and batch-scaling IDs `/2`, `/8`, `/32`; current benches use `ring_ntt_crt_round_trip_d64_q32_2xi32` and `/8`, `/32`, `/128`.

### One-Hot Batched Commit / Opening (M4, removed benches)

| Benchmark | Time interval |
| --- | ---: |
| `akita/onehot_commit_breakdown/single_full_commit_nv34` | `[4.1214 s 4.1842 s 4.2833 s]` |
| `akita/onehot_opening/single_1xnv34/prove` | `[7.2512 s 7.2702 s 7.2922 s]` |
| `akita/onehot_opening/batched_32xnv29/prove` | `[6.9099 s 6.9328 s 6.9555 s]` |

### Profile Matrix (M4, selected)

Command template:

```bash
AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=error AKITA_PROFILE_ANSI=0 \
  AKITA_MODE=<mode> AKITA_NUM_VARS=<nv> \
  cargo run --release -p akita-pcs --example profile
```

| Mode | Setup | Commit | Prove | Verify | Proof total | Levels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `onehot_fp128_d32` (nv32, no setup split) | `0.753792 s` | `0.183935 s` | `0.631495 s` | `0.022312 s` | `61,300 B` | `7` |
| `onehot_fp32_d32` | `0.985262 s` | `0.547446 s` | `3.335166 s` | `0.051551 s` | `38,352 B` | `6` |
| `dense_fp64_d32` | `0.219134 s` | `2.091091 s` | `1.788981 s` | `0.018076 s` | `41,696 B` | `5` |

Dense fp64 nv26 setup preparation (refreshed on the cutover branch):

| Mode | Setup expand | Backend prepare |
| --- | ---: | ---: |
| `dense_fp64_d32` | `0.158866 s` | `0.060265 s` |

Additional one-hot and dense rows from the same session live under `/tmp/akita-metal-baselines/extra-profiles/`.
