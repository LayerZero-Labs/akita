# Compute Backend Baselines

These are local CPU baselines for the first compute-backend cutover. They are
short-run baselines meant to catch large regressions while the runtime boundary
is being introduced; rerun with default Criterion settings before treating a
sub-2% delta as meaningful.

## Environment

- Date captured: 2026-05-24 local time
- Commit: `324d14b731d624abfb60fbb8010f4df907f3501f`
- Host: `Darwin Quangs-MacBook-Pro.local 25.5.0 ... RELEASE_ARM64_T6041 arm64`
- CPU: Apple M4 Max
- Logical CPUs: 16
- Memory: 68,719,476,736 bytes
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- Features: workspace defaults
- Raw local logs: `/tmp/akita-metal-baselines/`

## Commands

```bash
cargo bench -p akita-pcs --bench root_kernels -- --sample-size 10 --warm-up-time 1 --measurement-time 3
cargo bench -p akita-pcs --bench ring_ntt -- --sample-size 10 --warm-up-time 1 --measurement-time 2
cargo bench -p akita-pcs --bench onehot_batched_commit -- --sample-size 10 --warm-up-time 1 --measurement-time 2
cargo bench -p akita-pcs --bench onehot_batched_opening -- --sample-size 10 --warm-up-time 1 --measurement-time 2
AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=error AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release -p akita-pcs --example profile
```

Criterion reported "Gnuplot not found, using plotters backend". Some one-hot
benches use their own internal timing configuration; the command-line
measurement-time argument did not prevent those cases from collecting longer
samples.

## Root Kernels

| Benchmark | Time interval |
| --- | ---: |
| `root_kernels/dense_root_matvec_full_nv25_d32` | `[1.9827 s 2.0213 s 2.0570 s]` |
| `root_kernels/dense_root_matvec_full_nv25_d32_single_row_subkernel` | `[1.6988 s 1.7103 s 1.7227 s]` |
| `root_kernels/dense_root_predecomp_digit_matvec_full_nv25_d32` | `[2.2886 s 2.3411 s 2.4018 s]` |

## Ring NTT

| Benchmark | Time interval |
| --- | ---: |
| `ring_schoolbook_mul_d64` | `[4.6332 us 4.6436 us 4.6544 us]` |
| `ntt_single_prime_forward_inverse_d64` | `[274.45 ns 275.65 ns 276.54 ns]` |
| `ring_ntt_crt_round_trip_d64_k6` | `[5.6906 us 5.7043 us 5.7348 us]` |
| `ring_schoolbook_mul_d32_q128m159` | `[2.7027 us 2.7084 us 2.7128 us]` |
| `ring_partial_split_mul_d32_q128m159` | `[1.0120 us 1.0172 us 1.0205 us]` |
| `ring_crt_ntt_mul_d32_q128m159_k5` | `[3.4254 us 3.4333 us 3.4390 us]` |
| `ring_partial_split_mul_i8_rhs_d32_q128m159` | `[1.0224 us 1.0260 us 1.0293 us]` |
| `ring_crt_ntt_mul_i8_rhs_d32_q128m159_k5` | `[3.1237 us 3.1387 us 3.1558 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_scalar/2` | `[972.94 ns 979.70 ns 987.38 ns]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_packed/2` | `[832.50 ns 834.48 ns 837.30 ns]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/crt_simd/2` | `[4.0765 us 4.0897 us 4.1047 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_scalar/8` | `[3.7759 us 3.7904 us 3.8039 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_packed/8` | `[3.2513 us 3.2615 us 3.2693 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/crt_simd/8` | `[16.179 us 16.786 us 18.011 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_scalar/32` | `[14.905 us 15.013 us 15.091 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/split_packed/32` | `[12.976 us 13.026 us 13.097 us]` |
| `ring_cached_mul_batch_scaling_d32_q128m159/crt_simd/32` | `[64.565 us 64.742 us 64.886 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_scalar/2` | `[964.13 ns 970.83 ns 975.14 ns]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_packed/2` | `[829.36 ns 831.36 ns 835.20 ns]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/crt_simd/2` | `[4.0602 us 4.0717 us 4.0876 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_scalar/8` | `[3.7886 us 3.8012 us 3.8101 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_packed/8` | `[3.2490 us 3.2611 us 3.2687 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/crt_simd/8` | `[15.890 us 15.996 us 16.108 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_scalar/32` | `[14.892 us 14.956 us 15.008 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/split_packed/32` | `[12.965 us 13.016 us 13.046 us]` |
| `ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159/crt_simd/32` | `[63.518 us 63.706 us 63.916 us]` |
| `ring_partial_split_cyclic_mul_d32_q128m159` | `[893.64 ns 895.55 ns 898.79 ns]` |
| `ring_crt_ntt_cyclic_mul_d32_q128m159_k5` | `[3.2949 us 3.3044 us 3.3137 us]` |
| `ring_partial_split_quotient_d32_q128m159` | `[1.8936 us 1.9456 us 2.0608 us]` |
| `ring_crt_ntt_quotient_d32_q128m159_k5` | `[6.6767 us 6.9220 us 7.1920 us]` |
| `ring_partial_split_cached_matvec_d32_q128m159` | `[28.843 us 29.462 us 30.516 us]` |
| `ring_partial_split_packed_cached_matvec_d32_q128m159` | `[22.729 us 23.011 us 23.394 us]` |
| `ring_crt_ntt_simd_cached_matvec_d32_q128m159_k5` | `[21.057 us 21.166 us 21.286 us]` |
| `ring_partial_split_cached_matvec_i8_rhs_d32_q128m159` | `[28.971 us 29.157 us 29.365 us]` |
| `ring_partial_split_packed_cached_matvec_i8_rhs_d32_q128m159` | `[23.050 us 23.114 us 23.187 us]` |
| `ring_crt_ntt_simd_cached_matvec_i8_rhs_d32_q128m159_k5` | `[20.942 us 21.089 us 21.386 us]` |

## One-Hot Batched Commit

| Benchmark | Time interval |
| --- | ---: |
| `akita/onehot_commit_breakdown/single_full_commit_nv34` | `[4.1214 s 4.1842 s 4.2833 s]` |
| `akita/onehot_commit_breakdown/single_inner_witness_nv34` | `[4.0607 s 4.1072 s 4.1551 s]` |
| `akita/onehot_commit_breakdown/single_decompose_only_nv34` | `[20.942 ms 21.569 ms 22.166 ms]` |
| `akita/onehot_commit_breakdown/single_outer_only_nv34` | `[19.923 ms 20.252 ms 20.582 ms]` |
| `akita/onehot_commit_breakdown/batched_full_commit_32xnv29` | `[4.1158 s 4.1463 s 4.1742 s]` |
| `akita/onehot_commit_breakdown/batched_inner_witness_32xnv29` | `[4.5607 s 4.6253 s 4.6971 s]` |
| `akita/onehot_commit_breakdown/batched_decompose_only_32xnv29` | `[15.871 ms 16.576 ms 17.338 ms]` |
| `akita/onehot_commit_breakdown/batched_outer_only_32xnv29` | `[13.730 ms 15.526 ms 17.690 ms]` |

## One-Hot Batched Opening

| Benchmark | Time interval |
| --- | ---: |
| `akita/onehot_opening/single_1xnv34/prove` | `[7.2512 s 7.2702 s 7.2922 s]` |
| `akita/onehot_opening/single_1xnv34/verify` | `[58.743 ms 58.865 ms 58.977 ms]` |
| `akita/onehot_opening/batched_32xnv29/prove` | `[6.9099 s 6.9328 s 6.9555 s]` |
| `akita/onehot_opening/batched_32xnv29/verify` | `[49.073 ms 49.209 ms 49.350 ms]` |

## Profile Example

Command:

```bash
AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=error AKITA_PROFILE_ANSI=0 AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release -p akita-pcs --example profile
```

| Metric | Result |
| --- | ---: |
| Setup | `0.753792 s` |
| Commit | `0.183935 s` |
| Prove | `0.631495 s` |
| Verify | `0.022312 s` |
| Proof total | `61,300 bytes` |
| Fold bytes | `29,920 bytes` |
| Tail bytes | `31,380 bytes` |
| Levels | `7` |

## Notes For Comparison

- These numbers include current setup-owned CPU NTT cache behavior. After the
  cutover, compare setup preparation separately from repeated commit/prove
  execution.
- Treat large one-hot commit/opening changes as high signal. Treat small
  sub-2% Criterion deltas as suspect until rerun with longer default settings.
- If a later run changes hardware load, Rust version, feature flags, or
  `RAYON_NUM_THREADS`, record a new baseline instead of comparing directly.
