# Compute Backend Baselines

> **Historical snapshot.** Frozen local baselines from the compute-backend
> cutover (dated host/commit; some referenced benches no longer exist). For live
> numbers use the profile harness and CI matrix (Akita Book
> `book/src/usage/profiling.md` and `specs/profile-bench-coverage-matrix.md`).
> Scheduled to move to `docs/archive/` (see `specs/PRUNING.md`).

These are local CPU baselines for the first compute-backend cutover. They are
short-run baselines meant to catch large regressions while the backend
operation boundary is being introduced; rerun with default Criterion settings
before treating a sub-2% delta as meaningful.

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

These are low-level CPU-kernel baselines. They intentionally build CPU NTT
state directly and do not measure the new backend operation boundary.

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
| `ring_crt_ntt_mul_d32_q128m159_k5` | `[3.4254 us 3.4333 us 3.4390 us]` |
| `ring_crt_ntt_mul_i8_rhs_d32_q128m159_k5` | `[3.1237 us 3.1387 us 3.1558 us]` |
| `ring_crt_ntt_cyclic_mul_d32_q128m159_k5` | `[3.2949 us 3.3044 us 3.3137 us]` |
| `ring_crt_ntt_quotient_d32_q128m159_k5` | `[6.6767 us 6.9220 us 7.1920 us]` |
| `ring_crt_ntt_simd_cached_matvec_d32_q128m159_k5` | `[21.057 us 21.166 us 21.286 us]` |
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

## Profile Example: One-Hot nv32 fp128

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

After the compute-backend cutover, the profile example also emits
`setup_expand` and `backend_prepare` timing rows before the aggregate `setup`
row. The baseline above predates that split and should be refreshed before
using setup preparation as a regression gate.

## Additional Profile Matrix

Additional profiles were initially captured from commit `223d32fa`
(`docs(compute): record cutover prep`), which is a docs-only descendant of the
original baseline commit. The dense fp64 nv26 rows were refreshed on this
compute-backend cutover branch after removing the misplaced `EqPolynomial`
table cap and adding the explicit backend preparation split. Raw logs are under
`/tmp/akita-metal-baselines/extra-profiles/`; refreshed fp64 rows are from the
terminal reruns recorded in `WORKLOG-NEVER-COMMIT.md`.

Command template:

```bash
AKITA_PROFILE_TRACE=0 AKITA_PROFILE_LOG=error AKITA_PROFILE_ANSI=0 AKITA_MODE=<mode> AKITA_NUM_VARS=<nv> cargo run --release -p akita-pcs --example profile
```

### One-Hot nv32 Small-Field Modes

| Mode | Setup | Commit | Prove | Verify | Proof total | Fold bytes | Tail bytes | Levels | Claim/challenge ext degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `onehot_fp32_d32` | `0.985262 s` | `0.547446 s` | `3.335166 s` | `0.051551 s` | `38,352 B` | `19,472 B` | `18,880 B` | `6` | `4/4` |
| `onehot_fp32_d64` | `0.294832 s` | `0.569073 s` | `3.056431 s` | `0.037627 s` | `43,248 B` | `20,624 B` | `22,624 B` | `6` | `4/4` |
| `onehot_fp64_d32` | `0.440407 s` | `0.409220 s` | `2.144096 s` | `0.037579 s` | `43,248 B` | `20,624 B` | `22,624 B` | `6` | `2/2` |
| `onehot_fp64_d64` | `0.195142 s` | `0.385249 s` | `2.191310 s` | `0.040556 s` | `54,528 B` | `23,776 B` | `30,752 B` | `6` | `2/2` |

### Dense nv26 Modes

| Field / mode | Setup | Commit | Prove | Verify | Proof total | Fold bytes | Tail bytes | Levels | Claim/challenge ext degree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp128 `full_d32` | `0.753363 s` | `5.920908 s` | `4.434486 s` | `0.019827 s` | `61,300 B` | `29,920 B` | `31,380 B` | `7` | `1/1` |
| fp128 `dense_d64` | `0.110167 s` | `3.208263 s` | `3.498260 s` | `0.016489 s` | `71,688 B` | `32,048 B` | `39,640 B` | `6` | `1/1` |
| fp32 `dense_fp32_d32` | `0.244446 s` | `6.687460 s` | `1.579170 s` | `0.021254 s` | `37,600 B` | `19,040 B` | `18,560 B` | `6` | `4/4` |
| fp32 `dense_fp32_d64` | `0.111561 s` | `1.023532 s` | `1.044095 s` | `0.016854 s` | `41,008 B` | `19,360 B` | `21,648 B` | `5` | `4/4` |
| fp64 `dense_fp64_d32` | `0.219134 s` | `2.091091 s` | `1.788981 s` | `0.018076 s` | `41,696 B` | `19,760 B` | `21,936 B` | `5` | `2/2` |
| fp64 `dense_fp64_d64` | `0.200086 s` | `2.503449 s` | `2.146537 s` | `0.027658 s` | `52,400 B` | `22,848 B` | `29,552 B` | `6` | `2/2` |

The refreshed dense fp64 rows also report setup preparation split out from the
aggregate setup time:

| Mode | Setup expand | Backend prepare |
| --- | ---: | ---: |
| `dense_fp64_d32` | `0.158866 s` | `0.060265 s` |
| `dense_fp64_d64` | `0.147241 s` | `0.052843 s` |

Earlier dense fp64 nv26 runs failed with
`InvalidSize { expected: 16777216, actual: 33554432 }` after the commit phase.
That failure was the misplaced algebra table cap now removed in this PR, not an
invalid dense fp64 profile shape.

## Notes For Comparison

- Rows captured before the compute-backend cutover include setup-owned CPU NTT
  cache behavior. Rows captured after the cutover should compare setup
  expansion, backend preparation, and repeated commit/prove execution
  separately when those timing lines are available.
- Treat large one-hot commit/opening changes as high signal. Treat small
  sub-2% Criterion deltas as suspect until rerun with longer default settings.
- If a later run changes hardware load, Rust version, feature flags, or
  `RAYON_NUM_THREADS`, record a new baseline instead of comparing directly.
