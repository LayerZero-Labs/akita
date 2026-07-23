# Profiling

Operational runbook for the `examples/profile` harness: local timings, Perfetto
traces, and the CI benchmark matrix.

## Canonical command

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 \
  cargo run --release --no-default-features \
  --features parallel,profile-onehot-fp128-d64 --example profile
```

Run from `crates/akita-pcs/`. The harness refuses debug builds unless
`AKITA_ALLOW_DEBUG_PROFILE=1`.

Always use the feature-pruned command above when profiling this path or
measuring its binary size/codegen time. An unpruned default-feature build of
the `profile` example retains every locally supported profile mode; it is a
multi-mode developer artifact, not a like-for-like onehot fp128/D64 binary.
Mixing the two build surfaces can roughly double the example binary and make a
normal release link look like a verifier regression.

## Presets and ring degrees

Under committed-fold A-role SIS pricing, **fp128** production is **D=64**
(signed-sparse; ~20% smaller than D128).
Shipped tables: `fp128_d64_onehot`, `fp128_d64_dense`, `fp128_d128_*`.
**fp128 D=32** is not a valid A-role fold degree (`d_a ≥ 64`); there is no
`D32OneHot` preset.
**fp32/fp64** D32/D64 are not securable; smallest secure choice is **D128
one-hot** (CI benches at `nv=28`).

Compare ring degrees with
`akita_config::proof_optimized::fp128::best_onehot_schedule` /
`best_dense_schedule`.

## Environment knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `AKITA_MODE` | `onehot_fp128_d64` | Preset family and representation |
| `AKITA_NUM_VARS` | `25` in code (`32` in canonical command above) | Witness size |
| `AKITA_NUM_POLYS` | `1` | Batched opening count |
| `AKITA_PROFILE_TRACE` | `1` | Chrome/Perfetto trace output |
| `AKITA_PROFILE_LOG` | `trace` | `tracing` filter |
| `AKITA_PROFILE_ANSI` | `1` | Colored log output |
| `AKITA_PROFILE_SPAN_CLOSES` | `1` | Log span close events |
| `AKITA_PROFILE_COMMIT_ONLY` | unset | Exit after the singleton commitment, skipping prove and verify |
| `AKITA_PROFILE_PROVE_THREADS` | `RAYON_NUM_THREADS` or Rayon default | Global prove pool size (`0` = Rayon default) |
| `AKITA_PROFILE_VERIFY_THREADS` | `RAYON_NUM_THREADS` or Rayon default | Verify pool when it differs from prove (`0` = Rayon default) |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset | Bypass `--release` guard |
| `RAYON_NUM_THREADS` | Rayon default | Fallback when profile thread vars are unset |

Implementation: `crates/akita-pcs/examples/profile/main.rs`.
Disable parallel while retaining the same pruned workload:

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 \
  cargo run --release --no-default-features \
  --features profile-onehot-fp128-d64 --example profile
```

## CI benchmark matrix

Workflow: `.github/workflows/profile-bench.yml`.
CI builds use `--no-default-features --features parallel,profile-ci`.
When adding a bench case, extend the mode→feature table in
`scripts/check_profile_ci_features.sh`.

Committed-fold A-role pricing (every cell folds securely):

| Case | nv | np | Setup mode |
|------|----|----|------------|
| `onehot_fp32_d128` | 28 | 1 | `direct` |
| `onehot_fp64_d128` | 28 | 1 | `direct` |
| `dense_fp128_d64` | 24 | 1 | `direct` |
| `onehot_fp128_d64` | 32 | 1 | `direct` |
| `onehot_fp128_d64` | 32 | 1 | `recursive` |
| `onehot_fp128_d64` | 30 | 4 | `direct` |
| `onehot_fp128_d64_multi_chunk_w2r2` | 32 | 1 | `direct` |
| `onehot_fp128_d64_multi_chunk_w4r2` | 32 | 1 | `direct` |
| `onehot_fp128_d64_multi_chunk_w8r2` | 32 | 1 | `direct` |

fp32/fp64 use `nv=28` because the ext-degree-4 challenge schedule exceeds the 1
GiB `MAX_MATERIALIZED_EQ_TABLE_BYTES` budget at higher `num_vars`.
The multi-chunk row runs in its own parallel CI group. It exercises the
distributed chunked relation shape on a single hosted runner; after the
introducing PR lands, it is compared against merge-base like the other rows.

Report pipeline: `scripts/profile_bench_report.py`.
Coverage matrix spec: `specs/profile-bench-coverage-matrix.md`.

## NTT matvec microbenchmarks

Use the `ntt_matvec` Criterion target to compare the production i8/L8
commitment kernel with the unified i16 kernel independently of proof setup,
transcript work, and planner policy:

```bash
cargo bench -p akita-pcs --bench ntt_matvec -- rank_ring_dim
cargo bench -p akita-pcs --bench ntt_matvec -- width
cargo bench -p akita-pcs --bench ntt_matvec -- equal_output
cargo bench -p akita-pcs --bench ntt_matvec -- equal_io
```

The first group sweeps ring degrees 64, 128, 256, and 512 and output ranks 1,
2, 4, and 8 at width 128. The second sweeps widths 128 through 1024 at D64 and
rank 4. Every shape includes the current i8/L8 prover path and unified i16
L8/L10/L11 paths. Labels state whether the exact i16 path uses only the base
CRT residues or also the optional i16 tail.

The equal-output group compares D64/rank-8, D128/rank-4, D256/rank-2, and
D512/rank-1 at widths 128, 256, 512, and 1024. All four return 512 field
coefficients, but their scalar input sizes differ because each input ring
contains D coefficients. The equal-I/O group compares
D64/rank-8/width-1024, D128/rank-4/width-512, D256/rank-2/width-256, and
D512/rank-1/width-128. Those shapes fix both the input at 65,536 coefficients
and the output at 512 coefficients. Both groups compare i8 and i16 at common
bases L2 through L8 and include i16-only L10 and L11 cases. Criterion uses 10
samples, a 200 ms warmup, and a 1 second measurement window for these large
matrices.

Prepared-cache construction is not timed. The measured work includes digit
validation and transformation, pointwise accumulation, inverse transforms,
CRT reconstruction, and output allocation. Criterion throughput counts
`rank * width * D` coefficient-products. Use a shape filter for quick paired
measurements:

```bash
cargo bench -p akita-pcs --bench ntt_matvec -- d64_r4_w128
```

These are kernel measurements, not protocol timings. Use the profile harness
above for end-to-end proof measurements.

### Interpret ring-degree scaling

Let `n = width * D` be the scalar input dimension and `m = rank * D` be the
scalar output dimension. An unstructured dense matvec costs `O(m * n)`. Akita
represents the matrix as `rank * width` negacyclic ring blocks. With a prepared
matrix, the hot NTT matvec has the approximate per-residue cost

```text
input transforms:  n * log D
pointwise products: m * n / D
output transforms: m * log D
```

The structural saving is the `1 / D` factor in the pointwise term. A ring
column count of `width` is not a scalar width: holding it fixed while raising
D also raises `n`. Use `equal_output` to measure that growing-input scenario.
Use `equal_io` to hold `m` and `n` fixed and expose the actual structure versus
transform tradeoff.

Larger D is useful because it reduces both pointwise work and prepared-matrix
storage. It is not free: transform work grows with `log D`, the exact CRT bound
grows with D, and fewer independent ring rows and columns can reduce
parallelism or cache efficiency. The fastest supported D is therefore a
measured balance, not necessarily the largest available degree.
