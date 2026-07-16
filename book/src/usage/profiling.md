# Profiling

Operational runbook for the `examples/profile` harness: local timings, Perfetto
traces, and the CI benchmark matrix.

## Canonical command

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile
```

Run from `crates/akita-pcs/`. The harness refuses debug builds unless
`AKITA_ALLOW_DEBUG_PROFILE=1`.

## Presets and ring degrees

Under committed-fold A-role SIS pricing, **fp128** production is **D=64**
(signed-sparse; ~20% smaller than D128).
Shipped tables: `fp128_d64_onehot`, `fp128_d64_full`, `fp128_d128_*`.
**fp128 D=32** is not a valid A-role fold degree (`d_a ≥ 64`); there is no
`D32OneHot` preset.
**fp32/fp64** D32/D64 are not securable; smallest secure choice is **D128
one-hot** (CI benches at `nv=28`).

Compare ring degrees with
`akita_config::proof_optimized::fp128::best_onehot_schedule` /
`best_full_schedule`.

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
| `AKITA_PROFILE_PROVE_THREADS` | `RAYON_NUM_THREADS` or Rayon default | Global prove pool size (`0` = Rayon default) |
| `AKITA_PROFILE_VERIFY_THREADS` | `RAYON_NUM_THREADS` or Rayon default | Verify pool when it differs from prove (`0` = Rayon default) |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset | Bypass `--release` guard |
| `RAYON_NUM_THREADS` | Rayon default | Fallback when profile thread vars are unset |

Implementation: `crates/akita-pcs/examples/profile/main.rs`.
Disable parallel: `cargo run --no-default-features --release --example profile`.

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
GiB `MAX_MATERIALIZED_EQ_TABLE_BYTES` budget at higher `nuposition_index_bits`.
The multi-chunk row runs in its own parallel CI group. It exercises the
distributed chunked relation shape on a single hosted runner; after the
introducing PR lands, it is compared against merge-base like the other rows.

Report pipeline: `scripts/profile_bench_report.py`.
Coverage matrix spec: `specs/profile-bench-coverage-matrix.md`.
