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
(exact-shell; ~20% smaller than D128).
Shipped tables: `fp128_d64_onehot`, `fp128_d64_full`, `fp128_d128_*`.
**D32** presets always use the planner DP (no shipped table).
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
| `AKITA_ALLOW_DEBUG_PROFILE` | unset | Bypass `--release` guard |
| `RAYON_NUM_THREADS` | Rayon default | Cap parallel threads |

Implementation: `crates/akita-pcs/examples/profile/main.rs`.
Disable parallel: `cargo run --no-default-features --release --example profile`.

## CI benchmark matrix

Workflow: `.github/workflows/profile-bench.yml`.
CI builds use `--no-default-features --features parallel,profile-ci`.
When adding a bench case, extend the mode→feature table in
`scripts/check_profile_ci_features.sh`.

Committed-fold A-role pricing (every cell folds securely):

| Case | nv | np |
|------|----|----|
| `onehot_fp32_d128` | 28 | 1 |
| `onehot_fp64_d128` | 28 | 1 |
| `dense_fp128_d64` | 24 | 1 |
| `onehot_fp128_d64` | 32 | 1 |
| `onehot_fp128_d64` | 30 | 4 |

fp32/fp64 use `nv=28` because the ext-degree-4 challenge schedule exceeds the 1
GiB `MAX_MATERIALIZED_EQ_TABLE_BYTES` budget at higher `num_vars`.

Report pipeline: `scripts/profile_bench_report.py`.
Coverage matrix spec: `specs/profile-bench-coverage-matrix.md`.
