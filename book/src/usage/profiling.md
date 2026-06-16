# Profiling

> **Status:** stub. Part of the initial Akita Book scaffold.

Operational runbook for the profile harness: the `examples/profile` binary, the
`AKITA_MODE` / `AKITA_NUM_VARS` / trace knobs, Perfetto traces, the
`profile_bench_report.py` workflow, and the CI benchmark matrix.

**CI benchmark matrix** (`.github/workflows/profile-bench.yml`, committed-fold
A-role pricing, every cell folds securely):

| Case | nv | np |
|------|----|----|
| `onehot_fp32_d128` | 28 | 1 |
| `onehot_fp64_d128` | 28 | 1 |
| `dense_fp128_d64` | 24 | 1 |
| `onehot_fp128_d64` | 32 | 1 |
| `onehot_fp128_d64` | 30 | 4 |

fp32/fp64 use nv=28 because the ext-degree-4 challenge schedule blows the 1 GiB
`MAX_MATERIALIZED_EQ_TABLE_BYTES` budget at higher `num_vars`.

## Sources to fold in

- `crates/akita-pcs/examples/profile/` (`main.rs`, `modes.rs`, `workload.rs`, `report.rs`)
- `AGENTS.md` (Profiling), `scripts/profile_bench_report.py`
- `.github/workflows/profile-bench.yml`, `specs/profile-bench-coverage-matrix.md`
- `crates/akita-pcs/benches/`, `bench-data/`
