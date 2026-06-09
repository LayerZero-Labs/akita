# Profiling

> **Status:** stub. Part of the initial Akita Book scaffold.

Operational runbook for the profile harness: the `examples/profile` binary, the
`AKITA_MODE` / `AKITA_NUM_VARS` / trace knobs, the registered modes, Perfetto
traces, the `profile_bench_report.py` workflow, the Criterion benches, and the
CI benchmark matrix. Include runtime advice (`RAYON_NUM_THREADS`, release guard,
eq-table budget at high `num_vars`).

## Sources to fold in

- `crates/akita-pcs/examples/profile/` (`main.rs`, `modes.rs`, `workload.rs`, `report.rs`)
- `AGENTS.md` (Profiling), `scripts/profile_bench_report.py`
- `specs/profile-bench-coverage-matrix.md` (Active Benchmark Matrix)
- `crates/akita-pcs/benches/`, `bench-data/`
