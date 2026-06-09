# Quickstart and configuration

> **Status:** stub. Part of the initial Akita Book scaffold.

The smallest path to a working commit → prove → verify, then how to pick the
configuration that matches your field and proof-size goals.

## Quickstart

Build/test commands, the smallest end-to-end template, and the default presets
a newcomer should reach for first.

**Sources to fold in**

- `crates/akita-pcs/tests/single_poly_e2e.rs` (smallest E2E template).
- `crates/akita-pcs/tests/common/mod.rs` (default presets: `fp128::D64OneHot`, `fp128::D128Full`).
- Council usage report: concrete Quickstart outline (build/test/profile commands).

## Choosing a configuration

How the `fp32` / `fp64` / `fp128` preset families differ, when to choose
one-hot vs full (dense), and how `D` (ring dimension) trades proof size against
prover time. Under committed-fold A-role pricing the planner's `total_bytes`
optimum is D=32 or D=64; D128 is resolved via runtime DP only.

**Sources to fold in**

- `crates/akita-config/src/proof_optimized/fp128.rs`, `fp32.rs`, `fp64.rs`.
- `crates/akita-config/src/lib.rs` (`CommitmentConfig`).
- Paper §3.7 `sec:akita-params` (ring dimension, why D=32) and the planner §3.10.
- `AGENTS.md` (Profiling), `specs/profile-bench-coverage-matrix.md`.
