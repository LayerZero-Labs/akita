# Quickstart and configuration

> **Status:** stub. Part of the initial Akita Book scaffold.

The smallest path to a working `batched_commit` → `batched_prove` →
`batched_verify`, then how to pick the `CommitmentConfig` preset that matches
your field and proof-size goals.

## Quickstart

Build/test commands, the smallest end-to-end template, and the profile default
a newcomer should reach for first.

**Sources to fold in**

- `crates/akita-pcs/tests/single_poly_e2e.rs` (smallest E2E template).
- `AGENTS.md` (Essential Commands); `crates/akita-pcs/examples/profile/main.rs`
  (`AKITA_MODE=onehot_fp128_d64`, `AKITA_NUM_VARS=32`).

## Choosing a configuration

How the `fp32` / `fp64` / `fp128` preset families differ, when to choose one-hot
vs full (dense), and how ring dimension `D` trades proof size against prover time
and setup memory.

**Paper framing (§3.5 `sec:akita-params`).** Production uses **d=64** with the
exact-shell challenge family. **d=32** and **d=128** are alternate profiles with
different challenge families; the planner prices the tradeoff per workload but
does **not** retune `D` per recursion level (pick a preset, then plan).

**Proof-size / CI reality (committed-fold A-role SIS pricing).**

| Field | Typical production choice | Notes |
|-------|---------------------------|--------|
| **fp128** | **D64 one-hot** (`onehot_fp128_d64`) | **Production default** (Paper §3.5 exact-shell at d=64). Planner picks **D64 over D128** (~20% smaller proof); both fold securely. Shipped tables: D128 full/onehot, D64 full/onehot. **D32** can win marginally on bytes but has **no shipped table** (always DP) and uses bounded-L₁ challenges; Jolt cycle benches pin **D32OneHot**. |
| **fp32 / fp64** | **D128 one-hot** | D32/D64 are **not securable** under the reprice (collapse to cleartext root-direct). CI benches at **nv=28** (eq-table memory budget). Shipped: fp32 D128/D256 onehot; fp64 D128 full/onehot and D256 onehot. |

Use `akita_config::proof_optimized::fp128::best_onehot_schedule` /
`best_full_schedule` to compare fp128 D32/D64/D128 for a lookup key. Every preset
falls back to the verifier-reachable DP on table miss.

**Test harness vs profile defaults.** `crates/akita-pcs/tests/common/mod.rs` uses
`fp128::D64OneHot` (one-hot) and `fp128::D128Full` (dense tests); profile/CI
canonical dense is **`fp128::D64Full`** at D64.

**Sources to fold in**

- `crates/akita-config/src/proof_optimized/`, `crates/akita-config/src/generated_families.rs`.
- `crates/akita-planner/src/resolve.rs` (`shipped_table`).
- Paper §3.5 `sec:akita-params`; App B `sec:akita-d32-profile` (d=32 sampler).
- Paper §3.11 `sec:akita-planner` (tables + identical DP on miss).
- `.github/workflows/profile-bench.yml` (`AKITA_BENCH_CASES`); `specs/profile-bench-coverage-matrix.md`.
- `AGENTS.md` (Profiling).
