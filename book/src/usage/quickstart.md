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
  (`AKITA_MODE=onehot_fp128`, `AKITA_NUM_VARS=32`).

## Choosing a configuration

How the `fp32` / `fp64` / `fp128` preset families differ, when to choose one-hot
vs dense, and how ring dimension `D` trades proof size against prover time
and setup memory.

**Paper framing (§3.5 `sec:akita-params`).** The uniform production profile uses
**d=64** with the signed-sparse challenge family. The default direct fp128
one-hot preset now chooses dimensions per fold from generated adaptive tables;
**d=32** remains invalid for the A-role fold degree (`d_a ≥ 64`).

**Proof-size / CI reality (committed-fold A-role SIS pricing).**

| Field | Typical production choice | Notes |
|-------|---------------------------|--------|
| **fp128** | **One-hot** (`fp128::OneHot`) | **Default direct one-hot preset.** The generated schedule chooses dimensions for the first two fold levels and uses D64 afterward. Explicit uniform `fp128::D64OneHot` remains available, and Jolt/recursive presets remain pinned to D64. Shipped tables include the default and D64 one-hot families plus D64 dense. |
| **fp32 / fp64** | **D128 one-hot** | D32/D64 are **not securable** under the reprice and unsupported schedules fail fast. CI benches at **nv=28** (eq-table memory budget). Shipped: fp32 D128/D256 onehot; fp64 D128 dense/onehot and D256 onehot. |

Use `akita_config::proof_optimized::fp128::best_onehot_schedule` to compare the
available fp128 one-hot presets for a lookup key. Dense fp128 uses
`fp128::D64Dense` directly.

**Test harness vs profile defaults.** `crates/akita-pcs/tests/common/mod.rs` uses
`fp128::D64OneHot` (one-hot) and `fp128::D64Dense` (dense tests); profile/CI
canonical dense is **`fp128::D64Dense`** at D64.

**Sources to fold in**

- `crates/akita-config/src/proof_optimized/`, `crates/akita-config/src/generated_families.rs`.
- `crates/akita-planner/src/resolve.rs` (`resolve_schedule`) and `crates/akita-schedules/src/generated/`.
- Paper §3.5 `sec:akita-params`.
- Paper §3.11 `sec:akita-planner` (tables + identical DP on miss).
- `.github/workflows/profile-bench.yml` (`AKITA_BENCH_CASES`); `specs/profile-bench-coverage-matrix.md`.
- `AGENTS.md` (Profiling).
