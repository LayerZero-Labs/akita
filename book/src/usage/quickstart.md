# Quickstart and configuration

> **Status:** stub. Part of the initial Akita Book scaffold.

The smallest path to a working
`commit(GroupContext::scheduler_without_precommitted_groups())` → `batched_prove` →
`batched_verify`, then how to pick the `CommitmentConfig` preset that matches
your field and proof-size goals.

## Quickstart

Build/test commands, the smallest end-to-end template, and the profile default
a newcomer should reach for first.

**Sources to fold in**

- `crates/akita-pcs/tests/akita_fp128_e2e.rs` (smallest E2E template).
- `AGENTS.md` (Essential Commands); `crates/akita-pcs/examples/profile/main.rs`
  (`AKITA_MODE=onehot_fp128`, `AKITA_NUM_VARS=32`).

## Choosing a configuration

How the `fp31` / `fp32` / `fp63` / `fp64` / `fp128` preset families differ,
when to choose one-hot vs dense, and how ring dimension `D` trades proof size
against prover time and setup memory.

**Paper framing (§3.5 `sec:akita-params`).** The uniform production profile uses
**d=64** with the signed-sparse challenge family. The default direct fp128
one-hot preset now chooses dimensions per fold from generated adaptive tables;
**d=32** remains invalid for the A-role fold degree (`d_a ≥ 64`).

**Proof-size / CI reality (committed-fold A-role SIS pricing).**

| Field | Typical production choice | Notes |
|-------|---------------------------|--------|
| **fp128** | **One-hot** (`fp128::OneHot`) | **Default direct one-hot preset.** Generated schedules choose dimensions for the first two fold levels and use the D64 suffix domain. Direct dense uses `fp128::Dense`; recursive and multi-chunk companions inherit the adaptive policy and have their own generated catalog keys. |
| **fp31** | **Adaptive one-hot** (`fp31::OneHot`) | Uses the prime `2^31 - 19`. It has the fp32 catalog geometry while leaving one carry bit for addition and multiplication kernels. CI benches at **nv=30**. |
| **fp32** | **Adaptive one-hot** (`fp32::OneHot`) | Searches A at D64 through D1024, B/D at D64 through D256, and the monotone D64/D128 suffix domain. CI benches at **nv=30**. |
| **fp63** | **Adaptive one-hot** (`fp63::OneHot`) | Uses the prime `2^63 - 259`. It has the fp64 catalog geometry and permits four unreduced products in a `u128` accumulator. CI benches at **nv=30**. |
| **fp64** | **Adaptive one-hot** (`fp64::OneHot`) | Searches A at D64 through D512, B/D at D64 through D256, and the D64 suffix domain. CI benches at **nv=30**. |

### Why fp63 instead of fp62

The fp63 modulus leaves enough `u128` headroom for four raw base-field
products. The largest coefficient in quadratic extension multiplication needs
three, so fp63 already permits the fused four-product, two-reduction algorithm.
Fp62 would increase the raw-product limit to sixteen without removing another
reduction from this hot path.

The largest admissible fp62 prime is `2^62 - 87`, but it needs quadratic
non-residue 5 and was slower in packed extension multiplication. The largest
fp62 candidate retaining non-residue 2 is `2^62 - 171`; it gave only a small
base arithmetic improvement and no lower extension-field instruction count.
No admissible fp62 candidate enters the one-word Solinas fold. Reaching that
reduction shape requires dropping to 58 bits with `2^58 - 27`.

Use `fp128::OneHot` for direct one-hot and `fp128::Dense` for direct dense.

**Test harness vs profile defaults.** Direct protocol tests should use
`fp128::OneHot` and `fp128::Dense`. Recursive and multi-chunk tests use their
dedicated companion configs and generated catalogs.

**Sources to fold in**

- `crates/akita-config/src/proof_optimized/` and `crates/akita-planner/src/generated_families.rs`.
- `crates/akita-schedules/src/resolve.rs` and `crates/akita-schedules/src/generated/`.
- Paper §3.5 `sec:akita-params`.
- Paper §3.11 `sec:akita-planner` (generated tables + offline DP parity guard).
- `.github/workflows/profile-bench.yml` and [`profiling.md`](profiling.md).
- `AGENTS.md` (Profiling).
