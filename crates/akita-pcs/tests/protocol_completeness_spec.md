# Akita Protocol Completeness Tests — Spec

## Branch

`feat/akita-e2e-correctness-matrix`

---

## 1. Target file

**`crates/akita-pcs/tests/protocol_completeness.rs`**

This file is the single home for every test whose primary assertion is
"an honest prove→verify cycle succeeds." Tests that primarily check rejection, structural
invariants, commit-without-prove, or scheduling logic stay where they are.

---

## 2. Tests to remove from other files

The following tests are correctness round-trips that currently live elsewhere.
They will be deleted from their current locations and replaced by the parameterised
tests in `protocol_completeness.rs`.

### `tests/single_poly_e2e.rs` — **delete entire file**

All 8 tests are pure prove→verify round-trips. The file has no other content.

| Removed test | Config | nv |
|---|---|---|
| `single_onehot_nv12` | `fp128::OneHot` | 12 |
| `single_onehot_nv15` | `fp128::OneHot` | 15 |
| `single_onehot_nv20` | `fp128::OneHot` | 20 |
| `single_dense_nv14` | `fp128::Dense` | 14 |
| `single_dense_nv16` | `fp128::Dense` | 16 |
| `single_dense_nv24` | `fp128::Dense` | 24 |
| `single_onehot_oversized_setup_15_12` | `fp128::OneHot` | setup=15, poly=12 |
| `single_onehot_oversized_setup_20_15` | `fp128::OneHot` | setup=20, poly=15 |

> The two oversized-setup cases expose a specific behaviour (setup capacity ≥ poly nv).
> They become dedicated tests in `protocol_completeness.rs` under `fp128_onehot_oversized_setup`.

### `tests/batched_aggregated_e2e.rs` — **delete entire file**

All 5 tests are prove→verify round-trips for the aggregated (batched commit) path.

| Removed test | Config | nv | batch |
|---|---|---|---|
| `aggregated_onehot_nv12_batch1` | `fp128::OneHot` | 12 | 1 |
| `aggregated_onehot_nv20_batch4` | `fp128::OneHot` | 20 | 4 |
| `aggregated_dense_nv14_batch1` | `fp128::Dense` | 14 | 1 |
| `aggregated_dense_nv17_batch4` | `fp128::Dense` | 17 | 4 |
| `aggregated_mixed_dense_and_onehot_under_dense_cfg` | `fp128::Dense` | 17 | 4 (mixed) |

> Aggregated/batched-commit cases are added as a dedicated driver in `protocol_completeness.rs`.

### `tests/heterogeneous_prove_e2e.rs` — **delete entire file**

| Removed test | Config | nv |
|---|---|---|
| `heterogeneous_delegating_clusters_batched_prove_and_verify` | `fp128::Dense` | 16 |

### `src/scheme/tests/fp32_ext4.rs` — **delete entire file**

Both tests are correctness round-trips. The file has no non-correctness content.

| Removed test | Config | nv |
|---|---|---|
| `fp32_ext4_folded_eor_batched_roundtrip_and_rejections` | `fp32::OneHot` | 16, batch=2 |
| `fp32_ext4_multi_group_uses_one_batched_eor_sumcheck` | `fp32::OneHot` | pre=14, final=20 |

### `src/scheme/tests/heterogeneous_group.rs` — **delete entire file**

| Removed test | Config | nv |
|---|---|---|
| `heterogeneous_polynomial_groups_round_trip_with_group_local_points` | `fp128::OneHot` + `fp128::Dense` | pre=14/15, final=16 |

### `src/scheme/tests/single.rs` — **remove 3 tests, keep 4**

| Action | Test | Reason |
|---|---|---|
| Remove | `verify_passes_for_consistent_opening` | pure correctness |
| Remove | `fp128_degree_one_batched_proof_roundtrip_is_stable` | pure correctness + serialization stability |
| Remove | `monomial_basis_prove_verify_round_trip` | pure correctness (monomial basis) |
| **Keep** | `verify_rejects_wrong_opening` | tamper rejection |
| **Keep** | `verify_rejects_malformed_v_dimension_without_panicking` | error path |
| **Keep** | `folded_payload_commitments_and_digits_stay_base_field` | structural type check |
| **Keep** | `folded_root_rejects_unchecked_extension_opening_reduction_payload` | tamper rejection |

### `src/scheme/tests/batched.rs` — **remove 1 test, keep 1**

| Action | Test | Reason |
|---|---|---|
| Remove | `batched_verify_accepts_consistent_openings_and_rejects_bad_inputs` | primary assertion is honest verify passes |
| **Keep** | `batched_commit_matches_individual_commits` | commit-only, no prove/verify |

### `src/scheme/tests/onehot.rs` — **remove 5 tests, keep 5**

| Action | Test | Reason |
|---|---|---|
| Remove | `multi_group_root_folded_group_binding_round_trips` | prove→verify |
| Remove | `multi_group_root_allows_precommitted_arity_above_final_group` | prove→verify |
| Remove | `multi_group_root_opens_multi_polynomial_precommitted_group` | prove→verify |
| Remove | `multi_group_multi_chunk_fold_round_trips` | prove→verify |
| Remove | `batched_onehot_roundtrip_matches_public_shape_context` | prove→verify |
| **Keep** | `profile_native_commit_group_returns_exact_frozen_layout` | commit-only, layout check |
| **Keep** | `profile_native_commit_group_allows_independent_groups` | commit-only, layout check |
| **Keep** | `group_batch_schedule_preserves_precommitted_order` | scheduling structural check |
| **Keep** | `group_batch_commits_independent_arity_precommitteds` | commit-only, structural check |
| **Keep** | `commit_group_returns_frozen_exact_layout` | commit-only, layout check |

### `tests/akita_e2e.rs` — **remove 6 tests, keep 6**

| Action | Test | Reason |
|---|---|---|
| Remove | `adaptive_dense_prove_verify` | pure correctness |
| Remove | `adaptive_dense_generated_prove_verify_nv24` | pure correctness |
| Remove | `chunked_multi_chunk_prove_verify` | pure correctness |
| Remove | `adaptive_onehot_direct_tail_uses_terminal_schedule_basis` | pure correctness |
| Remove | `batched_onehot_same_point_round_trip` | pure correctness |
| Remove | `adaptive_dense_mixed_basis_roundtrip_and_serialization` | pure correctness + serialization |
| **Keep** | `trace_internalization_rejects_tampered_root_fold_handle` | tamper rejection |
| **Keep** | `trace_internalization_rejects_tampered_recursive_fold_handle` | tamper rejection |
| **Keep** | `trace_internalization_rejects_tampered_terminal_e_hat_digit` | tamper rejection |
| **Keep** | `small_field_dense_uncataloged_roots_fail_fast` | error path |
| **Keep** | `adaptive_dense_tiny_roots_and_setup_capacities_are_rejected` | error path |
| **Keep** | `batched_onehot_same_point_rejects_tampered_root_stage1_range_image_evaluation` | tamper rejection |

> `akita_e2e.rs` retains its helper functions (`make_dense_fixture`, tamper helpers, etc.) since
> the kept tests still use them.

---

## 3. Catalog-derived nv values

Each test only accepts nv values present in the generated schedule catalog.
Values were read directly from the catalog source files.

| Catalog | Valid nv (final group) |
|---|---|
| `fp128_dense` | 14, 16, 24, 26, 28, 30, 32, 44, 50 |
| `fp128_onehot` | 12, 14, 15, 16, 18, 20, 28, 30, 32, 36, 40, 44, 50 |
| `fp128_dense_multi_chunk` | 16 |
| `fp128_onehot_multi_chunk` | 32 |
| `fp128_dense_precommitted` | precommit profiles at nv = 14, 15, 16 |
| `fp128_onehot_precommitted` | precommit profiles at nv = 14, 15, 16, 20 |
| `fp128_onehot_recursive` | final = 32 (2-poly), precommits = 2 × nv=16 |
| `fp128_onehot_recursive_multi_chunk_w8r2` | final = 32 (2-poly), precommits = 2 × nv=16 |
| `fp64_dense` | 14, 20, 26 |
| `fp64_onehot` | 28, 30 |
| `fp32_dense` | 20, 26 |
| `fp32_onehot` | 14, 16, 20, 28, 30 |

---

## 4. The complete test list

Tests are grouped by driver type. Each non-recursive test receives `nv_values: &[usize]`
and loops. Production-sized nv that require `--release` are split into a separate `#[ignore]` fn.

### Group A — single-chunk, no precommit

| # | Function | Config | Normal nv | `#[ignore]` nv | Sourced from |
|---|---|---|---|---|---|
| 1 | `fp128_dense` | `fp128::Dense` | [14, 16, 24] | — | `single_poly_e2e`, `akita_e2e` |
| 2 | `fp128_onehot` | `fp128::OneHot` | [12, 15, 20] | — | `single_poly_e2e`, `akita_e2e` |
| 3 | `fp64_dense` | `fp64::Dense` | [14, 20] | — | new |
| 4 | `fp64_onehot` | `fp64::OneHot` | — | [28, 30] | new |
| 5 | `fp32_dense` | `fp32::Dense` | — | [20, 26] | new |
| 6 | `fp32_onehot` | `fp32::OneHot` | [14, 16, 20] | — | `fp32_ext4.rs` (scheme-level) |
| 6b | `fp32_onehot_large` | `fp32::OneHot` | — | [28, 30] | new |

### Group B — multi-chunk, no precommit

| # | Function | Config | Normal nv | `#[ignore]` nv | Sourced from |
|---|---|---|---|---|---|
| 7 | `fp128_dense_multi_chunk` | `fp128::DenseMultiChunk` | [16] | — | `akita_e2e` |
| 8 | `fp128_onehot_multi_chunk` | `fp128::OneHotMultiChunk` | — | [32] | new |

### Group C — precommitted (non-recursive)

The nv list is the **final group** nv. The precommitted group nv is resolved internally.

| # | Function | Config | Final nv (normal) | `#[ignore]` | Sourced from |
|---|---|---|---|---|---|
| 9 | `fp128_dense_precommitted` | `fp128::Dense` | [14, 16] | — | new |
| 10 | `fp128_onehot_precommitted` | `fp128::OneHot` | [15, 16, 20] | — | `onehot.rs` (scheme-level) |
| 11 | `fp128_onehot_multi_chunk_precommitted` | `fp128::OneHotMultiChunk` | — | [32] | new |

### Group D — aggregated (batched commit)

| # | Function | Config | nv | batch sizes | Sourced from |
|---|---|---|---|---|---|
| 12 | `fp128_dense_aggregated` | `fp128::Dense` | [14, 17] | [1, 4] | `batched_aggregated_e2e` |
| 13 | `fp128_onehot_aggregated` | `fp128::OneHot` | [12, 20] | [1, 4] | `batched_aggregated_e2e` |
| 14 | `fp128_mixed_aggregated` | `fp128::Dense` | [17] | [4 mixed] | `batched_aggregated_e2e` |

### Group E — special configurations

| # | Function | What it tests | Sourced from |
|---|---|---|---|
| 15 | `fp128_onehot_oversized_setup` | Setup capacity ≥ poly nv (setup=15, poly=12 and setup=20, poly=15) | `single_poly_e2e` |
| 16 | `fp128_dense_monomial_basis` | Prove→verify in monomial basis instead of Lagrange | `scheme/tests/single.rs` |
| 17 | `fp128_onehot_multi_group_precommitted` | Multi-group: 1 precommit + 1 final, same-field | `scheme/tests/onehot.rs` |
| 18 | `fp128_onehot_precommit_arity_above_final` | Pre-committed nv > final group nv | `scheme/tests/onehot.rs` |
| 19 | `fp128_onehot_multi_poly_precommitted_group` | Precommitted group with 2 polynomials | `scheme/tests/onehot.rs` |
| 20 | `fp128_onehot_multi_chunk_multi_group` | Multi-chunk + precommitted group | `scheme/tests/onehot.rs` |
| 21 | `fp32_onehot_multi_group` | fp32 extension-field opening, multi-group | `scheme/tests/fp32_ext4.rs` |
| 22 | `heterogeneous_group_types` | Mixed OneHot + Dense groups in one proof | `scheme/tests/heterogeneous_group.rs` |
| 23 | `heterogeneous_compute_backends` | Four delegating cluster backends | `heterogeneous_prove_e2e.rs` |

### Group F — recursive (fixed geometry, no nv list, always `#[ignore]`)

| # | Function | Base config | Geometry | Sourced from |
|---|---|---|---|---|
| 24 | `fp128_onehot_recursive` | `fp128::OneHot` | final=32 (2-poly) + 2×nv=16 precommit | `recursive_setup_e2e` (non-ignored copy) |
| 25 | `fp128_onehot_recursive_multi_chunk` | `fp128::OneHotMultiChunk` | final=32 (2-poly) + 2×nv=16 precommit | new |

---

## 5. Code reduction strategy

### What `common/mod.rs` already provides

| Helper | Used for |
|---|---|
| `prove_input`, `verify_input`, `selected_prover_data`, `selected_statement` | claim construction |
| `make_dense_poly(nv, seed)`, `make_onehot_poly(nv, seed)` | poly generation (fp128 only) |
| `opening_from_poly_for_layout(poly, point, layout)` | MLE evaluation (fp128 only) |
| `init_rayon_pool()`, `run_on_large_stack(f)` | thread setup |
| `recursive_multi_group_round_trip::<BaseCfg>(label, cb)` | full recursive cycle |

### New additions to `common/mod.rs`

| Addition | Purpose |
|---|---|
| `prove_verify_dense_roundtrip::<Cfg>(nv_values, label)` | Groups A+B dense driver |
| `prove_verify_onehot_roundtrip::<Cfg>(nv_values, k, label)` | Groups A+B onehot driver |
| `prove_verify_dense_precommitted_roundtrip::<Cfg>(final_nvs, label)` | Group C dense driver |
| `prove_verify_onehot_precommitted_roundtrip::<Cfg>(final_nvs, k, label)` | Group C onehot driver |
| `prove_verify_aggregated_roundtrip::<Cfg>(cases, label)` | Group D driver |
| `generic_opening_from_poly` | field-generic MLE eval for fp32/fp64 drivers |

The recursive case uses the already-existing `recursive_multi_group_round_trip` — no new driver.

### `matrix_test!` macro arms

```rust
macro_rules! matrix_test {
    (dense; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]) => { ... };
    (onehot; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]; k=$k:expr) => { ... };
    (dense_precommitted; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]) => { ... };
    (onehot_precommitted; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]; k=$k:expr) => { ... };
    (recursive; $name:ident; $base_cfg:ty) => { ... };  // always #[ignore]
}
```

Groups D and E (aggregated, special) are plain `#[test]` functions — they are one-of-a-kind and
benefit more from clarity than from macro terseness.

---

## 6. How the final file looks

```rust
// crates/akita-pcs/tests/protocol_completeness.rs
//
// Single home for all Akita prove→verify correctness tests.
// Every supported (field × poly-type × nv × chunk-mode × precommit × setup-mode)
// combination must produce a proof the verifier accepts.
//
// Normal run (fast):
//   cargo test -p akita-pcs --test protocol_completeness --features schedules-default
//
// Full run including production-sized ignored tests:
//   cargo test --release -p akita-pcs --test protocol_completeness \
//     --features schedules-default -- --ignored

mod common;
use common::*;
use akita_config::proof_optimized::{fp32, fp64, fp128};

macro_rules! matrix_test {
    (dense; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_dense_roundtrip::<$cfg>(
                    &[$($nv),+],
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    (onehot; $name:ident; $cfg:ty; nvs=[$($nv:expr),+]; k=$k:expr) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_onehot_roundtrip::<$cfg>(
                    &[$($nv),+],
                    $k,
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    (dense_precommitted; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_dense_precommitted_roundtrip::<$cfg>(
                    &[$($nv),+],
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    (onehot_precommitted; $name:ident; $cfg:ty; final_nvs=[$($nv:expr),+]; k=$k:expr) => {
        #[test]
        fn $name() {
            init_rayon_pool();
            run_on_large_stack(|| {
                prove_verify_onehot_precommitted_roundtrip::<$cfg>(
                    &[$($nv),+],
                    $k,
                    concat!("completeness/", stringify!($name)).as_bytes(),
                );
            });
        }
    };
    (recursive; $name:ident; $base_cfg:ty) => {
        #[test]
        #[ignore = "production-sized; run explicitly with --release"]
        fn $name() {
            recursive_multi_group_round_trip::<$base_cfg>(
                concat!("completeness/", stringify!($name)).as_bytes(),
                |_| {},
            );
        }
    };
}

// ── Group A: single-chunk, no precommit ───────────────────────────────────

#[cfg(feature = "schedules-default")]
matrix_test!(dense;  fp128_dense;  fp128::Dense;  nvs=[14, 16, 24]);

#[cfg(feature = "schedules-default")]
matrix_test!(onehot; fp128_onehot; fp128::OneHot; nvs=[12, 15, 20]; k=256);

#[cfg(feature = "schedules-fp64-dense")]
matrix_test!(dense;  fp64_dense;   fp64::Dense;   nvs=[14, 20]);

#[cfg(feature = "schedules-fp64-onehot")]
matrix_test!(onehot; fp64_onehot;  fp64::OneHot;  nvs=[28, 30]; k=256);  // always ignored inside driver

#[cfg(feature = "schedules-fp32-dense")]
matrix_test!(dense;  fp32_dense;   fp32::Dense;   nvs=[20, 26]);          // always ignored inside driver

#[cfg(feature = "schedules-fp32-onehot")]
matrix_test!(onehot; fp32_onehot;       fp32::OneHot; nvs=[14, 16, 20]; k=256);

#[cfg(feature = "schedules-fp32-onehot")]
matrix_test!(onehot; fp32_onehot_large; fp32::OneHot; nvs=[28, 30];     k=256);  // ignored

// ── Group B: multi-chunk, no precommit ────────────────────────────────────

#[cfg(feature = "schedules-fp128-dense-multi-chunk")]
matrix_test!(dense;  fp128_dense_multi_chunk;  fp128::DenseMultiChunk;  nvs=[16]);

#[cfg(feature = "schedules-fp128-onehot-multi-chunk")]
matrix_test!(onehot; fp128_onehot_multi_chunk; fp128::OneHotMultiChunk; nvs=[32]; k=256); // ignored

// ── Group C: precommitted, non-recursive ──────────────────────────────────

#[cfg(feature = "schedules-fp128-dense-precommitted")]
matrix_test!(dense_precommitted;  fp128_dense_precommitted;  fp128::Dense;  final_nvs=[14, 16]);

#[cfg(feature = "schedules-fp128-onehot-precommitted")]
matrix_test!(onehot_precommitted; fp128_onehot_precommitted; fp128::OneHot; final_nvs=[15, 16, 20]; k=256);

#[cfg(feature = "schedules-fp128-onehot-multi-chunk-precommitted")]
matrix_test!(onehot_precommitted; fp128_onehot_multi_chunk_precommitted; fp128::OneHotMultiChunk; final_nvs=[32]; k=256); // ignored

// ── Group D: aggregated (batched commit) ──────────────────────────────────

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_dense_aggregated() { ... }

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_onehot_aggregated() { ... }

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_mixed_aggregated() { ... }

// ── Group E: special configurations ──────────────────────────────────────

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_onehot_oversized_setup() { ... }       // setup nv > poly nv

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_dense_monomial_basis() { ... }         // BasisMode::Monomial

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_onehot_multi_group_precommitted() { ... }

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_onehot_precommit_arity_above_final() { ... }

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_onehot_multi_poly_precommitted_group() { ... }

#[cfg(feature = "schedules-fp128-onehot-multi-chunk")]
#[test]
fn fp128_onehot_multi_chunk_multi_group() { ... }

#[cfg(feature = "schedules-fp32-onehot")]
#[test]
fn fp32_onehot_multi_group() { ... }            // fp32 extension-field opening, multi-group

#[test]
fn heterogeneous_group_types() { ... }          // mixed OneHot + Dense groups

#[test]
fn heterogeneous_compute_backends() { ... }     // four delegating cluster backends

// ── Group F: recursive ────────────────────────────────────────────────────

#[cfg(feature = "schedules-fp128-onehot-recursive")]
matrix_test!(recursive; fp128_onehot_recursive;             fp128::OneHot);

#[cfg(feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2")]
matrix_test!(recursive; fp128_onehot_recursive_multi_chunk; fp128::OneHotMultiChunk);
```

**Total: 25 test functions.** The macro-driven ones (Groups A–C, F) are ~2 lines each.
Groups D and E are plain `#[test]` functions using helpers from `common/`.
The file is roughly 130 lines excluding the macro definition and `common/` additions.

---

## 7. What changes where — summary

| File | Action |
|---|---|
| `tests/protocol_completeness.rs` | **Create** (this spec) |
| `tests/single_poly_e2e.rs` | **Delete** |
| `tests/batched_aggregated_e2e.rs` | **Delete** |
| `tests/heterogeneous_prove_e2e.rs` | **Delete** |
| `src/scheme/tests/fp32_ext4.rs` | **Delete** |
| `src/scheme/tests/heterogeneous_group.rs` | **Delete** |
| `src/scheme/tests/single.rs` | Remove 3 tests |
| `src/scheme/tests/batched.rs` | Remove 1 test |
| `src/scheme/tests/onehot.rs` | Remove 5 tests |
| `tests/akita_e2e.rs` | Remove 6 tests |
| `common/mod.rs` | Add 5 driver functions + `generic_opening_from_poly` |
| `tests/recursive_setup_e2e.rs` | No change |
| `src/scheme/tests/layout.rs` | No change |
| `src/scheme/tests/dense_group.rs` | No change |

---

## 8. Open questions — answer before writing code

1. **Feature flag names**: confirm the actual feature names in `crates/akita-schedules/Cargo.toml`
   match those used here (e.g. `schedules-fp64-dense`, `schedules-fp128-onehot-multi-chunk`, etc.).

2. **fp128_onehot_multi_chunk non-precommitted nv=32**: confirm that the `fp128_onehot_multi_chunk`
   catalog has an entry where `precommitteds` is empty at nv=32. If every nv=32 entry carries
   precommitted groups, tests 8 and 11 need to merge.

3. **Precommitted driver geometry**: does `prove_verify_*_precommitted_roundtrip` infer the
   precommit nv from `committed_group_profile::<Cfg>(&PolynomialGroupLayout::singleton(pre_nv))`
   by trying candidate pre_nv values, or does the catalog require a fixed (final_nv, pre_nv) pair?

4. **fp32/fp64 extension-field opening**: `generic_opening_from_poly` must handle
   `Cfg::ExtField ≠ Cfg::Field`. Confirm the MLE evaluation path for non-base-field openings.

5. **`#[ignore]` with `matrix_test!`**: confirm that `#[ignore]` as an outer attribute before the
   macro invocation attaches correctly to the generated `#[test]` fn, or whether it must be
   inside the macro arm.
