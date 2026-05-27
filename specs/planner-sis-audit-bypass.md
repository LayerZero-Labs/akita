# Findings: Planner emits SIS-insecure root layouts via zero-collision audit bypass

| Field       | Value                                                                                  |
|-------------|----------------------------------------------------------------------------------------|
| Author(s)   | Planner refactor investigation, 2026-05-27                                             |
| Status      | findings — investigation complete, fixes in progress                                   |
| PR          | (planner refactor branch)                                                              |

## Summary

The schedule planner's `derive_root_candidate` enumerates root candidates `(m, r)` for a fixed `log_basis` but reuses the SIS ranks `(n_a*, n_b*, n_d*)` from a *single* inner split picked by `root_level_layout_with_log_basis`'s internal `optimal_m_r_split`. The inherited ranks are correct for `(m*, r*)` but are stale for other `(m, r)` candidates, where the SIS-secure floor may be higher. `AjtaiKeyParams::try_new` is supposed to catch this — but its audit short-circuits when any of `(row_len, col_len, collision_inf)` is zero, and the planner's `root_lp` flows through `LevelParams::params_only` → `with_decomp` → `with_layout` in such a way that `a_key.collision_inf` is **always zero** on the emitted layout. So the audit is silently skipped, the planner emits SIS-insecure root layouts, and the shipped schedule tables in `crates/akita-types/src/generated/*` encode those insecure layouts on multiple keys.

The bug is not in any single function. It is the interaction of three lenient defaults that, in isolation, are all defensible, and that together open a security hole.

## Concrete example

Cfg: `fp16::D32Full`. Key: `num_vars = 32, num_t_vectors = 4, num_w_vectors = 4, num_z_vectors = 1, num_points = 1`. Active audit tables: `crates/akita-types/src/generated/sis_floor.rs` at `(family = Q16, d = 32, collision = 7)`:

```text
[3, 6, 9, 15, 47, 140, 377, 958, 2_273, 5_144, 11_184, 23_903, 48_739, 96_741,
 187_451, 355_415, 660_737, 682_696, 682_696, 682_696]   (rank → max secure col_len)
```

Inner split inside `root_level_layout_with_log_basis(log_basis = 2)`:

- `optimal_m_r_split` converges on `m* = 14, r* = 13`, with `inner_width(m*, r*) = 2^14 · num_digits_commit(8) = 131_072`.
- SIS-secure floor for that width at bucket 7: rank 15 (`table[14] = 187_451 ≥ 131_072`). So `n_a* = 15`.

Outer planner loop at `(m = 16, r = 11)`:

- `inner_width(16, 11) = 2^16 · 8 = 524_288`.
- SIS-secure floor for that width at bucket 7: rank **17** (`table[16] = 660_737 ≥ 524_288`, `table[15] = 355_415 < 524_288`).
- Planner uses `n_a = n_a* = 15` (inherited from the inner split).
- `AjtaiKeyParams::try_new(family, row_len = 15, col_len = 524_288, collision_inf = 0, d = 32)` — the `collision_inf = 0` short-circuit at `crates/akita-types/src/layout/params.rs` (see "Mechanism" below) makes the audit return "no violation".
- The emitted root step has `a_key { row_len: 15, col_len: 524_288, collision_inf: 0 }` despite the SIS-secure floor being 17. **Insecure.**

This entry is present in `FP16_D32_FULL_SCHEDULES` and is part of the shipped artifact reproduced by the drift guard (`crates/akita-planner/tests/generated_tables.rs`).

The same pattern shows up in every shipped family for some `(num_vars, num_polys)` combinations. The structural cause (collision_inf=0 propagation) is family-independent.

## Mechanism

Three places interact:

### 1. `LevelParams::params_only` creates AjtaiKey placeholders with `collision_inf = 0`

`crates/akita-types/src/layout/params.rs`:

```274:300:crates/akita-types/src/layout/params.rs
            a_key: AjtaiKeyParams {
                row_len: n_a,
                sis_family,
                ..Default::default()
            },
            b_key: AjtaiKeyParams {
                row_len: n_b,
                sis_family,
                ..Default::default()
            },
            d_key: AjtaiKeyParams {
                row_len: n_d,
                sis_family,
                ..Default::default()
            },
```

`Default::default()` zeroes `col_len` and `collision_inf`. This is intentional: `params_only` is a "ranks + ring + stage1" placeholder, layout fields are filled later. The placeholder is supposed to be sealed by `with_decomp` and `with_layout`. **It is not.**

### 2. `with_decomp` and `with_layout` propagate `collision_inf = 0` unchanged

`with_decomp` (line ~432 in the same file) copies `self.a_key.collision_inf` into the new layout:

```rust
a_key: AjtaiKeyParams::new_unchecked(
    self.a_key.sis_family,
    self.a_key.row_len,
    inner_width,
    self.a_key.collision_inf,  // ← carries 0 forward
    d,
),
```

`with_layout` (line ~502) is even worse: it takes `collision_inf` from the *layout* argument (`other`), not from `self` (which by that point has the SIS-secure bucket set):

```502:528:crates/akita-types/src/layout/params.rs
    pub fn with_layout(&self, other: &LevelParams) -> Self {
        let d = self.ring_dimension;
        Self {
            ring_dimension: d,
            log_basis: other.log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                self.a_key.sis_family,
                self.a_key.row_len,
                other.a_key.col_len,
                other.a_key.collision_inf,    // ← reads from layout-side, which still has 0
                d,
            ),
            // ...same for b_key, d_key
```

The fixed-point loop in `root_level_layout_with_log_basis` ends with `return Ok(derived_params.with_layout(&root_lp))`. `derived_params` carries the SIS-secure bucket (set by `sis_secure_level_params` via `AjtaiKeyParams::new`), and `root_lp` carries `collision_inf = 0` (from the `params_only` placeholder threaded through `with_decomp`). `with_layout` selects `other.a_key.collision_inf` — the zero — and the bucket is lost on the way out.

### 3. `try_new` (and `new_unchecked`) silently skip the SIS audit when `collision_inf == 0`

Pre-fix:

```49:73:crates/akita-types/src/layout/params.rs
    fn sis_security_violation(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Option<String> {
        if col_len > 0 && collision_inf > 0 && row_len > 0 {
            use crate::generated::sis_floor::min_rank_for_secure_width;
            if let Some(floor) = min_rank_for_secure_width(
                sis_family,
                ring_dimension as u32,
                collision_inf,
                col_len as u64,
            ) {
                if row_len < floor {
                    return Some(format!(/* violation */));
                }
            }
        }
        None
    }
```

`if col_len > 0 && collision_inf > 0 && row_len > 0` means: if any of those is zero, skip the audit and return "no violation". Combined with (2) above, this guarantees the planner's per-`(m, r)` `try_new` call never audits anything — the value it consults is always zero.

Even more subtly, the inner branch only emits a violation if `min_rank_for_secure_width` returns `Some(floor)`. If the configuration has no audited row at all (e.g., `collision_inf` is not a tabulated bucket, or `col_len` exceeds every audited row's max width), the function returns `None` and the audit again says "no violation". So even a non-zero, *unsupported* `collision_inf` would silently bypass the check.

## Why the drift guard passed before this investigation

`crates/akita-planner/tests/generated_tables.rs` asserts that `find_schedule::<Cfg>(key, false)` reproduces the shipped table entry exactly. Both branches (the shipped table and the from-scratch DP regen) consult the same buggy planner code, so they produce the same insecure plan. The guard catches drift between the shipped artifact and today's planner — it does not catch drift between today's planner and the SIS-floor tables that define security. As long as the bug is bit-for-bit reproducible, the guard is content.

## Scope of impact

For each `(Cfg, num_vars, num_polys)` row in the shipped tables under `crates/akita-types/src/generated/`:

- For root candidates where the outer loop's `(m, r)` matches the inner split's `(m*, r*)`, the inherited rank is already correct and the row is fine.
- For root candidates where the outer loop picks a *different* `(m, r)`, the rank is potentially below the audited floor, and the row may be insecure.

Across all shipped families a diagnostic comparison (transient `tests/proof_size_diagnostic.rs`, not committed) flagged on the order of 100+ rows whose emitted ranks lie below the current audited floor — exact counts depend on which row's `optimal_m_r_split` happens to land on the picked `(m, r)`. Every shipped family had at least one insecure row.

## What is being fixed

### `AjtaiKeyParams` audit (landed on this branch)

`sis_security_violation` is now strict, with explicit error messages on every failure mode:

- `row_len`, `col_len`, and `collision_inf` must all be non-zero. Any zero is an explicit violation, no silent-permissive bypass.
- `min_rank_for_secure_width` returning `None` is an explicit violation (no audited row covers the configuration → cannot certify security).
- The existing `row_len < floor` violation is unchanged.

`new_unchecked` now routes through `sis_security_violation` for its debug-build warning, so the same strict conditions trip the smell check (still `tracing::warn!`, not `Err`, by design — `new_unchecked` exists for legitimate intermediate construction).

After this change, `try_new` no longer accepts the planner's `collision_inf = 0` keys. The planner refactor (`derive_root_candidate` → `enumerate_root_candidates`) derives `collision_inf` from `ceil_supported_collision(...)` directly, so its emitted layouts pass the strict audit by construction.

### `LevelParams::with_layout` collision propagation (landed on this branch)

`with_layout` now preserves `collision_inf` from `self`, not from `other`. This matches the docstring's intent ("keeps rank/ring info from `self` but replaces all layout-derived fields") — `collision_inf` is a property of the SIS audit, not the layout. All in-tree callers feed `with_layout` a SIS-derived `self` and a layout-only `other` with `collision_inf = 0`, so flipping the propagation direction recovers the correct bucket end-to-end with no code-site churn.

`with_decomp` already preserves `self.a_key.collision_inf`, so it does not need changes.

### `sis_secure_level_params` placeholder construction (landed on this branch)

`sis_secure_level_params` previously called `AjtaiKeyParams::new(family, n, 0, collision, d)` with a placeholder `col_len = 0`. Under the strict audit, that constructor now panics. The fix is to call `AjtaiKeyParams::new_unchecked` here — `col_len = 0` is an intentional placeholder, the layout is filled in by a subsequent `with_layout` (which now preserves `collision_inf`), and the next strict-audit boundary (`try_new` in the planner or in `scale_batched_root_layout`) sees the correct bucket.

### `scale_batched_root_layout` fallback in `find_schedule` (landed on this branch)

`find_schedule` was calling `scale_batched_root_layout(...)?` to scale a singleton root layout for batched keys, populating `root_direct_commit_params`. Under the strict audit, scaling can legitimately fail when the singleton ranks no longer cover the batched widths — that is not an error, it just means the direct-baseline cannot be carried as a layout-typed hint. The DP comparator still has a valid upper bound from `direct_witness_bytes`. The `?` is now an `.ok()` so the schedule falls back to `commit_params: None` for those keys (which the existing `match` arm already supports).

### Shipped schedule tables (open follow-up)

The shipped tables under `crates/akita-types/src/generated/*` were generated against the buggy planner and the silently-permissive audit. With the audit and propagation fixes in place, `find_schedule(false)` now emits **different** schedules for keys whose old plans were silently insecure. Regenerate via:

```bash
cargo run --release -p akita-planner --bin gen_schedule_tables -- crates/akita-types/src/generated
```

Expectations:

- Some `(num_vars, num_polys)` keys lose their fold schedule under the current audit tables (no SIS-secure rank covers the required width — `fp16_d32_full num_vars=32 polys=4` is the canonical example). Those keys legitimately need a richer ring/decomposition Cfg, **not** a silent rank under-count. Decide per-Cfg whether to widen the audited collision tables, raise the `min_log_basis` floor, or accept a direct-witness baseline.
- Downstream tests in `akita-pcs`, `akita-scheme`, `akita-pcs/examples/profile`, etc. that consume the shipped tables may break wherever they rely on a specific witness/proof shape. Expected to be a one-time migration; subsequent regenerations are drift-stable.

This is left to a follow-up PR because (a) it has wide downstream impact, and (b) the per-Cfg decision about what to do when no SIS-secure schedule exists at a given `num_vars` is a product-level call.

### `LevelParams::params_only` typestate (open follow-up, optional)

`params_only` is still a public constructor that returns an `AjtaiKeyParams` with `col_len = 0, collision_inf = 0`. With the strict audit in place, leaking such a placeholder into a production code path now fails loudly at the next `try_new` boundary — which is the intended behavior. A typestate (`ParamsOnly` vs `Sealed`) would catch this at compile time, but the runtime audit is good enough for now. Track as a hygiene item, not a blocker.

## Invariants

The fixes guarantee:

1. **Audit-or-fail.** `AjtaiKeyParams::try_new` only returns `Ok(_)` when the SIS-floor tables explicitly certify the `(family, ring_dimension, collision_inf, col_len, row_len)` tuple. No silent-permissive paths. Enforced by the rewritten `sis_security_violation`.
2. **Bucket preservation.** Once `collision_inf` is set on an `AjtaiKeyParams`, `LevelParams::with_layout` and `LevelParams::with_decomp` carry it through. Enforced by the `with_layout` fix (was: `other.a_key.collision_inf`, now: `self.a_key.collision_inf`).
3. **Drift guard.** `tests/generated_tables.rs` continues to pin shipped tables to the planner output. It currently fails on this branch — that is the expected signal that the shipped tables need to be regenerated.

A new audit-guard test should be added in the follow-up PR that walks every shipped table entry, reconstructs its `AjtaiKeyParams` via `try_new`, and asserts success. The drift guard alone catches divergence between shipped tables and planner output but not divergence between planner output and the audit tables.

## Testing notes

A targeted regression test should pin the original failing case:

```rust
// at audit boundary
let bucket = 7;  // Q16 / d=32, log_basis=2
assert!(
    AjtaiKeyParams::try_new(SisModulusFamily::Q16, 15, 524_288, bucket, 32).is_err(),
    "rank 15 must not pass the SIS audit for col_len=524_288 at Q16/d=32/bucket=7"
);
```

Plus a fully-shipped-table audit walk:

```rust
for family in ALL_GENERATED_FAMILIES {
    let table = (family.schedule_table)().unwrap();
    for entry in table.entries.iter() {
        for step in entry.steps.iter() {
            // For each Fold step, reconstruct the three AjtaiKeyParams via
            // try_new and assert Ok. Currently this would fire on every
            // family.
        }
    }
}
```

## Open questions

- For families where the strict audit can no longer find any fold schedule under the current tables (e.g. `fp16_d32_full num_vars=32 polys=4`), what is the right product response? Drop the row, widen the audited collision tables in `crates/akita-types/src/generated/sis_floor.rs`, raise `PROOF_OPTIMIZED_LOG_BASIS_MIN` for that Cfg, or fall through to a direct-witness baseline?
- Should `params_only` get a `ParamsOnly` vs `Sealed` typestate to catch placeholder leaks at compile time? The runtime audit catches them now, but a static rule would be cleaner.
- Are any tests outside the `akita-types`/`akita-planner` crates relying on the shipped tables to encode specific `(n_a, n_b, n_d)` values? Worth a grep on the test corpus before regenerating.

## File-level diff summary

- `crates/akita-types/src/layout/params.rs`
  - `AjtaiKeyParams::sis_security_violation` — strict (no zero-bypass, no None-bypass, explicit error messages).
  - `AjtaiKeyParams::new_unchecked` — debug-mode warning now routes through `sis_security_violation`.
  - `LevelParams::with_layout` — `collision_inf` preserved from `self`, not `other`; doc-comment updated.
- `crates/akita-derive/src/derivation.rs`
  - `sis_secure_level_params` — placeholder key construction switched from `AjtaiKeyParams::new` to `AjtaiKeyParams::new_unchecked` (intentional `col_len = 0`).
- `crates/akita-planner/src/schedule_params.rs`
  - `derive_root_candidate` → `enumerate_root_candidates`: rank-from-width chain per `(m, r)`, no inherited stale ranks, returns `Vec`.
  - `find_schedule`: scores every `(root_lb, m, r)` candidate by total proof bytes (not greedy by `next_w_len`); falls back to `None` for `root_direct_commit_params` when `scale_batched_root_layout` rejects.
  - `CandidateLevelParams` collapsed (`proof_lp`/`lp` were always equal under the new derivation).
  - Removed the now-unused private helper `derive_batched_root_level_derivation`.
- `crates/akita-planner/src/generated_families.rs`
  - Added `regen_with_lookup` function pointer (used by the transient `proof_size_diagnostic` test; safe to keep or drop).
- `crates/akita-types/src/schedule.rs`
  - Test fixture `planned_batched_root_bytes_match_two_stage_payload_at_all_bases` switched to `AjtaiKeyParams::new_unchecked` for its synthetic byte-counting fixture.

## Test status on this branch

- `cargo build --workspace`: clean.
- `cargo fmt -q` / `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo test -p akita-types --lib`: 99 / 99 pass.
- `cargo test -p akita-derive`: 8 / 8 pass.
- `cargo test -p akita-planner --lib`: 2 / 2 pass.
- `cargo test -p akita-planner --test generated_tables` (drift guard): **fails as expected**. The new (correct) planner output diverges from the shipped tables on hundreds of keys. The follow-up step is regenerating the tables.
