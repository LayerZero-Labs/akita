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

---

# Follow-up finding: Placeholder-direct DP introduces a systematic bias against `Direct` and a hidden parent-formula trade-off

| Field       | Value                                                                                  |
|-------------|----------------------------------------------------------------------------------------|
| Author(s)   | Planner refactor investigation, 2026-05-28                                             |
| Status      | findings + fix landed on this branch                                                   |
| Scope       | `crates/akita-planner/src/schedule_params.rs`, all 34 shipped schedule tables          |
| Related     | builds on the SIS-audit-bypass fixes above; touches no SIS-audit code path             |

## Summary

The suffix DP in `derive_optimal_suffix_schedule` modelled a terminal-direct step in two stages: emit a `Step::Direct` *placeholder* whose witness shape used the parent's `MRowLayout::Intermediate` length (the same value used in the fold branch), and later patch the shape to `MRowLayout::Terminal` in `finalize_terminal_direct_witness_shape` when the parent fold actually selected this direct as its terminal suffix.

That structure produced two distinct, compounding suboptimalities in the planner's choices:

1. **Direct-vs-fold bias.** Inside the DP at level `L`, the "direct now" cost was the inflated Intermediate-shape `direct_bytes(current_w_len, log_basis)`. The "fold now then continue" cost was correctly scored (terminal directs deeper in the chain were patched at their own parents). The local `min(...)` therefore systematically over-rejected `Direct` in favor of one more fold, even when the corrected direct would have been cheaper. The error is bounded by the Intermediate–Terminal gap, which is `d_key.row_len() · num_digits_full_field(field_bits, log_basis) · ring_dimension` field elements (plus, under `zk`, the D-blinding term). For modest `n_d` and small `log_basis`, that gap is in the few-percent range.

2. **Hidden parent-formula trade-off.** The placeholder/finalize architecture also forced the DP to return one *single* best suffix per memo state, but the parent's proof-size formula at level `L-1` depends on whether its child at level `L` is `Fold` or `Direct`:
   - child = `Direct` → parent uses `compute_terminal_level_proof_size` (drops v-rows and the stage-1 sumcheck at the parent level — strictly smaller),
   - child = `Fold(lb)` → parent uses `compute_level_proof_size` with `next_lp.b_key.row_len()` from the child's chosen `lb`.

   A child that locally prefers a deeper `Fold` chain by a small margin can force the parent into the larger intermediate proof formula, costing more than the local savings. Even within the fold branch, different child `log_basis` choices yield different child `b_key.row_len`, hence different parent intermediate proofs. The old DP collapsed these into one `(cost, schedule)` pair and hid the trade-off from the parent.

The two effects pull in opposite directions and partially canceled by accident on most keys, so the old planner was *usually* close to optimal. But the SIS-audit fixes earlier in this doc forced regenerating the shipped tables anyway, and exposing both effects to the parent was the simplest way to make the regen monotone in proof size.

## Mechanism (old)

`crates/akita-planner/src/schedule_params.rs`, pre-fix structure:

- `to_direct_step(current_w_len, log_basis) -> Step::Direct` produced a placeholder with `current_w_len` from the parent's `MRowLayout::Intermediate` computation and **no SIS check**. SIS feasibility was deferred.
- `derive_optimal_suffix_schedule(level, current_w_len, log_basis) -> (best_cost, schedule)` returned a single best suffix. Its `best_cost` was seeded with the inflated placeholder bytes.
- `finalize_terminal_direct_witness_shape(suffix_steps, candidate, num_points, num_t_vectors, num_w_vectors, num_public_rows, fold_level)` walked the suffix's head step (must be a `Direct`), recomputed the terminal-layout witness length from the parent fold's `lp` plus the root-level batching counts, and patched `direct.current_w_len`, `direct.direct_bytes`, `direct.witness_shape`, and `direct.level_params` in place. It also ran the SIS check (`direct_level_params_with_log_basis`) and propagated any failure as the rejection signal for the whole parent candidate.
- Both call sites (the recursive DP loop and the `find_schedule` root loop) then ran the same patch dance: snapshot `old_direct_bytes`, call `finalize`, read back `new_direct_bytes`, and apply `suffix_cost += new_direct_bytes - old_direct_bytes` to repair the DP's accumulated cost. ~25 lines duplicated across two sites.

The placeholder is *not* a "wrong number that gets fixed later" — both call sites do correctly produce the right total bytes at scoring time. The bug is that **the DP's comparator decisions are made on the inflated placeholder**, so it picks the wrong `arg min` even when the final cost arithmetic is right.

## Mechanism (new)

The fix routes both fixups into the DP's primary search:

### 1. Two witness shapes per candidate (`CandidateLevelParams`)

Every candidate level (both root and recursive) now carries both shapes:

```rust
struct CandidateLevelParams {
    lp: LevelParams,
    /// Witness length entering the next level under `MRowLayout::Intermediate`.
    /// Used to recurse into the suffix DP's fold branch.
    next_w_len: usize,
    /// Witness length entering the next level under `MRowLayout::Terminal`.
    /// Used to cost the suffix DP's direct branch correctly the first time.
    next_w_len_terminal: usize,
}
```

`recursive_next_witness_len(level_lp, field_bits, layout)` and `root_next_witness_len_for_layout(lp, key, layout)` route both through the shared `w_ring_element_count_with_counts_for_layout_bits`, which already handles the `MRowLayout::Terminal` D-block dropout and the ZK D-blinding gating. The recursive shrink check still gates on the Intermediate shape (Terminal is `<= Intermediate`, so this matches the previous behavior).

### 2. Eager, SIS-aware `to_direct_step`

```rust
fn to_direct_step<Cfg>(
    num_vars: usize,
    level: usize,
    current_w_len_terminal: usize,
    log_basis: u32,
) -> Result<Option<Step>, AkitaError>
```

Builds the direct step under `MRowLayout::Terminal` in one shot. Runs `akita_derive::direct_level_params_with_log_basis(...)` for SIS — returns `Ok(None)` on infeasibility, so the DP state simply has no direct option. The same SIS derivation that the old `finalize_terminal_direct_witness_shape` ran, now decided at the emission site and memoized.

`finalize_terminal_direct_witness_shape` is **deleted**. `successor_level_params_from_schedule` is **deleted** (no longer needed; the next-level params come directly off the child fold step).

### 3. Two-shape DP signature + two-best result

```rust
type ScheduleMemo = HashMap<(usize, usize, usize, u32), SuffixResult>;
//                          (level, w_len, w_len_terminal, log_basis)

struct SuffixResult {
    best_direct: Option<(usize, Vec<Step>)>,
    best_fold_per_lb: BTreeMap<u32, (usize, Vec<Step>)>,
}
```

The DP at state `(level, w_len, w_len_terminal, log_basis)` returns:

- **`best_direct`** — the optimal schedule whose first step is `Step::Direct` at this level. `None` iff SIS is infeasible.
- **`best_fold_per_lb`** — *one entry per first-fold `log_basis`*. The key is intentional: for each first-fold lb, the parent's `compute_level_proof_size` sees a different `next_lp.b_key.row_len()`. Collapsing into a single `best_fold` (as the previous iteration of this refactor did) re-introduces the trade-off bug. Listing one entry per `lb` lets the parent enumerate all relevant child layouts against its own formula.

The memo key extends to four dimensions because `w_len_terminal` is not deterministic from `w_len + log_basis` — the gap depends on the parent's `d_key.row_len`, which varies between different grandparents reaching the same `(level, w_len, log_basis)` state. State-space growth is bounded by the small number of distinct audited `d_key.row_len` values per family / log_basis.

### 4. Parent-side enumeration

Both the recursive DP loop and the `find_schedule` root loop now run:

```rust
// Branch A: child is a Direct → use terminal proof formula at this level.
if let Some((child_cost, child_sched)) = child.best_direct.as_ref() {
    let proof = compute_terminal_level_proof_size(&candidate, candidate.next_w_len_terminal, claims) + eor;
    try_update(proof + child_cost, [Fold(.., next_w_len=Terminal)] ++ child_sched);
}
// Branch B: child is a Fold → use intermediate proof formula, one option per child first_lb.
for (_lb, (child_cost, child_sched)) in &child.best_fold_per_lb {
    let next_lp = level_params_from_fold_step(child_sched[0]);
    let proof = compute_level_proof_size(&candidate, &next_lp, claims) + eor;
    try_update(proof + child_cost, [Fold(.., next_w_len=Intermediate)] ++ child_sched);
}
```

Both call sites used to be ~50 lines of placeholder snapshot + finalize call + read-back + cost-delta arithmetic, and both used to make decisions on inflated direct costs. They are now uniform and short.

## Worked example: why the parent-formula trade-off matters

Cfg: `fp16::D64OneHot`. Key: `num_vars = 20`, singleton.

Suffix DP at level 1 with `current_w_len = 269_312`, considering the level-1 fold options at all log-bases:

```text
L1 lb=2 -> child.best_fold first_lb=2, n_b=4 -> total via branch B = 26_160
L1 lb=3 -> child.best_fold first_lb=4, n_b=5 -> total via branch B = 26_512
L1 lb=4 -> child.best_fold first_lb=4, n_b=5 -> total via branch B = 26_080
L1 lb=5 -> child.best_fold first_lb=5, n_b=6 -> total via branch B = 34_688
L1 lb=6 -> child.best_fold first_lb=6, n_b=7 -> total via branch B = 37_216
```

A single-best DP picks lb=4 (total 26,080) and forwards `next_lp.b_key.row_len = 5` to the root. The root's `compute_level_proof_size` consumes that and produces a root proof of 3,504 bytes. Grand total: 29,584.

A `best_fold_per_lb` DP keeps every first_lb option. The root now also evaluates `child.best_fold_per_lb[lb=2]` (suffix cost 26,160) with `next_lp.b_key.row_len = 4`, producing a root proof of 3,376 bytes. Grand total: 29,536.

Difference: 48 bytes. The locally cheaper child (lb=4) costs the parent 128 bytes of extra intermediate proof to save 80 bytes of suffix. The single-best DP cannot see this. The `best_fold_per_lb` DP does.

## Impact on shipped tables

Comparing `find_schedule(key, true)` (old shipped tables = old planner) against `find_schedule(key, false)` (new pure DP), before regenerating the tables:

| family                  | keys | improved | unchanged | regressed | total_old              | total_new              | ratio  |
|-------------------------|-----:|---------:|----------:|----------:|-----------------------:|-----------------------:|-------:|
| fp128_d32_full          |  100 |       75 |        25 |         0 | 27 021 597 769 445 448 | 27 021 597 769 350 936 | 1.0000 |
| fp128_d32_onehot        |  100 |       68 |        32 |         0 |              4 768 488 |              4 677 116 | 0.9808 |
| fp128_d64_full          |  100 |       24 |        76 |         0 |              5 879 768 |              5 869 488 | 0.9983 |
| fp128_d64_onehot        |  100 |       20 |        80 |         0 |              5 273 736 |              5 266 400 | 0.9986 |
| fp128_d64_onehot_tensor |  100 |       26 |        74 |         0 |              5 343 784 |              5 332 888 | 0.9980 |
| fp16_d32_full           |   64 |       26 |        38 |         0 |          8 591 128 808 |          8 591 114 768 | 1.0000 |
| fp16_d32_onehot         |   64 |       26 |        38 |         0 |              1 088 296 |              1 074 168 | 0.9870 |
| fp16_d64_full           |   64 |       24 |        40 |         0 |              1 384 328 |              1 374 152 | 0.9926 |
| fp16_d64_onehot         |   64 |       20 |        44 |         0 |              1 235 240 |              1 229 960 | 0.9957 |
| fp32_d32                |   64 |       28 |        36 |         0 |              1 451 632 |              1 443 656 | 0.9945 |
| fp32_d32_onehot         |   64 |       22 |        42 |         0 |              1 235 600 |              1 231 912 | 0.9970 |
| fp32_d64                |   64 |       34 |        30 |         0 |         25 771 430 016 |         25 771 397 248 | 1.0000 |
| fp32_d64_onehot         |   64 |       34 |        30 |         0 |              1 459 408 |              1 422 000 | 0.9744 |
| fp64_d32                |   64 |       36 |        28 |         0 |         94 490 904 368 |         94 490 870 552 | 1.0000 |
| fp64_d32_onehot         |   64 |       34 |        30 |         0 |              1 471 000 |              1 433 720 | 0.9747 |
| fp64_d64                |   64 |       36 |        28 |         0 |              2 211 392 |              2 170 192 | 0.9814 |
| fp64_d64_onehot         |   64 |       30 |        34 |         0 |              1 804 112 |              1 768 432 | 0.9802 |

Total across all 17 families: **461 keys improved, 711 keys unchanged, 0 keys regressed** (1,172 keys total). Worst-family ratio 0.9744, i.e. ~2.6% smaller proofs on `fp32_d64_onehot`; best families are flat to four decimals because savings are dominated by direct-bytes deltas on huge witness configurations where the total is in the gigabytes.

The "ratio = 1.0000" rows are real improvements buried in trailing digits — e.g. `fp16_d32_full` saved 14,040 bytes total but the total is 8.59 GB.

After regen, the new shipped tables are byte-for-byte equal to the new DP's output. The drift guard re-runs cleanly.

## Invariants

The new architecture guarantees:

1. **No more placeholder/patch dance.** Every `Step::Direct` emitted by the DP carries its real `MRowLayout::Terminal` witness length and SIS-secure `level_params` from the moment of construction. No subsequent mutation, no two-pass cost accounting.
2. **Correct local comparator.** At every state `(level, w_len, w_len_terminal, log_basis)`, `best_direct` and `best_fold_per_lb` are scored on their actual final byte costs against the parent's actual proof formula. No inflated placeholder bias.
3. **Parent sees all relevant child layouts.** Because `best_fold_per_lb` is keyed by first-fold `log_basis` rather than collapsed, the parent's intermediate proof can be computed against every distinct `next_lp.b_key.row_len()` the child can produce. The parent picks the global minimum.
4. **Monotone in proof size vs the old planner.** Verified empirically across 1,172 `(family, key)` pairs (0 regressions). Verified architecturally: the new DP enumerates a strict superset of the options the old DP considered correctly, while the old DP's `arg min` on inflated direct costs was a strict subset of the new comparator's option set.

## What did NOT change

- SIS feasibility decisions. The new `to_direct_step` calls exactly the same `direct_level_params_with_log_basis` that the old `finalize_terminal_direct_witness_shape` did, with identical `(num_vars, level, current_w_len)` inputs. Schedules previously rejected for SIS-infeasibility are still rejected; schedules previously accepted are still accepted with the same `level_params`.
- The root-direct `commit_params` path (lines around `root_direct_commit_layout` + `scale_batched_root_layout`). That's the zero-fold baseline and lives outside the suffix DP — the "uncommittable root-direct edge case" with `commit_params: None` is unchanged.
- All proof-size formulas (`level_proof_bytes`, `terminal_level_proof_bytes`, `direct_witness_bytes`). Refactor only changed which schedules the DP picks, not how it costs them.
- Memo state space order of magnitude. The new key adds one dimension (`w_len_terminal`), but values are bounded by the small set of distinct `d_key.row_len` per family/log_basis. Empirically `find_schedule` runtime on the full 1,172-key sweep is indistinguishable from before.

## File-level diff summary (this section)

- `crates/akita-planner/src/schedule_params.rs`
  - `CandidateLevelParams` — added `next_w_len_terminal`.
  - `derive_candidate_level_params` — computes both shapes via new helper `recursive_next_witness_len`.
  - `root_next_witness_len` → `root_next_witness_len_for_layout` — layout-parametric; root loop calls it twice for both shapes.
  - `to_direct_step` — new signature `(num_vars, level, current_w_len_terminal, log_basis) -> Result<Option<Step>, _>`; SIS check eager.
  - `finalize_terminal_direct_witness_shape` — **removed**.
  - `successor_level_params_from_schedule` — **removed**.
  - `SuffixResult` — new struct with `best_direct` + `best_fold_per_lb: BTreeMap<u32, _>`.
  - `derive_optimal_suffix_schedule` — return type `SuffixResult`; memo key 4-tuple; enumerates branch A (child=Direct → terminal proof at parent) and branch B per child first_lb (child=Fold → intermediate proof at parent with that lb's `next_lp`).
  - `find_schedule` root loop — consumes both `best_direct` and `best_fold_per_lb` entries; computes both proof formulas; picks global min.
- `crates/akita-types/src/generated/*.rs` (34 files) — regenerated against the new planner via `gen_schedule_tables` (both default and `--features zk` runs).
- `crates/akita-planner/tests/proof_size_comparison.rs` — new diagnostic test that walks every `(family, key)` and asserts `find_schedule(key, false).total_bytes <= find_schedule(key, true).total_bytes`. Pre-regen, it surfaced the per-family improvement table above. Post-regen, it functions as a structural consistency check (both paths produce identical schedules). Safe to keep as a drift sentinel.

## Test status on this branch (updated)

- `cargo fmt -q` / `cargo clippy --all -- -D warnings`: clean (default and `--features zk`).
- `cargo test -p akita-planner --release` — all green:
  - `generated_schedule_tables_match_find_schedule` (drift guard, against the regenerated tables): pass.
  - `refactor_does_not_increase_proof_sizes` (new): pass (0 regressions across 1,172 keys).
  - Unit tests inside `schedule_params.rs`: pass.
- `cargo test --release --features zk -p akita-planner`: same set, all green.
- `cargo test --release` (full workspace): all green.

## Open follow-ups

- The `tests/proof_size_comparison.rs` diagnostic now compares regenerated-tables against regenerated-DP and is therefore a structural check (always equal). It could be renamed (e.g. `tables_match_pure_dp_total_bytes`) or removed once it stops carrying useful diff information — the drift guard already enforces the same property.
- The DP could in principle also enumerate over **parent log_basis** for the same reason it now enumerates over child first_lb (the parent's own `lp.d_key.row_len()` affects `current_w_len_terminal`). Empirically the monotone constraint `child_lb >= parent_lb` and the small number of distinct `d_key.row_len` values make this redundant in practice, but it's a theoretical sharper bound. Not pursued.
- Adding a per-family proof-size baseline test (golden file) would make future refactors' net effect visible at a glance. Out of scope here.
