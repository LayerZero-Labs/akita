# Spec: Consolidate SIS / Ajtai logic into a new `akita-types::sis` module

| Field     | Value                            |
|-----------|----------------------------------|
| Author(s) | Omid Bodaghi, Cursor agent draft |
| Created   | 2026-06-02                       |
| Status    | superseded                       |
| PR        |                                  |
| Superseded-by | [`specs/sis-quantum128-scalar-n-table.md`](../../sis-quantum128-scalar-n-table.md) |

## Summary

The SIS / Ajtai logic — security-floor tables, secure-rank lookup, weak-binding
collision norms, gadget-decomposition digit counts, and Ajtai-key width
computation — is currently spread across at least four files in two crates:

- `crates/akita-planner/src/ajtai_params.rs` (`WitnessType::binding_norm`,
  `WitnessType::decomposed_num_digits`, `ajtai_{a,b,d}_width_bucket`,
  `compute_ajtai_key_params_{a,b,d}`, `compute_all_ajtai_keys_params`,
  `key_with_secure_rank`).
- `crates/akita-types/src/sis_offline.rs` (`a_role_witness_infinity_norm`,
  `a_role_collision_infinity_norm`, `sis_secure_level_params`,
  `sis_derived_root_params_for_layout`, `root_level_params_for_layout_with_log_basis`).
- `crates/akita-types/src/sis_floor.rs` (`SisModulusProfileId`, `sis_max_widths`,
  `min_rank_for_secure_width`, `ceil_supported_collision`).
- `crates/akita-types/src/layout/digit_math.rs` (`num_digits_for_bound`,
  `compute_num_digits*`, `ring_product_infinity_norm_bound`,
  `witness_block_l1_norm`, `fold_witness_norms`, `compute_num_digits_fold_with_claims`,
  `optimal_block_geometry_split`) and `crates/akita-types/src/layout/sis_derivation.rs`
  (`decomp_depths`, `level_layout_from_params`, `recursive_level_layout_from_params`).

The same A-role collision formula already exists in **two** copies
(`ajtai_params.rs` and `sis_offline.rs`); the same digit/width arithmetic is
re-derived in the planner DP and in the runtime table expansion. This duplication
is the source of the drift risk this spec eliminates.

**Goal:** create a new module **`akita_types::sis`** that owns every SIS/Ajtai
*leaf primitive* — norm bounds, Ajtai-key sizing, and decomposition digit/width
counts — behind a small, readable API. All other code (including
`akita-planner`) computes SIS/Ajtai quantities **only** through
`akita_types::sis`. The "connecting" wrappers (`compute_ajtai_key_params_*`,
`ajtai_*_width_bucket`, the `WitnessType` dispatcher) are deleted; each call site
instead wires the three leaf calls (`norm → width → rank → AjtaiKeyParams::try_new`)
explicitly.

> Note: this lives as a **module inside the existing `akita-types` crate**, not a
> new crate. `akita-types` already depends on `akita-challenges` and
> `akita-field`, so every input type the SIS APIs need (`SparseChallengeConfig`,
> `TensorChallengeShape`, `DecompositionParams`, `AjtaiKeyParams`, `AkitaError`)
> is already in scope — no new crate, no dependency-layering or cycle concerns.

## Intent

### Goal

Introduce `crates/akita-types/src/sis/` with three primitive submodules and make
it the single source of truth for SIS/Ajtai sizing:

- `norm_bound.rs` — weak-binding collision norms per witness role
  (`rounded_up_norm_s/t/w/z`).
- `ajtai_key.rs` — `min_secure_rank(...)`, the `AjtaiKeyParams` type, the
  collision-bucket rounding (`ceil_supported_collision`), and the generated
  SIS-floor tables (private).
- `decomposition_digits.rs` — gadget digit counts and the per-role committed
  widths (`decomposed_s_block_ring_count`, `decomposed_t_ring_count`,
  `decomposed_w_ring_count`).

No SIS/Ajtai sizing logic remains in `akita-planner`, nor scattered across
`akita-types`; the rest of the code only *orchestrates* (assembles
`AjtaiKeyParams` / `LevelParams` from the `akita_types::sis` primitives).

### Invariants

- **Single source of truth.** Exactly one implementation of: the SIS-floor
  tables, `min_secure_rank`, `ceil_supported_collision`, each role's collision
  norm, the `||c·s||_inf` ring-product bound, each role's digit count, and each
  role's committed width. Grep for the old symbol names must resolve only into
  `akita_types::sis` (plus its `akita_types::` re-exports). The two existing
  copies of the A-role collision formula collapse to one.
- **Behavior-preserving.** The SIS-floor tables and every numeric result are
  byte-for-byte identical to today. The generated schedule tables,
  `proof_size_comparison`, `generated_schedule_tables_match_find_schedule`
  drift-guard, and all prover/verifier round-trips must pass unchanged.
- **Verifier no-panic contract** (AGENTS.md) is preserved: every `sis` function
  reachable from `batched_verify` returns `Option`/`Result` and never panics on
  malformed input (the planner DP table-miss path is verifier-reachable).
- **Determinism.** Prover and verifier resolve identical schedules; the
  Fiat-Shamir `PlanSection`/descriptor bytes are unchanged (this refactor moves
  code, it does not change any wire/transcript encoding).

Protected by: `generated_tables` drift guard, `proof_size_comparison`,
`sis` unit tests (legacy-value + rank-cap, relocated from `sis_floor.rs`),
`akita_e2e`, `single_poly_e2e`, `zk`, and the full `cargo nextest` matrix.

### Non-Goals

- No new crate. (Considered and rejected — see Alternatives.)
- No change to the SIS security model, the collision formulas, the bucket set,
  or any numeric value. (The recent weak-binding-norm fix already changed the
  math; this is a pure relocation/refactor.)
- No change to wire formats, transcript binding, or proof structure.
- Not moving `LevelParams` or the schedule types into the `sis` module (it stays
  a leaf-primitive module, not a planning/layout module).

## Design

### Module layout

```
crates/akita-types/src/
  sis/
    mod.rs                     # declares submodules + curated `pub use` surface
    ajtai_key.rs               # SisModulusProfileId, AjtaiKeyParams, min_secure_rank, ceil_supported_collision
    floor.rs                   # generated SIS-floor tables (private; #[rustfmt::skip])
    norm_bound.rs              # rounded_up_norm_{s,t,w,z} + internal norm helpers
    decomposition_digits.rs    # digit counts + per-role widths
```

`crates/akita-types/src/lib.rs` adds `pub mod sis;` and re-exports the types that
are part of the `akita-types` public vocabulary at their **current** paths
(`akita_types::SisModulusProfileId`, `akita_types::AjtaiKeyParams`) so the ~32
references to `SisModulusProfileId` and ~10 to `AjtaiKeyParams` keep compiling
untouched:

```rust
// akita-types/src/lib.rs
pub mod sis;
pub use sis::{AjtaiKeyParams, SisModulusProfileId};
```

The deleted top-level files `sis_floor.rs` and `sis_offline.rs` are absorbed into
`sis/`. `layout/digit_math.rs` and `layout/sis_derivation.rs` lose their SIS
pieces to `sis/` and keep only non-SIS helpers (see the move table).

### Submodule: `sis/ajtai_key.rs`

Owns the Ajtai-key type, the secure-rank lookup, and bucket rounding. The
generated tables move into a private `sis/floor.rs` (kept compact with
`#[rustfmt::skip]`); `scripts/gen_sis_table.py`'s output target updates to that
file. `SisModulusProfileId` (today in `sis_floor.rs`) and `AjtaiKeyParams` (today in
`layout/params.rs`) move here.

```rust
pub enum SisModulusProfileId { Q16, Q32, Q64, Q128 }

pub struct AjtaiKeyParams { /* row_len, col_len, collision_inf, sis_modulus_profile */ }
impl AjtaiKeyParams {
    pub fn new(sis_modulus_profile, row_len, col_len, collision_inf, ring_dimension) -> Self;       // panics (prover-only)
    pub fn try_new(sis_modulus_profile, row_len, col_len, collision_inf, ring_dimension)
        -> Result<Self, AkitaError>;                                                       // verifier-safe
    pub fn new_unchecked(sis_modulus_profile, row_len, col_len, collision_inf, ring_dimension) -> Self;
    pub fn row_len(&self) -> usize;
    pub fn col_len(&self) -> usize;
    pub fn collision_inf(&self) -> u32;
    pub fn sis_modulus_profile(&self) -> SisModulusProfileId;
}

/// Minimum SIS-secure module rank that supports `width` ring columns at an
/// already-rounded-up collision bucket. (Renames `min_rank_for_secure_width`.)
pub fn min_secure_rank(
    sis_modulus_profile: SisModulusProfileId,
    d: u32,
    collision_inf_rounded_up: u32,
    width: u64,
) -> Option<usize>;

/// Round a raw collision infinity-norm up to the nearest audited SIS bucket.
pub fn ceil_supported_collision(sis_modulus_profile: SisModulusProfileId, d: u32, collision_inf: u32)
    -> Option<u32>;
```

`AjtaiKeyParams::{new,try_new}` keep their audit (they call `min_secure_rank`
internally), so the explicit `min_secure_rank` a caller also computes is a
deliberate, cheap double-check — it documents the rank the layout was sized for
and lets the planner reject infeasible candidates before building the key.

`AjtaiKeyParams::append_descriptor_bytes` (its Fiat-Shamir transcript encoding)
stays adjacent to the other descriptor helpers — either kept as a method using
`crate::descriptor_bytes::*`, or moved to a free fn in the descriptor module.
Same crate, so this is a trivial intra-crate reference (no boundary concern).

### Submodule: `sis/norm_bound.rs`

Owns the weak-binding collision norms (Hachi Lemma 7). The currently-internal
helpers `ring_product_infinity_norm_bound`, `witness_block_l1_norm`,
`a_role_witness_infinity_norm`, and `a_role_collision_infinity_norm` move here
(the first two stay `pub` for reuse/tests; the A-role helpers become private
implementation detail of `rounded_up_norm_s`).

The four public entry points return the **rounded-up** collision bucket ready to
feed `min_secure_rank` (s/t/w), or the folded-witness bound β (z):

```rust
/// A-role (committed witness `s`): ceil-bucket of `2·ω̄·β̄·ν`
/// with `β̄ = min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`.
pub fn rounded_up_norm_s(
    sis_modulus_profile: SisModulusProfileId,
    d: usize,
    decomposition: DecompositionParams,
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    is_root: bool,
    onehot_chunk_size: usize,
    ring_subfield_norm_bound: u32,
) -> Option<u32>;

/// B-role (`t̂`) and D-role (`ŵ`): ceil-bucket of the direct digit-difference
/// `2γ̄ = 2^lb − 1` (no challenge multiplication).
pub fn rounded_up_norm_t(sis_modulus_profile: SisModulusProfileId, d: usize, log_basis: u32) -> Option<u32>;
pub fn rounded_up_norm_w(sis_modulus_profile: SisModulusProfileId, d: usize, log_basis: u32) -> Option<u32>;

/// Folded witness `z = Σ c_i·s_i`: the L∞ bound
/// `β = num_claims · 2^block_index_bits · min(||c||_inf·||s||_1, ||c||_1·||s||_inf)`.
/// `z` is *not* Ajtai-committed, so this is the raw bound (no SIS bucket); it
/// feeds the next-level fold digit count in `decomposition_digits`.
pub fn rounded_up_norm_z(
    decomposition: DecompositionParams,
    fold_challenge_config: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    block_index_bits: usize,
    num_claims: usize,
    d: usize,
    onehot_chunk_size: usize,
    is_root: bool,
) -> u128;
```

Note the asymmetry (called out for the reviewer): `s/t/w` are Ajtai-committed, so
their norms are rounded up to an audited SIS bucket; `z` is decomposed for the
next level rather than committed, so `rounded_up_norm_z` returns the fold bound
β. If preferred, `rounded_up_norm_z` can instead return the *digit count*
directly and live in `decomposition_digits.rs` — see Open Questions.

`b/d` currently call `WitnessType::T/W::binding_norm` with `fold_shape = Flat`
because the collision is challenge-independent; `rounded_up_norm_t/w` drop the
challenge args entirely, which is strictly clearer.

### Submodule: `sis/decomposition_digits.rs`

Owns the gadget digit counts and the per-role committed widths. Moves the digit
primitives (`compute_num_digits`, `compute_num_digits_full_field`,
`num_digits_for_bound`, `balanced_digit_max`), `decomp_depths`, and the fold
digit count here.

```rust
// --- digit counts (per coefficient) ---
pub fn num_digits_for_bound(log_bound: u32, field_bits: u32, log_basis: u32) -> usize;

/// δ_commit for the committed witness `s` (level-dependent commit bound).
pub fn num_digits_s_commit(decomposition: DecompositionParams, log_basis: u32, is_root: bool) -> usize;
/// δ_open for `t̂` / `ŵ` (opened at the field level).
pub fn num_digits_open(decomposition: DecompositionParams, log_basis: u32) -> usize;
/// δ_fold for the folded witness `z`, from `rounded_up_norm_z`'s β.
pub fn num_digits_fold(beta: u128, field_bits: u32, log_basis: u32) -> usize;
/// (δ_commit, δ_open) pair (renames `decomp_depths`).
pub fn decomp_depths(decomposition: DecompositionParams) -> (usize, usize);

// --- per-role committed widths (ring-element column counts) ---
/// A width: `num_positions_per_block · δ_commit`.
pub fn decomposed_s_block_ring_count(num_positions_per_block: usize, num_digits_commit: usize) -> Option<usize>;
/// B width: `n_a · δ_open · num_live_blocks · t_vectors`.
pub fn decomposed_t_ring_count(n_a: usize, num_digits_open: usize, num_live_blocks: usize, t_vectors: usize)
    -> Option<usize>;
/// D width: `δ_open · num_live_blocks · t_vectors`.
pub fn decomposed_w_ring_count(num_digits_open: usize, num_live_blocks: usize, t_vectors: usize)
    -> Option<usize>;
```

All width helpers return `Option` (checked multiplication) so overflow is a
clean rejection, matching the current `ajtai_*_width_bucket` behavior.

### Call-site wiring (the pattern that replaces the deleted wrappers)

Every place that needs an Ajtai key does the explicit three-step the user
specified (no `compute_ajtai_key_params_*` wrapper):

```rust
use akita_types::sis::*;

// A key
let norm_s   = rounded_up_norm_s(family, d, decomp, &stage1, fold_shape, is_root, onehot_k, nu)
    .ok_or(/* InvalidSetup */)?;
let d_commit = num_digits_s_commit(decomp, log_basis, is_root);
let width_s  = decomposed_s_block_ring_count(num_positions_per_block, d_commit).ok_or(..)?;
let n_a      = min_secure_rank(family, d as u32, norm_s, width_s as u64).ok_or(..)?;
let a_key    = AjtaiKeyParams::try_new(family, n_a, width_s, norm_s, d)?;

// B key
let norm_t   = rounded_up_norm_t(family, d, log_basis).ok_or(..)?;
let d_open   = num_digits_open(decomp, log_basis);
let width_t  = decomposed_t_ring_count(n_a, d_open, num_live_blocks, t_vectors).ok_or(..)?;
let n_b      = min_secure_rank(family, d as u32, norm_t, width_t as u64).ok_or(..)?;
let b_key    = AjtaiKeyParams::try_new(family, n_b, width_t, norm_t, d)?;

// D key — analogous with rounded_up_norm_w + decomposed_w_ring_count.
```

This same block is used by the planner DP candidate evaluator
(`schedule_params.rs`), the runtime table expansion (`generated/expand.rs`), and
the root-layout derivation. To avoid re-typing it three times, a **single** thin
orchestration helper
`build_level_ajtai_keys(...) -> Result<(AjtaiKeyParams,AjtaiKeyParams,AjtaiKeyParams), _>`
may live in `akita-planner` (orchestration, not SIS logic — it contains no
formula, only the wiring above). See Open Questions on inline vs. one helper.

### What moves / changes / is removed (per file)

| Current location | Symbol(s) | Action |
|---|---|---|
| `akita-types/src/sis_floor.rs` | `SisModulusProfileId` | **move** → `sis/ajtai_key.rs`; `akita-types` re-exports at the current path |
| `akita-types/src/sis_floor.rs` | `sis_max_widths` (tables) | **move** → `sis/floor.rs` (private) |
| `akita-types/src/sis_floor.rs` | `min_rank_for_secure_width` | **move + rename** → `sis::min_secure_rank` |
| `akita-types/src/sis_floor.rs` | `ceil_supported_collision` | **move** → `sis/ajtai_key.rs` |
| `akita-types/src/sis_floor.rs` | (file) | **delete** once emptied |
| `akita-types/src/layout/params.rs` | `AjtaiKeyParams` | **move** → `sis/ajtai_key.rs`; `akita-types` re-exports; descriptor encoding stays in the descriptor module |
| `akita-types/src/sis_offline.rs` | `a_role_witness_infinity_norm`, `a_role_collision_infinity_norm` | **move** → `sis/norm_bound.rs` (private), folded into `rounded_up_norm_s` |
| `akita-types/src/sis_offline.rs` | `sis_secure_level_params`, `sis_derived_root_params_for_layout`, `root_level_params_for_layout_with_log_basis`, `SisRoleWidths`, `SisCollisionBounds` | **delete**; LevelParams assembly becomes orchestration (the 3-step pattern) at the call sites |
| `akita-types/src/sis_offline.rs` | (file) | **delete** once emptied |
| `akita-types/src/layout/digit_math.rs` | `compute_num_digits*`, `num_digits_for_bound`, `balanced_digit_max`, `ring_product_infinity_norm_bound`, `witness_block_l1_norm`, `fold_witness_norms`, `FoldWitnessNorms`, `FoldChallengeNorms`, `compute_num_digits_fold_with_claims` | **move** → `sis/` (`decomposition_digits.rs` + `norm_bound.rs`) |
| `akita-types/src/layout/digit_math.rs` | `gadget_row_scalars` | **stays** in `layout` (field/gadget helper, not SIS) |
| `akita-types/src/layout/digit_math.rs` | `optimal_block_geometry_split` | **move** → `akita-planner` (a planning *search*, not a leaf primitive; uses only `sis` primitives) |
| `akita-types/src/layout/sis_derivation.rs` | `decomp_depths` | **move** → `sis/decomposition_digits.rs` |
| `akita-types/src/layout/sis_derivation.rs` | `level_layout_from_params`, `recursive_level_layout_from_params` | **keep as orchestration**, rewired onto `sis` (these build `LevelParams`; see Open Questions on relocation) |
| `akita-planner/src/ajtai_params.rs` | `WitnessType`, `binding_norm`, `decomposed_num_digits`, `ajtai_{a,b,d}_width_bucket`, `compute_ajtai_key_params_{a,b,d}`, `compute_all_ajtai_keys_params`, `key_with_secure_rank` | **delete** the whole file; replace call sites with the 3-step pattern (optionally one `build_level_ajtai_keys` helper) |
| `akita-prover/src/protocol/ring_relation.rs` | `beta_linf_fold_bound_with_num_claims` | **rewire** to `sis::rounded_up_norm_z` (drops the duplicated β formula) |
| `akita-types/src/layout/params.rs` | `LevelParams::{num_digits_fold, fold_witness_norms, challenge_infinity_norm}` | **rewire** to delegate to `sis` (no inline formula) |
| `scripts/gen_sis_table.py` | output target | **update** to write `sis/floor.rs` |

### `akita_types::sis_floor` path

`min_rank_for_secure_width` / `ceil_supported_collision` are imported from
`akita_types::sis_floor` in ~6 files (planner + types). Decision: either keep a
thin `pub mod sis_floor { pub use crate::sis::{...}; }` shim, or migrate those
imports to `akita_types::sis::{...}` directly (recommended — one canonical path
for the SIS function surface). The widely-referenced **types**
(`SisModulusProfileId`, `AjtaiKeyParams`) stay re-exported at `akita_types::` to
keep their ~40 references untouched.

### Alternatives considered

- **A separate `akita-sis` crate** (the prior draft of this spec). Rejected per
  user preference: a module inside `akita-types` achieves the same single-source
  consolidation without a new crate, and avoids moving `DecompositionParams` /
  `AjtaiKeyParams` across a crate boundary or adding a dependency layer. The
  trade-off: the module relies on `akita-types`-internal discipline (rather than
  a hard crate boundary) to prevent SIS logic from leaking back out; the
  invariant grep check and code review enforce this.
- **Keep `WitnessType` as the dispatcher.** Rejected: the user wants explicit
  per-role functions; the enum-dispatch hides which role is being sized.
- **`sis` APIs take scalars instead of `DecompositionParams` / `SparseChallengeConfig`.**
  Rejected: same-crate access makes the struct-typed API free of any downside,
  and it is the readability win the user asked for.

## Evaluation

### Acceptance Criteria

- [ ] `akita_types::sis` module exists with `ajtai_key`, `norm_bound`,
  `decomposition_digits` (+ private `floor`), exposing exactly:
  `SisModulusProfileId`, `AjtaiKeyParams`, `min_secure_rank`,
  `ceil_supported_collision`, `rounded_up_norm_{s,t,w,z}`, `num_digits_*`,
  `decomp_depths`, `decomposed_{s_block,t,w}_ring_count` (+
  `ring_product_infinity_norm_bound`, `witness_block_l1_norm` for reuse).
- [ ] `akita-planner/src/ajtai_params.rs` is deleted; `WitnessType`,
  `compute_ajtai_key_params_*`, `ajtai_*_width_bucket`, `key_with_secure_rank`
  no longer exist anywhere.
- [ ] `sis_floor.rs` and `sis_offline.rs` are deleted; the A-role collision
  formula exists in exactly one place (`sis/norm_bound.rs`).
- [ ] `grep` for each moved symbol resolves into `akita_types::sis` (plus the
  documented type re-exports).
- [ ] All numeric outputs unchanged: `sis` legacy-value tests, the
  `generated_tables` drift guard, and `proof_size_comparison` pass without
  regenerating tables (or regenerate to a byte-identical result).
- [ ] Full CI green: `fmt`, `clippy` (all-features + no-default), file-line cap
  (note: `sis/floor.rs` must stay `#[rustfmt::skip]` and under 1500 lines),
  `machete`, `doc`, both `nextest` runs.

### Testing Strategy

- Relocate the existing `sis_floor.rs` unit tests into `sis/` and keep them green
  (legacy prefixes, rank caps, `ceil_supported_collision`).
- Add focused `sis` unit tests for `rounded_up_norm_{s,t,w,z}` and the width
  helpers against hand-computed values (one-hot single/multi-chunk, dense, root
  vs recursive), absorbing the assertions currently in
  `sis_offline.rs`/`digit_math.rs`.
- The drift guard `generated_schedule_tables_match_find_schedule` is the key
  end-to-end check that the planner DP + runtime expansion still agree after the
  call-site rewiring.
- `akita_e2e`, `single_poly_e2e`, `zk` confirm prover/verifier round-trips.

### Performance

Pure relocation — no runtime cost change. The extra explicit `min_secure_rank`
call at each site is already performed today (inside the deleted wrappers and
again inside `AjtaiKeyParams::try_new`); net lookups are unchanged or fewer.

## Documentation

- New module doc (`sis/mod.rs`) stating it is the single home for SIS/Ajtai
  primitives and that no SIS/Ajtai formula may live outside it.
- Update `AGENTS.md` `akita-types` crate entry to mention the `sis` module as the
  SIS/Ajtai single source of truth, and the `akita-planner` entry to note it no
  longer holds Ajtai sizing.
- Update `scripts/gen_sis_table.py` header + the floor-table provenance comment
  for the new `sis/floor.rs` path.
- Cross-link from `specs/weak-binding-norm-fix.md` (this consolidation is the
  maintainability follow-up to that fix).

## Open Questions

1. **`rounded_up_norm_z` shape.** Return β (`u128`) from `norm_bound.rs`, or the
   fold digit count directly from `decomposition_digits.rs`? β is more composable
   (the prover's abort check also needs β, not just the digit count), so the spec
   leans to β in `norm_bound.rs`.
2. **Orchestration home.** Move `optimal_block_geometry_split` + the `*_layout_from_params`
   builders + the old `sis_derived_*` assembly to `akita-planner` (recommended,
   bigger diff, concentrates all planning/derivation in the planner), or keep the
   layout builders in `akita-types/layout` rewired onto `sis` (smaller diff)?
   Either way, `optimal_block_geometry_split` is a *search* and should leave the `sis`
   module.
3. **`akita_types::sis_floor` path.** Delete it and migrate the ~6 importers to
   `akita_types::sis::{...}` (recommended), or keep a thin re-export shim?
4. **`build_level_ajtai_keys` helper.** Provide one thin orchestration helper in
   `akita-planner` for the three-key wiring, or fully inline the 3-step at each of
   the (≈3) call sites for maximum explicitness?
5. **`AjtaiKeyParams` descriptor encoding.** Keep `append_descriptor_bytes` as a
   method on the moved type (it references `crate::descriptor_bytes`), or move it
   to a free function in the descriptor module so `sis/ajtai_key.rs` holds only
   sizing/audit logic?

## References

- `specs/weak-binding-norm-fix.md` — the weak-binding fix that introduced the
  duplicated A-role formula this spec consolidates.
- Current code: `crates/akita-planner/src/ajtai_params.rs`,
  `crates/akita-types/src/{sis_floor.rs, sis_offline.rs}`,
  `crates/akita-types/src/layout/{digit_math.rs, sis_derivation.rs, params.rs}`,
  `crates/akita-prover/src/protocol/ring_relation.rs`,
  `scripts/gen_sis_table.py`.
