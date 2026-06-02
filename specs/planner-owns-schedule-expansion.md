# Spec: Planner Owns Schedule Tables + Expansion (Protocol-Agnostic Schedules)

| Field     | Value                          |
|-----------|--------------------------------|
| Author(s) |                                |
| Created   | 2026-06-01                     |
| Status    | implemented                    |
| PR        |                                |

## Summary

Today the schedule-table *data* (`GeneratedScheduleTableEntry` + the static
`akita-types/src/generated/*.rs` tables) lives in `akita-types`, the *expansion*
of that compact data into runtime `LevelParams` lives in
`akita-types::proof_size` (`schedule_from_entry_bits`), and the *search* that
produces the data lives in `akita-planner` (`find_schedule`). `akita-config`
stitches the three together in `CommitmentConfig::runtime_schedule`: a table hit
calls the `akita-types` walker, a table miss calls the `akita-planner` DP.

This spreads "how a compact schedule becomes `LevelParams`" across three crates
and forces the schedule-table representation (`GeneratedScheduleTableEntry`,
`GeneratedStep`, …) to be a public `akita-types` surface that several crates can
see. The protocol (`akita-prover`/`akita-verifier`) does not need that
representation — it only needs a resolved `Schedule` — but the type is reachable
from anywhere `akita-types` is.

This refactor makes **`akita-planner` the single owner of the
schedule-table representation, the shipped tables, the on-demand expansion, and
the cache-then-generate resolution**. `akita-config` keeps exactly what only it
can hold (preset `Cfg` types, the family list, the table generator), and the
protocol stays entirely schedule-table-agnostic — it consumes `Schedule` through
`CommitmentConfig::runtime_schedule`, which becomes a one-line delegation to a
new `akita_planner::resolve_schedule(...)` entry point.

The key risk the user flagged — a circular dependency — is real but localized,
and is resolved by splitting the `generated` module along its true seam:
`sis_floor` (a foundational SIS table that `akita-types` core depends on) stays
in `akita-types`; only the schedule-table representation + expansion moves to
`akita-planner`.

## Intent

### Goal

Move the schedule-table representation, the shipped `generated/*.rs` tables, the
compact→`LevelParams` expansion, and the entry-walking proof-size accounting out
of `akita-types` into `akita-planner`, and expose a single planner entry point
`resolve_schedule(key, &PlannerPolicy, stage1, fold_shape, table)` that checks
the table cache first and falls back to the DP, so that `akita-config` and the
protocol never name `GeneratedScheduleTableEntry`.

### Affected abstractions, types, and boundaries

Moved **from `akita-types` to `akita-planner`** (new home, e.g.
`akita_planner::schedule_table` + `akita_planner::generated`):

- Schedule-table representation: `GeneratedScheduleKey`,
  `GeneratedScheduleTableEntry`, `GeneratedStep`, `GeneratedFoldStep`,
  `GeneratedDirectStep`, `GeneratedScheduleTable`, `table_entry`,
  `GeneratedScheduleTableEntry::validate`.
- The static tables `akita-types/src/generated/{fp*}.rs` and the per-family
  `*_table()` constructors + `small_field_table_fn!` macro.
- The expansion module `generated/expand.rs`
  (`GeneratedFoldStep::expand_to_level_params`, `generated_level_buckets`,
  `scale_batched_root`).
- The entry walker + estimator `schedule_from_entry_bits` /
  `estimate_proof_bytes` (currently in `akita-types::proof_size`).
- The lookup-key translation `generated_schedule_lookup_key` (currently in
  `akita-types::schedule`).

Stays **in `akita-types`** (foundational, depended on by `akita-types` core and
by `akita-planner`):

- `sis_floor` (`SisModulusFamily`, `min_rank_for_secure_width`,
  `ceil_supported_collision`) — renamed out of the `generated` namespace to
  `akita_types::sis_floor` since it is a security-floor table, not a generated
  *schedule*. `akita_types::SisModulusFamily` re-export is unchanged.
- `level_proof_bytes` and the pure byte/layout helpers it uses
  (`field_bytes`, `proof_ring_vec_bytes`, `sumcheck_rounds`,
  `direct_witness_bytes`, `extension_opening_reduction_proof_bytes`,
  `root_extension_opening_partials`,
  `w_ring_element_count_with_counts*`, `root_direct_commit_layout`,
  `a_role_base_norm`, `AjtaiKeyParams`, `LevelParams`, `Schedule`, `Step`,
  `FoldStep`, `DirectStep`, `AkitaScheduleLookupKey`, `AkitaScheduleInputs`,
  `DecompositionParams`). These remain the shared vocabulary `akita-planner`
  imports — `level_proof_bytes` does **not** reference any `Generated*` type, so
  it is not part of the move.

New `akita-planner` public surface (the planner owns table *selection*, not
just the table data):

```rust
// akita-planner

/// The shipped table for a policy, if one exists. Keyed on the SIS family +
/// ring degree, plus `onehot` (`decomposition.log_commit_bound == 1`) and
/// `root_fold_is_tensor`. `(family, D)` with no shipped table → `None`.
pub fn shipped_table(
    policy: &PlannerPolicy,
    root_fold_is_tensor: bool,
) -> Option<GeneratedScheduleTable>;

/// Cache-then-generate. Selects the shipped table via `shipped_table`
/// (deriving `root_fold_is_tensor` by evaluating `fold_shape` at level 0),
/// expands a matching compact entry, or regenerates with `find_schedule`.
pub fn get_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    stage1: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError>;
```

Semantics: `get_schedule` consults the planner-owned shipped-table cache for
`policy`; on a hit it expands the compact entry via the moved walker, on a miss
(or no shipped table for the policy) it runs `find_schedule`. Every input the
walker needs (`sis_family`, `decomposition`, `challenge_field_bits`,
`claim_ext_degree`, `ring_subfield_norm_bound`) is derivable from
`PlannerPolicy`, so the call shape matches `find_schedule`'s and no new policy
fields are required. Because the planner already owns every shipped table, it
also owns the `policy → table` mapping; `akita-config` no longer carries a
`schedule_table()` hook.

#### Table selection (planner-owned registry)

`shipped_table` is a `match` on `(sis_family, ring_dimension)` plus two binary
discriminators, both derivable without naming any `Cfg`:

- `onehot = (decomposition.log_commit_bound == 1)` — full-field presets carry
  `log_commit_bound == field_bits` (16/32/64/128), onehot presets `== 1`.
- `root_fold_is_tensor` — evaluated from the `fold_shape` closure at level 0.
  This is the *only* discriminator between the otherwise byte-identical
  `fp128_d64_onehot` and `fp128_d64_onehot_tensor` policies.

`(family, D)` pairs with no shipped table (the `D ∈ {128,256,512}`
experimental presets, and any recursive-w `WCommitmentConfig` policy whose
`log_commit_bound` is its `log_basis`) fall through to `None` → regenerate. This
is safe because `WCommitmentConfig` never resolves a schedule through a table
today (recursive-w layout is built by `recursive_level_layout_from_params`), so
no behavior changes there.

### Invariants

1. **No dependency cycle.** After the move, `akita-types` names no
   `akita-planner` type. The crate graph stays strictly
   `akita-config → akita-planner → akita-types/akita-challenges/akita-field`.
   Protected by: the workspace compiling (`cargo build --workspace`); a cycle
   would fail Cargo's resolver.
2. **Schedule equivalence (table hit).** For every shipped `(family, key)`,
   `resolve_schedule(.., Some(table))` produces the byte-identical `Schedule`
   that today's `runtime_schedule` table-hit path produces (same `LevelParams`,
   same `total_bytes`). Protected by: the existing drift-guard
   (`akita-config/tests/generated_tables.rs`) and proof-size comparison tests,
   which must continue to pass unchanged.
3. **Schedule equivalence (table miss).** `resolve_schedule(.., None)` is exactly
   `find_schedule` — no behavioral change on the DP path. Protected by: the
   planner's own tests + the runtime-fallback test
   (`akita-config/tests/runtime_fallback.rs`).
4. **Prover/verifier consistency.** Both still resolve through
   `CommitmentConfig::runtime_schedule`, which now delegates to the single
   planner entry point; the Fiat-Shamir `PlanSection` digest of the resolved
   schedule is unchanged because the resolved `Schedule` is unchanged. Protected
   by: transcript-hardening tests (`akita-pcs/tests/transcript_hardening.rs`) and
   e2e (`akita_e2e.rs`).
5. **Protocol agnosticism.** `akita-prover` and `akita-verifier` contain zero
   references to any `Generated*` schedule-table type (they already only use
   `SisModulusFamily`, which stays in `akita-types`). Protected by: a grep-level
   check / review, and the crates compiling without depending on the planner's
   schedule-table module.
6. **SIS security + verifier no-panic.** The moved expansion is verifier-reachable
   (config resolves levels through it on replay), so every fallible step still
   returns `AkitaError`, never panics; the strict `AjtaiKeyParams::try_new` audit
   still fires at the layout boundary. Protected by: the SIS-audit walk guards in
   `akita-config/src/lib.rs` tests and the no-panic contract review.

### Non-Goals

- **Not** changing the DP search algorithm, the proof-size formulas, the SIS
  floor numbers, or any shipped table contents. This is a pure relocation +
  single-entry-point refactor; tables are byte-identical (they may be
  regenerated to confirm, but must not change).
- **Not** moving the preset `Cfg` types, the family list
  (`ALL_GENERATED_FAMILIES`), or the `gen_schedule_tables` binary out of
  `akita-config` — those require naming presets and the planner is `Cfg`-free by
  design.
- **Not** introducing a runtime in-memory schedule cache/memoization beyond the
  existing static shipped table; "check cache" here means "consult the shipped
  `GeneratedScheduleTable` first," matching today's behavior.
- **Not** changing transcript bytes, proof format, or setup envelopes.

## Evaluation

### Acceptance Criteria

- [ ] `akita-types` no longer exposes a `generated` schedule-table module;
      `akita_types::{schedule_from_entry_bits, estimate_proof_bytes,
      generated_schedule_lookup_key}` and the `Generated*` types are gone from
      its public surface. `akita_types::sis_floor::*` and
      `akita_types::SisModulusFamily` remain.
- [ ] `akita-planner` exposes `resolve_schedule`, the `Generated*` types, the
      shipped tables, the `*_table()` constructors, and `schedule_from_entry_bits`
      / `estimate_proof_bytes`.
- [ ] `CommitmentConfig::runtime_schedule` is a single delegation to
      `akita_planner::get_schedule(key, &policy_of::<Self>(),
      Self::stage1_challenge_config, Self::fold_challenge_shape_at_level)`.
- [ ] `CommitmentConfig::schedule_table()` and the trait `resolve_schedule`
      entry-lookup helper are **removed**; no preset, `WCommitmentConfig`, or
      test config implements them. The `$table` macro argument is gone.
- [ ] `akita_planner::shipped_table(policy, root_fold_is_tensor)` is the single
      `policy → table` registry; `get_schedule` consults it. The tensor preset
      resolves to `fp128_d64_onehot_tensor` and the flat onehot preset to
      `fp128_d64_onehot` (validated by `single_poly_tensor_e2e`).
- [ ] `gen_schedule_tables` writes into `akita-planner/src/generated/` and emits
      `use super::{...}` against the planner's representation; the drift guard
      compares against `akita_planner::generated::*`.
- [ ] `cargo build --workspace` and `cargo build --workspace --features zk`
      succeed (no cycle, both feature sets).
- [ ] `cargo test --workspace` and `... --features zk` pass with no table-content
      diffs; the drift guard reports zero drift.
- [ ] `akita-prover` and `akita-verifier` reference no `Generated*` type.

### Testing Strategy

Must continue passing unchanged (proves byte-for-byte equivalence):

- `akita-config/tests/generated_tables.rs` (drift guard, positional parity).
- `akita-config/tests/proof_size_comparison.rs`,
  `akita-config/tests/regen_diff.rs`, `akita-config/tests/runtime_fallback.rs`.
- `akita-types/src/proof_size.rs` byte-formula tests — these move with the walker
  into `akita-planner` (they exercise `level_proof_bytes` + the walker together;
  `level_proof_bytes` stays in types, so the tests that only touch it can stay,
  and the walker tests follow `schedule_from_entry_bits`).
- `akita-pcs/tests/{akita_e2e,transcript_hardening,setup}.rs`.

New / relocated tests:

- The `generated_schedule_lookup_key` aliasing test in
  `akita-types/src/schedule.rs` moves to `akita-planner`.
- A planner unit test that `resolve_schedule(.., Some(table))` and the prior
  two-call path (`table_entry` → `schedule_from_entry_bits`) agree, and that
  `resolve_schedule(.., None) == find_schedule`.

Feature combinations: run the full suite under both default and `zk` (the
walker's witness-length / proof-byte math differs under `zk`; regeneration must
match the shipped `zk` tables).

### Performance

No runtime performance change is expected: the table-hit path runs the same
expansion code (now linked from `akita-planner` instead of `akita-types`), and
the table-miss path is the unchanged DP. No proof-size, setup-size, or
transcript-byte change. The `profile` command
(`AKITA_MODE=onehot_fp128_d32 AKITA_NUM_VARS=32 cargo run --release --example
profile`) must report identical planned vs. serialized bytes as before the move.

## Design

### Architecture

The current `generated` module conflates two concerns under one name:

```
akita-types::generated
├── sis_floor            ← SIS security-floor table (SisModulusFamily, ranks)
│                          USED BY akita-types core: layout/params.rs,
│                          layout/digit_math.rs, sis_offline.rs, schedule.rs
└── schedule-table repr  ← GeneratedScheduleKey/Entry/Step + static tables
    + expand.rs            + expand_to_level_params + scale_batched_root
                           USED BY akita-config (runtime_schedule, gen bin)
                           and akita-types::proof_size walker
```

The naive "move all of `generated` to `akita-planner`" creates a cycle because
`akita-types` core depends on `sis_floor` (and `schedule.rs` depends on
`GeneratedScheduleKey` via `generated_schedule_lookup_key`). If those moved,
`akita-types → akita-planner` while `akita-planner → akita-types` — a cycle.

The fix is to cut along the true seam:

```
                 BEFORE                                  AFTER
akita-types::generated::sis_floor   ──►  akita-types::sis_floor        (stays)
akita-types::generated::{Entry,…}   ──►  akita-planner::generated      (moves)
akita-types::generated::expand      ──►  akita-planner::expand         (moves)
akita-types::proof_size::walker     ──►  akita-planner::schedule_table (moves)
akita-types::schedule::lookup_key   ──►  akita-planner                 (moves)
akita-types::proof_size::level_proof_bytes  (stays — no Generated* dep)
```

Resulting crate graph (unchanged shape, strictly one-directional):

```
akita-config ──► akita-planner ──► akita-types / akita-challenges / akita-field
  (presets,        (Cfg-free DP +
   family list,     schedule-table repr + tables + expansion +
   gen bin,         resolve_schedule cache-then-DP)
   runtime_schedule
   = 1-line delegate)
```

`runtime_schedule` collapses from the current branch (which straddled the crate
boundary and called `Self::schedule_table()` / `Self::resolve_schedule`):

```rust
// BEFORE (akita-config) — schedule_table() + resolve_schedule() trait hooks
fn runtime_schedule(key) -> Result<Option<Schedule>, AkitaError> {
    if let Some(entry) = Self::resolve_schedule(key)? {        // table_entry + validate (Cfg hook)
        return Ok(Some(akita_types::schedule_from_entry_bits(  // akita-types walker
            entry, key, sis_family, decomposition, challenge_field_bits,
            claim_ext_degree, ring_subfield_norm_bound, stage1, fold_shape)?));
    }
    Ok(Some(akita_planner::find_schedule(key, &policy_of::<Self>(), stage1, fold_shape)?))
}
```

to a true one-line delegation (no `schedule_table()` / `resolve_schedule` trait
methods at all):

```rust
// AFTER (akita-config)
fn runtime_schedule(key) -> Result<Option<Schedule>, AkitaError> {
    Ok(Some(akita_planner::get_schedule(
        key,
        &policy_of::<Self>(),
        Self::stage1_challenge_config,
        Self::fold_challenge_shape_at_level,
    )?))
}
```

The planner's `get_schedule` owns the entire cache-then-generate flow — table
selection, lookup, expansion, and DP fallback — deriving the walker's extra
arguments from `policy`:

```rust
// akita-planner
pub fn get_schedule(key, policy, stage1, fold_shape) -> Result<Schedule, AkitaError> {
    let root_fold_is_tensor = /* fold_shape(level 0) == Tensor */;
    if let Some(table) = shipped_table(policy, root_fold_is_tensor) {
        if let Some(entry) = table_entry(table, generated_schedule_lookup_key(key)) {
            return schedule_from_entry(entry, key, policy, stage1, fold_shape);
        }
    }
    find_schedule(key, policy, stage1, fold_shape)
}
```

`schedule_from_entry` is the relocated `schedule_from_entry_bits`, re-signed to
take `&PlannerPolicy` instead of the seven loose values
(`sis_family`, `root_decomp`, `challenge_field_bits`, `extension_opening_width`,
`ring_subfield_norm_bound`, …), all projections of `PlannerPolicy`:

- `sis_family` = `policy.sis_family`
- `root_decomp` = `policy.decomposition`
- `challenge_field_bits` = `policy.decomposition.field_bits() * policy.chal_ext_degree`
- `extension_opening_width` = `policy.claim_ext_degree`
- `ring_subfield_norm_bound` = `policy.ring_subfield_norm_bound`

This is the same projection `find_schedule` performs internally, so all three
entry points are symmetric.

#### `CommitmentConfig` surface shrinks

`schedule_table()` and the `resolve_schedule` entry-lookup helper are **removed**
from the trait. Presets no longer declare their table (the `$table` macro
argument is gone); the planner registry is the single source of the
`policy → table` binding. The drift-guard family list keeps a `schedule_table`
function pointer per family, now pointing directly at the concrete planner table
constructor (e.g. `|| Some(akita_planner::generated::fp128_d32_full_table())`)
rather than `Cfg::schedule_table`. The former trait-override test seam (a stub
`Cfg` injecting a synthetic uncommittable entry through `resolve_schedule`) is
replaced by a direct unit test of the extracted `root_commit_params(&Schedule)`
helper.

#### Visibility migration (the concrete `pub` work)

The moved `expand.rs` and walker consume several `akita-types` items that are
currently crate-internal or only reachable inside `akita-types`. Each must be
`pub` on `akita-types` after the move:

- `akita_types::sis_floor::{ceil_supported_collision, min_rank_for_secure_width}`
  (rename `generated::sis_floor` → `sis_floor`; keep re-exports).
- `a_role_base_norm`, `root_direct_commit_layout`,
  `w_ring_element_count_with_counts*`, `extension_opening_reduction_proof_bytes`,
  `root_extension_opening_partials`, `direct_witness_bytes`,
  `level_proof_bytes`, `field_bytes`, `proof_ring_vec_bytes`, `sumcheck_rounds`,
  `stage1_tree_stage_shapes` — confirm each is `pub` (most already are, since the
  planner DP imports a subset today).

This is the main mechanical risk and should be the first implementation step:
make the seam compile by widening visibility before relocating code.

#### Downstream type renaming

- `akita-scheme/src/tests/fp32_ring_subfield.rs` references
  `akita_types::generated::GeneratedScheduleTable` in two `schedule_table()`
  impls → switch to `akita_config::GeneratedScheduleTable` (a re-export of the
  planner type), avoiding a direct `akita-scheme → akita-planner` dependency
  edge.
- `akita-config` re-exports `GeneratedScheduleTable` (and any `Generated*` types
  test configs need) from `akita-planner` so config-level consumers have a stable
  name.

### On the circular-dependency concern in `generated_families.rs`

The user specifically worried about `generated_families.rs`. Analysis: it does
**not** introduce a cycle.

- `generated_families.rs` lives in `akita-config` and may stay there: it is the
  one place a preset `Cfg` is bound to its regen hook and shipped table, and only
  `akita-config` can name presets.
- Its dependencies all point **downward**: `akita_planner::find_schedule`
  (config → planner), the planner's `GeneratedScheduleTable` type (config →
  planner), and the in-crate presets (`fp128::D32Full`, …).
- The `schedule_table` function-pointer field becomes
  `fn() -> Option<akita_planner::GeneratedScheduleTable>` = `Cfg::schedule_table`;
  still downward.

So the family list, the generator binary, and the drift guard remain in
`akita-config`, calling down into the planner. The only real cycle hazard is the
`sis_floor` + `generated_schedule_lookup_key` coupling inside `akita-types`,
resolved by the seam split above.

### Alternatives Considered

1. **Move the entire `generated` module (including `sis_floor`) to
   `akita-planner`.** Rejected: `akita-types` core
   (`layout/params.rs`, `layout/digit_math.rs`, `sis_offline.rs`) depends on
   `sis_floor`, so this creates the `akita-types → akita-planner` cycle. The SIS
   floor is a foundational security table, not generated schedule data; it
   belongs in `akita-types`.

2. **Keep the walker in `akita-types`, only move the static tables.** Rejected:
   leaves `schedule_from_entry_bits` and the `Generated*` representation on the
   public `akita-types` surface, so the protocol-agnosticism goal (criterion 5)
   is not met and the three-crate straddle persists.

3. **Add a runtime in-memory cache (memoize resolved `Schedule`s) inside
   `resolve_schedule`.** Deferred (Non-Goal): the shipped static table already
   serves as the "cache." A keyed runtime memo is an orthogonal optimization that
   can be layered later without changing this API.

4. **Push the family list + gen binary down into `akita-planner` too.** Rejected:
   impossible without the planner naming preset `Cfg` types, which would break
   the `Cfg`-free invariant that the prior planner refactor established.

## Documentation

- Update `AGENTS.md` crate descriptions: move "schedule-table materialization
  (`schedule_plan_from_table`, …)" / generated-table ownership language from
  `akita-types` and `akita-config` to `akita-planner`; note `akita-planner` now
  owns `resolve_schedule`, the `Generated*` representation, the shipped tables,
  and the expansion, while `akita-config` keeps the family list + gen binary +
  `runtime_schedule` delegation.
- Update `specs/planner-refactor.md` references (this spec is a follow-on:
  "parameter flow" section's claim that expansion lives in
  `akita_types::schedule_from_entry_bits` becomes `akita_planner`).
- Update crate-level docs (`akita-planner/src/lib.rs`,
  `akita-config/src/lib.rs`, `akita-types/src/lib.rs`) to reflect the new module
  homes.

## Execution

Suggested ordering (each step should compile + test before the next):

1. **Rename the seam, no move yet.** `akita-types`: rename
   `generated::sis_floor` → `sis_floor`; widen visibility of every helper the
   walker/expand consume to `pub`. Confirm `cargo build --workspace` and `--zk`.
2. **Relocate the representation + tables.** Move `Generated*` types,
   `table_entry`, the `generated/*.rs` static tables, the `*_table()`
   constructors + macro, and `generated_schedule_lookup_key` into
   `akita-planner` (e.g. `akita_planner::generated`). Fix imports.
3. **Relocate expansion + walker.** Move `expand.rs` and
   `schedule_from_entry_bits` / `estimate_proof_bytes` into `akita-planner`;
   re-sign the walker to take `&PlannerPolicy`. Keep `level_proof_bytes` in
   `akita-types`.
4. **Add `resolve_schedule`** and collapse `CommitmentConfig::runtime_schedule`
   to the single delegation; remove the trait's `resolve_schedule` entry-lookup
   helper. Re-export `GeneratedScheduleTable` from `akita-config`.
5. **Repoint tooling.** `gen_schedule_tables` output dir → `akita-planner/src/
   generated/`; emitted `use super::{…}`; drift guard → `akita_planner::generated`.
   Regenerate tables; assert zero diff.
6. **Repoint downstream test configs** (`akita-scheme` `schedule_table()` impls)
   to the re-exported type.
7. **Full verification.** Build + test both feature sets; run the drift guard and
   the `profile` byte check; grep `akita-prover`/`akita-verifier` for `Generated`.

Risks to resolve first:

- Visibility widening (step 1) is the make-or-break: if some helper cannot be
  cleanly made `pub` (e.g. it leaks a `pub(crate)` type), that type must also be
  exported or the helper kept in `akita-types` and called from the planner.
- `zk`-gated table modules and the `#[cfg(feature = "zk")]` `*_table()` arms must
  move together to keep both feature sets compiling.

## References

- `specs/planner-refactor.md` — establishes the `Cfg`-free DP + dependency
  inversion this builds on (supersedes the materialization model).
- `specs/akita-pcs-crate-decomposition.md` — crate-boundary rationale.
- Key sources: `crates/akita-planner/src/{lib,schedule_params,ajtai_params}.rs`,
  `crates/akita-types/src/{proof_size,schedule,sis_offline}.rs`,
  `crates/akita-types/src/generated/{mod,expand}.rs`,
  `crates/akita-types/src/layout/{params,digit_math,mod}.rs`,
  `crates/akita-config/src/{lib,generated_families}.rs`,
  `crates/akita-config/src/bin/gen_schedule_tables.rs`,
  `crates/akita-config/tests/generated_tables.rs`.
- Profile command: `AKITA_MODE=onehot_fp128_d32 AKITA_NUM_VARS=32 cargo run --release --example profile`.
