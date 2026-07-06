**NEVER COMMIT THIS FILE.**

# Relation Weight Kernel Cutover Worklog

## Goal & Scope

Track the stacked PR that completes the prover-kernel cutover after
`quang/relation-weight-polynomial-spec`. The goal is to move Stage 2 from
legacy split/prefix-shaped implementation plumbing to flat relation-weight
indexing and pair-scan/fold-scan abstractions, guided by
`specs/sumcheck-kernel-cutover.md` and `specs/relation-weight-polynomial.md`.

## Starting State

- Branch: `quang/relation-weight-kernel-cutover`
- Base commit: `75f386dead6b6f11dc30c7fd4317264ae0bb177c` (`merge main into relation weight branch`)
- Related spec: `specs/sumcheck-kernel-cutover.md`
- Stacked on PR: `quang/relation-weight-polynomial-spec`

## Plan

1. Define the flat pair-scan interfaces and consolidate Stage 2 dense traversal
   behind them.
2. Rename or delete the remaining legacy x/y-facing Stage 2 implementation
   surfaces where that can be done without changing protocol bytes.
3. Keep `round_batching` as a fast path, but make its boundary match the same
   flat pair-scan contract.
4. Add equivalence tests that protect fast paths against the flat reference.

## Decisions

- **[2026-07-06] Worklog filename differs from the default.**
  Chose `RELATION-WEIGHT-KERNEL-WORKLOG-NEVER-COMMIT.md` because the shared git
  exclude currently ignores the default `WORKLOG-NEVER-COMMIT.md` name.
- **[2026-07-06] First stacked slice keeps the current visit order and changes
  the read boundary first.** Chose to introduce `WitnessPolynomial` as a
  borrowed flat read view before deleting prefix modules. Reason: this gives the
  dense pair scan the final live-address API while preserving proof bytes and
  keeping `round_batching` performance unchanged.
- **[2026-07-06] Second slice renames Stage 2 prefix vocabulary inside the
  module, but not shared round-batching signatures.** Chose
  `segment_prefix` / `coefficient_prefix` for the current local storage
  patterns. Reason: it removes x/y protocol language from Stage 2 round-state
  code while keeping the optimized round-batching API stable for a later,
  dedicated migration.
- **[2026-07-06] Third stacked commit introduced flat constructor fields.**
  Interim shape was `Stage2Layout { live_len, num_vars, uniform_tiling }`.
  Slice 1 replaces that bridge with `Stage2Geometry { live_len, num_vars,
  local_view }`, where `local_view: Option<ScalarLevelLocalView>` is
  prover-only fast-path metadata and never the Stage 2 contract.

## Deviations

- None yet.

## Tradeoffs

- **[2026-07-06] Considered renaming/deleting `x_prefix` and `y_prefix` in the
  first stacked commit.** Deferred that to a later slice because those modules
  still contain performance-sensitive prefix and fused-fold paths. The first
  slice instead establishes the flat read contract that those paths can migrate
  to.
- **[2026-07-06] Considered deleting all uniform prefix fast paths immediately.**
  Deferred deletion because the user asked to keep efficiency. Fast paths are
  opt-in through `local_view`; when no valid scalar-level embedding exists,
  Stage 2 uses the flat dense path only.

## Open Questions

- None currently.

## Strict Final End State

This section is the handoff contract after the 2026-07-06 subagent audit.
It overrides any softer interpretation in earlier notes. The final PR must
move strictly toward this shape.

### Canonical Stage 2 Statement

Stage 2 proves exactly:

```text
input_claim = relation_weight_claim + gamma * s_claim

expected_output_claim(r)
  = w(r) * RelationWeightPolynomial(r)
  + gamma * eq(stage1_point, r) * w(r) * (w(r) + 1)
```

Terminal Stage 2 has `gamma = 0`, so only the relation term remains.

### Canonical Types And APIs

- `RelationWeightPolynomial<E>` is prover-side, live-only, flat, and has exactly
  one semantic length: `evals.len() == live_len == witness.live_len()`.
- `WitnessPolynomial<'a, E>` is the prover storage view:
  `CompactDigits(&'a [i8]) | FieldEvals(&'a [E])`.
- `PreparedRelationWeightPolynomial` is verifier-side and exposes one semantic
  operation: evaluate the same flat relation-weight polynomial at the Stage 2
  final point.
- `PairStep` contains only flat addresses:

  ```rust
  struct PairStep {
      idx0: usize,
      idx1: usize,
  }
  ```

- `FlatPairStream` emits flat `PairStep`s only.
- `RangeBindingTerm` is optional and computes only
  `gamma * eq(stage1_point, z) * w(z) * (w(z) + 1)`.
- `RelationRowLayout` is the single row-family source of truth:
  `EvaluationTrace | FoldEvaluation | FoldConsistency | OuterConsistency |
  OpeningConsistency`.
- `RelationQuotientLayout` is derived from row families and their dimensions,
  never from a homogeneous global row/ring dimension.

### Allowed Local Concepts

- Flat address, live length, Boolean width, bind schedule, fold schedule.
- Segment layout and witness-segment offsets only when used to embed local
  witness positions into flat addresses.
- Row-family layout, row-family ring dimension, quotient slice.
- Family-local coordinates inside row-family evaluators, provided they are
  immediately embedded through:

  ```text
  embed_family(local row-family address) -> Option<flat_live_address>
  ```

- Initial `round_batching`, but only as a prover-only optimization over the
  same flat witness and relation-weight vectors.
- Cache/locality names such as blocked scan, embedded local-axis scan, fused
  fold-scan, and relation-family products.

### Prohibited Final Concepts

These must not survive in live Stage 2 kernel APIs, dispatch names, docs, tests,
or verifier-facing names:

- Protocol-level `x` / `y` axes.
- `prefix` as a generic Stage 2 sumcheck concept.
- `segment_prefix`, `coefficient_prefix`, `prefix_r_stage1`,
  `use_*_prefix_round`, `*_prefix_terms`, `*_prefix_polys`.
- `uniform_tiling` / `UniformStage2Tiling` as a Stage 2 contract. Slice 1
  replaced the interim bridge with `local_view: Option<ScalarLevelLocalView>`.
- `alpha_evals_y`, `m_evals_x`, or any split `alpha * m` Stage 2 handoff.
- `trace_table`, `trace_coeff`, `trace_opening_claim`, `gamma^2`, or trace-side
  Stage 2 summands.
- `fold_relation_weight_for_round` dispatch trees.
- Global `ring_len`, global ring dimension, or global `col_bits + ring_bits` as
  the Stage 2 abstraction.
- Public renames that only hide the old axes. A name is acceptable only if the
  API actually consumes/emits flat addresses or row-family layouts.

### Mixed Ring-Dimension Requirements

Final Stage 2 must support `d_a`, `d_b`, and `d_d` without assuming equality.

- Each row family carries its own `ring_dim`.
- Each quotient slice carries its own `ring_dim`, `digit_depth`, and
  `log_basis`.
- Relation-weight construction adds every row-family contribution into one flat
  `RelationWeightPolynomial`.
- Family-local coordinates may use that family's ring dimension only inside the
  evaluator/builder.
- The Stage 2 prover kernel never asks which ring dimension a round belongs to.
  It only folds/scans flat live vectors.
- Padded Boolean-domain addresses are fixed zero for both `w` and
  `RelationWeightPolynomial`.
- No relation-weight storage is materialized for padded addresses.
- Mixed dimensions must not route through a hidden uniform fallback. The final
  state removes all Stage 2 relation-weight proving boundaries that require
  `d_a == d_b == d_d`.

### Done Means

- Stage 2 round computation has exactly two top-level paths:
  `round_batching` or flat pair scan / fused fold-scan.
- All non-batching scans share one relation/range `accumulate_pair` kernel.
- Witness and relation-weight folds use the same flat fold schedule.
- Fused fold-scan is first-class and has no legacy axis names.
- `RelationWeightPolynomial` and `WitnessPolynomial` zero-extend identically.
- Ring switch emits `relation_weight_evals`, `relation_weight_claim`,
  `live_len`, and `num_vars` directly.
- The verifier evaluates `PreparedRelationWeightPolynomial(r)` as the same
  zero-extended flat polynomial.
- Grep for prohibited vocabulary in live Stage 2/protocol docs is clean, except
  historical specs explicitly marked superseded.
- Mixed-dimension fixtures prove `d_a != d_b != d_d` without changing Stage 2
  kernel APIs.
- `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`,
  `cargo test`, and doc guardrails pass.

## Cheat Paths And Footguns

### P0: Mixed Per-Role Ring Dims Still Rejected

- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`
  `ring_switch_build_w` dispatches on `dims.d_a()` and rejects/forces the
  opening role to use the same dimension.
- It calls `instance.ensure_ring_dim::<D>()` and `instance.y_trusted::<D>()`,
  which force a uniform ring view.

Why dangerous: code can pass mixed-D-per-level tests while still failing the
stricter same-fold shape `d_a != d_b != d_d`.

Guard:

- Add a positive prover ring-switch test with
  `CommitmentRingDims { d_a: 128, d_b: 64, d_d: 32 }`.
- The test must not call `uniform_dim()` or borrow `y_trusted::<d_a>()`.
- Add a focused grep guard for `uniform_dim()` in ring-switch/relation paths
  once the replacement is in place.

### P0: Production Stage 2 Still Enters Through Uniform Tiling

- **Resolved in Slice 1.** `fold.rs` now builds `Stage2Geometry::from_production_handoff`
  from `witness_len + stage1_point.len()` and only optionally attaches
  `ScalarLevelLocalView` when ring-switch tile metadata embeds consistently.

Guard retained:

- `stage2_production_handoff_geometry_uses_flat_contract` and
  `stage2_production_handoff_rejects_tile_product_as_semantic_live_len`.

### P0: Relation-Weight Materialization Still Uses One Global Ring Axis

- `crates/akita-prover/src/protocol/ring_switch/evals.rs`
  `build_relation_weight_evals` still takes `ring_bits` and `live_x_cols`,
  builds `alpha_evals_y`, materializes trace as `live_x_cols * y_len`, and emits
  `alpha_y * column_weight + trace`.

Why dangerous: this is the hidden same-ring-dimension assumption. Mixed row
families cannot be represented by one global coefficient axis.

Guard:

- Replace with direct flat-address accumulation into `RelationWeightPolynomial`.
- Test materialized prover weights against verifier closed form for
  `d_a != d_b != d_d`, including out-of-live addresses that must stay zero.

### P0: Verifier Prepared Evaluator Still Uses The Same Split

- `crates/akita-verifier/src/protocol/relation_weight.rs`
  `PreparedRelationWeightPolynomial::eval_at_point` splits challenges into
  old local halves, evaluates scalar powers, evaluates deferred rows separately,
  and returns an `alpha_val * row_val + trace_val` shape.

Why dangerous: prover and verifier remain tied to a global factorization and
will mis-evaluate the final flat relation polynomial once row-family embeddings
stop sharing one dimension.

Guard:

- Add verifier tests that compare `PreparedRelationWeightPolynomial::eval_at_point`
  to direct `multilinear_eval` on a small materialized flat
  `RelationWeightPolynomial`, including mixed family dimensions and a
  non-power-of-two live length.

### P1: Round Batching Still Encodes The Old Grid

- `crates/akita-prover/src/protocol/sumcheck/round_batching/stage2.rs`
  still takes `live_x_cols`, `col_bits`, and `ring_bits`, asserts rectangular
  lengths, and slices the stage-1 point by the old axis split.

Why dangerous: it is a hidden uniform tiling path and still contains panicking
asserts around shape assumptions.

Guard:

- Round batching must either sit behind a clearly local flat-address mapper or
  compare against `pair_scan` for the exact same flat live vector.
- Add `round_batching_matches_pair_scan` for non-rectangular live lengths.

### P1: Old Two-Axis Stage 2 Prefix Modules Remain Reachable

- `coefficient_prefix.rs`, `segment_prefix.rs`, and `round2_prefix.rs` still
  preserve separate fold/scan semantics for the old local axes.

Why dangerous: `pair_scan.rs` is the clean flat kernel, but these modules are
the main route back to the old model.

Guard:

- Acceptance condition: one Stage 2 scan entry point, except isolated
  `round_batching`.
- Add temporary equivalence tests before deletion:
  `pair_scan_matches_legacy_local_axis_inner`,
  `pair_scan_matches_legacy_local_axis_outer`, and
  `fused_fold_scan_matches_legacy_fused_path`.

### P1: Padded-Reference Tests Can Bless The Wrong Model

- Several Stage 2 tests compare prefix paths to padded full-hypercube
  references.

Why dangerous: the target invariant is not "padded advice happens to match";
  the target invariant is "padded entries are fixed implicit zero and cannot be
  supplied as advice."

Guard:

- Keep/extend boundary-style tests where arbitrary out-of-live witness advice is
  rejected at construction.
- Add full Stage 1/Stage 2 guard tests where out-of-live witness data cannot
  change relation or range terms because it cannot enter the boundary.

### P2: Duplicate Prepared Relation-Weight API Stub

- `crates/akita-types/src/relation_weight/prepared.rs` exports a placeholder
  `PreparedRelationWeightPolynomial` while the verifier owns the real evaluator.

Why dangerous: two canonical-looking APIs violate the no-wrapper/no-stub final
shape.

Guard:

- Delete the stub/export. The verifier evaluator should be the single prepared
  relation-weight implementation.

### P2: Old Quotient/Product Names Remain

- `crates/akita-prover/src/compute/plans.rs` still has old relation/quotient row
  product names.

Why dangerous: not directly unsound, but it keeps the old quotient framing alive
and makes mixed-family products easier to wire to the wrong abstraction.

Guard:

- Rename to relation-family product names from the spec and add an `rg` guard
  for old names.

### P2: Specs Still Contain Softer Non-Uniform Language

- `specs/relation-weight-polynomial.md` still says not to enable non-uniform
  `d_a/d_b/d_d` in this PR.
- `specs/runtime-ring-cutover.md` defers divergent per-role planner emission.
- `specs/y-ring-trace-internalization.md` still has old vocabulary despite
  being superseded.

Why dangerous: future agents can cite those lines to justify leaving the exact
paths the user now wants gone.

Guard:

- Update specs to the new strict target before or alongside the mixed-dimension
  implementation slices.
- Run `./scripts/check-doc-guardrails.sh`.
- Add focused grep checks for old protocol concepts once replacements land.

## Remaining Implementation Slices

These slices are ordered. Do not skip ahead by preserving a hidden uniform
fallback.

### Slice 1: Stage 2 Geometry Cutover

**Status: complete (2026-07-06).**

Goal: make flat live indexing the Stage 2 truth, with any scalar-level layout as
an explicit local fast-path view only.

Owned files/modules:

- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/{mod.rs,lifecycle.rs,round_flow.rs,pair_scan.rs}`
- `crates/akita-prover/src/protocol/core/fold.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`

Concrete changes:

- Replace `Stage2Layout::uniform(...)` and fields such as `live_segments`,
  `relation_coeff_len`, and `segment_bits` with a stricter
  `Stage2Geometry { live_len, num_vars, local_view: Option<ScalarLevelLocalView> }`.
- `prove_stage2` passes `live_len = w_evals_compact.len()` and
  `num_vars = ceil_log2(live_len)` or the already-public Boolean width, never
  `live_tiles * lane_len` as the semantic contract.
- The local view is selected only by stream/fold planning and cannot be required
  by Stage 2 construction.

Tests/verification:

- `cargo test -p akita-prover akita_stage2`
- Extend `stage2_flat_geometry_uses_dense_path_without_local_view` so the
  production Stage 2 path cannot regress to uniform construction.

Stopping condition:

- Constructor and round flow can run from `live_len + num_vars`.
- Local view is optional and only used for optimization planning.

Anti-cheat guardrails:

- Do not rename old axes while keeping the same abstraction.
- Do not infer protocol shape from `ring_bits`.
- Do not materialize full padded relation tables.

### Slice 2: One Stage 2 Pair-Scan Surface

**Status: complete (2026-07-06).**

Goal: all non-batched Stage 2 round messages go through one flat pair scanner.

Owned files/modules:

- `akita_stage2/pair_scan.rs`
- `akita_stage2/coefficient_prefix.rs`
- `akita_stage2/segment_prefix.rs`
- `akita_stage2/round_flow.rs`

Concrete changes:

- Add `EmbeddedLocalAxisPairs` or equivalent flat-address stream for local fast
  views.
- Add one `scan_round(stream, WitnessPolynomial, RelationWeightPolynomial,
  RangeBindingTerm)`.
- Move compact/full virtual accumulation from the prefix modules into the shared
  scanner.
- Prefix modules may temporarily only define local-address embedding helpers,
  then must disappear.

Tests/verification:

- Compare current local-axis paths against the new scanner on odd live lengths
  and multiple `b`.
- `cargo test -p akita-prover akita_stage2`

Stopping condition:

- `round_flow.rs` dispatch is `initial round batch | pair scan`.
- No separate `compute_round_*_prefix_terms`.

Anti-cheat guardrails:

- `PairStep` remains only `{ idx0, idx1 }`.
- No row/column/axis/equality-weight/relation-family metadata in pair identity.

### Slice 3: Unified Fold And Fused Fold-Scan

Goal: preserve current fused performance while removing prefix-shaped fold
branches.

Owned files/modules:

- `akita_stage2/{round_flow.rs,round2_prefix.rs,segment_prefix.rs,pair_scan.rs,mod.rs}`

Concrete changes:

- Introduce `fold_witness_polynomial`, `fold_relation_weight`, and
  `FusedFoldScan` over the same flat schedule.
- Port `fuse_compact_to_round2_and_compute_round` and
  `fuse_full_segment_prefix_and_compute_round` into that kernel.
- Keep compact lookup tables and hot-cache next-round computation.

Tests/verification:

- Existing fused transition tests.
- Add flat schedule cases where live length is odd.

Stopping condition:

- No `fold_relation_weight_for_round`, `fold_relation_weight_*prefix`, or
  `fuse_*prefix*` names remain in live Stage 2 code.

Anti-cheat guardrails:

- Do not replace fusion with a slow fold-then-rescan path.
- Do not materialize padded Boolean entries to make indexing easy.

### Slice 4: Row-Family Relation-Weight Materialization

Goal: implement mixed `d_a/d_b/d_d` relation weights directly by semantic row
family, not by any split factorization.

Owned files/modules:

- `crates/akita-prover/src/protocol/ring_switch/evals.rs`
- `crates/akita-prover/src/protocol/ring_switch/finalize.rs`
- `crates/akita-types/src/layout/relation_rows.rs`
- ring-switch tests

Concrete changes:

- Replace global-axis relation-weight materialization with a builder that
  iterates `RelationRowLayout` families.
- Evaluate each family in its own `ring_dim`.
- Embed each family contribution into the single flat witness address space.
- Keep `EvaluationTrace` as row 0.

Tests/verification:

- Uniform output equals current relation weights.
- Add mixed role-dimension fixtures where A/B/D family dimensions differ.
- `cargo test -p akita-pcs --test ring_switch`
- `mixed_d_per_level_e2e` or equivalent positive same-fold mixed-role test.

Stopping condition:

- Stage 2 handoff has only `relation_weight_evals`, `relation_weight_claim`,
  `live_len`, `num_vars`, and optional local fast-path view.

Anti-cheat guardrails:

- No production bridge from `m_evals_x` / `alpha_evals_y`.
- No global ring length.
- No uniform fallback for mixed dimensions.

### Slice 5: Verifier Prepared Evaluator Becomes Semantic

Goal: verifier evaluates the same `RelationWeightPolynomial(r)` as the prover,
including mixed dimensions.

Owned files/modules:

- `crates/akita-verifier/src/protocol/relation_weight.rs`
- `crates/akita-verifier/src/stages/stage2.rs`
- verifier ring-switch code

Concrete changes:

- Replace challenge splitting as a Stage 2 API with a flat point plus
  row-family evaluators.
- Local family evaluators may split coordinates only behind
  `PreparedRelationWeightPolynomial::eval_at_point`.
- Delete old alpha-row-trace final evaluation shape.

Tests/verification:

- `cargo test -p akita-verifier`
- E2E direct and recursive setup modes.
- Malformed mixed-shape tests return `AkitaError`.

Stopping condition:

- `AkitaStage2Verifier::expected_output_claim` only computes
  `w(r) * relation_weight_eval + gamma * eq * w(w+1)`.

Anti-cheat guardrails:

- No `alpha_val * row_val` as a verifier-stage concept.
- No trace-side summand.

### Slice 6: Stage 1 Shared Flat Range Scan And Final Cleanup

Goal: remove remaining old prover vocabulary and share flat range machinery
without changing Stage 1 protocol.

Owned files/modules:

- `akita_stage1/{lifecycle.rs,round_flow.rs,x_prefix.rs,sparse_y.rs,round2_prefix.rs}`
- shared `sumcheck/kernel`
- docs/tests

Concrete changes:

- Move common `WitnessPolynomial`, flat streams, range accumulation, and folding
  schedule into shared kernel.
- Route Stage 1 through it.
- Delete Stage 1 old prefix/sparse scan duplication.

Tests/verification:

- `cargo test -p akita-prover akita_stage1`
- `cargo test`
- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- doc guardrails if docs touched

Stopping condition:

- Grep live code under `protocol/sumcheck` for forbidden concepts and fix all
  production hits.

Anti-cheat guardrails:

- Do not change Stage 1 wire format.
- Do not hide old names behind wrappers.
- Do not delete files merely to silence tests before replacement coverage
  exists.

## Slice Retrospectives

### 2026-07-06 retrospective: flat witness read boundary

**Bottom line:** no blockers. The dense Stage 2 scan now reads compact and
field witness storage through a `WitnessPolynomial` flat live-address view, and
the module docs no longer describe the protocol as an x/y product.

- `Deferred:` Prefix and fused-fold paths still use legacy local names and
  layout concepts. This is intentional for the first stacked slice; they should
  migrate onto the same flat pair/fold vocabulary in later commits.
- `Risk:` `WitnessPolynomial` currently exposes separate `compact_pair` and
  `field_pair` methods. This keeps the small-digit arithmetic optimized, but it
  is not yet the final single kernel abstraction.
- `Non-issue checked:` The dense scan still produces the same Stage 2 behavior.
  The full Stage 2 prover test filter passed after the read-boundary change and
  `WitnessTable` rename.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover witness_polynomial -- --nocapture`
    -> `test result: ok. 2 passed; 0 failed`
  - `cargo fmt -q && cargo test -p akita-prover akita_stage2 -- --nocapture`
    -> `test result: ok. 21 passed; 0 failed`

### 2026-07-06 retrospective: segment/coefficient prefix vocabulary

**Bottom line:** no blockers. Stage 2 production round-state helpers now use
segment/coefficient prefix terminology instead of x/y prefix module names.

- `Deferred:` Shared `round_batching` functions still use the old
  `live_x_cols` / `col_bits` / `ring_bits` parameter names. They are outside
  the Stage 2 module and should move in a dedicated follow-up so Stage 1 and
  Stage 2 batching stay aligned.
- `Deferred:` Some local iterator variables in prefix internals and test
  helpers still use `x` or `y`. They are local coordinates, not public protocol
  APIs, but the final cleanup should rename them once the flat stream/fold-scan
  abstractions replace those loops.
- `Non-issue checked:` The rename did not change prefix behavior. Full Stage 2
  prover tests and clippy passed after the file/module renames.
- `Verification:`
  - `cargo fmt -q && cargo test -p akita-prover witness_polynomial -- --nocapture`
    -> `test result: ok. 2 passed; 0 failed`
  - `cargo test -p akita-prover akita_stage2 -- --nocapture`
    -> `test result: ok. 21 passed; 0 failed`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> exited 0

### 2026-07-06 retrospective: flat Stage 2 layout contract (superseded by Slice 1)

**Bottom line:** interim commit introduced flat fields but production still
entered through `Stage2Layout::uniform(...)`. Slice 1 completes the geometry
cutover: `Stage2Geometry` is the constructor contract; `local_view` is
optional fast-path metadata only.

- `Superseded:` `uniform_tiling` / `UniformStage2Tiling` naming and production
  `Stage2Layout::uniform(...)` entry. See Slice 1 retrospective after completion.
- `Carried forward:` Prefix and round-batching fast paths remain behind
  `local_view` until Slices 2–3 unify scan/fold surfaces.
- `Carried forward:` `RingSwitchOutput` field names (`live_x_cols`, etc.) stay
  upstream until a dedicated handoff rename; Slice 1 only changes what Stage 2
  consumes semantically (`live_len + num_vars`).

### 2026-07-06 retrospective: Slice 1 geometry cutover

**Bottom line:** no blockers. Production Stage 2 now enters through
`Stage2Geometry::from_production_handoff(witness_len, num_vars, ...)`.
`ScalarLevelLocalView` is optional fast-path metadata only.

- `Done:` Renamed `Stage2Layout` / `uniform_tiling` to `Stage2Geometry` /
  `local_view`. `fold.rs` no longer calls a uniform constructor.
- `Done:` Added production-handoff regression tests for flat contract and
  embedding mismatch.
- `Deferred:` Prefix modules and `round_batching` still use scalar-level names
  internally. Slice 2 unifies the pair-scan surface.
- `Deferred:` `RingSwitchOutput` still exposes `live_x_cols`, `col_bits`,
  `ring_bits`; they are consulted only when attaching `local_view`.
- `Verification:`
  - `cargo fmt -q`
  - `cargo test -p akita-prover akita_stage2 -- --nocapture`
    -> `test result: ok. 24 passed; 0 failed`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> exited 0

### 2026-07-06 retrospective: Slice 2 pair-scan surface

**Bottom line:** no blockers. Non-batched rounds now enter only through
`scan_round`; `round_flow` dispatches `initial round batch | scan_round`.

- `Done:` Renamed prefix scan bodies to `scan_embedded_*` and routed them only
  through the `scan_round` dispatcher.
- `Done:` Dense blocked scans remain the default flat path when no local-view
  fast path is active.
- `Deferred:` Single sequential accumulation kernel over embedded local-axis
  pair streams (Slice 3 fuses fold-scan).
- `Deferred:` Prefix modules still hold fold helpers and fused segment paths.
- `Verification:`
  - `cargo test -p akita-prover akita_stage2 -- --nocapture`
    -> `test result: ok. 24 passed; 0 failed`
  - `cargo clippy --all --message-format=short -q -- -D warnings`
    -> exited 0

## Follow-ups

- Decide after the first implementation slice whether the stacked PR is ready
  as its own reviewable unit or should continue through round-batching cleanup.
