# Spec: Core Protocol Naming Cleanup

Author(s): Quang Dao

Created: 2026-06-01

Status: implemented

PR: https://github.com/LayerZero-Labs/akita/pull/143

## Summary

Several identifiers in the core prover/verifier protocol flow carry names that do not match their actual mathematical content or protocol role.
The names mostly leaked from paper section titles or from single-letter paper notation, and they actively mislead new readers.
The clearest examples are `QuadraticEquation`, which holds a linear matrix relation rather than a quadratic equation, and `compute_r_split_eq`, whose "split-eq" is an unrelated gadget factorization that collides with the genuine `GruenSplitEq` sumcheck construction.

This spec proposes a behavior-preserving change in two parts.
Part 1 is a rename pass over a fixed set of core-protocol identifiers so the names describe the math and role.
Part 2 splits the prover's `QuadraticEquation` into a shared public statement type, `RingRelationInstance`, and a prover-only `RingRelationWitness`, and makes the verifier assemble and consume the same `RingRelationInstance` instead of threading the statement through loose arguments.
Neither part alters any algorithm, serialized byte layout, or transcript byte stream.
Part 1 is a near-zero-net-line rename; part 2 is a genuine refactor whose review surface is the constructor return types, the verifier statement assembly, and the witness-column segment layout.

## Intent

### Goal

Rename a fixed set of core prover/verifier protocol identifiers so each name matches its actual mathematical content and protocol role, and split the prover relation bundle into a shared `RingRelationInstance` statement plus a prover-only `RingRelationWitness`, without changing any runtime behavior, proof bytes, or transcript bytes.

The rename targets are types, functions, struct fields, and enum variants in:

- `crates/akita-prover/src/protocol/` (the `ring_relation`, `ring_switch`, and `flow` modules),
- `crates/akita-verifier/src/protocol/` and `crates/akita-verifier/src/stages/`,
- `crates/akita-types/src/proof/` and `crates/akita-types/src/layout/`.

It explicitly does not rename transcript label constants, since labels are diagnostics that must not enter production sponge bytes, and it does not rename the `_hat` digit-decomposition convention (see Non-Goals).

### Rename Table (Tier 1, definite)

Each row is `current -> proposed`, with the math/role that justifies it.
Proposed names are recommendations; the exact spelling is the main thing to settle in review.

1. `QuadraticEquation` (was `crates/akita-prover/src/protocol/quadratic_equation.rs`) is split, not just renamed, into `RingRelationInstance` (public statement) and `RingRelationWitness` (prover secret).
   The prover module is now `ring_relation.rs` with a zero-sized [`RingRelationProver`](crates/akita-prover/src/protocol/ring_relation.rs) builder (`::new`, `::new_recursive_multipoint`) replacing the old free constructors and crate-root `new_ring_relation_*` re-exports.
   Today it bundles three things for the per-fold-level negacyclic-ring relation `M * z = y + (X^D + 1) * r`: the RHS targets (`y` at `:254`, `v = D * w_hat` at `:247`), the structure that defines the virtual `M` (`challenges`, `opening_points` / `ring_multiplier_points`, `gamma` / `row_coefficient_rings`, the claim-incidence maps, `num_public_rows`, `relation_matrix_row_layout`), and the prover's witness (`z_pre`, `w_hat`, `w_folded`, `hint`, and the `zk`-gated `d_blinding_digits`).
   `M` and `z` are never materialized, and the commitment `u` is consumed during construction rather than stored.
   The old struct used `Option` witness fields only so the prover could `.take()` them out during `ring_switch_build_w`; that accessor cluster is gone. The verifier never held the bundle (zero references in `akita-verifier` before the split).
   That latent split is the design fix:
   - `RingRelationInstance` holds the public statement (challenges, opening points, incidence maps, `gamma` / `row_coefficient_rings`, `num_public_rows`, `relation_matrix_row_layout`, and targets `y`, `v`) with no `Option` witness fields.
   - `RingRelationWitness` holds the prover secret (`z_pre`, `w_hat`, `w_folded`, `hint`, and the `zk`-gated `d_blinding_digits`), passed by value into the prover.
     `d_blinding_digits` is the D-side blinding for `v = D · ŵ` and is `#[cfg(feature = "zk")]` on the witness, carried under the same gate rather than folded into a generic "zk blinding".
   The `Ring` prefix is deliberate: it marks the negacyclic-ring relation, distinct from the stage-1 norm/range check.
   "Quadratic" was paper-section jargon (the module doc said "Quadratic equation builder ... §4.2") describing the proof-system relation degree, not the struct's content.
   Crate homes: `RingRelationInstance` lives in `akita-types` so both prover and verifier can build it.
   It needs `LevelParams`, `RelationMatrixRowLayout`, `CyclotomicRing`, the opening points (`RingOpeningPoint`, `RingMultiplierOpeningPoint` in `proof/batch.rs`), and the incidence types, all defined in `akita-types`, plus `Challenges`, which is defined in `akita-challenges` (already a dependency of `akita-types`, see `akita-types/Cargo.toml`), so no new crate edge is added.
   `RingRelationWitness` stays in `akita-prover` and cannot move: its `z_folded_rings` field has type `DecomposeFoldWitness`, which is defined in `akita-prover`, so the witness type is structurally prover-only and pulling it into `akita-types` would force a prover-only type down into the shared layer.

2. `compute_r_split_eq` (was `crates/akita-prover/src/protocol/quadratic_equation/r_split.rs`) -> `compute_relation_quotient`.
   It computes the divisibility quotient `r` of `M * z = y + (X^D + 1) * r` via gadget/Kronecker factorization, without materializing `M` or `z`.
   The "eq" is not an equality polynomial, and "split-eq" collides with the unrelated `GruenSplitEq` sumcheck factorization in `crates/akita-algebra/src/split_eq.rs`.
   The file is now `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`, re-exported from `ring_relation.rs`.

3. `r_stage1` (verifier, e.g. `crates/akita-verifier/src/stages/stage2.rs:181`) -> `stage1_point`.
   It is the random point output by the stage-1 norm sumcheck, not the divisibility quotient `r`.
   This removes the collision with quotient `r` and with the verifier helper `compute_r_contribution` (`crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs:334`).

4. `RelationMatrixRowLayout::Intermediate` / `RelationMatrixRowLayout::Terminal` (`crates/akita-types/src/layout/params.rs:26`) -> `RelationMatrixRowLayout::WithDBlock` / `RelationMatrixRowLayout::WithoutDBlock`.
   The variant selects whether the `D`-block rows `v = D * w_hat` appear in the relation, not the position in the fold chain.
   The current names collide with `AkitaProofStep::Intermediate` / `Terminal` (`crates/akita-types/src/proof/levels.rs:806`), which genuinely do mean position.

5. `z_pre` (field on `DecomposeFoldWitness`, `crates/akita-prover/src/lib.rs:83`, and on `RingRelationWitness`) -> `z_folded_rings`.
   It is the challenge-folded witness in ring form before balanced digit decomposition ("Folded witness rows in ring form").
   The "pre" suffix is opaque; the new name states both that it is folded and that it is still in ring (pre-digit) form.
   The companion fields `centered_coeffs` and `centered_inf_norm` keep their names.

6. The "Direct" overload.
   Three unrelated meanings share the word, and the proposed terminal-direct-relation mode (`specs/terminal-direct-ring-relation.md`) would add a fourth.
   - `DirectWitnessProof` (`crates/akita-types/src/proof/direct_witness.rs:20`) -> `CleartextWitnessProof`.
   - `AkitaBatchedRootProof::Direct` (`crates/akita-types/src/proof/levels.rs:464`) -> `AkitaBatchedRootProof::ZeroFold`.
   - `AkitaPlannedStep::Direct` (`crates/akita-types/src/schedule.rs:265`) -> `AkitaPlannedStep::ZeroFold` (and the related `AkitaPlannedDirectStep` -> `AkitaPlannedZeroFoldStep` / `direct_step()` -> `zero_fold_step()` accessors).
   Both 0-fold variants take the same `ZeroFold` spelling: they live on distinct enums, so there is no collision, and the shared name reflects that they are the same protocol concept at the root and in the schedule.
   These rename Rust identifiers only; variant selection on the wire is driven by the externally supplied shape/context, not by an in-stream discriminant or variant name, so the renames are byte-safe (see Invariants).

### Rename Candidates (Tier 2, lower priority)

These are real improvements but more invasive or more debatable.
Include them only if review agrees; otherwise they move to a follow-up.

- `build_w_coeffs` (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs:253`) -> `pack_recursive_witness_digits`.
  The output is packed i8 digit planes, not field coefficients.
  Also fix the stale doc comment that references a non-existent `e-hat` identifier (`coeffs.rs:229-237`).
- `repeated_b_commitment_rows` / `repeated_b_planes_per_claim` (`crates/akita-prover/src/protocol/ring_relation/repeated_b.rs`) -> `multi_group_b_rows` / `b_planes_per_claim`.
  "Repeated" means per-commitment-group repetition for multipoint batching.
- `RelationMatrixRowLayout` D/B/A row vocabulary.
  The `D`/`B`/`A` row labels are named after the Ajtai keys `d_key`/`b_key`/`a_key`, and `D` rows collide visually with the ring degree `const D`.
  At minimum document the key mapping at `generate_relation_rhs` (`crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`) and `LevelParams::relation_matrix_row_count_for`; optionally rename `D` rows to `v_rows`.
- `alpha` in ring-switch output structs (`crates/akita-verifier/src/protocol/ring_switch.rs`) -> `ring_eval_challenge`, since it evaluates cyclotomic rows at `alpha` and is neither a batching coefficient nor a fold challenge.

### Explicitly Keep (Non-Goals do not cover these well enough to leave implicit)

- The `_hat` suffix (`w_hat`, `t_hat`, `r_hat`, and similar).
  In the lattice setting `_hat` signals balanced base-`b` digit decomposition, matching the source papers, so it is correct domain notation and stays.
- `GruenSplitEq` and `split_eq` in the sumcheck/algebra layer.
  This is standard literature terminology for the equality-polynomial factorization and is correctly named.
- `LevelParams`, the `flow/*` fold/level functions, the `ring_switch_finalize_*` family, `RingSwitchOutput`, `relation_claim_from_rows_extension`, `ExtensionOpeningReductionProof`.
  These already match their role.

### Invariants

- **Behavior preserving.**
  No algorithm, control flow, numeric behavior, or serialized data layout changes.
  Part 1 edits identifier names, call sites, module paths, and doc comments only.
  Part 2 moves fields between two types and shifts where the verifier assembles the statement, but computes the same values in the same order.

- **Proof bytes unchanged.**
  Serialized proof bytes for every existing fixture are byte-identical before and after.
  The enum-variant renames (Tier 1 item 6) are inherently byte-safe: the proof wire form does not encode a Rust variant name or an in-stream discriminant tag for these enums.
  Variant selection on deserialization is driven by the externally supplied shape/context (for example `AkitaProofStepShape::Intermediate` / `Terminal` in `crates/akita-types/src/proof/wire.rs`, and `DirectWitnessShape` for the cleartext witness), which is reconstructed from params rather than read from the byte stream.
  There is no `serde`-style name-based serialization anywhere on the proof path, so renaming a Rust variant cannot move any byte.
  The instance/witness split touches no serialized type: `RingRelationInstance` and `RingRelationWitness` are in-memory prover/verifier state, not wire types.

- **Transcript bytes unchanged.**
  No transcript label constants are renamed, and label text never enters production sponge bytes, so Fiat-Shamir output is identical.
  Building `RingRelationInstance` must not change the order in which challenges are sampled or values absorbed; the instance is populated from already-sampled challenges, not by moving sampling into its constructor.
  The `logging-transcript` event streams stay equal.

- **Verifier no-panic boundary preserved.**
  Renames must not remove or weaken any existing validation, bound check, or error path on verifier-reachable code.

- **No new public aliases.**
  This is a full cutover with no deprecated aliases or re-export shims; all call sites are updated in one pass.

### Non-Goals

- Do not change any algorithm, proof structure, serialization format, or transcript schedule.
- Do not rename transcript label constants.
- Do not rename the `_hat` digit-decomposition convention.
- Do not rename `GruenSplitEq` or the sumcheck `split_eq` machinery.
- Do not introduce backward-compatibility aliases for the old names.
- Do not bundle these renames into an unrelated feature PR; this is its own reviewable unit.
- Do not change the prover/verifier crate dependency direction.
  Both crates continue to depend on `akita-types`; `akita-prover` does not gain a dependency on `akita-verifier`, and the `akita-types` shared layer is not collapsed.
  Restructuring the dependency graph is a separate investigation, deliberately excluded from this PR.
- Do not unify the ring-switch Fiat-Shamir sampling, the fold-challenge wrapper, or the M-table evaluation between prover and verifier beyond the segment-layout sharing described in Design; deeper unification is a follow-up.
- Do not enable true recursive multipoint (one commitment opened at multiple points); the routing contract is deliberately kept single-point pending a soundness argument and a row-eval spec entry, see the Deferred note in Design.

## Evaluation

### Acceptance Criteria

- All Tier 1 renames are applied with every call site updated; the workspace builds with no references to the old names.
- `QuadraticEquation` is split: `RingRelationInstance` exists in `akita-types`, `RingRelationWitness` exists in `akita-prover`, and the prover constructors return both.
- The verifier builds and consumes `RingRelationInstance`; the witness-column segment layout is derived from it on both sides (one `segment_layout` method) rather than duplicated between the verifier's `RingSwitchSegmentLayout` and the prover's inline offsets.
- The per-claim incidence cluster is consolidated into two named axes: `RingRelationInstance` holds a `ClaimIncidenceSummary` (evaluation axis) and a new `CommitmentRouting` (commitment axis), replacing the loose `claim_to_point` / `claim_poly_indices` / `num_polys_per_point` / `num_public_rows` / `claim_to_point_poly` fields. `CommitmentRouting` validates its indices at construction and adds no transcript bytes.
- The `Option` witness fields and the `take_*` / `*_centered` accessor cluster on the old `QuadraticEquation` are gone.
- `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and `cargo test` pass on default features.
- `cargo test --features zk` and `cargo test --no-default-features` pass.
- Serialized proof bytes for existing proof fixtures are unchanged (see Testing Strategy).
- `logging-transcript` prover/verifier event-stream equality tests pass unchanged.
- A repository search shows the old identifiers are fully gone (no `QuadraticEquation`, `compute_r_split_eq`, `r_split`, `quadratic_equation` module, `new_ring_relation_*`, `prove_*_from_quadratic`, `RelationMatrixRowLayout::Intermediate`, `RelationMatrixRowLayout::Terminal`, `z_pre`, `DirectWitnessProof` remaining, except in this spec and in historical specs).
- `RingRelationProver` is the sole prover entry point for building `(RingRelationInstance, RingRelationWitness)`; flow helpers are named `prove_*_from_ring_relation`.

### Testing Strategy

This is behavior-preserving (a rename plus an internal type split), so the strongest evidence is that existing tests pass unchanged and that serialized artifacts do not move.

- Run the full existing suite: `cargo test`, then `cargo test --features zk`, then `cargo test --no-default-features`.
- Byte-identity check: before the rename, capture serialized bytes of a representative proof for one dense and one one-hot profile (for example via an `akita_e2e` fixture or a small scratch test) and after the rename assert the bytes are identical.
  If a proof-bytes golden test already exists, that is sufficient; otherwise add a temporary local check during implementation and do not commit it.
- Wire-stability check for Tier 1 item 6: confirm a proof using `CleartextWitnessProof`, `AkitaBatchedRootProof::ZeroFold`, and `AkitaPlannedStep::ZeroFold` deserializes byte-for-byte from bytes produced before the rename.
  This is the same round-trip byte-identity check as above, exercised on a 0-fold / cleartext-witness fixture; because variant selection is shape-driven (no in-stream variant tag or name), a successful round trip is sufficient evidence that the rename moved no byte.
- Run `logging-transcript` tests (`cargo test -p akita-pcs transcript_hardening --features logging-transcript`) to confirm event-stream equality.

Minimum acceptance commands:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `cargo test --features zk`
- `cargo test --no-default-features`

### Performance

No performance change is expected or allowed.
This is a rename; there is no new arithmetic, allocation, or branching.
No benchmark or proof-size movement should occur, and the byte-identity check above is the guard.

## Design

### Architecture

The renames are mechanical and touch identifiers, call sites, module paths, and doc comments only.
Recommended order so each step compiles:

1. Rename leaf types and enum variants (`RelationMatrixRowLayout` variants, `DirectWitnessProof`, root/schedule `Direct` variants) and fix call sites.
2. Rename `z_pre` field and its uses.
3. Rename `r_stage1` and its uses.
4. Rename `compute_r_split_eq` and move `r_split.rs` to `relation_quotient.rs`, updating the re-export.
5. Split the old `QuadraticEquation` into `RingRelationInstance` / `RingRelationWitness`, rename the module to `ring_relation`, and introduce `RingRelationProver` last, since it is the most widely referenced surface.
6. Apply Tier 2 renames if approved.
7. Run fmt, clippy, the full test matrix, and the byte-identity check.

Because the repo makes no backward-compatibility guarantees, every rename is a full cutover with no aliases.

### Relation Instance/Witness Split and Verifier Adoption

This is part 2, the refactor.

**Split.**
`QuadraticEquation` becomes two types: a shared public statement `RingRelationInstance` (in `akita-types`) and a prover-only secret `RingRelationWitness` (in `akita-prover`).
The verifier already builds and consumes the statement; today it does so from loose arguments rather than a named type, so the split also makes the verifier a first-class builder of the statement instead of a place that re-derives it ad hoc.

**`RingRelationInstance` shape and ownership.**
The instance owns the fields it needs; it does not hold a `&LevelParams` borrow.
Owning a handful of scalars plus the vectors it already materializes keeps the instance a self-contained value that both prover and verifier can pass by reference into the row-evaluation paths without threading a `LevelParams` lifetime through every evaluator.
The `LevelParams`-derived quantity it consumes (`RelationMatrixRowLayout`) is a small `Copy` scalar, so copying it in is cheaper than carrying a borrow.

The instance folds the per-claim incidence into two named axes rather than carrying five loose vectors.
The loose vectors today encode two distinct things that the names hide.
One axis is the **evaluation** incidence (which claim is opened at which point): `claim_to_point`, `num_polys_per_point`, `num_public_rows`, and `claim_poly_indices`, which are exactly the accessors of `ClaimIncidenceSummary` (`crates/akita-types/src/proof/incidence.rs:94`) and are copied out of the summary in `RingRelationProver::new`.
The other axis is the **commitment** routing (which committed-polynomial bundle holds a claim's witness columns): `claim_to_point_poly` plus the commitment-view of `claim_poly_indices`, consumed in `compute_relation_matrix_col_evals` to build the `t`-segment (`evals.rs:140-144`) entirely separately from the point side.
These axes coincide in the root (one commitment per point) but diverge in recursive multipoint (one shared commitment opened at many points: evaluation routing is per-claim while commitment routing sends every claim to bundle 0), which is why a single summary cannot represent both.
So the instance holds the evaluation axis as a `ClaimIncidenceSummary` (already constructed and threaded by both sides: verifier `levels.rs:283`, `recursive.rs:886`, `batched.rs`) and the commitment axis as a new `CommitmentRouting` type (see the consolidation note for its shape).

```rust
// in akita-types
pub struct RingRelationInstance<F: FieldCore, const D: usize> {
    // shape (copied out of LevelParams, no borrow)
    relation_matrix_row_layout: RelationMatrixRowLayout,
    // structure that defines the virtual M
    challenges: Challenges,                                  // from akita-challenges
    opening_points: Vec<RingOpeningPoint<F>>,
    ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F, D>>,
    incidence: ClaimIncidenceSummary,                        // evaluation axis: claim -> opening point
    commitment_routing: CommitmentRouting,                   // commitment axis: claim -> committed bundle
    gamma: Vec<F>,
    row_coefficient_rings: Vec<CyclotomicRing<F, D>>,        // derived from gamma
    // RHS targets
    y: Vec<CyclotomicRing<F, D>>,
    v: Vec<CyclotomicRing<F, D>>,                            // stored target; see provenance note
}
```

`RingRelationWitness` (in `akita-prover`) holds the prover secret: `z_folded_rings` (was `z_pre`), `w_hat`, `w_folded`, `hint`, and the `#[cfg(feature = "zk")]` `d_blinding_digits`.

**Why the witness must stay in `akita-prover`.**
This is not just a placement preference; the witness is structurally prover-only.
`z_folded_rings` has type `DecomposeFoldWitness`, which is defined in `akita-prover` (`crates/akita-prover/src/lib.rs`), and the bundle also carries prover-only commitment state (`hint: AkitaCommitmentHint`).
Moving `RingRelationWitness` into `akita-types` would force `DecomposeFoldWitness` (and the digit-decomposition machinery around it) down into the shared layer for no verifier benefit, since the verifier never holds the witness.
So the instance goes up to `akita-types` and the witness stays in `akita-prover`; that asymmetry is the whole point of the split.

**Constructors and witness consumption.**
`RingRelationProver::new` and `RingRelationProver::new_recursive_multipoint` return `(RingRelationInstance, RingRelationWitness)`.
`ring_switch_build_w` takes `(&RingRelationInstance, RingRelationWitness)` and consumes the witness by value.
The old `Option` witness fields and `take_*` / `*_centered` accessor cluster are gone; the prover reads witness fields directly after construction.
Flow entrypoints that already held a built instance are named `prove_*_from_ring_relation` (replacing `prove_*_from_quadratic`).

**Constructor contract (transcript invariance).**
Building `RingRelationInstance` must not move any Fiat-Shamir sampling or transcript absorb into its constructor.
The instance is populated from values that have already been sampled and absorbed: the constructor only copies/derives.
The derived field `row_coefficient_rings` (from `gamma`) is a pure deterministic function of already-fixed inputs and is computed identically on both sides.
This is what keeps proof bytes and the `logging-transcript` event stream byte-identical.

**Target `v` provenance.**
`v` is a stored target on the instance, not a value the instance recomputes from the witness.
On the prover, `v = D · ŵ` is computed during construction (it has `w_hat` in hand) and stored; on the verifier, `v` is read from the proof.
The instance never reaches into `RingRelationWitness` to recompute `v`, so the instance/witness boundary stays clean and `v` is identical on both sides by construction.

**Incidence consolidation (`CommitmentRouting`).**
The evaluation axis is already consolidated: the instance moves in the `ClaimIncidenceSummary` both sides already construct, with no re-flattening.
For the commitment axis, this spec introduces a named `CommitmentRouting` type in `akita-types` rather than leaving `claim_to_point_poly` as a bare, unexplained `Vec<usize>`:

```rust
// in akita-types, next to ClaimIncidenceSummary
pub struct CommitmentRouting {
    claim_to_group: Vec<usize>,       // was claim_to_point_poly: claim -> committed bundle
    claim_poly_in_group: Vec<usize>,  // was the commitment-view of claim_poly_indices
    num_polys_per_group: Vec<usize>,  // bundle p -> # polys committed in it
}
```

This is the right shape because it names the second axis that the loose `claim_to_point_poly` left implicit, and because the verifier already carries this exact pairing as `PreparedRingSwitchRowEval.claim_to_point_poly: Vec<(usize, usize)>` (`crates/akita-verifier/src/protocol/ring_switch.rs:133`); `CommitmentRouting` is the named, owned form of that tuple list, shared by both sides.
`compute_relation_matrix_col_evals` builds its `t`-vector index from exactly these three (`evals.rs:118-144`), so the type captures a real computation, not a passive bag.

Why a sibling type and not an extension of `ClaimIncidenceSummary`: the summary's model is one-commitment-per-point, enforced by its `validate()` (`incidence.rs:110-166`), and it is a verifier-reachable no-panic type. Folding a decoupled commitment axis into it would either weaken those invariants or require a second validation mode on the same type. Keeping `CommitmentRouting` separate preserves the summary's invariants and states the commitment axis as its own checked object.

`CommitmentRouting` carries no new transcript bytes: the absorbed incidence shape stays exactly `append_claim_incidence_shape_to_transcript` (`incidence.rs:430`), which binds only `nuposition_index_bits` / `num_points` / `num_claims` / `num_polys_per_point` / `claim_to_point` / `claim_poly_indices`. `CommitmentRouting` is in-memory routing only, so adding it changes no Fiat-Shamir output.
`CommitmentRouting` must validate at construction (lengths equal `num_claims`, `claim_to_group[i] < num_polys_per_group.len()`, `claim_poly_in_group[i] < num_polys_per_group[claim_to_group[i]]`) to keep the verifier no-panic boundary; this mirrors the bounds `compute_relation_matrix_col_evals` checks today at `evals.rs:96-111`.

**Verifier adoption (the payoff).**
The verifier today assembles the fold statement from loose pieces: `LevelParams`, the proof's `y_rings` / `v` / commitment rows, `ClaimIncidenceSummary`, and transcript-derived challenges, threaded by hand through `verify_root_level_inner` (`crates/akita-verifier/src/protocol/levels.rs:271`) and `verify_one_level_inner` (`crates/akita-verifier/src/protocol/levels/recursive.rs:120`).
There is no single statement object today, so the same incidence/opening/challenge plumbing is spelled out separately from the prover's.
Under this spec the verifier builds one `RingRelationInstance` from those same pieces and threads `&RingRelationInstance` into `prepare_relation_matrix_evaluator` (`crates/akita-verifier/src/protocol/ring_switch.rs`) and `relation_claim_from_rows_extension` (`crates/akita-types/src/proof/relation.rs`).
Every field is already in the verifier's hands at this point, so this is a regrouping, not new computation. Concretely:

- `relation_matrix_row_layout`: from the level's `LevelParams` (copied out).
- `challenges`: the witness fold challenges already sampled from the transcript.
- `opening_points` / `ring_multiplier_points`: the verifier's reconstructed opening points (`RingMultiplierOpeningPoint::from_base` is already used in `proof/batch.rs`).
- `incidence`: the `ClaimIncidenceSummary` the verifier already threads through `verify_root_level_inner` (`levels.rs:283`) and `verify_one_level_inner` (`recursive.rs:886`); this supplies `num_public_rows` and the three summary maps.
- `commitment_routing`: built from the same per-path data the verifier already derives for `PreparedRingSwitchRowEval.claim_to_point_poly` (`ring_switch.rs:133`); root collapses to the point routing, recursive multipoint routes all claims to group 0.
- `gamma` / `row_coefficient_rings`: sampled via the same shared `sample_public_row_coefficients::<F, C, T>(incidence_summary, transcript)` the prover calls. The verifier already invokes it at `levels.rs:360`; the prover calls it at `root_fold.rs:286,449,806`. Same function, same transcript position, so the values match by construction.
- `y` / `v`: the proof's `y_rings` / `v`.

The verifier gains a single named statement it constructs once per level and reuses, instead of re-deriving incidence and row-coefficient data inline at each evaluation site, and prover and verifier now share one definition of what the ring-relation statement is.
This availability is verified against the current code, not assumed: the verifier already constructs `ClaimIncidenceSummary` and already samples the row coefficients from the same shared helper, so building the instance reorders no sampling and absorbs nothing new.

**Concrete deduplication in scope: the witness-column segment layout.**
The witness-column segment layout is currently duplicated: the verifier has an explicit `RingSwitchSegmentLayout` (`crates/akita-verifier/src/protocol/ring_switch.rs:140`, built by `segment_layout()` at `:605`), while the prover open-codes the same offsets inline in `compute_relation_matrix_col_evals` (`crates/akita-prover/src/protocol/ring_switch/evals.rs`).
This spec moves that layout onto the instance as `RingRelationInstance::segment_layout(&self) -> Result<RingRelationSegmentLayout, AkitaError>`, computed from `relation_matrix_row_layout`, `num_public_rows`, `num_polys_per_point`, and the opening-point counts it already owns.
The verifier's `RingSwitchSegmentLayout` is replaced by this method's return type, and the prover's inline offset arithmetic in `compute_relation_matrix_col_evals` calls the same method, so the column layout has exactly one definition.
This is the deduplication that earns `RingRelationInstance` its place in `akita-types`: without it, the shared type would be a passive bag of fields rather than the single source of the layout both sides must agree on.

**Deferred (enabled but not in this PR).**
Unifying the ring-switch Fiat-Shamir sampling blocks (prover `finalize.rs` versus verifier `ring_switch_verifier_core`) and the fold-challenge wrapper (`derive_witness_fold_challenges` versus the prover inline path) is enabled by the shared instance but is left to a follow-up to keep this PR's review surface bounded.
Unifying the prover's materialized M-table build with the verifier's deferred slice-MLE evaluators is explicitly not pursued; they are intentionally different cost models.

**True recursive multipoint (one commitment opened at many points): deferred.**
`CommitmentRouting` is shaped to express split routing, where the evaluation axis is per-claim and the commitment axis collapses every claim to bundle 0.
That is exactly the shape needed to open a single recursive witness at `k > 1` points, and the row-evaluation math on both sides already separates the two axes: the prover sizes the B-block by commitment groups (`num_polys_per_commitment_group.len()`) and the public rows by opening points (`relation_quotient.rs`), and the verifier routes T-vectors by `claim_to_commitment_group` and W-rows by `claim_to_point` (`ring_switch.rs`).
This PR deliberately does not enable it.
`RingRelationProver::new_recursive_multipoint` keeps an explicit `num_claims != 1` guard, and `RingRelationInstance::new` enforces `CommitmentRouting::check_matches_incidence` (which requires `claim_to_commitment_group == claim_to_point`), so the only routing shape accepted today is the one the current callers produce: a single opening point per recursive level.
Both recursive call sites build a single-element opening-point set (`flow/recursive.rs`), so no behavior is lost by the guard.
Enabling true multipoint is a follow-up that requires three things: relaxing `check_matches_incidence` to validate split routing rather than require axis equality (and dropping the `num_claims != 1` guard); a caller that actually produces multiple opening points for one recursive witness, with the matching Fiat-Shamir absorbs and identical verifier replay order; and a soundness argument plus a `book/src/how/verifying/matrix_evaluation.md` row-eval entry for the `k`-public-row layout.
Until that analysis exists the contract stays single-point.

**Why `akita-types`, not a dependency flip.**
`RingRelationInstance` goes in `akita-types`, the existing lowest common dependency of both prover and verifier, next to `relation_claim_from_rows_extension` and `LevelParams::relation_matrix_row_count_for` which already live there.
Every field type it needs is reachable from `akita-types` without a new crate edge (`Challenges` via the existing `akita-challenges` dependency; everything else native to `akita-types`).
This spec does not change the crate dependency direction; see Non-Goals.

### Diff Surface (estimated)

The numbers below are literal-substring match counts on the worktree (`*.rs`, excluding `specs/`).
They are an upper bound: they include doc comments, tests, and derived identifiers, and they are the surface a full implementation would touch.

Per-symbol (Tier 1):

| Symbol | Occurrences | Files |
| --- | ---: | ---: |
| `QuadraticEquation` | 35 | 12 |
| `compute_r_split_eq` (+ `r_split` module/path) | ~22 | 12 |
| `z_pre` (incl. derived `z_pre_centered`, `z_pre_centered_inf_norm`) | 146 | 15 |
| `r_stage1` | 155 | 22 |
| `RelationMatrixRowLayout::Intermediate` | 47 | 20 |
| `RelationMatrixRowLayout::Terminal` | 38 | 18 |
| `DirectWitnessProof` | 77 | 27 |
| `AkitaBatchedRootProof::Direct` | 17 | 9 |
| `AkitaPlannedStep::Direct` (+ `AkitaPlannedDirectStep`, `direct_step()`) | ~12 | few |

Totals (Tier 1):

- Unique files touched: about 70, across 9 crates.
- Identifier occurrences: roughly 550 to 600.
- Net line delta is near zero; this is almost entirely single-token replacement plus module-path and doc updates, not added or removed logic.

Per-crate distribution of the ~70 files:

| Crate | Files |
| --- | ---: |
| `akita-prover` | 32 |
| `akita-types` | 12 |
| `akita-verifier` | 11 |
| `akita-pcs` | 6 |
| `akita-scheme` | 4 |
| `akita-config` | 2 |
| `akita-planner` | 1 |
| `akita-derive` | 1 |
| `akita-challenges` | 1 |

The blast radius is concentrated in `akita-prover`, `akita-types`, and `akita-verifier`; downstream crates are touched only lightly through re-exports and type references.

Two symbols dominate: `z_pre` and `r_stage1` together account for roughly 300 of the ~560 occurrences.
If a smaller review unit is wanted, those two renames split cleanly into their own commits within the single PR, without blocking the rest.

Tier 2 adds a small increment on top: `build_w_coeffs` (about 8 occurrences across 3 files), `repeated_b*` (about 15 across 4 files), plus the `alpha` and doc-only changes.

The spec documents themselves are out of this count: `specs/terminal-direct-ring-relation.md` references some old names and would be updated if this lands first.

Part 2 (the split and verifier adoption) is not captured by these literal counts.
Its surface is the two prover constructors, the `ring_switch_build_w` signature and its callers, the deletion of the `Option` / `take_*` / `*_centered` accessor cluster, the new `RingRelationInstance` and `CommitmentRouting` types in `akita-types` (the instance holding a `ClaimIncidenceSummary` for the evaluation axis and a `CommitmentRouting` for the commitment axis), the `segment_layout` method that replaces the verifier's `RingSwitchSegmentLayout` and the prover's inline `compute_relation_matrix_col_evals` offsets, and the verifier statement-assembly sites in `crates/akita-verifier/src/protocol/levels.rs` and `levels/recursive.rs`.
This is a real refactor with net line movement, unlike part 1, and is the part where reviewer attention belongs.

### Alternatives Considered

**Keep the names and only add doc comments.**
A glossary pass is cheaper and lower risk, but it leaves the misleading identifiers in code, so readers still hit the `QuadraticEquation` and `split_eq` traps on first contact.
This spec chooses renames for the Tier 1 set and allows the doc-only treatment for the Tier 2 row-vocabulary items.

**Rename `_hat` to spell out digit decomposition.**
Rejected: in the lattice setting `_hat` is the established paper convention for digit decomposition, so spelling it out would diverge from the source material without adding clarity for the intended audience.

**Fold these renames into the terminal-direct-ring-relation PR (#141).**
Rejected: that is a spec-only docs PR for an unrelated feature, and a cross-cutting rename sweep would destroy its reviewability.
The `RelationMatrixRowLayout` and `Direct` renames are most valuable to land before that feature's implementation, since it adds another `Terminal`/`Direct` meaning.

**Introduce deprecated aliases for a transition period.**
Rejected by the repo's no-backward-compatibility rule; all call sites are updated in one pass.

**Put the shared instance in `akita-verifier` and have `akita-prover` depend on it.**
Considered because the prover is the more comprehensive crate, so a prover-depends-on-verifier edge is plausible.
Rejected for this PR: the shared statement belongs in the lowest common dependency, and `akita-types` already plays that role (it holds `relation_claim_from_rows_extension`, `LevelParams`, `RelationMatrixRowLayout`, and the incidence types).
Flipping the dependency direction would not remove the duplication that motivates the split unless the shared code lived in the verifier, and it would pull verifier replay into the prover build for no gain over placing the type in `akita-types`.
The broader question of whether to restructure the crate graph is a separate investigation, not part of this PR.

**Split `QuadraticEquation` into instance and witness without verifier adoption.**
Considered as a smaller part 2.
This still removes the `Option`/`take_*` muddle and gives textbook instance/witness naming, but it leaves the verifier statement distributed and the segment-layout duplication in place.
The spec keeps verifier adoption in scope because it is the change that makes the shared `RingRelationInstance` earn its place in `akita-types`; if a reviewer insists on a smaller unit, verifier adoption is the natural cut line, kept as a collapsible child commit rather than a separately merged PR.

## Documentation

- Add a short "Naming" or glossary note (either in this spec's wake or in a prover module doc) that pins: `_hat` = digit decomposition, the two distinct `split_eq` meanings, `M` as a virtual row-combined relation table versus the Ajtai keys `A`/`B`/`D`, and the distinction between the matrix-relation layer and the norm sumcheck (`AkitaStage1Prover`).
- Update `specs/terminal-direct-ring-relation.md` references that mention `RelationMatrixRowLayout::Terminal`, `DirectWitnessProof`, or `compute_r_split_eq` if this rename lands first.
- No external or README changes are required.

## Execution

Land this as a single PR, ordered as part 1 (renames) commits first and part 2 (the split and verifier adoption) commits second, so the history reads cleanly without forcing a reviewer to context-switch across two PRs.
Splitting into two independent PRs is explicitly not the plan: part 2 changes call sites that part 1 renames, so a hard split would make the second PR a large rebase against the first and double the review setup.
If the diff feels too large to land atomically, use a collapsible stack (a base branch for part 1, a child branch for part 2) that merges as one unit, rather than two separately reviewed-and-merged PRs.
Within the PR, keep the commit boundaries aligned to the recommended rename order below and to the part 1 / part 2 split, so a reviewer can read commit-by-commit.

Important risks to resolve first:

- Confirm the proof path has no `serde`-style name-based variant serialization before renaming any serialized enum variant (Tier 1 item 6).
  The current wire form selects variants from the externally supplied shape/context (`proof/wire.rs`), not from an in-stream discriminant or name, so the renames are byte-safe; if any path is later found to serialize variant names, that path must be fixed or the variant rename dropped.
- The byte-identity check (including a 0-fold / cleartext-witness fixture) is the gate that this stayed behavior-preserving; run it before opening the PR, especially for part 2.
- Before applying the renames, grep the workspace for the proposed new names (`stage1_point`, `compute_relation_quotient`, `z_folded_rings`, `ZeroFold`, `CleartextWitnessProof`, `WithDBlock` / `WithoutDBlock`) to confirm none already exist and collide; if one does, settle the spelling in review first.
- `QuadraticEquation` is referenced across prover, tests, and re-exports; do the split last and lean on the compiler to find call sites.
- For part 2, building `RingRelationInstance` on the verifier side must not move any challenge sampling or transcript absorb; populate the instance from already-sampled values so the `logging-transcript` event stream is unchanged.

## References

- `specs/terminal-direct-ring-relation.md`, the feature that motivated auditing these names and that adds another `Terminal`/`Direct` meaning.
- `crates/akita-prover/src/protocol/ring_relation.rs`, `RingRelationProver` and the ring-relation builder module.
- `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs`, `compute_relation_quotient` and the gadget-factorization doc.
- `crates/akita-algebra/src/split_eq.rs`, the genuine `GruenSplitEq` construction that collides by name.
- `crates/akita-types/src/layout/params.rs`, `RelationMatrixRowLayout` and its doc.
- `crates/akita-types/src/proof/levels.rs`, `AkitaProofStep`, `AkitaBatchedRootProof`, and `TerminalLevelProof`.
- `crates/akita-types/src/proof/direct_witness.rs`, `DirectWitnessProof`.
- `crates/akita-prover/src/lib.rs`, `DecomposeFoldWitness` and `z_pre`.
- `crates/akita-verifier/src/stages/stage2.rs` and `crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs`, `r_stage1` versus quotient `r`.
