# Spec: Core Protocol Naming Cleanup

Author(s): Quang Dao

Created: 2026-06-01

Status: proposed

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

- `crates/akita-prover/src/protocol/` (the `quadratic_equation`, `ring_switch`, and `flow` modules),
- `crates/akita-verifier/src/protocol/` and `crates/akita-verifier/src/stages/`,
- `crates/akita-types/src/proof/` and `crates/akita-types/src/layout/`.

It explicitly does not rename transcript label constants, since labels are diagnostics that must not enter production sponge bytes, and it does not rename the `_hat` digit-decomposition convention (see Non-Goals).

### Rename Table (Tier 1, definite)

Each row is `current -> proposed`, with the math/role that justifies it.
Proposed names are recommendations; the exact spelling is the main thing to settle in review.

1. `QuadraticEquation` (`crates/akita-prover/src/protocol/quadratic_equation.rs:245`) is split, not just renamed, into `RingRelationInstance` (public statement) and `RingRelationWitness` (prover secret).
   Today it bundles three things for the per-fold-level negacyclic-ring relation `M * z = y + (X^D + 1) * r`: the RHS targets (`y` at `:254`, `v = D * w_hat` at `:247`), the structure that defines the virtual `M` (`challenges`, `opening_points` / `ring_multiplier_points`, `gamma` / `row_coefficient_rings`, the claim-incidence maps, `num_public_rows`, `m_row_layout`), and the prover's witness (`z_pre`, `w_hat`, `w_folded`, `hint`, and zk blinding).
   `M` and `z` are never materialized, and the commitment `u` is consumed during construction rather than stored.
   The witness fields are `Option` today only so the prover can `.take()` them out during `ring_switch_build_w` (`take_z_pre`/`take_w_hat`/`take_w_folded`/`take_hint` at `quadratic_equation.rs:996-1047`); they are always `Some` after construction and the verifier never builds this type (zero references in `akita-verifier`).
   That latent split is the design fix:
   - `RingRelationInstance` holds the public statement (challenges, opening points, incidence maps, `gamma` / `row_coefficient_rings`, `num_public_rows`, `m_row_layout`, and targets `y`, `v`) with no `Option` witness fields.
   - `RingRelationWitness` holds the prover secret (`z_pre`, `w_hat`, `w_folded`, `hint`, zk blinding), passed by value into the prover.
   The `Ring` prefix is deliberate: it marks the negacyclic-ring relation, distinct from the stage-1 norm/range check.
   "Quadratic" was paper-section jargon (the module doc said "Quadratic equation builder ... §4.2") describing the proof-system relation degree, not the struct's content.
   Crate homes: `RingRelationInstance` lives in `akita-types` so both prover and verifier can build it (it needs only `LevelParams`, `MRowLayout`, `Challenges`, `CyclotomicRing`, opening points, and incidence, all already in `akita-types`); `RingRelationWitness` stays in `akita-prover`.

2. `compute_r_split_eq` (`crates/akita-prover/src/protocol/quadratic_equation/r_split.rs:240`) -> `compute_relation_quotient`.
   It computes the divisibility quotient `r` of `M * z = y + (X^D + 1) * r` via gadget/Kronecker factorization, without materializing `M` or `z`.
   The "eq" is not an equality polynomial, and "split-eq" collides with the unrelated `GruenSplitEq` sumcheck factorization in `crates/akita-algebra/src/split_eq.rs`.
   The file `r_split.rs` is renamed to `relation_quotient.rs` and the re-export in `quadratic_equation.rs:36` is updated.

3. `r_stage1` (verifier, e.g. `crates/akita-verifier/src/stages/stage2.rs:181`) -> `stage1_point`.
   It is the random point output by the stage-1 norm sumcheck, not the divisibility quotient `r`.
   This removes the collision with quotient `r` and with the verifier helper `compute_r_contribution` (`crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs:331`).

4. `MRowLayout::Intermediate` / `MRowLayout::Terminal` (`crates/akita-types/src/layout/params.rs:26`) -> `MRowLayout::WithDBlock` / `MRowLayout::WithoutDBlock`.
   The variant selects whether the `D`-block rows `v = D * w_hat` appear in the relation, not the position in the fold chain.
   The current names collide with `AkitaProofStep::Intermediate` / `Terminal` (`crates/akita-types/src/proof/levels.rs:806`), which genuinely do mean position.

5. `z_pre` (field on `DecomposeFoldWitness`, `crates/akita-prover/src/lib.rs:83`, and on `RingRelationWitness`) -> `z_folded_rings`.
   It is the challenge-folded witness in ring form before balanced digit decomposition ("Folded witness rows in ring form").
   The "pre" suffix is opaque; the new name states both that it is folded and that it is still in ring (pre-digit) form.
   The companion fields `centered_coeffs` and `centered_inf_norm` keep their names.

6. The "Direct" overload.
   Three unrelated meanings share the word, and the proposed terminal-direct-relation mode (`specs/terminal-direct-ring-relation.md`) would add a fourth.
   - `DirectWitnessProof` (`crates/akita-types/src/proof/direct_witness.rs:20`) -> `CleartextWitnessProof`.
   - `AkitaBatchedRootProof::Direct` (`crates/akita-types/src/proof/levels.rs:464`) -> `RootZeroFold`.
   - `AkitaPlannedStep::Direct` (`crates/akita-types/src/schedule.rs:265`) -> `AkitaPlannedStep::ZeroFold` (and the related `AkitaPlannedDirectStep` / `direct_step()` accessors).
   These rename Rust identifiers only; serialization must remain by discriminant index, not by variant name (see Invariants).

### Rename Candidates (Tier 2, lower priority)

These are real improvements but more invasive or more debatable.
Include them only if review agrees; otherwise they move to a follow-up.

- `build_w_coeffs` (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs:253`) -> `pack_recursive_witness_digits`.
  The output is packed i8 digit planes, not field coefficients.
  Also fix the stale doc comment that references a non-existent `e-hat` identifier (`coeffs.rs:229-237`).
- `repeated_b_commitment_rows` / `repeated_b_planes_per_claim` (`crates/akita-prover/src/protocol/quadratic_equation/repeated_b.rs`) -> `multi_group_b_rows` / `b_planes_per_claim`.
  "Repeated" means per-commitment-group repetition for multipoint batching.
- `MRowLayout` D/B/A row vocabulary.
  The `D`/`B`/`A` row labels are named after the Ajtai keys `d_key`/`b_key`/`a_key`, and `D` rows collide visually with the ring degree `const D`.
  At minimum document the key mapping at `generate_y` (`crates/akita-prover/src/protocol/quadratic_equation/r_split.rs:587`) and `LevelParams::m_row_count_for`; optionally rename `D` rows to `v_rows`.
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
  Enum-variant renames (Tier 1 item 6) must not change serialized discriminants; serialization stays index-based, not name-based.
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

## Evaluation

### Acceptance Criteria

- All Tier 1 renames are applied with every call site updated; the workspace builds with no references to the old names.
- `QuadraticEquation` is split: `RingRelationInstance` exists in `akita-types`, `RingRelationWitness` exists in `akita-prover`, and the prover constructors return both.
- The verifier builds and consumes `RingRelationInstance`; the witness-column segment layout is derived from it on both sides rather than duplicated.
- The `Option` witness fields and the `take_*` / `*_centered` accessor cluster on the old `QuadraticEquation` are gone.
- `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and `cargo test` pass on default features.
- `cargo test --features zk` and `cargo test --no-default-features` pass.
- Serialized proof bytes for existing proof fixtures are unchanged (see Testing Strategy).
- `logging-transcript` prover/verifier event-stream equality tests pass unchanged.
- A repository search shows the old identifiers are fully gone (no `QuadraticEquation`, `compute_r_split_eq`, `r_split`, `MRowLayout::Intermediate`, `MRowLayout::Terminal`, `z_pre`, `DirectWitnessProof` remaining, except in this spec and in historical specs).

### Testing Strategy

This is behavior-preserving (a rename plus an internal type split), so the strongest evidence is that existing tests pass unchanged and that serialized artifacts do not move.

- Run the full existing suite: `cargo test`, then `cargo test --features zk`, then `cargo test --no-default-features`.
- Byte-identity check: before the rename, capture serialized bytes of a representative proof for one dense and one one-hot profile (for example via an `akita_e2e` fixture or a small scratch test) and after the rename assert the bytes are identical.
  If a proof-bytes golden test already exists, that is sufficient; otherwise add a temporary local check during implementation and do not commit it.
- Discriminant-stability check for Tier 1 item 6: confirm `CleartextWitnessProof`, `RootZeroFold`, and `Step::ZeroFold` deserialize from bytes produced before the rename, proving the variant index did not move.
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

1. Rename leaf types and enum variants (`MRowLayout` variants, `DirectWitnessProof`, root/schedule `Direct` variants) and fix call sites.
2. Rename `z_pre` field and its uses.
3. Rename `r_stage1` and its uses.
4. Rename `compute_r_split_eq` and move `r_split.rs` to `relation_quotient.rs`, updating the re-export.
5. Rename `QuadraticEquation` last, since it is the most widely referenced type.
6. Apply Tier 2 renames if approved.
7. Run fmt, clippy, the full test matrix, and the byte-identity check.

Because the repo makes no backward-compatibility guarantees, every rename is a full cutover with no aliases.

### Relation Instance/Witness Split and Verifier Adoption

This is part 2, the refactor.

**Split.**
`QuadraticEquation` becomes two types.
`RingRelationInstance` (in `akita-types`) holds the public statement: `LevelParams` reference or the fields it needs, `MRowLayout`, fold `Challenges`, opening points (`RingOpeningPoint` / `RingMultiplierOpeningPoint`), the claim-incidence maps (`claim_to_point`, `claim_to_point_poly`, `claim_poly_indices`, `num_polys_per_point`, `num_public_rows`), `gamma` / `row_coefficient_rings`, and the targets `y` and `v`.
`RingRelationWitness` (in `akita-prover`) holds the prover secret: `z_folded_rings` (was `z_pre`), `w_hat`, `w_folded`, `hint`, and zk blinding.
The two prover constructors (`new_prover`, `new_recursive_multipoint_prover`) return `(RingRelationInstance, RingRelationWitness)`.
`ring_switch_build_w` takes `(&RingRelationInstance, RingRelationWitness)` and consumes the witness by value, replacing the `Option` fields and the `take_*` / `*_centered` accessor cluster (about ten methods) that exist only to move the witness out.

**Verifier adoption.**
The verifier today assembles the fold statement from loose pieces: `LevelParams`, the proof's `y_rings` / `v` / commitment rows, `ClaimIncidenceSummary`, and transcript-derived challenges, threaded through `verify_root_level_inner` (`crates/akita-verifier/src/protocol/levels.rs:271`) and `verify_one_level_inner` (`crates/akita-verifier/src/protocol/levels/recursive.rs:120`).
There is no single statement object today.
Under this spec the verifier builds a `RingRelationInstance` from those same pieces and threads it into `prepare_ring_switch_row_eval` and `relation_claim_from_rows_extension`, so prover and verifier share one definition of the ring-relation statement.

**Concrete deduplication in scope.**
The witness-column segment layout is currently the verifier's `RingSwitchSegmentLayout` (`crates/akita-verifier/src/protocol/ring_switch.rs:140,605`) versus the prover's inline offset arithmetic in `compute_m_evals_x` (`crates/akita-prover/src/protocol/ring_switch/evals.rs`).
This spec moves that segment layout to a method on `RingRelationInstance` so both sides derive it from one place.

**Deferred (enabled but not in this PR).**
Unifying the ring-switch Fiat-Shamir sampling blocks (prover `finalize.rs` versus verifier `ring_switch_verifier_core`) and the fold-challenge wrapper (`derive_stage1_challenges` versus the prover inline path) is enabled by the shared instance but is left to a follow-up to keep this PR's review surface bounded.
Unifying the prover's materialized M-table build with the verifier's deferred slice-MLE evaluators is explicitly not pursued; they are intentionally different cost models.

**Why `akita-types`, not a dependency flip.**
`RingRelationInstance` goes in `akita-types`, the existing lowest common dependency of both prover and verifier, next to `relation_claim_from_rows_extension` and `LevelParams::m_row_count_for` which already live there.
This spec does not change the crate dependency direction; see Non-Goals.

### Diff Surface (estimated)

The numbers below are literal-substring match counts on the worktree (`*.rs`, excluding `specs/`).
They are an upper bound: they include doc comments, tests, and derived identifiers, and they are the surface a full implementation would touch.

Per-symbol (Tier 1):

| Symbol | Occurrences | Files |
| --- | ---: | ---: |
| `QuadraticEquation` | 35 | 12 |
| `compute_r_split_eq` (+ `r_split` module/path) | ~22 | 12 |
| `z_pre` (incl. derived `z_pre_centered`, `z_pre_centered_inf_norm`) | 157 | 15 |
| `r_stage1` | 173 | 22 |
| `MRowLayout::Intermediate` | 47 | 20 |
| `MRowLayout::Terminal` | 38 | 18 |
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

Two symbols dominate: `z_pre` and `r_stage1` together account for roughly 330 of the ~560 occurrences.
If a smaller first PR is preferred, those two renames split cleanly into their own commits or a separate PR without blocking the rest.

Tier 2 adds a small increment on top: `build_w_coeffs` (about 8 occurrences across 3 files), `repeated_b*` (about 15 across 4 files), plus the `alpha` and doc-only changes.

The spec documents themselves are out of this count: `specs/terminal-direct-ring-relation.md` references some old names and would be updated if this lands first.

Part 2 (the split and verifier adoption) is not captured by these literal counts.
Its surface is the two prover constructors, the `ring_switch_build_w` signature and its callers, the deletion of the `Option` / `take_*` / `*_centered` accessor cluster, the new `RingRelationInstance` type in `akita-types`, and the verifier statement-assembly sites in `crates/akita-verifier/src/protocol/levels.rs` and `levels/recursive.rs`.
This is a real refactor with net line movement, unlike part 1, and is the part where reviewer attention belongs.

### Alternatives Considered

**Keep the names and only add doc comments.**
A glossary pass is cheaper and lower risk, but it leaves the misleading identifiers in code, so readers still hit the `QuadraticEquation` and `split_eq` traps on first contact.
This spec chooses renames for the Tier 1 set and allows the doc-only treatment for the Tier 2 row-vocabulary items.

**Rename `_hat` to spell out digit decomposition.**
Rejected: in the lattice setting `_hat` is the established paper convention for digit decomposition, so spelling it out would diverge from the source material without adding clarity for the intended audience.

**Fold these renames into the terminal-direct-ring-relation PR (#141).**
Rejected: that is a spec-only docs PR for an unrelated feature, and a cross-cutting rename sweep would destroy its reviewability.
The `MRowLayout` and `Direct` renames are most valuable to land before that feature's implementation, since it adds another `Terminal`/`Direct` meaning.

**Introduce deprecated aliases for a transition period.**
Rejected by the repo's no-backward-compatibility rule; all call sites are updated in one pass.

**Put the shared instance in `akita-verifier` and have `akita-prover` depend on it.**
Considered because the prover is the more comprehensive crate, so a prover-depends-on-verifier edge is plausible.
Rejected for this PR: the shared statement belongs in the lowest common dependency, and `akita-types` already plays that role (it holds `relation_claim_from_rows_extension`, `LevelParams`, `MRowLayout`, and the incidence types).
Flipping the dependency direction would not remove the duplication that motivates the split unless the shared code lived in the verifier, and it would pull verifier replay into the prover build for no gain over placing the type in `akita-types`.
The broader question of whether to restructure the crate graph is a separate investigation, not part of this PR.

**Split `QuadraticEquation` into instance and witness without verifier adoption.**
Considered as a smaller part 2.
This still removes the `Option`/`take_*` muddle and gives textbook instance/witness naming, but it leaves the verifier statement distributed and the segment-layout duplication in place.
The spec keeps verifier adoption in scope because it is the change that makes the shared `RingRelationInstance` earn its place in `akita-types`; if review prefers a smaller PR, verifier adoption is the natural cut line.

## Documentation

- Add a short "Naming" or glossary note (either in this spec's wake or in a prover module doc) that pins: `_hat` = digit decomposition, the two distinct `split_eq` meanings, `M` as a virtual row-combined relation table versus the Ajtai keys `A`/`B`/`D`, and the distinction between the matrix-relation layer and the norm sumcheck (`AkitaStage1Prover`).
- Update `specs/terminal-direct-ring-relation.md` references that mention `MRowLayout::Terminal`, `DirectWitnessProof`, or `compute_r_split_eq` if this rename lands first.
- No external or README changes are required.

## Execution

Suggested sequencing: land part 1 (renames) first, then part 2 (the split and verifier adoption) as a separate commit or PR, since part 2 carries the design judgment and the larger review surface.

Important risks to resolve first:

- Confirm proof and shape serialization is by discriminant index, not by variant name, before renaming any serialized enum variant (Tier 1 item 6).
  If any path serializes variant names, that path must be fixed or the variant rename dropped.
- The byte-identity and discriminant-stability checks are the gate that this stayed behavior-preserving; run them before opening the PR, especially for part 2.
- `QuadraticEquation` is referenced across prover, tests, and re-exports; do the split last and lean on the compiler to find call sites.
- For part 2, building `RingRelationInstance` on the verifier side must not move any challenge sampling or transcript absorb; populate the instance from already-sampled values so the `logging-transcript` event stream is unchanged.

## References

- `specs/terminal-direct-ring-relation.md`, the feature that motivated auditing these names and that adds another `Terminal`/`Direct` meaning.
- `crates/akita-prover/src/protocol/quadratic_equation.rs`, `QuadraticEquation` and the "quadratic equation builder" module doc.
- `crates/akita-prover/src/protocol/quadratic_equation/r_split.rs`, `compute_r_split_eq` and the gadget-factorization doc.
- `crates/akita-algebra/src/split_eq.rs`, the genuine `GruenSplitEq` construction that collides by name.
- `crates/akita-types/src/layout/params.rs`, `MRowLayout` and its doc.
- `crates/akita-types/src/proof/levels.rs`, `AkitaProofStep`, `AkitaBatchedRootProof`, and `TerminalLevelProof`.
- `crates/akita-types/src/proof/direct_witness.rs`, `DirectWitnessProof`.
- `crates/akita-prover/src/lib.rs`, `DecomposeFoldWitness` and `z_pre`.
- `crates/akita-verifier/src/stages/stage2.rs` and `crates/akita-verifier/src/protocol/slice_mle/structured_slice.rs`, `r_stage1` versus quotient `r`.
