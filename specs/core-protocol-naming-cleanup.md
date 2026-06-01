# Spec: Core Protocol Naming Cleanup

Author(s): Quang Dao

Created: 2026-06-01

Status: proposed

PR: TBD

## Summary

Several identifiers in the core prover/verifier protocol flow carry names that do not match their actual mathematical content or protocol role.
The names mostly leaked from paper section titles or from single-letter paper notation, and they actively mislead new readers.
The clearest examples are `QuadraticEquation`, which holds a linear matrix relation rather than a quadratic equation, and `compute_r_split_eq`, whose "split-eq" is an unrelated gadget factorization that collides with the genuine `GruenSplitEq` sumcheck construction.

This spec proposes a behavior-preserving rename pass over a fixed set of core-protocol identifiers so the names describe the math and role.
It is a pure rename and doc change.
It does not alter any algorithm, serialized byte layout, or transcript byte stream.

## Intent

### Goal

Rename a fixed set of core prover/verifier protocol identifiers so each name matches its actual mathematical content and protocol role, without changing any runtime behavior, proof bytes, or transcript bytes.

The rename targets are types, functions, struct fields, and enum variants in:

- `crates/akita-prover/src/protocol/` (the `quadratic_equation`, `ring_switch`, and `flow` modules),
- `crates/akita-verifier/src/protocol/` and `crates/akita-verifier/src/stages/`,
- `crates/akita-types/src/proof/` and `crates/akita-types/src/layout/`.

It explicitly does not rename transcript label constants, since labels are diagnostics that must not enter production sponge bytes, and it does not rename the `_hat` digit-decomposition convention (see Non-Goals).

### Rename Table (Tier 1, definite)

Each row is `current -> proposed`, with the math/role that justifies it.
Proposed names are recommendations; the exact spelling is the main thing to settle in review.

1. `QuadraticEquation` (`crates/akita-prover/src/protocol/quadratic_equation.rs:245`) -> `RelationWitness`.
   It is prover state for the linear negacyclic-ring relation `M * z = y + (X^D + 1) * r` plus fold challenges and witness data (`z_pre`, `w_hat`, `v`, `y`).
   "Quadratic" is paper-section jargon (the module doc says "Quadratic equation builder ... §4.2") and describes the proof-system relation degree, not the struct's content.
   Alternatives: `RingRelationWitness`, `MatrixRelationState`.

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

5. `z_pre` (field on `DecomposeFoldWitness`, `crates/akita-prover/src/lib.rs:83`, and on `RelationWitness`) -> `z_folded_rings`.
   It is the challenge-folded witness in ring form before balanced digit decomposition ("Folded witness rows in ring form").
   The "pre" suffix is opaque; the new name states both that it is folded and that it is still in ring (pre-digit) form.
   The companion fields `centered_coeffs` and `centered_inf_norm` keep their names.

6. The "Direct" overload.
   Three unrelated meanings share the word, and the proposed terminal-direct-relation mode (`specs/terminal-direct-ring-relation.md`) would add a fourth.
   - `DirectWitnessProof` (`crates/akita-types/src/proof/direct_witness.rs:20`) -> `CleartextWitnessProof`.
   - `AkitaBatchedRootProof::Direct` (`crates/akita-types/src/proof/levels.rs:457`) -> `RootZeroFold`.
   - schedule `Step::Direct` -> `Step::ZeroFold`.
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

- **Pure rename.**
  No algorithm, control flow, numeric behavior, or data layout changes.
  The only edits are identifier names, their call sites, module paths, and doc comments.

- **Proof bytes unchanged.**
  Serialized proof bytes for every existing fixture are byte-identical before and after the rename.
  Enum-variant renames (Tier 1 item 6) must not change serialized discriminants; serialization stays index-based, not name-based.

- **Transcript bytes unchanged.**
  No transcript label constants are renamed, and label text never enters production sponge bytes, so Fiat-Shamir output is identical.
  The `logging-transcript` event streams stay equal across the rename.

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

## Evaluation

### Acceptance Criteria

- All Tier 1 renames are applied with every call site updated; the workspace builds with no references to the old names.
- `cargo fmt -q`, `cargo clippy --all -- -D warnings`, and `cargo test` pass on default features.
- `cargo test --features zk` and `cargo test --no-default-features` pass.
- Serialized proof bytes for existing proof fixtures are unchanged (see Testing Strategy).
- `logging-transcript` prover/verifier event-stream equality tests pass unchanged.
- A repository search shows the old identifiers are fully gone (no `QuadraticEquation`, `compute_r_split_eq`, `r_split`, `MRowLayout::Intermediate`, `MRowLayout::Terminal`, `z_pre`, `DirectWitnessProof` remaining, except in this spec and in historical specs).

### Testing Strategy

This is a rename, so the strongest evidence is that existing tests pass unchanged and that serialized artifacts do not move.

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

## Documentation

- Add a short "Naming" or glossary note (either in this spec's wake or in a prover module doc) that pins: `_hat` = digit decomposition, the two distinct `split_eq` meanings, `M` as a virtual row-combined relation table versus the Ajtai keys `A`/`B`/`D`, and the distinction between the matrix-relation layer and the norm sumcheck (`AkitaStage1Prover`).
- Update `specs/terminal-direct-ring-relation.md` references that mention `MRowLayout::Terminal`, `DirectWitnessProof`, or `compute_r_split_eq` if this rename lands first.
- No external or README changes are required.

## Execution

Important risks to resolve first:

- Confirm proof and shape serialization is by discriminant index, not by variant name, before renaming any serialized enum variant (Tier 1 item 6).
  If any path serializes variant names, that path must be fixed or the variant rename dropped.
- The byte-identity and discriminant-stability checks are the gate that this stayed a pure rename; run them before opening the PR.
- `QuadraticEquation` is referenced across prover, tests, and re-exports; rename it last and lean on the compiler to find call sites.

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
