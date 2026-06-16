# Spec: Protocol Core EOR Consolidation

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     | Amirhossein Khajehpour                     |
| Created       | 2026-06-15                                 |
| Status        | implemented                                |
| PR            |                                            |
| Supersedes    |                                            |
| Superseded-by |                                            |
| Book-chapter  | how/proving/extension-opening-reduction.md |

## Summary

The prover protocol flow has been restructured around the same root/suffix/core
vocabulary used by the verifier, and the extension-opening-reduction (EOR) path
has been consolidated so root and suffix fold preparation share one reduction
machinery. The main implementation change is that EOR materialization no longer
depends on a root-only projected-polynomial path: `FoldInputPoly` is the single
fold-facing wrapper for original witnesses, dense tensor projections, and sparse
tensor projections, and it implements `AkitaPolyOps` so both root folds and
recursive suffix witnesses can pass through the same relation-building code.

## Intent

### Goal

Make prover root and suffix fold orchestration mirror the verifier's core layout,
and make EOR preparation a shared fold input transformation rather than a
separate root-extension flow.

The refactor is implemented in the prover and verifier protocol crates:

- Prover top-level orchestration is now `crates/akita-prover/src/protocol/core.rs`
  with submodules under `crates/akita-prover/src/protocol/core/`.
- Verifier fold replay is now `crates/akita-verifier/src/protocol/core.rs` with
  matching `root_fold` and `suffix` submodules.
- The old prover `protocol/flow/root_extension.rs` split is removed; EOR logic is
  centralized in `protocol/core/extension_opening_reduction.rs`.
- `FoldInputPoly` lives in `crates/akita-prover/src/backend/field_reduction.rs`
  and is re-exported from `akita-prover` and `backend`.

### Invariants

- The refactor must not change proof bytes or transcript bytes. EOR still absorbs
  logical openings, row coefficients, partials, and sumcheck messages in the same
  order; only the module homes and in-memory preparation path changed.
- Prover and verifier must keep the same root/suffix boundary. Root folds use
  `BlockOrder::RowMajor`; suffix folds use `BlockOrder::ColumnMajor`.
- EOR is optional and shape-driven. Non-EOR roots use `FoldInputPoly::Original`;
  EOR roots transform into projected `FoldInputPoly` values before ring-relation
  construction.
- `FoldInputPoly` must preserve storage shape. Dense tensor projections stay
  dense, sparse one-hot tensor projections stay sparse, and original non-EOR
  witnesses remain borrowed from the caller.
- Suffix witnesses must be able to use the same `AkitaPolyOps` surface as root
  fold inputs. The wrapper cannot be root-specific.
- The verifier no-panic boundary is unchanged: malformed proof, schedule, setup,
  or terminal witness shapes still reject with `AkitaError`.

### Non-Goals

- Do not change the EOR sumcheck protocol, challenge derivation, transcript
  labels, serialized proof layout, or terminal witness wire encoding.
- Do not add compatibility aliases for the removed `flow` module paths.
- Do not move prover-only witness materialization into `akita-types` or the
  verifier crate.
- Do not replace the dense/sparse EOR prover optimizations; this refactor only
  provides a common input abstraction for them.

## Evaluation

### Acceptance Criteria

- `crates/akita-prover/src/protocol/mod.rs` exposes `core`, not the old
  root/suffix `flow` module.
- `crates/akita-prover/src/protocol/core.rs` owns the shared fold state and
  re-exports the public prove entry points from `core/prove.rs`,
  `core/root_fold.rs`, and `core/suffix.rs`.
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`
  contains the shared EOR preparation and proof helpers used by both root and
  recursive fold paths.
- `crates/akita-prover/src/protocol/flow/root_extension.rs` is deleted.
- `FoldInputPoly::{Original, ProjectedDense, ProjectedSparse}` implements
  `AkitaPolyOps` and dispatches every fold-facing operation to the wrapped
  representation.
- Root EOR materialization calls
  `AkitaPolyOps::tensor_packed_extension_fold_input` and then builds the
  ring-relation over `&FoldInputPoly` references.
- Non-EOR root folds still borrow original polynomials through
  `FoldInputPoly::Original`.
- The default validation suite passes:
  `cargo fmt -q`,
  `cargo clippy --all --message-format=short -q -- -D warnings`, and
  `cargo test`.

### Testing Strategy

The change is intended to be behavior-preserving, so the primary evidence is that
existing root, suffix, EOR, transcript, and end-to-end tests pass unchanged.

Required checks:

- `cargo fmt -q`
- `cargo check --all --message-format=short -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`

Additional review checks:

- Search for stale module references to `protocol/flow/root_extension`.
- Search for root-only EOR materialization paths that bypass `FoldInputPoly`.
- Confirm terminal witness shape checks still use the segment-aware
  `admits_realized` relation after merging main's tail encoding work.

### Performance

No proof-size or transcript performance movement is expected from the module
restructure itself. The performance-sensitive part is preserving sparse EOR
storage: one-hot tensor projections must remain `ProjectedSparse` so the sparse
sumcheck path is not accidentally densified before relation construction.

## Design

### Directory Restructuring

The prover protocol layout was renamed and split to match the verifier's current
root/suffix terminology:

- `protocol/core.rs` is the shared prover core module. It owns imports, shared
  fold helpers, trace-table helpers, and the private submodule wiring.
- `protocol/core/prove.rs` owns top-level batched prove orchestration after the
  schedule and setup inputs are selected.
- `protocol/core/root_fold.rs` owns folded-root and terminal-root preparation.
  It performs opening-batch transcript setup, optional EOR preparation, root
  opening-point preparation, root ring-relation construction, and dispatch into
  the shared fold prover.
- `protocol/core/suffix.rs` owns recursive suffix folds. It prepares recursive
  EOR when needed, builds suffix ring relations, handles terminal witness
  materialization, and carries `SuffixProverState`.
- `protocol/core/extension_opening_reduction.rs` owns common EOR preparation:
  partial opening preparation, row coefficient sampling, sparse/dense term
  construction, sumcheck proving, and conversion from reduction challenges to the
  protocol point.

The verifier mirrors the same conceptual split:

- `crates/akita-verifier/src/protocol/core.rs` owns shared fold replay helpers.
- `core/root_fold.rs` verifies the folded-root or one-fold terminal-root payload.
- `core/suffix.rs` verifies recursive suffix folds after the root handoff.
- `core/verify.rs` remains the public batched verifier orchestration once the
  schedule is selected.

This removes the old prover-only `flow/root_extension.rs` island. EOR is no
longer a separate root-extension flow; it is part of core fold preparation.

### EOR Consolidation

Before consolidation, the root EOR path had a distinct root-extension module that
prepared tensor partials, sampled row coefficients, built dense or sparse EOR
terms, and then separately transformed root polynomials into a projected form for
ring-relation construction.

The consolidated path has one EOR lifecycle:

1. `prepare_extension_opening_reduction` pads the logical opening point, derives
   tensor column partials, appends logical openings, samples row coefficients,
   absorbs proof partials, samples the EOR `eta` challenges, and builds the EOR
   sumcheck input claim.
2. `build_extension_opening_reduction_terms` selects sparse terms when every
   input supports sparse tensor packed extension evaluations; otherwise it builds
   dense terms from base evaluations and tensor equality factors.
3. `prove_extension_opening_reduction` proves the reduction and returns the
   protocol point used by the ordinary fold relation.
4. Root fold preparation transforms the committed polynomials with
   `tensor_packed_extension_fold_input` and then treats those transformed values
   as ordinary fold inputs.
5. Suffix fold preparation uses the same EOR machinery against the logical
   recursive witness when the recursive opening field has extension degree
   greater than one.

The important boundary is that EOR proves the logical opening reduction, while
`FoldInputPoly` provides the committed witness representation consumed by the
ring relation after reduction.

### `FoldInputPoly`

`FoldInputPoly<'a, F, P, D>` is the common input type for fold materialization:

```rust
pub enum FoldInputPoly<'a, F: FieldCore, P, const D: usize> {
    Original(&'a P),
    ProjectedDense(DensePoly<F, D>),
    ProjectedSparse(Arc<SparseRingPoly<F, D>>),
}
```

The variants intentionally represent storage, not protocol phase:

- `Original(&P)` is the borrowed non-EOR input.
- `ProjectedDense(DensePoly)` is the dense tensor-projected input used when the
  backend cannot preserve sparse structure.
- `ProjectedSparse(Arc<SparseRingPoly>)` is the sparse tensor-projected input used
  by one-hot-style backends.

`FoldInputPoly` implements `AkitaPolyOps`, so downstream fold code can call the
same operations regardless of whether the witness is original, dense-projected,
or sparse-projected. This includes root evaluation and folding,
`tensor_extension_column_partials`, `tensor_packed_extension_evals`,
`tensor_packed_extension_poly`, `decompose_fold`, and the optional batched fold
kernels.

The trait method `AkitaPolyOps::tensor_packed_extension_fold_input` is the
factory. The default implementation projects into `ProjectedDense`. One-hot
backends override it to return `ProjectedSparse`. The blanket implementation for
`&P` preserves the wrapper variant so borrowed suffix or grouped commitment paths
do not accidentally collapse sparse projections into dense values.

This makes the EOR materialization generic enough for suffix-round witnesses:
once a recursive witness view implements `AkitaPolyOps`, the suffix path can use
the same EOR preparation and ring-relation construction surface as the root path.

### Transcript and Proof Compatibility

The refactor changes where code lives, not the protocol byte stream.

- Opening-batch shape, commitments, shared opening points, logical openings,
  row coefficients, EOR partials, EOR challenges, ring-switch messages, stage-1
  messages, stage-2 messages, and terminal witness payloads remain in the same
  order.
- `ExtensionOpeningReductionProof` remains the wire proof object.
- `PreparedFold` carries the optional EOR proof beside the ordinary
  `RingRelationInstance` and `RingRelationWitness`.
- Verifier replay continues to derive the same prepared points and trace claims
  from the proof and public inputs.

## Documentation

This spec is the durable implementation note for the refactor. The book chapter
`book/src/how/proving/extension-opening-reduction.md` should own the long-term
protocol explanation; this spec should remain as the code-structure and migration
record.

## References

- `specs/core-protocol-naming-cleanup.md`
- `specs/eor-streamed-prover.md`
- `specs/tail-wire-encoding.md`
- `crates/akita-prover/src/backend/field_reduction.rs`
- `crates/akita-prover/src/protocol/core.rs`
- `crates/akita-prover/src/protocol/core/extension_opening_reduction.rs`
- `crates/akita-prover/src/protocol/core/root_fold.rs`
- `crates/akita-prover/src/protocol/core/suffix.rs`
