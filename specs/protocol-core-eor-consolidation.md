# Spec: Protocol Core EOR Consolidation

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     | Amirhossein Khajehpour                     |
| Created       | 2026-06-15                                 |
| Status        | implemented                                |
| PR            | #194                                       |
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

- Serialized proof layout and `ExtensionOpeningReductionProof` wire encoding are
  unchanged. Only module homes and in-memory preparation paths moved.
- Fiat–Shamir transcript layout is path-specific and must stay aligned with the
  shipped prover/verifier replay:
  - **Root EOR** (`pad_base_evals = false`): absorb logical openings, sample row
    coefficients γ, absorb proof partials, sample EOR η, then run the EOR
    sumcheck.
  - **Recursive suffix EOR** (`pad_base_evals = true`, single claim): absorb
    proof partials only, sample EOR η, with row coefficient fixed to `[1]` (no
    opening absorb and no γ squeeze). The verifier suffix path must not
    pre-absorb the carried opening before `verify_fold_eor`.
- Root and suffix share one implementation (`prepare_extension_opening_reduction`
  / `verify_fold_eor`); the `pad_base_evals` flag selects the branch above.
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
- `crates/akita-prover/src/protocol/core.rs` owns shared core wiring and
  re-exports the public prove entry points from `core/prove.rs`,
  `core/root_fold.rs`, `core/suffix.rs`, and `core/fold.rs`.
- `crates/akita-prover/src/protocol/core/fold.rs` owns the shared per-fold
  prover engine (`PreparedFold`, `prove_fold`, stage-1/2/3, ring-switch binding).
- `crates/akita-prover/src/protocol/core/suffix.rs` owns suffix state,
  `prove_suffix`, and per-level `prepare_fold_data`.
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
- `crates/akita-verifier/src/protocol/core/fold.rs` owns shared per-fold replay
  (`verify_fold_eor`, `verify_fold`, stage verifiers).
- Recursive suffix EOR uses partials-first transcript replay (`pad_base_evals =
  true`); root EOR keeps openings-then-γ-then-partials.
- The default validation suite passes:
  `cargo fmt -q`,
  `cargo clippy --all --message-format=short -q -- -D warnings`, and
  `cargo test`.

### Testing Strategy

Primary evidence is existing root, suffix, EOR, and end-to-end tests passing
after the refactor. Because suffix EOR transcript layout is path-specific,
regressions there will not always surface in root-only or non-EOR tests; include
at least one recursive extension-field suffix round-trip when touching EOR
transcript code.

Required checks:

- `cargo fmt -q`
- `cargo check --all --message-format=short -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`

Additional review checks:

- Search for stale module references to `protocol/flow/root_extension`.
- Search for root-only EOR materialization paths that bypass `FoldInputPoly`.
- Confirm suffix EOR does not pre-absorb openings on the verifier or absorb them
  on the prover when `pad_base_evals = true`.
- Confirm terminal witness shape checks still use the segment-aware
  `admits_realized` relation after merging main's tail encoding work.

### Performance

No proof-size movement is expected from the module restructure itself. Two
performance boundaries matter:

- **Sparse EOR storage:** one-hot tensor projections must remain
  `ProjectedSparse` so the sparse sumcheck path is not accidentally densified
  before relation construction. Root EOR must call the one-hot batched partial
  APIs (`tensor_extension_column_partials_batch`,
  `tensor_packed_extension_sparse_linear_combination`), not the dense base-eval
  fallback.
- **Fold engine locality:** moving `prove_fold` / `verify_fold` into
  `core/fold.rs` is structural only; hot paths stay the same once sparse EOR
  dispatch is preserved.

## Design

### Directory Restructuring

The prover protocol layout was renamed and split to match the verifier's current
root/suffix terminology:

- `protocol/core.rs` is the shared prover core module. It owns imports, ZK hiding
  state, and private submodule wiring.
- `protocol/core/fold.rs` owns the shared per-fold prover engine: trace-table
  helpers, `PreparedFold`, `prove_fold`, stage-1/2/3, and ring-switch binding.
- `protocol/core/prove.rs` owns top-level batched prove orchestration after the
  schedule and setup inputs are selected.
- `protocol/core/root_fold.rs` owns folded-root and terminal-root preparation.
  It performs opening-batch transcript setup, optional EOR preparation, root
  opening-point preparation, root ring-relation construction, and dispatch into
  `prove_fold`.
- `protocol/core/suffix.rs` owns recursive suffix orchestration: it carries
  `SuffixProverState`, runs `prove_suffix`, and per-level `prepare_fold_data`
  before dispatching into `prove_fold`.
- `protocol/core/extension_opening_reduction.rs` owns common EOR preparation:
  partial opening preparation, path-selected transcript replay (root vs suffix),
  sparse/dense term construction, sumcheck proving, and conversion from reduction
  challenges to the protocol point.

The verifier mirrors the same conceptual split:

- `crates/akita-verifier/src/protocol/core.rs` owns shared imports, terminal
  witness replay, and submodule wiring.
- `core/fold.rs` owns shared per-fold replay: EOR verification, stage-1/2/3
  replay, ring-switch replay, `verify_fold`.
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

The consolidated path has one EOR lifecycle, with transcript branching on
`pad_base_evals`:

1. `prepare_extension_opening_reduction` pads the logical opening point and
   derives tensor column partials. On the **root** path it then absorbs logical
   openings, samples row coefficients γ, absorbs proof partials, and samples EOR
   η. On the **recursive suffix** path it skips opening absorb and γ, fixes the
   row coefficient to `[1]`, absorbs proof partials, and samples EOR η.
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

**Proof bytes.** The refactor does not change serialized proof objects. EOR proof
segments, fold proofs, and terminal witness encodings are unchanged on the wire.

**Transcript bytes.** Layout is unchanged for the root fold and for non-EOR
rounds. Recursive suffix EOR is the subtle case: main never absorbed the carried
single opening before partials. An intermediate consolidation step accidentally
replayed the root opening-first layout on suffix rounds (changing Fiat–Shamir
challenges). The shipped fix restores partials-first suffix replay on both prover
and verifier.

| Phase | Root EOR | Recursive suffix EOR |
|-------|----------|----------------------|
| Openings absorbed before partials | yes | no |
| Row coefficient γ sampled | yes (batching) | no (implicit `[1]`) |
| Partials absorbed | yes | yes |
| EOR η sampled | yes | yes |
| ZK masked opening on suffix path | N/A (root uses public openings when provided) | prover passes `None`; verifier checks opening against partials inside `verify_fold_eor` |

Everything after EOR (ring switch, stage-1/2/3, terminal witness) keeps the
pre-refactor order. Labels and sumcheck message shapes are unchanged.

**Review note.** Do not claim “transcript bytes unchanged” without the root vs
suffix split above. This spec is orthogonal to `specs/fold-linf-rejection.md`
(Golomb `z` / `β_inf` tail rejection); that work does not alter EOR transcript
layout.

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
- `crates/akita-prover/src/protocol/core/fold.rs`
- `crates/akita-prover/src/protocol/core/root_fold.rs`
- `crates/akita-prover/src/protocol/core/suffix.rs`
- `crates/akita-verifier/src/protocol/core/fold.rs`
