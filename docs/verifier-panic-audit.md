# Verifier Panic-Hardening Audit

> **Historical snapshot.** This is a point-in-time audit artifact (PR #81), not
> a maintained reference. It predates the setup-layout rename (it still refers to
> `setup.seed.max_stride`; runtime uses `max_setup_len`). The durable verifier
> no-panic contract lives in the Akita Book
> ([`book/src/how/verification.md`](../book/src/how/verification.md)) and
> [`docs/verifier-contract.md`](verifier-contract.md). Scheduled to move to
> `docs/archive/` in the spec/doc archive pass (see `specs/PRUNING.md`).

This document records the verifier no-panic audit for the security-hardening
work in PR #81. The verifier boundary is defined in
`specs/security-hardening.md`: malformed verifier-facing proof, setup,
schedule, claim, opening, commitment, direct-witness, transcript, or prepared
state data must return `AkitaError` or `SerializationError` instead of
panicking or allocating from an unchecked shape.

## Completion Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Setup matrix metadata rejects zero generation dimensions, incompatible ring dimensions, wrapped field counts, and insufficient view capacity before row views are used. | `FlatMatrix::check`, `FlatMatrix::deserialize_with_mode`, `FlatMatrix::total_ring_elements_at`, `FlatMatrix::ring_view`, and `RingMatrixView::row` in `crates/akita-types/src/layout/flat_matrix.rs`; matrix-capacity checks in `crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs` and root-direct recommitment validation in `crates/akita-verifier/src/protocol/batched.rs`. | Guarded |
| Schedule and `LevelParams`-derived verifier layouts reject invalid ring dimensions, `log_basis`, zero block geometry, zero digit depths, row-count overflow, and undersized matrix widths before replay. | `LevelParams::relation_matrix_row_count_for` with explicit `RelationMatrixRowLayout` in `crates/akita-types/src/layout/params.rs`; witness-size helpers in `crates/akita-types/src/schedule.rs`; ring-switch layout checks in `prepare_relation_weight_evaluator` and `RelationWeightEvaluator::eval_flat_at_point`. | Guarded |
| Ring-switch preparation rejects inconsistent opening-point, challenge, gamma, group-routing, block, and row-weight shapes before constructing row-eval state. | `ring_switch_verifier`, `prepare_relation_weight_evaluator`, and the semantic event builder in `crates/akita-types/src/proof/relation_weights.rs`; tests `ring_switch_prepare_rejects_invalid_log_basis` and `ring_switch_prepare_rejects_zero_num_live_blocks`. | Guarded |
| Claim opening-batch transcript absorption validates routing/count vectors. | `append_opening_batch_shape_to_transcript` in `crates/akita-types/src/proof/opening_batch.rs`, with malformed-shape tests in the same module. | Guarded |
| In-memory segment-typed terminal witnesses cannot panic on malformed metadata/buffers. | `SegmentTypedWitness` decode paths return `Result`; malformed-shape tests live in `crates/akita-types/src/proof/tail_segments/tests.rs`, and terminal replay validates the scheduled layout in `crates/akita-verifier/src/protocol/core/terminal_direct.rs`. | Guarded |
| Stage challenge dimensions are checked before equality-polynomial and multilinear evaluators. | `EqPolynomial::mle`, `EqPolynomial::evals*`, `multilinear_eval`, and `AkitaStage2Verifier::new`; stage-2 witness evaluators use checked shifts; `packed_witness_eval_rejects_challenge_dimension_mismatch` covers malformed stage wiring. | Guarded |
| Folded-root verifier rejects a root layout whose ring dimension does not match per-role dispatch before alpha splitting. | Per-role `dispatch_for_field!` boundaries in `verify_fold` and related paths in `crates/akita-verifier/src/protocol/core/fold.rs`; `per_role_dispatch_rejects_wrong_stack_d` covers the mismatch. | Guarded |
| Root-direct commitment recomputation validates direct witness length and setup stride envelope once before matrix row views or digit reads. | `validate_root_direct_recommitment_shape` in `crates/akita-verifier/src/protocol/batched.rs`; tests `root_direct_recommitment_rejects_undersized_setup_stride` and `root_direct_recommitment_rejects_wrong_witness_dimension`. | Guarded |
| Split batched-sumcheck APIs reject inconsistent verifier count, batching coefficient count, max rounds, and challenge length before slicing. | `compute_batched_expected_output_claim` in `crates/akita-sumcheck/src/batched_sumcheck.rs`, with `batched_expected_claim_rejects_malformed_shapes`. | Guarded |
| Offset-equality evaluators reject factor/challenge dimension mismatch without fallback arithmetic. | [`eval_offset_eq_interval`](crates/akita-algebra/src/offset_eq.rs), [`summarize_pow2_block_carries`](crates/akita-algebra/src/offset_eq.rs), and [`eq_eval_at_index`](crates/akita-algebra/src/offset_eq.rs) (interval paths reject overflow and out-of-domain indices); verifier callers in [`relation_weights.rs`](crates/akita-types/src/proof/relation_weights.rs) and [`setup_contribution.rs`](crates/akita-types/src/setup_contribution.rs). | Guarded |
| Valid proof semantics remain unchanged. | `cargo test --workspace --all-features`; targeted verifier/dependency tests. | Verified locally |

## Remaining Panic-Shaped Operations

The following remaining panic-shaped operations are allowed by the contract
because they are either outside verifier-reachable public malformed-input
paths, test-only, or guarded by earlier validation. If future code makes any
of these verifier-reachable without preserving the stated guard, that is a
regression.

| Pattern / site class | Classification | Guard or reason |
| --- | --- | --- |
| Test `unwrap`, `expect`, `assert!`, `assert_eq!`, and fixed-size shifts under `#[cfg(test)]` in verifier/dependency crates. | Test-only | Fixture data is not verifier-facing runtime input. |
| Prover-only compact witness paths, including `CompactPairFoldLut` in `crates/akita-sumcheck/src/compact_fold.rs` and its users under `crates/akita-prover/src/protocol/sumcheck`. | Prover-only | `akita-verifier` does not construct or call these helpers. The spec explicitly excludes prover-only panic freedom in this PR. |
| NTT/CRT/SIMD assertions and shifts in `crates/akita-algebra/src/ntt` and prover kernels. | Setup/prover-only | Verifier replay does not execute these kernels on public proof data. |
| `RingMatrixView::rows` and `RingMatrixView::as_slice` unchecked internal slicing. | Guarded | Only constructed by `FlatMatrix::ring_view`, which validates row count, column count, ring dimension divisibility, and backing length first. |
| `EqPolynomial` table indexing inside table construction. | Guarded | Public table constructors now validate the implied table length before allocation and mutation. |
| Opening-point weight indexing in `lagrange_weights` and `monomial_weights`. | Guarded | Public weight constructors now validate the implied table length before allocation and mutation. |
| Relation-event row, claim, block, and opening-point indexing. | Guarded | `ring_switch_verifier`, `prepare_relation_weight_evaluator`, and `build_relation_weight_events` validate layout shape, route shape, challenge counts, opening-point lengths, row counts, and setup stride before evaluation. |
| `setup_contribution` and ZK blinding row/column indexing. | Guarded | The functions check derived row/column widths and `setup.seed.max_stride` once before indexing the validated flat matrix slice. |
| Transcript serialization fail-fast paths. | Intentional fail-fast | Transcript append serialization has no recoverable continuation; this is documented as outside the verifier malformed-input recovery path. |

## Local Evidence

Commands run locally during this pass:

```bash
cargo fmt --all
git diff --check
cargo check --workspace --all-targets --all-features
cargo check -p akita-verifier --no-default-features
cargo test --workspace --all-features
cargo test -p akita-algebra
cargo test -p akita-types -p akita-algebra -p akita-sumcheck -p akita-verifier --all-features
cargo test -p akita-verifier --all-features
cargo check -p akita-verifier --no-default-features
cargo clippy --all --message-format=short -q -- -D warnings
git diff --check
```

CI-only security, fuzzing, and supply-chain gates remain separate from this
local verifier panic-hardening audit.
