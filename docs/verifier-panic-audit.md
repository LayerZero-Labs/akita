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
| Setup matrix metadata rejects zero generation dimensions, incompatible ring dimensions, wrapped field counts, and insufficient view capacity before row views are used. | `FlatMatrix::check`, `FlatMatrix::deserialize_with_mode`, `FlatMatrix::total_ring_elements_at`, `FlatMatrix::ring_view`, and `RingMatrixView::row` in `crates/akita-types/src/layout/flat_matrix.rs`; matrix-capacity checks in `crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`, `zk_blinding.rs`, and root-direct recommitment validation in `crates/akita-verifier/src/protocol/batched.rs`. | Guarded |
| Schedule and `LevelParams`-derived verifier layouts reject invalid ring dimensions, `log_basis`, zero block geometry, zero digit depths, row-count overflow, and undersized matrix widths before replay. | `LevelParams::m_row_count` in `crates/akita-types/src/layout/params.rs`; witness-size helpers in `crates/akita-types/src/schedule.rs`; ring-switch layout checks in `prepare_ring_switch_row_eval` and `RingSwitchDeferredRowEval::eval_at_point`. | Guarded |
| Ring-switch preparation rejects inconsistent opening-point, challenge, gamma, group-routing, block, and row-weight shapes before constructing row-eval state. | `ring_switch_verifier`, `prepare_ring_switch_row_eval`, and `RingSwitchDeferredRowEval::eval_at_point` in `crates/akita-verifier/src/protocol/ring_switch.rs`; tests `ring_switch_prepare_rejects_invalid_log_basis` and `ring_switch_prepare_rejects_zero_num_blocks`. | Guarded |
| Claim opening-batch transcript absorption validates routing/count vectors. | `append_opening_batch_shape_to_transcript` in `crates/akita-types/src/proof/opening_batch.rs`, with malformed-shape tests in the same module. | Guarded |
| In-memory packed direct witnesses cannot panic on malformed metadata/buffers. | `PackedDigits::digit_at` returns `None`; `PackedDigits::to_field_elems` returns `Result`; tests in `crates/akita-types/src/proof/mod.rs` and `packed_witness_eval_rejects_truncated_data` in `crates/akita-verifier/src/stages/stage2.rs`. Direct terminal witness length is checked in `crates/akita-verifier/src/protocol/levels.rs`. | Guarded |
| Stage challenge dimensions are checked before equality-polynomial and multilinear evaluators. | `EqPolynomial::mle`, `EqPolynomial::evals*`, `multilinear_eval`, and `AkitaStage2Verifier::new`; stage-2 witness evaluators use checked shifts; `packed_witness_eval_rejects_challenge_dimension_mismatch` covers malformed stage wiring. | Guarded |
| Folded-root verifier rejects a root layout whose ring dimension does not match the const-generic dispatch before alpha splitting. | `validate_level_dispatch` in `crates/akita-verifier/src/protocol/mod.rs` is called before `alpha_bits` derivation in `verify_fold_batched_proof`, `verify_root_level`, and `verify_one_level`. | Guarded |
| Root-direct commitment recomputation validates direct witness length and setup stride envelope once before matrix row views or digit reads. | `validate_root_direct_recommitment_shape` in `crates/akita-verifier/src/protocol/batched.rs`; tests `root_direct_recommitment_rejects_undersized_setup_stride` and `root_direct_recommitment_rejects_wrong_witness_dimension`. | Guarded |
| Split batched-sumcheck APIs reject inconsistent verifier count, batching coefficient count, max rounds, and challenge length before slicing. | `compute_batched_expected_output_claim` in `crates/akita-sumcheck/src/batched_sumcheck.rs`, with `batched_expected_claim_rejects_malformed_shapes`. | Guarded |
| Offset-equality structured slices reject factor/challenge dimension mismatch without fallback arithmetic. | `eval_offset_eq_tensor` and `factor_summary` in `crates/akita-algebra/src/offset_eq.rs`, plus verifier callers in `structured_slice.rs`, `setup_contribution.rs`, and `zk_blinding.rs`. | Guarded |
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
| `RingSwitchDeferredRowEval` internal row, claim, block, and opening-point indexing. | Guarded | `prepare_ring_switch_row_eval`, `ring_switch_verifier`, and `eval_at_point` validate layout shape, route shape, challenge counts, opening-point lengths, row counts, and setup stride before the hot evaluators. |
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
