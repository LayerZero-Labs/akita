# Fourth-Root Verifier Optimization Roadmap

Date: 2026-05-07  
Branch reviewed: `feat/tensor-challenges`

## Goal

The optimization goal is verifier time. Some extra prover work is expected and
acceptable if verifier replay drops substantially after the full optimization
set lands. Tensor challenges alone should not be judged as the final result:
they reduce challenge-dependent work, while the current dominant cost is still
setup-dependent M-table evaluation.

The target from the updated Jolt book has two techniques:

1. Tensor-structured folding challenges.
2. Claim-reduction sumcheck for the setup-dependent M-table contribution.

This branch implements most of the first technique and none of the second.

## Optimizations Already Implemented

### Tensor Stage 1 Challenge Shape

Implemented pieces:

- `Stage1ChallengeShape::{Flat, Tensor}` in `akita-challenges`.
- `TensorStage1Challenges` carrying left/right sparse challenge vectors.
- Balanced tensor split of `num_blocks = 2^r` into left and right dimensions.
- Transcript sampling with separate left and right labels.
- Left challenge digest absorbed before sampling right challenges.
- Logical expansion to integer challenges for existing prover fold kernels.
- Shape-aware challenge mass and SIS extraction accounting in `LevelParams`.
- Generated schedule gating through `allow_tensor_stage1_schedules`.

Current limitations:

- The prover materializes all logical tensor products before folding.
- Tensor configs are test/bench wrappers, not production defaults.
- D32 tensor is not ready in the benchmark matrix because tensor products can
  exceed the current i8 narrowing assumptions.

### Exact Negacyclic Tensor Aggregate Evaluation

Implemented pieces:

- `IntegerChallenge::tensor_product` performs negacyclic multiplication in
  `Z[X] / (X^D + 1)`.
- `TensorStage1Challenges::eval_factored_aggregate_at_pows` evaluates weighted
  tensor aggregates without materializing all `p,q` products.
- The evaluator includes the exact correction term:

```text
eval(reduce(L * R), alpha)
  = L(alpha) * R(alpha) - (alpha^D + 1) * Q(alpha)
```

Why it matters:

- This preserves the existing generic ring-switch challenge model.
- It avoids relying on the book's simplified product-only formula.
- It is the right foundation for verifier-side tensor summaries.

### Tensor-Aware Verifier M-Eval Summaries

Implemented pieces:

- `PreparedChallengeEvals::Tensor` stores tensor challenges compactly instead of
  eagerly storing every `c_{p,q}(alpha)`.
- `summarize_tensor_all_block_carries` splits low block bits into right and left
  tensor dimensions, handles carry propagation, and calls the exact aggregate
  evaluator.
- Recent commits share tensor carry summaries across claims and reuse weight
  buffers.
- `PreparedMEval::debug_expanded_challenge_evals` remains as a reference bridge.

Current limitations:

- `PreparedMEval::eval_at_point` still computes a full low-bit equality table.
- Opening point block weights are still flat `RingOpeningPoint::b` vectors.
- Direct setup matrix row evaluation remains in the verifier.

### Offset-EQ Helpers

Implemented pieces:

- `offset_eq::eval_offset_eq_tensor` evaluates shifted tensor inner products
  with carry DP.
- `offset_eq::eval_offset_eq_peeled_carry_terms` supports a peeled low-bit block
  plus coarse carry terms.
- Verifier M-eval uses these helpers for structured `w`, `t`, `z`, and `r`
  segments.

Current limitation:

- This is not yet the full sliced tensor transducer described in Section 5.
  It is a useful primitive and partial application, not a complete replacement
  for all shifted equality-table work.

### Tests and Bench Harness

Implemented pieces:

- Challenge determinism and transcript-shape tests.
- Exact aggregate vs expanded tensor-product tests.
- Product-only factorization negative test.
- Dense tensor Stage 1 E2E test.
- One-hot tensor E2E test.
- Same-point batched and multipoint tensor E2E tests.
- Schedule mismatch rejection tests.
- Verifier-only benchmark cases comparing flat and tensor Stage 1 schedules.
- Benchmark metadata printing for proof size, root shape, fold count, and final
  direct witness shape.

Current limitation:

- The branch does not include benchmark result files or thresholds.
- The benchmark matrix measures tensor-only behavior before claim reduction.

## Missing Optimizations

### 1. Claim-Reduction Sumcheck

Missing protocol pieces:

- A split of the current M-eval into:

```text
m_tau1(x) = m_alg(x) + m_setup(x)
```

- A verifier-computable `m_alg(r_x)` path.
- A setup-dependent `m_setup(r_x)` claim.
- A scaled claim:

```text
lambda = w_eval * alpha_eval
lambda * m_setup(r_x)
```

- A short setup-side sumcheck over row/column/coefficient variables.
- A final setup matrix polynomial opening claim.
- Transcript labels and proof structures for the new sumcheck.
- Verifier checks for the zero-`lambda` case without division.

This is the main missing verifier-time optimization.

### 2. Setup Matrix Polynomial and Opening Path

Missing data model pieces:

- A public polynomial view `S(row, col, coeff)` over the shared setup matrix.
- A deterministic mapping from D/B/A role prefixes into the shared envelope.
- A way to commit to or otherwise open `S` at a verifier challenge point.
- Recursive batching of the setup opening into the next level, or another
  binding mechanism with equivalent soundness.

Without this, the verifier must keep reading setup matrix rows directly.

### 3. Batched Stage 1 Relation Flow

Current flow:

- Stage 1 proves the range-check carried claim.
- Stage 2 is the existing fused relation/range continuation sumcheck over
  `col_bits + ring_bits` variables and degree 3.

Missing flow:

- Stage 1 should batch range-check and relation claims under the range-check
  degree envelope.
- Stage 2 should become a setup-side claim-reduction sumcheck, not the current
  full witness-domain fused sumcheck.
- The final combined output equality should be deferred until the setup-side
  term is supplied.

### 4. Complete Shifted-EQ / Sliced Tensor Contraction

If the intended optimization is the full Section 5 sliced tensor transducer,
the following are still missing:

- Factored representation of opening point block weights instead of only flat
  `RingOpeningPoint::b`.
- Avoiding unconditional full `EqPolynomial::evals` for the low block bits in
  tensor verifier paths.
- A generic transducer abstraction for slices beyond the currently hand-coded
  offset/carry cases.
- Tests that prove equivalence to materialized shifted equality tables across
  non-aligned offsets, multipoint batches, and recursive layouts.

### 5. Prover-Side Tensor Kernels

Missing prover efficiency pieces:

- Two-stage tensor fold:

```text
tmp_p = sum_q beta_q * s_{p,q}
z     = sum_p alpha_p * tmp_p
```

- Dense and one-hot kernels that consume tensor factors directly.
- Avoidance of full `IntegerChallenge` tensor-product materialization when
  possible.
- Prover-side support for the setup-side claim-reduction polynomial.

This is secondary to verifier time, but it should be measured once verifier
changes are in place.

### 6. Tiered Commitment / Cascade Control

The book's claim-reduction design introduces setup polynomial opening work that
can bloat the next recursive witness. The repo does not yet include:

- tiered chunk commitments;
- meta-commitment checks;
- split `D`/`B` commitment handling for witness plus setup polynomial;
- schedule search that accounts for the setup-opening cascade.

This can be postponed until a root-only claim-reduction prototype works, but it
is needed before applying the optimization across multiple levels.

## Incremental Plan

### Phase 0: Lock Measurements Before More Optimizing

Deliverables:

- Add a small reproducible benchmark note with exact commands and profiles.
- Capture verifier-only flat vs tensor results for representative profiles:
  onehot D64 at nv 12/15/20/25 and dense D64/D128 where feasible.
- Run with and without Rayon if the goal is to isolate algorithmic verifier
  work.
- Record per-span timing for `stage1_sumcheck`, `stage2_sumcheck`,
  `stage2_m_eval`, and the M-eval subspans.

Acceptance criterion:

- We know exactly which verifier subspan tensor currently improves or worsens.

### Phase 1: Align the Book Text With the Implemented Ring-Switch Model

Deliverables:

- Update Section 5 to remove product-only post-ring-switch factorization, or add
  the `(alpha^D + 1)` correction.
- Clearly distinguish "tensor challenges" from "claim reduction"; state that
  Technique 1 alone is not expected to lower total verifier time much.
- Mark shifted-eq/transducer work as partial if the book is tracking code status.

Acceptance criterion:

- The book no longer implies an optimization that the code does not implement or
  an algebraic identity that is false for generic `alpha`.

### Phase 2: Finish Tensor Verifier Cleanup

Deliverables:

- Remove unconditional full low-bit equality table construction from tensor
  verifier paths where not needed.
- Add a factored opening-point block-weight representation, or a helper that can
  evaluate shifted `RingOpeningPoint::b` weights from coordinates without storing
  all `2^r` values.
- Add targeted tests comparing tensor M-eval against materialized M-eval for
  random offsets, odd `r`, multipoint batches, and recursive layouts.

Acceptance criterion:

- Tensor verifier work scales with roughly `2^(r/2)` for challenge-dependent
  summaries, not with an accidental full `2^r` equality table.

### Phase 3: Split M-Eval Into `m_alg` and `m_setup`

Deliverables:

- Introduce reference functions:

```text
materialized_m_tau1(x)
m_alg_direct(x)
m_setup_direct(x)
```

- Prove by tests that:

```text
m_tau1(x) = m_alg(x) + m_setup(x)
```

for flat and tensor challenges, all row families, and batched/multipoint roots.

Acceptance criterion:

- The existing verifier can still call the direct combined path, while tests
  validate the split independently.

### Phase 4: Add Setup Polynomial View

Deliverables:

- Define `S(row, col, coeff)` over `setup.shared_matrix`.
- Implement deterministic mapping from D/B/A role-specific row/column prefixes
  to `S`.
- Add a direct evaluator for `S(r_row, r_col, r_coeff)` and small reference
  tests.

Acceptance criterion:

- Setup matrix contributions can be expressed as evaluations of one shared
  polynomial interface without changing the current proof.

### Phase 5: Prototype Setup-Side Claim Reduction

Deliverables:

- Add a standalone setup-side sumcheck prover/verifier over tiny test matrices.
- Carry the scaled claim `lambda * m_setup`; never divide by `lambda`.
- Add tests for `lambda = 0`, tampered setup rows, tampered sumcheck messages,
  and bad final openings.

Acceptance criterion:

- A direct setup-table claim and the claim-reduction proof agree on small
  instances.

### Phase 6: Integrate Claim Reduction Behind a Config Flag

Deliverables:

- Extend proof types with the setup-side sumcheck and setup opening claim.
- Add transcript labels for the new reduction challenges.
- Add a verifier path that:
  1. replays the batched witness-domain sumcheck,
  2. computes `lambda`,
  3. subtracts `lambda * m_alg`,
  4. verifies setup-side claim reduction,
  5. closes the deferred output equality.
- Keep the old fused Stage 2 path as the default until benchmark and soundness
  review are complete.

Acceptance criterion:

- E2E proofs verify under both old and new paths for small root-level schedules.

### Phase 7: Measure Verifier Target, Then Optimize Prover

Deliverables:

- Re-run the Phase 0 verifier matrix with claim reduction enabled.
- Compare flat, tensor-only, claim-reduction-only, and tensor-plus-claim-reduction.
- Only after verifier gains are visible, add prover tensor folding kernels and
  setup-side prover optimizations.

Acceptance criterion:

- Verifier time improves in the profiles that motivated the Section 5 work, and
  prover regression is quantified rather than hidden.

### Phase 8: Add Cascade Control and Production Schedules

Deliverables:

- Implement the tiered commitment or another bounded setup-opening strategy.
- Extend planner/search cost models for setup-opening cascade.
- Generate audited schedules for the target profiles.
- Enable `allow_tensor_stage1_schedules` only for reviewed configs.

Acceptance criterion:

- Multi-level verifier improvements survive recursive witness growth, and
  generated schedules pass SIS, headroom, proof-size, and verifier-time checks.

## Implementation Log

### 2026-05-07: Reference M-Eval Split

Implemented the first prerequisite for claim reduction: verifier-side prepared
M-evaluation can now return an additive split between algebraic terms and setup
matrix terms.

Where:

- `crates/akita-verifier/src/protocol/ring_switch.rs`
  - Added `PreparedMEvalSplit`.
  - Added `PreparedMEval::eval_split_at_point`.
  - Kept `PreparedMEval::eval_at_point` behavior unchanged by recombining the
    split.
  - Classified public/opening weights, tensor/flat challenge summaries, gadget
    scalars, quotient rows, and non-setup z terms as `algebraic`.
  - Classified direct D/B/A shared-matrix row reads as `setup`.
- `crates/akita-pcs/tests/ring_switch.rs`
  - Extended existing materialized-M-eval tests to assert that the split
    recombines to the current materialized multilinear evaluation for both flat
    and tensor Stage 1 layouts.

Validation:

```bash
cargo fmt -q
cargo test -p akita-pcs --test ring_switch prepared_m_eval -- --nocapture
```

Result: 2 tests passed.

### 2026-05-07: Setup Matrix Polynomial View

Implemented the direct `S(row, col, coeff)` surface needed by the setup-side
claim-reduction sumcheck.

Where:

- `crates/akita-types/src/layout/flat_matrix.rs`
  - Added `FlatMatrix::setup_polynomial_view`.
  - Added `SetupMatrixPolynomialView`.
  - Added coefficient access with row/column zero-padding.
  - Added direct multilinear evaluation over row, column, and coefficient
    variables.
- `crates/akita-types/src/layout/mod.rs` and `crates/akita-types/src/lib.rs`
  - Re-exported `SetupMatrixPolynomialView`.

Validation:

```bash
cargo fmt -q
cargo test -p akita-types layout::flat_matrix::tests::setup_polynomial_view -- --nocapture
```

Result: 2 tests passed.

### 2026-05-07: Eq-Weighted Setup Claim-Reduction Prototype

Implemented the protocol-independent sumcheck core for claims of the form
`scale * sum_z eq(target, z) * table(z)`. This is the reusable primitive for
reducing a setup-side weighted table claim to a final point claim on the setup
polynomial.

Where:

- `crates/akita-sumcheck/src/eq_weighted_table.rs`
  - Added `EqWeightedTableProver`.
  - Added `EqWeightedTableVerifier`.
  - Added `eq_eval` for verifier-side final weight evaluation.
- `crates/akita-sumcheck/src/lib.rs`
  - Exported the new prototype.
- `crates/akita-types/src/layout/flat_matrix.rs`
  - Added a roundtrip test that builds a table from `SetupMatrixPolynomialView`,
    proves the scaled eq-weighted claim, and verifies it through the generic
    sumcheck driver.

Validation:

```bash
cargo fmt -q
cargo test -p akita-types layout::flat_matrix::tests::setup_polynomial_claim_reduction_roundtrip -- --nocapture
```

Result: 1 test passed.

### 2026-05-07: Prepared M-Eval Setup-Claim Bridge

Connected the M-eval split to the claim-reduction prototype through a reference
table materialization path. This does not change the verifier hot path yet; it
proves that the setup component extracted from `PreparedMEval` can be reduced by
the new eq-weighted sumcheck at the same point used by Stage 2.

Where:

- `crates/akita-verifier/src/protocol/ring_switch.rs`
  - Added `PreparedMEval::debug_split_eval_table`, a reference-only method that
    materializes algebraic/setup split values over the padded M-eval x-domain by
    evaluating every Boolean point.
- `crates/akita-pcs/tests/ring_switch.rs`
  - Extended flat and tensor prepared-M-eval tests to assert that the split table
    recombines to the materialized M-eval table.
  - Proved and verified the setup table contribution with
    `EqWeightedTableProver` / `EqWeightedTableVerifier`.

Validation:

```bash
cargo fmt -q
cargo test -p akita-pcs --test ring_switch prepared_m_eval -- --nocapture
```

Result: 2 tests passed.

### 2026-05-07: Optional Stage 2 Setup-Claim Payload

Prepared the serialized proof surface for an opt-in setup-side claim-reduction
path without changing default proofs.

Where:

- `crates/akita-types/src/proof/mod.rs`
  - Added `AkitaStage2Proof::setup_claim_reduction: Option<SumcheckProof<_>>`.
  - Added the optional shape field to `LevelProofShape`.
  - Threaded optional serialization, deserialization, validation, and size
    accounting.
  - Existing constructors continue to set the field to `None`.
- `crates/akita-types/src/schedule.rs`
  - Updated dummy proof sizing construction to use `None`.

Validation:

```bash
cargo fmt -q
cargo test -p akita-pcs --test single_poly_e2e onehot_tensor_stage1_prove_verify -- --nocapture
```

Result: 1 test passed.

## Recommended Near-Term Order

1. Correct the Section 5 text around ring-switch factorization and current code
   status.
2. Measure current verifier subspans to confirm the remaining bottleneck.
3. Implement the `m_alg + m_setup` split as a reference-only change.
4. Prototype setup-side claim reduction on small matrices.
5. Integrate claim reduction behind a config flag.
6. Revisit tensor prover kernels and tiered commitments after verifier results
   justify the path.

Initial report creation was document-only; implementation log entries above list
the tests run for each landed code step.
