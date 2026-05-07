# Fourth-Root Verifier Implementation Audit

Date: 2026-05-07  
Branch reviewed: `feat/tensor-challenges`  
External reference: `Lattice_Jolt/sections/5_fourth_root_verifier.tex`

## Executive Summary

The branch implements a meaningful subset of the first fourth-root verifier
technique: tensor-structured Stage 1 folding challenges. It also implements an
important correction not captured by the simplified book formula: because Hachi
ring-switches at a generic verifier-field point, evaluating
`reduce(L_p * R_q)` is not equal to `L_p(alpha) * R_q(alpha)` unless
`alpha^D = -1`. The code correctly uses an explicit `(alpha^D + 1)` quotient
correction in tensor aggregate evaluation.

The second technique from the updated book, the claim-reduction sumcheck, is not
implemented. The verifier still runs the existing Stage 2 fused sumcheck over
the full witness-domain variables and still evaluates the setup-dependent
`M_alpha` contribution directly from `A`/`B`/`D` shared matrix rows at the final
sumcheck point. There is no setup-side claim-reduction proof, no split into
`m_alg + m_setup`, no setup-matrix polynomial opening, and no tiered commitment
machinery.

The "shifted eq" / sliced-eq optimization is only partially present. The repo has
an `offset_eq` helper and tensor carry summaries used in verifier M-evaluation,
but the current verifier path still materializes a full low-bit equality table
and still stores opening-point block weights as flat `2^r` vectors. If the book
section is claiming a complete shifted-eq/tensor-slice contraction, the code does
not yet implement that full optimization.

This matches the prior empirical observation: tensor challenges alone are not
expected to lower verifier time reliably because they target only the
challenge-dependent minority of the verifier cost while leaving the dominant
setup-table work in place. Verifier time should be the primary metric after the
claim-reduction path lands; prover time may regress and should be tracked as a
secondary cost.

## Book Claims vs Code Status

| Book optimization / requirement | Status in this branch | Evidence and notes |
|---|---:|---|
| Tensor Stage 1 challenge shape: sample left and right vectors instead of a flat vector. | Implemented | `Stage1ChallengeShape::Tensor`, `TensorStage1Challenges`, `sample_stage1_challenges`, and tensor transcript labels are in `crates/akita-challenges/src/stage1.rs`. |
| Fiat-Shamir two-round tensor shape with an empty prover round between left and right. | Implemented in transcript form | The code samples left challenges, absorbs `tensor_stage1_left_digest`, then samples right challenges. This gives the right challenge dependency without serializing an empty proof message. |
| Tensor product challenges for prover folding. | Implemented functionally, not optimized | The prover calls `Stage1Challenges::expand_integer` and materializes all logical tensor products before folding in `QuadraticEquation::new_prover` and `ring_switch_build_w`. This is correct for compatibility but does not implement the two-stage prover fold described in the book/spec. |
| Simplified ring-switch factorization `c_alpha(p,q) = c_alpha^L(p) * c_alpha^R(q)`. | Correctly rejected by implementation | `tensor-exact-aggregate-evaluator.md` and tests document the missing negacyclic correction. `TensorStage1Challenges::eval_factored_aggregate_at_pows` implements the exact corrected identity. |
| Exact tensor aggregate verifier path that avoids materializing all tensor products. | Partially implemented | `PreparedChallengeEvals::Tensor` stores tensor challenges compactly, and `summarize_tensor_all_block_carries` calls `eval_factored_aggregate_at_pows` for factored carry weights. This avoids full tensor expansion for selected M-eval challenge summaries. |
| Shifted/sliced equality contraction for tensor evaluations. | Partial | `crates/akita-algebra/src/offset_eq.rs` implements offset-eq tensor and peeled carry helpers. `PreparedMEval::eval_at_point` uses these helpers, but still computes full `EqPolynomial::evals(&x_challenges[..block_bits])` and opening-point `b` remains a flat vector. This is not a complete shifted-eq replacement. |
| Share/reuse tensor carry summaries and buffers. | Implemented | Recent commits refactor tensor carry summaries across claims and reuse `u_weights`/`v_weights` buffers in `summarize_tensor_all_block_carries`. |
| Claim-reduction sumcheck replacing setup-dependent M-table evaluation. | Not implemented | `AkitaStage2Verifier` still has `num_rounds = col_bits + ring_bits`, degree `3`, and computes `relation_oracle = w_eval * alpha_val * m_val` using direct `PreparedMEval::eval_at_point`. |
| Split `m_tau1 = m_alg + m_setup`. | Not implemented | There are no `m_alg`, `m_setup`, or setup-side claim APIs in source. Searches find only the existing Stage 2 fused path. |
| Setup polynomial `S(row, col, coeff)` and setup opening. | Not implemented | The code directly reads `setup.shared_matrix.ring_view` in prover and verifier M-eval paths. There is no committed setup polynomial proof object or opening claim. |
| Tiered/split commitment for carrying setup polynomial openings recursively. | Not implemented | No tiered commitment or meta-commitment structures were found in runtime proof/setup types. |
| Production-gated tensor schedules. | Implemented conservatively | `ScheduleProvider::allow_tensor_stage1_schedules` defaults to `false`, and generated tensor tables are rejected unless a config opts in. Current tensor configs are test/bench wrappers rather than production defaults. |

## Critical Findings

### 1. Tensor-only does not target the dominant verifier bottleneck

The updated book explicitly says setup-dependent M-table evaluation is the
dominant cost. This branch mainly reduces challenge aggregation in the verifier.
The direct setup matrix terms remain in `PreparedMEval::eval_at_point`:

- D rows are evaluated through `eval_d_matrix_w_residual_direct`.
- B rows are evaluated through `eval_b_matrix_t_residual_direct`.
- A rows are evaluated while building `z_base`.

That means the implementation is not expected to reach the fourth-root verifier
goal yet. The most likely benchmark outcome remains: tensor can be neutral or
worse until claim reduction removes the setup-side work.

### 2. The code correctly fixes a book-level algebra simplification

The LaTeX currently states the tensor challenge contribution factorizes after
ring switching. In Hachi's current ring-switch model, this is false at a generic
`alpha`. The code's exact aggregate evaluator computes:

```text
eval(reduce(L * R), alpha)
  = L(alpha) * R(alpha) - (alpha^D + 1) * Q(alpha)
```

This is the right correction if `alpha` remains generic for ring-switch
soundness. The book section should either include this correction or explicitly
change the ring-switch model, which would require a separate soundness review.

### 3. The shifted-eq optimization is not complete

`offset_eq` implements useful offset/carry contraction primitives, and the
verifier uses them for several structured M-eval segments. However,
`PreparedMEval::eval_at_point` still computes a full `block_low_eq` over
`block_bits` before it knows whether challenge evaluation is tensor. It also
summarizes each public opening-point block vector from a flat
`RingOpeningPoint::b`.

So if "shifted eq" means the full sliced tensor transducer from Section 5, this
branch only has a lower-level helper and selected applications. It does not yet
eliminate all `O(2^r)` equality-table/opening-vector work in verifier replay.

### 4. Prover tensor support is functional but intentionally not optimized

The prover expands tensor challenges into all logical block challenges via
`expand_integer` before folding. This preserves existing `AkitaPolyOps`
interfaces and tests the algebra, but it does not implement:

- two-stage tensor folding `tmp_p = sum_q beta_q s_{p,q}`, then
  `z = sum_p alpha_p tmp_p`;
- tensor-aware dense or one-hot kernels that avoid full tensor product
  materialization;
- a prover-side strategy for the later claim-reduction setup polynomial.

This is acceptable for an incremental verifier-first path, but prover cost
should not be interpreted as final.

### 5. Schedule/security accounting was updated, but tensor is not production

`LevelParams` tracks tensor challenge shape, effective challenge mass, and the
`4 * omega` extraction degradation used for A-role SIS sizing. Generated tensor
schedules are gated behind `allow_tensor_stage1_schedules`. This is the right
direction.

However, current tensor use is mainly through test and benchmark wrapper configs.
D32 tensor benchmarks are intentionally omitted in `akita_e2e.rs` because the
current D32 challenge family can produce tensor-product coefficients that exceed
the prover's i8 narrowing path. Tensor should remain opt-in until schedules,
headroom, and verifier-time results are audited per production profile.

## Correctness Assessment

What looks logically correct:

- Tensor left/right challenge transcript ordering is clear and domain separated.
- The left digest binds the right challenge sampling to the sampled left vector.
- Tensor challenge expansion uses negacyclic multiplication in
  `IntegerChallenge::tensor_product`.
- The exact aggregate evaluator is bilinear in the left/right weighted sums and
  includes the quotient correction for generic `alpha`.
- Tests cover exact aggregate vs expanded products and demonstrate that
  product-only factorization is not exact.
- E2E tests cover dense, one-hot, same-point batching, multipoint batching, and
  schedule shape mismatches.

Main residual risks:

- The verifier still has unconditional full equality table work in
  `PreparedMEval::eval_at_point`, which may hide or erase tensor savings.
- Tensor-specific verifier aggregation is specialized to the current block
  layout and carry splitting; more layout variants need tests before production.
- The benchmark harness compares flat vs tensor verifier replay, but the branch
  does not include committed benchmark results or acceptance thresholds.
- Claim reduction is not present, so performance conclusions from this branch
  should not be extrapolated to the fourth-root target.
- The docs and code disagree if the docs imply product-only factorization or a
  complete shifted-eq optimization.

## Incremental Change Review

The recent commits form a sensible incremental sequence for Technique 1:

1. Add tensor challenge plumbing and transcript labels.
2. Add exact tensor aggregate evaluation after discovering product-only
   factorization is invalid for generic ring-switch `alpha`.
3. Store tensor challenge data compactly in verifier preparation.
4. Decompose tensor carry summaries and use them in M-eval.
5. Add E2E, tamper, and schedule-shape tests.
6. Add production gating for generated tensor schedules.
7. Add micro-optimizations for stack buffers and shared carry summaries.
8. Expand verifier-only benchmark cases.

That sequence is logically sound for a tensor-challenge prototype. It should not
be described as completing the fourth-root verifier optimization because the
claim-reduction sumcheck and setup-opening protocol are still missing.

## Audit Conclusion

This branch is best characterized as "Technique 1 plumbing plus exact tensor
aggregate verifier support," not as the full Section 5 fourth-root verifier. The
most important implementation gap is the claim-reduction sumcheck. The most
important documentation gap is that Section 5 should not claim simple
post-ring-switch tensor factorization unless it also accounts for the
negacyclic quotient correction or changes the ring-switch model.

No tests or benchmarks were run as part of this document-only audit.
