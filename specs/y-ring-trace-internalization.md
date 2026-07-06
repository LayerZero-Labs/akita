# Spec: Internalize the §3.1 Trace Check and Drop On-Wire `y_ring`

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-06-05                     |
| Status      | superseded (trace mechanism)   |
| PR          | #154                           |

> **Superseded trace mechanism.** The `gamma^2` fused trace summand described
> below is replaced by the `EvaluationTrace` relation row inside
> [`relation-weight-polynomial.md`](relation-weight-polynomial.md). Stage 2 now
> has two summands only (`RelationWeightPolynomial` + range virtual term); there
> is no separate `trace_coeff`, no terminal `trace_gamma` squeeze, and no
> third batching slot. The goal of this spec (drop on-wire `y_ring`, bind opening
> through committed witness) is preserved; only the encoding changed.

## Summary

Every Akita fold level ships the §3.1 ring opening value `y_ring` (`Y` in the paper) on the wire, and the verifier reads it both to form the ring-switch relation claim `V_alpha` and to run a separate public trace check `Tr_H(y_ring · sigma_{-1}(packed_inner_point)) = (D/K) · opening` that ties the ring value back to the incoming extension-field claim.
This is one base-field ring element per intermediate level, plus `P` at the root and one at the terminal.
The opening value is fully determined by the already-committed fold witness: today the public-output row recomposes the committed `e_hat` digits into per-block `e_folded` rings and checks `y_ring = sum_j b_j · e_folded_j`.
This spec removes `y_ring` from the proof entirely by internalizing the trace projection as one extra fused term in the stage-2 sum-check, enforcing the same normalized trace functional that `recover_ring_subfield_inner_product(y_ring, packed_inner_point)` computes directly against the committed witness.
The new fused addend does not increase the stage-2 degree bound or add committed data, so the net effect is at least a clean save of one ring element per level and, in ZK builds, removal of that ring element's hiding mask.
If the public-output rows are removed from `M` rather than kept as inert padding, the quotient tail can shrink too.

## Intent

### Goal

Eliminate the `y_ring` / `y_rings` payload from `AkitaLevelProof`, `AkitaBatchedFoldRoot`, and `TerminalLevelProof`, and replace the external `recover_ring_subfield_inner_product` trace check with a fused stage-2 sum-check term that binds the committed fold witness to the public opening through the trace projection, with no extra committed data.

### Background: what `y_ring` does today

`y_ring` plays two roles per level (recursive case, `num_claims = 1`):

1. **Relation RHS.** It is the public right-hand side of the "public-output" rows of `M`, and it enters `V_alpha` only through its evaluation `y_ring(alpha)` (`crates/akita-types/src/proof/relation.rs:75-81`, the public-output loop in `relation_claim_from_rows_extension`; RHS layout built by `generate_y` at `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:595-630`).
2. **Trace bridge.** The verifier reads the cleartext `y_ring` off the wire and checks `Tr_H(y_ring · sigma_{-1}(packed_inner_point)) = (d/k) · opening`, where `opening` is the incoming extension-field claim (`crates/akita-verifier/src/protocol/levels/recursive.rs:313-357`, via `recover_ring_subfield_inner_product` / `check_trace_inner_product` in `crates/akita-types/src/field_reduction.rs:535-579,600-621`; paper `sections/akita/3_basic_akita.tex:382-390,488-498`).

The fold relation already proves `y_ring = sum_j b_j · e_folded_j` through the public-output row.
Here each `e_folded_j` is recomposed from the committed `e_hat` digit planes inside the next-level witness `w = (e_hat, t_hat, z_hat, r_hat)`.
The earlier shorthand `b^T z` is therefore misleading for implementation: the trace term must target the `e_hat` segment, not the `z_hat` segment.
So `y_ring` is recomputable from committed data; it is sent only because the verifier needs a handle to run the trace check and to form `V_alpha`.

### Why this is not "just add a row to `M`"

A row of `M` acts on the witness by negacyclic ring multiplication, and the ring-switch only ever exposes the committed witness as its evaluation at the random ring-switch challenge `alpha` (the `alpha~(y)` weighting in the stage-2 oracle `w(r) · alpha(r_y) · row(r_x)`, `crates/akita-verifier/src/stages/stage2.rs:444-461`).
The trace `Tr_H(Y) = sum_{sigma in H} sigma(Y)` is a fixed `Z_q`-linear **projection** `R_q -> R_q^H ≅ F_{q^k}` that collapses `D` ring coordinates to `k` subfield coordinates.
It is not negacyclic multiplication by any fixed ring element, and it is not recoverable from `y_ring(alpha)` alone (it would need `y_ring(alpha^a)` for every conjugate `a in H`).
Therefore the trace check cannot be expressed as one extra `M`-row evaluated at `alpha`; it must be a fused inner-product term against a fixed, `alpha`-independent public weighting derived from the opening point.

### Design: the fused trace term (superseded encoding)

> **Historical.** This section describes the `gamma^2 · W · TraceWeight` fused
> addend. The implemented cutover uses an `EvaluationTrace` row of
> `RelationWeightPolynomial` instead; see
> [`relation-weight-polynomial.md`](relation-weight-polynomial.md).

Replace roles (1) and (2) with a single fused stage-2 term, batched as the `γ²` addend of the existing stage-2 sum-check challenge `γ` (`CHALLENGE_SUMCHECK_BATCH`), that enforces

```text
TraceOpen( sum_j b_j · e_folded_j ) = opening
```

directly over the committed witness.
Here `opening` is the public incoming claim (already transcript-bound), and `packed_inner_point` is the public ring element that packs the inner opening-point weights.
In code this is `prepared_point.packed_inner_point`: `prepare_recursive_opening_point_ext` or `prepare_root_opening_point_ext` splits the verifier's opening point into outer block weights and inner ring-slot weights, then embeds the inner weights into a `CyclotomicRing`.
This `packed_inner_point` is the paper's `řᵢₙ` (`\check{r}_{in}`): the ψ-packed *inner block* of the opening point `r`, used as the public weight in the Hachi trace identity.
It is **not** the opening commitment `\mathbf{v} = D · \hat{\mathbf{e}}` nor the eval claim scalar `v`; the three roles are deliberately separate (see Notation).
With that convention,

```text
TraceOpen(Y) := recover_ring_subfield_inner_product(Y, packed_inner_point).
```

Equivalently, in the paper's raw ring identity,

```text
Tr_H( Y · sigma_{-1}(packed_inner_point) ) = (D/K) · embed_subfield(opening).
```

The implementation should use the normalized `TraceOpen` convention, so the stage-2 input contribution is `trace_coeff · opening` with `trace_coeff = γ²`.
If an implementation instead uses raw trace coordinates, it must multiply both the public input and the trace-weight table by the same `(D/K)` scale.

Concretely, the stage-2 fused oracle gains a third addend over the same Boolean domain as the current stage-2 relation.
Let `y ∈ {0,1}^{ring_bits}` index the ring coefficient and `x ∈ {0,1}^{col_bits}` index the witness column.
The per-corner oracle is:

```text
gamma     · eq(stage1_point, (y, x)) · W(y, x) · (W(y, x) + 1)      [range, unchanged]
+           W(y, x) · alpha_eval(y) · M_without_public(x)            [relation, public-output row dropped]
+ gamma^2 · W(y, x) · TraceWeight(y, x)                            [new: trace projection]
```

with the matching input-claim contribution `trace_coeff · opening` (`trace_coeff = γ²`) under the normalized convention above.
No new Fiat-Shamir challenge is introduced: `γ` is sampled after the next-level witness is bound, so the trace sub-claim is correctly randomized against the committed witness.
`TraceWeight(y, x)` is a fully public multilinear table.
It is nonzero only on the `e_hat` segment of the committed witness.
On an `e_hat` column for block `j` and open-digit plane `h`, and on ring coordinate `c`, it equals

```text
g_open[h] · TraceOpen( b_j · X^c )
```

where `g_open[h]` is the open-digit gadget scalar, `b_j` is the public ring multiplier for block `j`, and `X^c` is the ring basis element represented by witness coordinate `c`.
For a root public row, `b_j` already includes the same per-claim row coefficient used when batching claims at one opening point; equivalently the trace weights for all claims in the public row are summed under those row coefficients.

The sum-check instance is therefore:

```text
input_claim =
  gamma · s_claim
+ relation_claim_without_public_rows
+ trace_coeff · opening    (trace_coeff = gamma^2)

input_claim =
sum_{(y,x) in {0,1}^{ring_bits} × {0,1}^{col_bits}} [
  gamma · eq(stage1_point, (y,x)) · W(y,x) · (W(y,x) + 1)
+ W(y,x) · alpha_eval(y) · M_without_public(x)
+ gamma^2 · W(y,x) · TraceWeight(y,x)
].
```

During the sum-check, the verifier samples one challenge per variable.
Only after all rounds are complete does it have the final random point `r = (r_y, r_x)`.
At that final point, it checks the oracle value

```text
gamma · eq(stage1_point, (r_y,r_x)) · W(r_y,r_x) · (W(r_y,r_x) + 1)
+ W(r_y,r_x) · alpha_eval(r_y) · M_without_public(r_x)
+ gamma^2 · W(r_y,r_x) · TraceWeight(r_y,r_x).
```

For `K = 1` (the `fp128_d128` production optimum) the `y`-weighting is exactly `packed_inner_point` itself: `TraceOpen(X^c) = packed_inner_point[c]` (no reversal), so the table inherits the tensor structure of the opening point and the verifier's final-point evaluation is a pure product of eq / gadget tensors with no ring arithmetic.
For `K > 1`, `trace_weight` routes through the `|H|` Galois conjugates of `packed_inner_point`, but the verifier collapses them into a single `Tr_H` of one ring product, so its work stays `O(|H| · D)` and proof size is unchanged.
The exact closed forms the verifier evaluates are derived in *Verifier final-point evaluation* below.

Because the new term is (public table) × (witness MLE), it is degree ≤ the existing relation term and does not add stage-2 proof bytes.
The mandatory saving is the removed `y_ring`; a full public-row removal can additionally shrink `r_hat` and may reduce padded witness width when it crosses a power-of-two boundary.

### Verifier final-point evaluation

The verifier never materializes the `TraceWeight` table; it only evaluates it once, at the final sum-check point `r = (r_y, r_x)`, where `r_y ∈ E^{ring_bits}` selects the ring coordinate `c` and `r_x ∈ E^{col_bits}` selects the witness column.
Note `ring_bits = alpha_bits = log2(D)`, so `r_y` and the inner opening coordinates live over the same variables.

The enabling fact is that `TraceOpen` is `E`-linear in its ring argument.
With the eq-weight ring element

```text
EQ(r_y) := sum_c eq(r_y, c) · X^c
```

(tensor-structured, since `eq(r_y, c) = prod_t (c_t · r_y[t] + (1 - c_t)(1 - r_y[t]))`), the per-`c` sum folds into a single ring argument:

```text
sum_c eq(r_y, c) · TraceOpen(b_j · X^c) = TraceOpen(b_j · EQ(r_y)).
```

The `e_hat` columns are emitted plane-major as `col = offset_e + h · num_blocks + j` (`crates/akita-prover/src/protocol/ring_switch/coeffs.rs:146-165`, with `num_blocks = 2^{r_vars}`), so the column eq factors as `eq(r_x, x) = eq_seg(r_x) · eq_block(r_x_blk, j) · eq_plane(r_x_pl, h)`.
This yields the master factorization:

```text
TraceWeight(r_y, r_x)
  = eq_seg(r_x)                                                   [e_hat segment selector]
    · ( sum_h eq_plane(r_x_pl, h) · base^h )                      [gadget factor  G]
    · ( sum_j eq_block(r_x_blk, j) · TraceOpen(b_j · EQ(r_y)) ) [block / inner factor  B]
```

`G` is the MLE of the gadget vector `[1, base, base^2, …]` (`base = 2^{log_basis}`, `crates/akita-types/src/layout/digit_math.rs:17-26`).
When `num_digits_open` is a power of two it tensor-factors as `prod_t ((1 - r_x_pl[t]) + r_x_pl[t] · base^{2^t})` (`O(log num_digits_open)`); otherwise it is a short explicit sum over `h < num_digits_open`.

**K = 1 (the production target).**
Here the block weights `b_j` are scalars (`b_j = eq(b_open, j)`, the Lagrange block weights threaded through `evaluate_and_fold`, `crates/akita-prover/src/backend/recursive_witness.rs:179-194`), so they pull straight through `TraceOpen`.
The inner trace collapses to a plain coefficient dot product `TraceOpen(Y) = <coeffs(Y), packed_inner_point>` (`crates/akita-types/src/field_reduction.rs:608-615`; the `(D/K)` scale cancels at `K = 1`).
Since `packed_inner_point[c] = eq(inner_open, c)` in the Lagrange basis, `TraceOpen(EQ(r_y)) = <eq(r_y, ·), packed_inner_point> = eq(r_y, inner_open)`.
The entire weight is then a product of per-variable tensor factors, with no ring multiplication:

```text
TraceWeight(r_y, r_x)
  = eq_seg(r_x)                                              [segment selector]
    · eq(r_x_blk, b_open)                                    [block weights]
    · prod_t ((1 - r_x_pl[t]) + r_x_pl[t] · base^{2^t})      [gadget powers]
    · eq(r_y, inner_open)                                    [inner / packed_inner_point]
```

with `inner_open = opening_point[..alpha_bits]` and `b_open = opening_point[alpha_bits .. alpha_bits + r_vars]` (`crates/akita-types/src/proof/batch.rs:660-683`).
Cost is `O(num_vars)` field operations, the same order as the eq-evals the verifier already does for the relation and range terms, and strictly cheaper than today's per-level `recover_ring_subfield_inner_product` ring contraction.

**K > 1.**
The `b_j` are ring multipliers, so scalars no longer pull through `TraceOpen`; instead fold the blocks *before* the trace.
Because `Tr_H` is `E`-linear, with the block-weight ring element `B_blk(r_x_blk) := sum_j eq_block(r_x_blk, j) · b_j` the block / inner factor `B` is a single trace of one triple ring product:

```text
B = (K / D) · Tr_H( B_blk(r_x_blk) · EQ(r_y) · sigma_{-1}(packed_inner_point) ).
```

The conjugate sum is taken once per level, not once per block, so verifier cost is `O(|H| · D)` independent of `num_blocks · num_digits_open`.
The term remains a single `E`-valued sum-check addend whose matching input contribution is `trace_coeff · opening` (`trace_coeff = γ²`) under the normalized convention.

**Layout preconditions and witness column order.**
The `y`-axis factor `eq(r_y, inner_open)` is always clean.
On the `x`-axis the witness layout places `ẑ` at `offset_z = 0` and `ŵ` at `offset_e = z_len` (`crates/akita-types/src/proof/ring_relation.rs`, `segment_layout`).
The low `r_vars` block window may carry: `block_offset_low = z_len mod num_blocks`.
When `m_vars >= r_vars` that carry is zero and `eq(r_x_blk, b_open)` is exact on the `ŵ` block axis; when `m_vars < r_vars` the verifier uses the same carry-bucket peel as the existing row-MLE evaluators (the Matrix evaluation chapter, `book/src/how/verifying/matrix_evaluation.md`).
The high index of the `ŵ` segment carries `O = offset_e / num_blocks`, which need not be a multiple of `num_digits_open`; that factor uses a single `eval_offset_eq_tensor` (carry) call instead of a product, still `O(col_bits)`.
This is the same offset / carry treatment already applied to the `e_hat` (`ŵ`) segment, so the trace term adds no new column-alignment constraint.
The only obligation carried into step 2 of Execution is the `K > 1` weighting derivation.

### Invariants

- **Proof-size strictly smaller.** Per intermediate level, the proof shrinks by at least `D · base_elem_bytes` (one ring element; `proof_ring_vec_bytes(num_claims=1, D, base_elem_bytes)`, `crates/akita-types/src/proof_size.rs:83`); the root shrinks by at least `P · D · base_elem_bytes`; the terminal by at least `D · base_elem_bytes`. The stage-1 tree and stage-2 degree bound are unchanged. If public-output rows are removed from `M` rather than kept as inert padding, the `r_hat` quotient tail also shrinks because `r` has one fewer row per removed public output; the planner DP and witness-shape checks must treat this as an intentional layout cutover, not as byte-for-byte witness stability.
- **No new committed data.** `w = (e_hat, t_hat, z_hat, r_hat)` gains no new semantic segment and no committed `y_ring`. In the full row-removal variant, `w` may become smaller because the removed public-output rows no longer contribute quotient digits to `r_hat`.
- **Soundness preserved.** The new term must bind the committed fold witness to the public `opening` at least as tightly as today's `sum_j b_j · e_folded_j = y_ring` plus external trace check. The extraction in the `batched-root-cwss` theorem must be re-derived for the dropped public-output row and the added fused term; soundness loss must remain bounded by the existing sum-check / field-size terms. This is the gating obligation (see Execution).
- **Prover/verifier transcript consistency.** Removing the `ABSORB_EVALUATION_CLAIMS` absorb of `y_ring` (`recursive.rs:313-315`) and deriving `trace_coeff = γ²` from `CHALLENGE_SUMCHECK_BATCH` sampled after witness binding must be mirrored on both sides; `logging-transcript` event-stream equality and wire-before-squeeze checks must stay green (`crates/akita-pcs/tests/transcript_hardening*.rs`).
- **K-path parity.** `K = 1` keeps a fully tensor-factored weighting (`O(num_vars)` final-point eval, no ring arithmetic) and is the first implementation target. `K in {2,4,8}` route through the conjugate-sum weighting (one `Tr_H` of a single ring product, `O(|H| · D)`) and also require moving the extension-opening-reduction final binding away from on-wire `y_ring`. The existing trace identity tests in `crates/akita-types/src/field_reduction.rs` (and the `K`-generic dispatcher) remain the algebraic anchor for the weighting derivation.
- **ZK.** The `y_ring` hiding masks (`zk_base_mask_lcs(y_rings.len() * D, …)`, `recursive.rs:316-317`) and the `relation_claim_mask` `y`-contribution are removed; the fused trace term gets its own deferred ZK relation analogous to the stage-2 final relation (`stage2.rs:464-533`). The hiding-witness cursor accounting must still close (`zk_hiding_cursor == hiding_witness.len()`).
- **End-to-end roundtrip.** All batched / recursive / terminal / zero-fold e2e tests pass for every active profile (`fp128_d128`, the `fp32`/`fp64` extension profiles, dense + onehot, ZK and non-ZK).

### Soundness derivation gate (superseded batching model)

> **Historical.** The `gamma^2` degree-2 batching argument below applied to the
> three-summand oracle. The `EvaluationTrace` row model folds the trace
> obligation into `relation_weight_claim`; soundness is the ordinary row-batched
> relation extraction plus stage-1 range binding.

The old protocol proves two facts about the same folded opening:

```text
public-output M row:  Y = sum_j b_j · e_folded_j
external trace check: TraceOpen(Y) = target
```

where `target` is the incoming opening in the non-EOR path. The new protocol
does not expose `Y`; instead it proves the single linear statement

```text
sum_{(y,x)} W(y,x) · TraceWeight_target(y,x) = target
```

inside stage 2, batched as the `γ²` addend of the existing stage-2 batching
challenge `γ` (no new Fiat-Shamir label).

The stage-2 sum-check input becomes

```text
gamma · s_claim + relation_claim_without_public_rows + trace_coeff · target
```

with `trace_coeff = γ²`, and the corner oracle becomes

```text
gamma · eq · W · (W+1) + W · alpha_eval · M_without_public + gamma^2 · W · TraceWeight_target
```

Let `Delta_rel`, `Delta_range`, and `Delta_tr` be the prover's errors in the
relation, range, and trace terms respectively. A prover that passes the
stage-2 input check with an incorrect trace target satisfies

```text
Delta_rel + gamma · Delta_range + gamma^2 · Delta_tr = 0.
```

For any fixed transcript prefix and committed witness with `Delta_tr != 0`,
this is a degree-2 polynomial in `γ` with at most two roots. Thus the fused
check adds at most a `2 / |challenge field|` batching failure term on top of
the existing sum-check low-degree soundness error. If `Delta_tr = 0`,
the committed `e_hat` segment already projects to the target; the remaining
condition is the ordinary relation soundness for `M_without_public`.

The extractor therefore obtains the same witness as before, except that the
opening relation is extracted from the committed `e_hat` digits directly:

```text
TraceOpen(sum_j b_j · e_folded_j) = target.
```

Because stage 1 still range-checks the committed digit witness and stage 2
still binds the final oracle value of the same witness table, dropping the
public-output row does not introduce a new witness degree of freedom. The
removed `Y` was an auxiliary linear image of `e_hat`; the fused term queries
that image through a public multilinear functional instead of receiving it as
wire data.

For EOR paths the target is not the raw protocol-point opening. The EOR
sum-check final oracle already includes the transparent tail equality factor:

```text
root:      final_claim =
             sum_claim row_coeff[claim]
               · witness_claim(rho)
               · factor_by_point[claim_to_point[claim]]

recursive: final_claim = final_witness(rho) · final_factor.
```

The no-y-ring fused trace term must therefore bind `final_claim` directly. It
does this by using the protocol point derived from `rho` for the packed inner
opening and scaling the public trace weights by the same EOR tail factor(s):

```text
root trace weights:      row_coeff[claim] · factor_by_point[point] · b_claim,j
recursive trace weights: final_factor · b_j
trace input claim:       trace_coeff · final_claim   (trace_coeff = γ²)
```

Equivalently, one could divide by a nonzero factor and bind the unscaled
protocol-point opening, but the scaled-weight formulation avoids an inversion
and also handles the root multipoint sum uniformly. This is the acceptance gate
for the non-ZK EOR implementation: the verifier must no longer reconstruct an
EOR output from `y_ring`; it must use the EOR final claim as the fused trace
target.

### Non-Goals

- Changing the ring-switch challenge `alpha`, the `tau0`/`tau1` row batching, the digit-decomposition bases, or the stage-1 range check.
- Changing the extension-opening reduction partials or its degree-two sum-check. However, `K > 1` implementation must still redesign the EOR final binding, because the verifier currently recovers the EOR output from on-wire `y_ring`.
- The zero-fold (`AkitaBatchedRootProof::ZeroFold`) fast path, which sends no `y_ring`.
- Committing `y_ring` explicitly (the "commit the ring element" framing). That variant is recorded under Alternatives Considered and is deliberately not the chosen design.
- Witness column ordering (`ẑ ‖ ê ‖ t̂ ‖ …`). This work neither depends on nor changes that layout. Removing `y_ring` only drops a materialized *row* family (no column-alignment constraint), and the fused trace term reuses the same offset/carry treatment on the `e_hat` column segment.
- Any change to setup, SIS sizing, or the security floor.

## Evaluation

### Acceptance Criteria

- [ ] `AkitaLevelProof`, `AkitaBatchedFoldRoot`, and `TerminalLevelProof` no longer carry a `y_ring` / `y_rings` field; all constructors, shapes (`level_proof_shape`, `TerminalLevelProofShape`), serialization, and `can_decode_vec` shape guards are updated.
- [ ] `relation_claim_from_rows_extension` (and `relation_claim_from_rows`) no longer take `y_rings`; the public-output rows are removed from the `M` RHS layout in `generate_y` and the verifier `RingRelationInstance` construction.
- [ ] The verifier enforces `TraceOpen(sum_j b_j · e_folded_j) = opening` in non-EOR paths, and the scaled EOR final-claim variant above in EOR paths, via a fused stage-2 term batched as `trace_coeff = γ²`; it no longer calls `recover_ring_subfield_inner_product` / the standalone `internal_claims[0] == opening` check on on-wire `y_ring` (`recursive.rs:319-357`).
- [ ] `level_proof_bytes` drops the `y_bytes` term; `crates/akita-types/src/proof_size.rs` tests and the planner DP scoring are updated; shipped schedule tables regenerated with `regen_diff` reflecting the new (smaller) sizing.
- [ ] Non-ZK and ZK e2e suites are green: `cargo nextest run --profile ci-non-zk` and `--profile ci-all-features`.
- [ ] `cargo test -p akita-pcs --features logging-transcript --test transcript_hardening` green (event-stream equality after the `y_ring` absorb removal and `trace_coeff = γ²` derivation from post-witness `CHALLENGE_SUMCHECK_BATCH`).
- [ ] A negative test: tampering the committed `e_hat` digits so that `sum_j b_j · e_folded_j` projects to the wrong subfield value is rejected (replaces the role of the current `y_ring` trace-mismatch rejection paths, e.g. `crates/akita-pcs/src/scheme/tests/batched.rs:419-421`).
- [ ] Profile shows the expected per-level shrink: `AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile` reports `y_ring_bytes = 0` at every level and total proof size reduced by the predicted amount.

### Testing Strategy

Must continue passing: the full batched/recursive/terminal/zero-fold e2e set (`crates/akita-pcs/tests/*`), `akita-types` `field_reduction` trace tests, `relation.rs` claim tests, `proof_size.rs` formula tests, `regen_diff`, and the transcript-hardening + proptest suites.

New tests:

- Algebraic anchor: a unit test that the fused `trace_weight` MLE, contracted against a witness whose `e_hat` recomposes to chosen `e_folded` blocks, equals `TraceOpen(sum_j b_j · e_folded_j)` for `K in {1,2,4}` (mirrors and reuses the existing trace-identity harness in `field_reduction.rs`).
- Soundness smoke: a tampered-witness negative test at recursive, root, and terminal levels.
- Size-delta fixture: serialize a proof before/after on a fixed fixture and assert the removed `y_ring` blocks account for the mandatory delta; if the implementation also removes public-output rows from `M`, include the expected `r_hat` / row-count shrink in the formula.

### Performance

Proof size: strictly smaller, by at least one base-field ring element per level (+ `P` at root). For `fp128_d128` the mandatory `y_ring` saving is `D · 16 = 2048` bytes per level; for `fp64_d64` it is `512` bytes. If public-output rows are removed from `M`, each removed row also deletes its quotient digits from `r_hat`, subject to the next-power-of-two witness padding used by stage 2. The exact total per profile is read from the profile command above and from the updated planner DP.
Prover/verifier time: negligible change, and for `K = 1` strictly favorable. The verifier replaces one `recover_ring_subfield_inner_product` (`O(|H| · D)`) per level with one `trace_weight(r)` final-point evaluation plus one extra fused-oracle addend per sum-check round; no new rounds. For `K = 1` that final-point evaluation is a pure product of eq / gadget tensors (`O(num_vars)`, no ring arithmetic); for `K > 1` it is one `Tr_H` of a single ring product (`O(|H| · D)`). The closed forms are in *Verifier final-point evaluation*. The prover adds one public weighting table and one term in the per-round stage-2 evaluation, both `O(witness)`.
Planner: the proof-size optimum may shift slightly (every level is cheaper); re-run the schedule generation and confirm via `regen_diff` and the profile.

## Design

### Architecture

Affected surfaces:

- `akita-types`: proof structs and shapes (`src/proof/levels.rs`, `src/proof/shapes.rs`, `src/proof/relation.rs`), the RHS layout helpers, `src/proof_size.rs` and `src/layout/proof_size.rs`. New `src/trace_weight/` (`layout`, `build`, `eval`, `stage2`, `trace_table`) owns the public `TraceWeight` table, prover `TraceTable`, and verifier closed-form `eval_trace_terms_closed`. `batched_eval_target_from_incidence` lives in `src/proof/incidence.rs`. The trace primitives in `src/field_reduction.rs` remain the algebraic anchor for `Tr_H` / `TraceOpen`.
- `akita-prover`: `src/protocol/ring_relation.rs` + `ring_relation/relation_quotient.rs` (drop the public-output row and its quotient contribution), the stage-2 prover (`src/protocol/sumcheck/akita_stage2/`) to add the `γ²`-batched trace term and `trace_weight` table, and `src/protocol/ring_switch/finalize.rs` for the claim assembly.
- `akita-verifier`: `src/protocol/levels.rs` and `levels/recursive.rs` (drop the `y_ring` absorb + external trace check, derive `trace_coeff = γ²` after witness binding, feed the trace term), `src/stages/stage2.rs` (`expected_output_claim`, `input_claim`, and the ZK final relation gain the trace addend), `src/protocol/ring_switch.rs` (relation-claim assembly without the public-output row).
- `akita-planner` / `akita-config`: re-score and regenerate shipped schedule tables.

The fused trace term reuses the existing stage-2 Boolean domain and final witness oracle.
It is structurally a sibling of the relation term: over Boolean corners it replaces `alpha_eval(y) · M_without_public(x)` with the public table `TraceWeight(y,x)`, and at the final verifier point it evaluates that table at `(r_y,r_x)`.

### Alternatives Considered

**Commit `y_ring` and add two constraints (the original framing).**
Append `y_ring`'s balanced digits to `e_hat` so it is bound by `v = D · e_hat` and recursed inside `w`, make the public-output row homogeneous (`sum_j b_j · e_folded_j - y_ring = 0`), and add the trace constraint as a fused term over the committed `y_ring`.
This matches the "commit the ring element + two extra constraints" intuition and isolates `y_ring` cleanly.
Rejected as primary because it pays `δ = log_b q` extra committed digit planes per level (range-checked in stage 1 and recursed), partially eating the saving, and it has no `v`/D-block at the terminal level (`MRowLayout::WithoutDBlock`) so the terminal would need a special case.
The no-commit variant binds the same opening through `e_hat`, which is already committed and range-checked, with zero extra committed data.

**Keep `y_ring` but Frobenius-compress it.**
Sends fewer than `D` coordinates by exploiting subfield structure.
Rejected: it still sends a payload and still needs the external trace check; the fused-term approach sends nothing and is simpler downstream.

**Leave it alone.**
The saving is modest (single-digit to low-double-digit KB total), but the discussion (Jiapeng's suggestion; "every kb is worth it") and the recursive-friendliness (smaller per-level payload compounds across levels and matters for the in-Jolt verifier) justify it, provided the soundness re-derivation is clean.

## Execution

Recommended order, soundness first:

1. **Soundness derivation (gating).** Write the special-soundness/extraction argument for: dropping the public-output `M` row, and adding the fused term `γ² · w(r) · trace_weight_v(r)` with input contribution `trace_coeff · opening` (`trace_coeff = γ²`) under the normalized non-EOR trace convention, or `trace_coeff · final_claim` with EOR-scaled trace weights. Confirm the extractor still recovers a witness whose folded opening projects to the selected target, and bound the added soundness error (`2/|E|` from the degree-2 batching polynomial). Do this before code.
2. **Weighting derivation + unit anchor.** Derive `trace_weight_v(x, y)` for `K = 1` against the `e_hat` segment and current `build_w_coeffs` ordering, then `K in {2,4,8}` via `SubfieldParams::h_exponents`. The closed-form final-point evaluations are already worked out in *Verifier final-point evaluation*; the unit test should check the round table's final-point contraction against them. Land the algebraic unit test against `field_reduction` before wiring the sum-check.
3. **Verifier non-ZK.** Derive `trace_coeff = γ²`, add the trace addend in `AkitaStage2Verifier::{input_claim, expected_output_claim}`, and remove the `y_ring` absorb + external check across recursive/root/terminal.
4. **Prover non-ZK.** Build `trace_weight`, add the term to stage-2 per-round evals, drop `y_rings` from the RHS / proof structs.
5. **Proof structs, shapes, serialization, sizing.** Remove the fields; update `level_proof_bytes`, planner DP, regenerate schedules.
6. **ZK.** Remove the `y_ring` masks; add the deferred trace relation; close the cursor accounting.
7. **Tests + profile.** Negative tests, byte-delta test, transcript-hardening, and the profile shrink check.

Risks to resolve first: the exact `M`-row bookkeeping when the public-output row is removed (does the consistency row or commitment binding implicitly depend on it?), whether the intentional `r_hat` shrink should be accepted in the first PR or temporarily avoided with inert padding, and the `e_hat` segment alignment under the witness layout (`offset_e = z_len` folds into `eq_seg`; see *Verifier final-point evaluation*). The `K > 1` per-round eval cost is resolved: the conjugate sum collapses into one `Tr_H` of a single ring product, `O(|H| · D)` per level, independent of `num_blocks · num_digits_open`. These are flagged for step 1–2 before committing to the wiring.

## Notation

Authoritative symbol map for this cutover and the fused trace term.
Older specs may still say `w_hat` or use bare `v` in trace formulas; this
section supersedes those fragments for trace-internalization work.
Full rationale and cross-scheme survey:
`~/Documents/Notes/akita-v-notation-and-zfirst-rationale.md`.

### Locked symbols (paper ↔ Rust)

| Object | Paper | Rust | Never confuse with |
|--------|-------|------|-------------------|
| Opening/evaluation point | `\mathbf{r}` | `opening_point`, `prepared_point` | `packed_inner_point` |
| Eval/opening claim (scalar) | `v`, `\bar{v}`, `v'` | `opening`, `openings`, `input_claim` | `.v` (commitment vector) |
| Opening commitment | `\mathbf{v} = D\hat{\mathbf{e}}` | `.v`, `RingRelationInstance::v`, `ABSORB_PROVER_V` | scalar `opening` |
| Packed inner trace weight | `\check{r}_{\mathrm{in}}` | `packed_inner_point` | commitment `.v` |
| Relation RHS vector | `y` in `Mz = y + \cdots` | `RingRelationInstance::y`, `generate_y` | `y_ring` / `Y`, stage-2 `y` axis |
| Removed folded ring output | `Y`, `y_{\mathrm{ring}}` | *(dropped on wire)* | commitment `.v`, scalar `opening` |
| Sum-check Boolean ring index | `y \in \{0,1\}^{\mathrm{ring\_bits}}` | `ring_bits`, `alpha_evals_y` | eval claim `opening` |
| Sum-check Boolean column index | `x \in \{0,1\}^{\mathrm{col\_bits}}` | `col_bits`, `x_challenges` | |
| Sum-check random point | `(r_y, r_x)` | stage-2 challenges | opening point `\mathbf{r}` |
| Ajtai witness commitment | `\mathbf{u}'` | `commitment`, `next_w_commitment` | opening commitment `.v` |

**Path A (locked).** Greyhound/Hachi write `f(\mathbf{r}) = y` because they are
univariate ring-eval schemes with no fused 2D stage-2 hypercube.
Akita reserves `y` for the stage-2 ring axis (see fused-term design above), so
the public eval claim is scalar `v \in E` in paper notation (Thaler/Spartan:
point `\mathbf{r}`, value `v`).
Rust keeps **`opening`** for that scalar and **`v` / `.v`** only for the
commitment vector `\mathbf{v} = D\hat{\mathbf{e}}`; do not rename eval claims to
`v` in code.

In trace formulas, write **`packed_inner_point`** (paper `\check{r}_{\mathrm{in}}`)
for the public ψ-packed inner opening block.
Do not use bare `v` for that role; it collides with both paper eval-claim `v`
and commitment `\mathbf{v}`.

### Trace-term Rust names (this PR)

| Role | Rust |
|------|------|
| `trace_weight` module | `akita-types/src/trace_weight/{layout,build,eval,stage2,trace_table}.rs` |
| Prover trace table (`K=1` sparse, `K>1` dense) | `TraceTable`, `build_trace_table_scaled` |
| Public block weights + verifier closed terms | `TracePublicWeights`, `TraceTerm`, `TraceClaim` |
| Scalar bound by the fused trace term | `trace_eval_target` (equals `opening` on ordinary paths, `final_claim` on EOR) |
| Fused trace coefficient | `stage2_trace_coeff(batching_coeff, trace_gamma, is_terminal)` |
| `trace_coeff` (`γ²`) times the trace target | `trace_opening_claim = trace_coeff * trace_eval_target` |
| Verifier final-point trace MLE | `eval_trace_terms_closed` |
| Root / recursive trace claim assembly | `build_trace_claim_root`, `build_trace_claim_recursive` |
| Batched root eval target from incidence | `batched_eval_target_from_incidence` in `proof/incidence.rs` |
| Per-claim root incidence iteration | shared driver for `trace_public_weights_root_terms` and `trace_terms_root` |

## References

- Paper: `sections/akita/3_basic_akita.tex:382-390` (send `Y`, verifier trace check), `:488-498` (per-claim opening summary), `:757-828` (stage-2 fused sum-check).
- Proof structs: `crates/akita-types/src/proof/levels.rs:139-152,375-391,451-467`.
- Relation claim / RHS: `crates/akita-types/src/proof/relation.rs:59-97`; `crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs:595-630`.
- Trace primitives: `crates/akita-types/src/field_reduction.rs:185-194,535-579,600-621`.
- Opening-point split / `packed_inner_point`: `crates/akita-types/src/proof/batch.rs:624-734` (`reduce_inner_opening_to_ring_element`: `crates/akita-types/src/layout/opening_point.rs:185-197`).
- Notation: see the Notation section above and `~/Documents/Notes/akita-v-notation-and-zfirst-rationale.md`.
- Fold producing `y_ring = sum_j b_j · e_folded_j`, `e_folded_j = <a, block_j>`: `crates/akita-prover/src/backend/recursive_witness.rs:179-211`.
- Plane-major `e_hat` column layout (`col = offset_e + h · num_blocks + j`): `crates/akita-prover/src/protocol/ring_switch/coeffs.rs:146-165`; `segment_layout`: `crates/akita-types/src/proof/ring_relation.rs`.
- Gadget powers `g_open[h] = base^h`: `crates/akita-types/src/layout/digit_math.rs:17-26`.
- Verifier level + fused trace claim: `crates/akita-verifier/src/protocol/levels.rs`, `levels/recursive.rs`; stage-2 oracle: `crates/akita-verifier/src/stages/stage2.rs`.
- Proof sizing: `crates/akita-types/src/proof_size.rs:72-104`.
- Notation: `specs/archive/2026-Q2/w-to-e-notation.md`; trace cutover lineage: `specs/extension-field-trace-cutover.md`, `specs/extension-field-opening-batching.md`.
