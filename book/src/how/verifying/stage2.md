# The Stage 2 fused check

Stage 2 proves three facts about the same flat digit witness. Keeping them in
one sumcheck binds them to one polynomial and one final evaluation.

The three facts are:

1. The witness agrees with the Stage 1 range result.
2. The witness satisfies the relation matrix.
3. The opening digits encode the claimed polynomial evaluations under the
   scheduled opening method.

On a fold that takes the L2 route, Stage 2 adds a fourth fact: the committed
digits match the physical norm that Stage 1 proved.

The verifier replays a degree three sumcheck over the complete relation witness
domain.

## Input claim

This chapter writes the Stage 1 point, the range image evaluation, and the
batching challenge as the
[sumcheck stages](../proving/sumcheck-stages.md) chapter does. Let $r_1$ be the
final Stage 1 point and $s_1$ the range image evaluation that Stage 1 returns
at $r_1$. Let $\gamma$ be the batching challenge that the transcript samples
after it absorbs $s_1$. Let $v_M$ be the public relation claim after row and
ring evaluation. That chapter writes it $h_\tau$. Let $v_{open}$ be the
weighted opening claim, which is the claimed opening value multiplied by the
weight of its opening row. That chapter writes it
$\beta_{\mathrm{open}}v_{\mathrm{open}}$. On a fold that takes the L2 route,
the verifier also derives a physical norm claim $v_{L2}$ from the virtual
evaluations that Stage 1 returns and a batching challenge $\eta$. On every
other fold $v_{L2}=0$. The Stage 2 input claim is

$$
\gamma s_1+v_M+v_{open}+v_{L2}.
$$

The transcript fixes the witness, $r_1$, $\gamma$, the row batching
coefficients, and the opening claims before this sumcheck is replayed.

## Final point equation

Let $r_2$ be the final Stage 2 point and let $w(r_2)$ be the witness
evaluation carried by the proof. The verifier checks that the last sumcheck
claim $c_{out}$ equals the Stage 2 polynomial at $r_2$:

$$
c_{out}=
\gamma\,eq(r_1,r_2)\,w(r_2)(w(r_2)+1)
+w(r_2)\widetilde M_{native}(\tau_1,r_2)
+C_{compression}(r_2)
+w(r_2)C_{method}(r_2)
+w(r_2)W_{L2}(r_1,r_2).
$$

The terms have separate owners:

- `EqPolynomial` evaluates the range image equality factor.
- `RelationMatrixEvaluator` evaluates the ordinary relation matrix.
- `CompressionRelationWeights` and `NegativeBinarySupport` evaluate the F and
  H compression relations and their digit constraints.
- `PreparedEvaluationTrace` evaluates an evaluation trace opening.
- `CoefficientPackingCompactFactors` evaluates the packed E and Q relation
  weights and the packing Z and direct-opening structured terms.
- `eval_boolean_pair_tensor_families` evaluates the physical norm weight
  $W_{L2}$. The term is present only on the L2 route.

If $\gamma$ is zero, the verifier skips the range equality evaluation. This is
an arithmetic shortcut only. The transcript and proof shape do not change.

## Range image term

Stage 1 checks the range product over the same digit witness and returns the
claim $s_1$ at $r_1$. Stage 2 uses the equality polynomial $eq(r_1,r_2)$ to
connect that claim to the witness value at $r_2$. The factor
$w(r_2)(w(r_2)+1)$ is the negative binary constraint in the field convention
used by the protocol.

Compressed layouts have additional negative binary digit spans in the F and H
suffix. `NegativeBinarySupport` evaluates the equality polynomial only over
those checked intervals. It does not allocate a dense support bitmap. Inside
$C_{compression}(r_2)$ this constraint is
$\rho\,B_{comp}(r_1,r_2)\,w(r_2)(w(r_2)+1)$. Here $B_{comp}$ is the restricted
equality weight and $\rho$ is a separate batching challenge that the
transcript samples before $\gamma$. The input claim of this constraint is
zero.

## Physical norm term

A fold on the L2 route proves the squared norm of the complete physical folded
response in Stage 1. Stage 2 then ties that proof to the committed digits. The
weight $W_{L2}(r_1,r_2)$ is a public tensor formula. It depends on the
schedule, the batching challenge $\eta$, the Stage 1 point, and the Stage 2
point. The
[fused Stage-2 claim](../proving/sumcheck-stages.md#the-fused-stage-2-claim)
section describes how the term enters the claim.

The schedule selects the route. The proof does not. On a fold without this
route the verifier reads no norm values and sets $v_{L2}=0$.

## Relation term

The native relation term is

$$
w(r_2)\widetilde M_{native}(\tau_1,r_2).
$$

The previous chapter explains `M`. Direct setup mode reads the public setup
during this evaluation. Deferred setup mode substitutes the Stage 3 setup claim;
Stage 3 then prepares its setup-index weight from the same challenge point.

Compressed F and H rows are evaluated separately because their digit support
and native dimensions differ from the ordinary A, B, and D roles. Their result
is still part of the same Stage 2 polynomial.

## Method-dependent structured terms

The relation matrix checks algebraic consistency of the witness. It does not by
itself prove that the `E` digits encode the opening values stated by the caller.
The scheduled method supplies that binding. Evaluation trace uses

$$
\lambda_{trace}w(r_2)\widetilde T(r_2).
$$

`lambda_trace` is the row weight assigned to the virtual trace row. The trace
is a virtual row because the verifier evaluates its public tensor formula at
the final point instead of adding a physical matrix row and quotient digits.

Subring coefficient packing has two contributions: the packed E and Q relation
weights, and two ordered structured sources, packing-Z and direct opening.
The CPU prover expands both into explicit relation events and Stage 2 term
tables from the checked group semantics. The
verifier never builds those tables. It builds `CoefficientPackingCompactFactors`
from the same validated packing groups, a small set of tensor families over the
block point, tail point, extension basis coordinates, and opening digit
weights, and evaluates them directly at the final point. The packing-Z weights
do not share the native A row's low alpha factor, so they remain a separate
factorized term. Every contribution still evaluates the same flat witness at
the same final point.

## Why the terms share one point

All terms use the same $w(r_2)$. A prover cannot use one witness for the range
proof, another for the relation, and a third for the opening claims. The shared
point also lets every verifier term consume the same checked coefficient and
address split.

## Code map

- Stage 2 verifier: `crates/akita-verifier/src/stages/stage2.rs`.
- Stage 1 verifier: `crates/akita-verifier/src/stages/stage1.rs`.
- Relation preparation: `crates/akita-verifier/src/relation/mod.rs`, with the relation claim in
  `crates/akita-verifier/src/stages/relation_claim.rs`.
- Coefficient-packing compact factors:
  `crates/akita-verifier/src/coefficient_packing_relation/`.
- Opening semantics: `crates/akita-verifier/src/stages/opening_semantics.rs`.
- Evaluation trace: `crates/akita-verifier/src/relation/evaluation_trace.rs`.
- Prover stage: `crates/akita-cpu-backend/src/opaque/sumcheck/relation_range_image/`.

The next chapter derives the evaluation trace path and explains its compact
contraction. The packing equations are in
[Root fold and ring switching](../proving/root-fold-ring-switch.md).
