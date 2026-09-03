# The Stage 2 fused check

Stage 2 proves three facts about the same flat digit witness. Keeping them in
one sumcheck binds them to one polynomial and one final evaluation.

The three facts are:

1. The witness agrees with the Stage 1 range result.
2. The witness satisfies the relation matrix.
3. The opening digits encode the claimed polynomial evaluations under the
   scheduled opening method.

The verifier replays a degree three sumcheck over the complete relation witness
domain.

## Input claim

Let `eta` be the batching challenge for the range image. Let `R(tau0)` be the
Stage 1 range image claim, `v_M` the public relation claim after row and ring
evaluation, and `v_open` the claimed opening value. The Stage 2 input is

```math
\eta R(\tau_0)+v_M+v_{open}.
```

The transcript fixes the witness, `alpha`, `tau0`, `tau1`, the row batching
coefficients, and the opening claims before this sumcheck is replayed.

## Final point equation

Let `r` be the final Stage 2 point and let `w(r)` be the witness evaluation
carried by the proof. The verifier checks

```math
\eta\,eq(\tau_0,r)w(r)(w(r)+1)
+w(r)\widetilde M_{native}(\tau_1,r)
+C_{compression}(r)
+w(r)C_{method}(r).
```

The terms have separate owners:

- `EqPolynomial` evaluates the range image equality factor.
- `RelationMatrixEvaluator` evaluates the ordinary relation matrix.
- `CompressionRelationWeights` and `NegativeBinarySupport` evaluate the F and
  H compression relations and their digit constraints.
- `PreparedEvaluationTrace` evaluates an evaluation trace opening.
- `CoefficientPackingRelationEvents` evaluates the packed E and Q relation
  events at the checked physical coefficient blocks.
- `CoefficientPackingStage2Terms` evaluates the packing Z and direct-opening
  structured linear terms.

If `eta` is zero, the verifier skips the range equality evaluation. This is an
arithmetic shortcut only. The transcript and proof shape do not change.

## Range image term

Stage 1 checks the range product over the same digit witness and returns a claim
at `tau0`. Stage 2 uses the equality polynomial `eq(tau0,r)` to connect that
claim to the witness value at `r`. The factor `w(r)(w(r)+1)` is the negative
binary constraint in the field convention used by the protocol.

Compressed layouts have additional negative binary digit spans in the F and H
suffix. `NegativeBinarySupport` evaluates the equality polynomial only over
those checked intervals. It does not allocate a dense support bitmap.

## Relation term

The native relation term is

```math
w(r)\widetilde M_{native}(\tau_1,r).
```

The previous chapter explains `M`. Direct setup mode reads the public setup
during this evaluation. Deferred setup mode substitutes the Stage 3 setup claim
and caches the exact `SetupContributionPlan` for Stage 3.

Compressed F and H rows are evaluated separately because their digit support
and native dimensions differ from the ordinary A, B, and D roles. Their result
is still part of the same Stage 2 polynomial.

## Method-dependent structured terms

The relation matrix checks algebraic consistency of the witness. It does not by
itself prove that the `E` digits encode the opening values stated by the caller.
The scheduled method supplies that binding. Evaluation trace uses

```math
\lambda_{trace}w(r)\widetilde T(r).
```

`lambda_trace` is the row weight assigned to the virtual trace row. The trace
is a virtual row because the verifier evaluates its public tensor formula at
the final point instead of adding a physical matrix row and quotient digits.

Subring coefficient packing splits its contribution across the common relation
path and the structured path. `CoefficientPackingRelationEvents` supplies the
packed E and Q weights to the common relation-weight factorization.
`CoefficientPackingStage2Terms` supplies the two ordered structured sources:
packing-Z and direct opening. Their weights come from the block point, tail
point, extension basis coordinates, and opening digit weights. The packing-Z
weights do not share the native A row's low alpha factor, so they remain a
separate factorized term. Every contribution still evaluates the same flat
witness at the same final point.

## Why the terms share one point

All terms use the same `w(r)`. A prover cannot use one witness for the range
proof, another for the relation, and a third for the opening claims. The shared
point also lets every verifier term consume the same checked coefficient and
address split.

## Code map

- Stage 2 verifier: `crates/akita-verifier/src/stages/stage2.rs`.
- Stage 1 verifier: `crates/akita-verifier/src/stages/stage1.rs`.
- Relation preparation: `crates/akita-verifier/src/protocol/ring_switch.rs`.
- Evaluation trace: `crates/akita-verifier/src/protocol/evaluation_trace.rs`.
- Prover stage: `crates/akita-prover/src/protocol/sumcheck/relation_range_image/`.

The next chapter derives the evaluation trace path and explains its compact
contraction. The packing equations are in
[Root fold and ring switching](../proving/root-fold-ring-switch.md).
