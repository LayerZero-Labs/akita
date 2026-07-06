# Sumcheck stages

The fused sumcheck that proves one fold: stage 1 (the digit range check on the
balanced witness), stage 2 (the fused relation sumcheck), and the optional
stage 3 (the setup product sumcheck used by verifier offloading).

## Stage 1

Digit range check on the balanced witness. Carries a virtual claim
`s_claim = w(stage1_point) * (w(stage1_point) + 1)` into stage 2.

## Stage 2

Stage 2 proves two summands over the next-witness Boolean hypercube:

```text
relation_weight_claim + gamma * s_claim
  = sum_x [
      w(x) * RelationWeightPolynomial(x)
    + gamma * eq(stage1_point, x) * w(x) * (w(x) + 1)
  ]
```

`RelationWeightPolynomial` is the single multilinear polynomial whose
evaluations are the field-level, `tau1`-batched relation weights. It includes
every ring-switched relation row: fold rows, setup rows, quotient rows, and the
`EvaluationTrace` row that binds the committed fold witness to the public
opening (replacing the old on-wire `y_ring` plus separate trace check).

The verifier evaluates `expected_output_claim(r)` as

```text
w(r) * RelationWeightPolynomial(r)
  + gamma * eq(stage1_point, r) * w(r) * (w(r) + 1)
```

Canonical spec: [`specs/relation-weight-polynomial.md`](../../../specs/relation-weight-polynomial.md).

## Stage 3

Optional setup product sumcheck for verifier offloading. See
[`specs/setup-product-sumcheck.md`](../../../specs/setup-product-sumcheck.md).

**Sources**

- `crates/akita-prover/src/protocol/sumcheck/akita_stage1/`, `akita_stage1_tree.rs`, `akita_stage2/`, `round_batching/`.
- `crates/akita-verifier/src/stages/`.
- Paper §3.5 (`fig:akita-sumcheck`), §3.5.1 `sec:akita-range-check`, §4.3 `sec:claim-reduction`.
- `specs/packed-sumcheck.md`, `specs/setup-product-sumcheck.md`.
