# Setup contribution and Stage 3

The relation matrix contains products with the public A, B, and D setup
matrices. The verifier can check those products directly during Stage 2 or
defer the public setup inner product to Stage 3. Both modes use one
`SetupContributionPlan` and one relation formula.

## One plan for three setup roles

The setup contribution contains:

```text
A setup columns paired with Z weights
B setup columns paired with T weights
D setup columns paired with E weights
```

The plan derives every role from the same checked inputs:

- authenticated group order and row ranges;
- exact chunk and block ranges from `WitnessLayout`;
- native A, B, and D ring dimensions;
- decomposition gadgets;
- the prepared relation address point; and
- the ring switch challenge `alpha`.

There is no separate Stage 3 layout and no verifier-only copy of the setup
address rules.

## Structured role weights

For each group and nonempty unit, the plan builds paired equality tensors for
the role columns and relation witness addresses. The main axes are claims,
blocks, digits, native role subcolumns, and setup rows.

Equal affine units can be combined into one extra unit axis. Before combining
them, the plan checks the complete axis shape, scalar, and setup and relation
offset strides. Unequal dyadic units keep their original families. No physical
padding is introduced for the optimization.

## Mixed dimensions

Let `d_0` be the common coefficient block and let a role use ring dimension
`d_R`. Its projection ratio is `q = d_R/d_0`.

For `q = 1`, the plan evaluates the relation equality directly. For `q > 1`,
it factors aligned low relation coordinates once. If the physical relation
offset is not aligned, it records an explicit dense role projection axis. The
setup side and relation side therefore use the same native role interpretation
without padding smaller rings.

The prepared state records whether a batch is still in native coordinates or
has already factored the relation coordinates. This state is encoded by the
`ProjectedEqPairTensor` enum, so evaluation cannot apply the transform twice.

## Direct Stage 2 mode

In direct mode, the plan materializes compact setup index weights and scans the
required public setup prefix. For each setup ring it evaluates the ring at
`alpha`, multiplies by the structured index weight, and accumulates the result.

The scan is linear in the required public setup size. This is necessary because
the setup coefficients are arbitrary. The verifier factors the setup equality
table by contiguous outer blocks to avoid repeated index division and repeated
outer equality multiplications.

Direct mode returns the setup contribution as part of
`RelationMatrixEvaluator::eval_flat_at_point`.

## Deferred Stage 3 mode

In deferred mode, Stage 2 receives a claimed setup contribution. It uses that
claim in the same relation evaluation and caches the exact prepared plan.

Stage 3 then proves a product over two coordinates:

- `rho_y` selects the coefficient inside a setup ring;
- `rho_setup_idx` selects the setup ring index.

At the final Stage 3 point, the verifier checks

```math
setup(\rho_y,\rho_{setup})
\cdot
weight(\rho_{setup})
\cdot
pow_\alpha(\rho_y).
```

The first factor comes from the public setup or from an authenticated setup
prefix opening. The second factor is the compact MLE of the same
`SetupContributionPlan` used by Stage 2. The final factor evaluates the powers
of `alpha` inside one ring.

When a setup prefix is selected, the first factor is an evaluation of the actual
full power-of-two setup prefix `S[0..n_prefix]`. The active support
`natural_len` belongs to the setup-index weight: rows outside that support have
zero weight even though the committed setup coefficients in the tail are real.
The Boolean setup-product claim is therefore unchanged, while the deferred
opening claim is bound to the full-prefix commitment.

Stage 3 returns the full challenge point for the recursive suffix. It does not
replace or modify the next witness claim produced by Stage 2.

## Setup prefix offloading

A schedule may select an authenticated setup prefix slot. The verifier checks
that the selected slot covers the active setup length required by the plan and
the full power-of-two prefix needed by its commitment domain, absorbs the slot
identity, and uses the proof's setup prefix evaluation. If no slot is selected,
it evaluates the local public setup directly.

The slot changes where the public setup polynomial is opened. It does not
change the setup index weights or relation geometry.

## Safety and ownership

Plan construction checks every role dimension, projection ratio, row span,
unit range, and address product. The verifier caches a plan only when deferred
mode is active, and Stage 3 consumes it only for the exact Stage 2 challenge
point.

The main implementation owners are:

- `crates/akita-types/src/setup_contribution/` for geometry and tensors;
- `crates/akita-verifier/src/protocol/ring_switch/relation_evaluation.rs` for
  direct or deferred selection; and
- `crates/akita-verifier/src/stages/stage3.rs` for the setup product check.
