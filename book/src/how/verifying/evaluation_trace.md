# Evaluation trace

The evaluation trace binds each public opening claim to the `E` digits inside
the next witness when the schedule selects `EvaluationTrace`. Early folds may
select `SubringCoefficientPacking` instead. Both methods bind the claimed
opening in Stage 2, but they use different coefficient geometry.

The verifier never constructs the prover's trace table. It prepares a small
description of each group and exact witness unit, then evaluates the trace once
at the final Stage 2 point.

## What the trace checks

For each commitment group, claim, and global live block, the trace combines:

- the claim batching coefficient;
- the block opening weight from the public opening point;
- the opening digit gadget weight;
- the source ring trace or packed inner opening factor; and
- the equality weight of the physical `E` address.

The Boolean corner values are exactly the coefficients that the prover used to
construct the opening evaluation from `E`. Their multilinear extension is the
public trace polynomial `T` used by Stage 2.

## Prepared group state

`PreparedEvaluationTrace` stores one group record with:

- the group-local block opening point and basis;
- source and opening ring dimensions;
- the common coefficient block length;
- shared opening digit weights;
- the source ring inner trace;
- one coefficient per claim;
- one exact descriptor per nonempty witness unit; and
- a prepared contraction plan.

A unit descriptor contains the physical start of its first claim, the stride
between claims, the first global block, and its exact block count. The verifier
derives these values from `WitnessLayout`. It does not recompute chunk
boundaries.

Preparation also chooses the concrete contraction kernel. For a short affine
range, it stores the two small block-weight tables. For a long range, it stores
the exact dyadic unit segments used by the paired recurrence. Evaluation does
not rebuild either plan.

## Coefficient and address factors

At the final Stage 2 point, the verifier first splits off the common low
coefficient coordinates. It evaluates each source lane of the inner trace at
that coefficient point. The remaining column point selects the physical `E`
address.

For one claim and unit, the address is affine in these coordinates:

- local block;
- opening digit;
- source subcolumn; and
- source lane.

The block opening weight is Boolean in the global block bits. The physical
column weight is Boolean in the relation address bits. A paired tensor
recurrence contracts both factors without enumerating all live blocks.

Both Lagrange and monomial opening bases are supported. The monomial path uses
its Boolean factor directly and does not invert a challenge.

## Unequal dyadic chunks

The canonical partition does not pad chunks. A group with `B` blocks and `C`
chunks uses boundaries `floor(cB/C)`. Unit block counts can therefore differ,
and empty units can appear when `C > B`.

Preparation skips empty `E` units. For nonempty units, the evaluator searches
for consecutive units with the same block count, claim stride, and affine
offset differences. It combines only those exact runs. Each run is divided
into power of two tensor segments. An irregular unit remains separate.

This compaction changes verifier work only. It does not change witness layout,
proof bytes, transcript order, opening schedule, or protocol claims.

## Concrete crossover

The paired recurrence has better asymptotic behavior for long block ranges,
but a short affine scan has a lower constant cost for small ranges. The
verifier uses the scan for 2 to 32 blocks per unit. It uses the paired recurrence
for singleton units and for at least 64 blocks per unit.

This is a local kernel choice. Both paths evaluate the same prepared unit
formula, and dense tests compare them with the materialized trace definition.

## Extension field openings

When the challenge field is a proper extension of the base field, block and
position weights are stored as canonical subfield coordinates. The verifier
keeps `K` coordinates for extension degree `K` instead of expanding each value
to a dense ring of dimension `D`.

The trace preparation recovers the needed extension values directly from those
coordinates. Invalid dimensions, coordinate counts, or noncanonical subfield
images are rejected before arithmetic.

## Cost

For fixed protocol dimensions, the compact path is logarithmic in the block
range of one affine unit. Its explicit outer work is proportional to groups,
claims, nonempty unit segments, source subcolumns, and source lanes. The chunk
count is capped at 64.

The verifier stores no witness-sized trace table and no dense block basis table.

## Code map

- Verifier preparation and contraction:
  `crates/akita-verifier/src/protocol/evaluation_trace.rs`.
- Shared checked trace geometry:
  `crates/akita-types/src/trace_weight/`.
- Exact `E` addresses: `crates/akita-types/src/witness.rs`.
- Paired tensor model and kernels:
  `crates/akita-algebra/src/offset_eq/tensor_pair/`.
- Ring and subfield trace primitives:
  `crates/akita-types/src/field_reduction.rs`.

The next chapter explains how the same relation geometry drives direct setup
evaluation and the deferred Stage 3 check.
