# Spec: Two-stage setup offload

| Field | Value |
|---|---|
| Author(s) | |
| Created | 2026-07-22 |
| Status | proposed |
| Branch | `quang/stage3-setup-contribution-refactor` |
| Base | PR #318 integration head `1275bc63` (including PR #317) |
| Supersedes | The recursive path in `setup-product-sumcheck.md` and `batched-stage3-setup-opening.md` after cutover |
| Superseded-by | |
| Book-chapter | |

## Summary

Recursive setup offload currently appends a third sum-check after the direct
relation/range-image sum-check. Stage 3 proves the setup product and carries the
Stage 2 witness opening to a new point. This placement is correct for uniform
ring dimensions, but it creates a second point-routing protocol and blocks
mixed ring dimensions because it reconstructs the setup projection from
`d_a` rather than the checked common relation geometry.

This spec replaces the recursive three-stage path with the two-stage protocol
described by the Akita paper:

1. Stage 1 runs the range tree and adds relation plus evaluation trace to the
   final range sum-check.
2. Stage 2 batches the setup product with a witness claim reduction. The claim
   reduction binds the carried range image to the witness and carries the
   Stage 1 witness opening to one fresh witness point.

Direct folds MAY retain the current Stage 1 range proof followed by the fused
relation/range-image Stage 2 proof. Recursive setup offload MUST use the
two-stage shape in this document. Stage 3 and its proof field are deleted when
the recursive cutover is complete.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14
(RFC 2119 and RFC 8174) when, and only when, they appear in all capitals.

## Motivation

The setup contribution depends on the random relation point. It cannot be
proved before relation evaluation fixes that point. The current implementation
therefore runs:

```text
Stage 1: range tree
Stage 2: relation + evaluation trace + range-image refold
Stage 3: setup product + carried Stage 2 witness opening
```

The paper observes that relation is linear in the witness and can ride the
final Stage 1 range leaf without increasing its degree. This fixes the relation
point one stage earlier. The setup product can then run beside the range-image
refold in Stage 2:

```text
Stage 1: range tree; final leaf also proves relation + evaluation trace
Stage 2: setup product + witness claim reduction
```

This placement has four advantages:

- recursive offload adds no third transcript stage;
- setup contribution consumes the checked relation point directly;
- the witness exits at one point rather than carrying unrelated Stage 1 and
  range-image claims; and
- mixed role dimensions no longer need a Stage 3-only `log2(d_a)` projection.

## Scope

This spec covers:

- the recursive setup-offload proof shape;
- prover and verifier transcript order;
- relation and evaluation-trace placement in Stage 1;
- setup-product and witness-claim-reduction placement in Stage 2;
- mixed nested role dimensions in the two sum-checks;
- setup-prefix and next-witness openings carried to the successor fold; and
- deletion of the old Stage 3 protocol after parity is established.

This spec does not cover:

- planner objective changes;
- new commitment matrices or setup-prefix layouts;
- non-nested or non-power-of-two role dimensions;
- range-image microkernel specialization; or
- zero-knowledge masking for recursive setup offload.

Range-image kernel specialization remains a separate performance PR. It MUST
not change the equations in this document.

## Typed topology authority

PR #317 makes setup offload a successor-owned edge property:

```text
RecursiveFoldParams.incoming_setup_prefix: Option<SetupPrefixSlotId>
```

This field MUST be the sole schedule authority for whether the predecessor uses
the recursive two-stage protocol. Runtime code MAY derive a local protocol
shape from this edge, but it MUST NOT store an independent producer-side setup
mode.

The mirrored `CommittedGroupParams.setup_prefix` field is transitional. Until
it is removed, schedule validation MUST require exact equality with
`incoming_setup_prefix`. Consumers MUST read the successor edge, not the
mirror, to select the protocol.

The terminal successor MUST NOT request an incoming setup prefix because no
later committed fold can consume the carried prefix opening.

## Prepared setup authority from PR #318

PR #318 makes `SetupContributionPlan` the canonical prepared setup-weight
authority in both direct and deferred setup modes. Stage 2 constructs the plan
while evaluating the relation, uses its prepared E/T/Z equality slices, and
caches the challenge-bound plan for the existing Stage 3 verifier.

The two-stage cutover MUST reuse this plan and its role-native setup geometry.
It MUST NOT add another setup-weight builder, equality-table representation, or
projection API solely for the relocated setup product. Once Stage 3 is deleted,
the plan passes directly from offloaded Stage 1 relation evaluation into the
new Stage 2 setup-product verifier instead of through the Stage 3 cache.

PR #318 also changes recursive schedule selection to prioritize the first
remaining direct setup footprint before proof bytes. That planner policy is
part of this spec's baseline but is not changed by the protocol cutover.

## Current boundary

The current code deliberately rejects recursive setup offload when predecessor
role dimensions differ from the successor inner ring dimension. That rejection
MUST remain until the complete prover, verifier, proof, transcript, and suffix
state cutover in this spec passes end to end.

The implementation MUST NOT remove the guard by replacing `log2(d_a)` with
`log2(coeff_count)` inside the old Stage 3. Such a change would still be wrong:
the A, B, and D setup weights have different native coefficient boundaries, so
one Stage 3 slice cannot represent every role projection.

## Shared geometry

Let:

```text
coeff_count = min(d_a, d_b, d_d, outgoing_witness_ring_dimension)
```

as checked by `RelationRangeImagePlan`. Let the flat digit witness use the
LSB-first point:

```text
r = (r_coeff, r_lane)
```

where `r_coeff` has `log2(coeff_count)` coordinates. This common split is a
storage and contraction boundary for the witness. It is not the setup
coefficient boundary for every role.

The setup contribution MUST evaluate each role at its own native boundary:

```text
role R uses log2(d_R) low coefficient coordinates
```

The checked setup projection MUST therefore accept the complete relation point
and derive each A/B/D role view from the flat address mapping. It MUST NOT drop
a fixed prefix based only on `d_a` or `coeff_count`.

Nested dimensions MAY reuse prefixes of one alpha-power ladder. The semantic
projection remains role-specific even when allocations are shared.

## Stage 1: range tree plus relation

All non-final range product levels remain unchanged. They continue to use the
equality-factored proofs selected by `DigitRangePlan`.

For a recursive offload edge, the final range leaf MUST be an ordinary
sum-check because the relation term does not share the range term's equality
factor. Let:

- `V_leaf` be the final range-tree input claim;
- `tau_leaf` be its equality anchor;
- `L(S(x))` be the final range-leaf polynomial;
- `V_rel` be the row-batched relation claim, including evaluation trace;
- `m_tau(r)` be the row-batched relation weight; and
- `zeta` be a fresh batching challenge sampled after both input claims are
  transcript-bound.

Stage 1 proves:

```text
V_leaf + zeta * V_rel
  = sum_x [
      eq(tau_leaf, x) * L(S(x))
      + zeta * W(x) * m_tau(x)
    ].
```

For bases 4 and 8, this is the only range-tree stage. For larger bases, only
the final leaf changes shape.

The final Stage 1 point is `r1`. Stage 1 outputs three claims:

```text
range_image_claim = S(r1)
witness_claim     = W(r1)
deferred_setup_claim = sigma_setup(r1)
```

`deferred_setup_claim` is the setup-dependent summand removed from the direct
relation evaluator. The prover binds this claimed scalar after Stage 1. The
verifier uses it with the local relation and evaluation-trace summands to check
the Stage 1 terminal equation, then accepts it only after Stage 2 proves the
setup product. The scalar is deferred, not trusted.

Evaluation trace MUST move with relation. It MUST NOT remain in offloaded Stage
2 or be represented by a second trace protocol.

### Stage 1 proof shape

The proof format MUST distinguish:

- equality-factored range stages; and
- the ordinary offloaded final leaf.

The implementation SHOULD use one explicit typed stage-proof enum. It MUST NOT
encode the distinction through sentinel degrees, empty coefficients, or a
schedule-external mode byte.

Direct folds MAY continue to serialize equality-factored Stage 1 leaves.

## Stage 2: setup product plus witness claim reduction

Stage 2 receives `r1`, `S(r1)`, `W(r1)`, and the deferred setup claim. It runs
two terms over one padded Boolean cube.

### Witness claim reduction

After sampling a fresh `theta`, the witness term proves:

```text
theta * W(r1) + S(r1)
  = sum_x eq(r1, x) * [theta * W(x) + W(x) * (W(x) + 1)].
```

This term simultaneously:

- binds the multilinear range-image table to the committed digit witness; and
- carries the independent Stage 1 witness opening to the same fresh point.

It outputs one opening `W(r_star)`.

### Setup product

The setup term proves:

```text
sigma_setup(r1)
  = sum_j SetupPrefix(j) * setup_weight(r1, j).
```

The setup weight factors into the setup-index component and the role-native
alpha component. Its evaluator MUST use the checked full relation point and
the typed setup-prefix slot selected by the successor edge.

The setup term outputs one opening `SetupPrefix(rho_setup)`.

### Batched Stage 2 geometry

The witness reduction and setup product MAY have different native round
counts. The existing padded-cube lifting rule remains valid:

```text
Lift_n_to_N(f) = 2^(-(N - n)) * f
```

where `N` is the maximum native round count. A fresh batching challenge MUST
combine the two input claims before the first Stage 2 round.

The verifier's final relation MUST check both projected terms and MUST return:

```text
next_witness_point = r_star
setup_prefix_point = rho_setup
```

Both points MUST be projections of the same Stage 2 challenge vector. The
existing suffix opening batch MAY reuse `BatchedStage3Geometry` only after that
type is renamed around its protocol-independent padded-product meaning. No
Stage 3 name or proof field may remain after cutover.

## Direct folds

Direct folds do not carry a setup-prefix opening. They MAY retain the current
architecture:

```text
Stage 1: range tree
Stage 2: relation + evaluation trace + range-image refold
```

The direct relation evaluator includes the setup contribution by scanning the
configured setup envelope.

The direct and offloaded paths MUST share:

- `RelationRangeImagePlan` geometry;
- semantic relation events;
- evaluation-trace inputs;
- witness layout and group/chunk order; and
- verifier setup-weight primitives.

They MUST NOT share mutable prover state or introduce a generic expression
engine solely to hide their different transcript placement.

## Transcript order

The recursive offload transcript MUST use this order:

1. Bind the outgoing witness commitment or terminal state.
2. Bind ring-switch claims and sample the existing relation row challenge.
3. Run all non-final Stage 1 range-tree levels.
4. Bind the final range input claim and row-batched relation claim.
5. Sample the Stage 1 relation batching challenge.
6. Run the ordinary final Stage 1 leaf.
7. Bind `S(r1)` and `W(r1)`.
8. Bind the deferred setup claim.
9. Sample the witness-claim-reduction challenge and the Stage 2 term-batching
   challenge.
10. Run the batched Stage 2 sum-check.
11. Bind `W(r_star)` and `SetupPrefix(rho_setup)`.
12. Pass the shared projected opening state to the successor fold.

Prover and verifier MUST use the same labels and order. The implementation
MUST add logging-transcript parity before deleting Stage 3.

## Proof and state changes

The recursive proof shape changes intentionally:

- Stage 1 gains an ordinary final-leaf variant and carries `W(r1)`.
- Stage 2 carries the setup claim, setup-prefix evaluation, next-witness
  evaluation, and one batched sum-check.
- `FoldLevelProof.stage3_sumcheck_proof` is deleted.
- Stage 3 proof shapes and deserialization branches are deleted.
- suffix state receives both projected Stage 2 openings directly.

No compatibility wrapper or pass-through alias is permitted. This repository
does not promise backward compatibility.

## Mixed ring dimensions

After this cutover, recursive setup offload SHOULD support the same nested
power-of-two role dimensions as direct Stage 2.

The acceptance geometry includes:

```text
(d_a, d_b, d_d) = (128, 64, 32)
```

with an independently selected successor inner dimension. Correctness depends
on role-native setup projection, not equality between predecessor roles and the
successor ring.

The implementation MUST retain the current rejection until all of these pass:

- setup-weight materialization versus succinct evaluation for mixed roles;
- direct versus offloaded relation-claim parity;
- prover/verifier final-relation parity;
- mixed recursive setup E2E; and
- malformed point, slot, and proof-shape rejection.

Mixed-role multigroup and multichunk schedules are not automatically claimed.
They become supported only after planner-authenticated schedules and E2E tests
exist for those combinations.

## Implementation sequence

### Slice 1: checked equations and proof topology

- Add dense-oracle tests for the offloaded Stage 1 and Stage 2 equations.
- Record the exact proof variants, claims, points, and transcript order needed
  by the atomic recursive cutover.
- Keep the current runtime rejection and Stage 3 implementation unchanged.

### Slice 2: offloaded Stage 1

- Add the typed ordinary final-leaf proof variant as part of the working
  recursive path; do not merge a dormant proof variant.
- Prepare relation factorization and evaluation trace before Stage 1.
- Fuse them into the final range leaf.
- Verify the ordinary final leaf and close local relation terms.
- Preserve all earlier range-tree stages.

### Slice 3: offloaded Stage 2

- Reuse the current range-image/witness machinery for claim reduction.
- Reuse the setup product table and structured setup-weight evaluator.
- Batch both terms over one padded cube.
- Return the two projected openings directly to suffix state.

### Slice 4: mixed setup projection

- Replace fixed `d_a` slicing with checked role-native projection.
- Add mixed direct/offloaded parity and E2E coverage.
- Remove the schedule rejection only after these tests pass.

### Slice 5: deletion and performance

- Delete Stage 3 types, modules, labels, proof fields, and sizing branches.
- Delete obsolete producer-side setup mode reconstruction.
- Regenerate proof-size and schedule-shape expectations.
- Run the complete CI feature matrix and path-specific setup-offload tests.
- Benchmark direct and offloaded proving, verification, proof bytes, and peak
  memory against pinned baselines.

Range-image specialization begins only after this protocol cutover is stable.

## Acceptance criteria

- The recursive setup-offload path runs exactly two sum-check stages.
- Relation and evaluation trace occur in the final Stage 1 range leaf.
- Offloaded Stage 2 contains only witness claim reduction and setup product.
- The recursive proof has no Stage 3 field or transcript frame.
- The successor edge is the sole setup-offload topology authority.
- Direct folds retain exact relation, trace, and range-image semantics.
- The setup contribution matches the direct flat matrix scan.
- The witness claim reduction matches a dense materialized oracle.
- Prover and verifier derive identical Stage 1 and Stage 2 points.
- The carried setup-prefix and witness openings share one padded-cube challenge.
- Uniform recursive proofs pass transcript and proof-shape tests.
- Mixed `(128, 64, 32)` recursive setup offload passes end to end before its
  schedule guard is removed.
- Malformed slots, points, proof variants, and missing carried claims return
  `AkitaError` on verifier-reachable paths.
- No unsupported mixed-shape claim is introduced by test-only schedule
  mutation.
- Range-image specialization remains absent from this PR.

## Security considerations

Moving relation changes when its point is sampled, not the relation being
proved. The Stage 1 batching challenge MUST be sampled after both input claims
are bound. Evaluation trace MUST use the same row challenge and relation point
as the other relation rows.

The range-image table and witness are independent multilinear tables away from
the Boolean cube. Stage 2 MUST prove the claim-reduction identity; checking
`S(r1) = W(r1)(W(r1)+1)` directly is unsound.

The setup prefix is transcript-independent, while alpha evaluation is
transcript-dependent. The commitment remains over flat setup coefficients; all
role-native alpha and address projections stay on the weight side.

Removing the mixed/offload guard before role-native projection is verified can
silently omit A/B lane coordinates. The guard is a security boundary, not a
temporary usability limitation.

## References

- `specs/relation-range-image-sumcheck.md`
- `specs/typed-schedule-topology-cutover.md`
- `specs/setup-product-sumcheck.md`
- `specs/batched-stage3-setup-opening.md`
- `specs/setup-layout-repack.md`
- `book/src/how/proving/sumcheck-stages.md`
- Akita paper, “Verifier offloading,” especially “Protocol placement: the
  two-stage form”
