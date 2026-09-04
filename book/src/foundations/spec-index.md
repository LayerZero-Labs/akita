# Spec index

This index names the specifications that still define current behavior or
unresolved work. Shipped and superseded records live in
[`specs/archive/README.md`](../../../specs/archive/README.md). The lifecycle
policy and the checker use the same live set in
[`specs/PRUNING.md`](../../../specs/PRUNING.md) and
`scripts/check-spec-references.sh`.

## Live specifications

| Spec | Status | Why it remains live |
|------|--------|---------------------|
| [`akita-compute-backend-metal`](../../../specs/akita-compute-backend-metal.md) | active | Metal and hybrid backend work remains open. |
| [`quotient-free-tail-ring-relations`](../../../specs/quotient-free-tail-ring-relations.md) | active | Defines reduced evaluation and its bounded quotient-free tail cutover. |
| [`quotient-free-tail-ring-relations-implementation`](../../../specs/quotient-free-tail-ring-relations-implementation.md) | active | Defines the implementation architecture and acceptance contract for the quotient-free tail cutover. |
| [`dyadic-chunk-partition`](../../../specs/dyadic-chunk-partition.md) | implemented | Defines the current witness chunk partition contract. |
| [`external-schedule-catalog-ownership`](../../../specs/external-schedule-catalog-ownership.md) | active | Owns the in-flight cut from compiled schedule rows to explicitly supplied full-catalog artifacts. |
| [`guided-schedule-adaptation`](../../../specs/guided-schedule-adaptation.md) | active | Defines bounded offline adaptation from trusted scalar rows to exact grouped rows. |
| [`flat-public-matrix-and-exact-ntt-cache`](../../../specs/flat-public-matrix-and-exact-ntt-cache.md) | implemented | Load-bearing setup layout with follow-up provenance and artifact work. |
| [`fold-linf-rejection`](../../../specs/fold-linf-rejection.md) | implemented | Its sizing formula is used by the SIS cap implementation. |
| [`heterogeneous-group-source-contracts`](../../../specs/heterogeneous-group-source-contracts.md) | implemented | Defines current source-free group and fold-admission rules. |
| [`large-digit-ntt-infrastructure`](../../../specs/large-digit-ntt-infrastructure.md) | implemented | Load-bearing large-digit NTT and terminal verification contract. |
| [`packed-sumcheck`](../../../specs/packed-sumcheck.md) | approved | Approved packed EOR and sum-check implementation; earlier Stage 1 and Stage 2 prerequisite gates are complete. |
| [`role-native-projected-digit-layout`](../../../specs/role-native-projected-digit-layout.md) | implemented | Normative witness and verifier layout source. |
| [`runtime-ring-cutover`](../../../specs/runtime-ring-cutover.md) | implemented | Normative runtime ring contract cited by the architecture chapter. |
| [`selective-l2-fold-security-sizing`](../../../specs/selective-l2-fold-security-sizing.md) | implemented | Current security sizing source; deferred alternatives remain recorded. |
| [`setup-offloading-planner`](../../../specs/setup-offloading-planner.md) | implemented | Current recursive setup selection policy and external artifact contract. |
| [`sis-quantum128-scalar-n-table`](../../../specs/sis-quantum128-scalar-n-table.md) | implemented | Current 128-bit SIS security policy source. |
| [`structured-e-term`](../../../specs/structured-e-term.md) | implemented | Current structured verifier E-term contract. |
| [`subring-coefficient-packing`](../../../specs/subring-coefficient-packing.md) | active | Merged implementation still has an unresolved proof blocker. |
| [`transcript-grinding`](../../../specs/transcript-grinding.md) | proposed | Defines the planned public proof-of-work policy and packed nonce stream. |

## Archived records

The archive contains the historical design record for shipped work and the
superseded alternatives that explain why the live contracts have their current
shape. Update the owning Book chapter for current narrative text. Do not use an
archived record as a source for new behavior.

## Maintenance

When a PR changes a specification, update its status and links in the same PR.
When a specification is shipped and its durable content is in the Book, move it
to the archive and add an entry to the archive index. Run
`scripts/check-spec-references.sh --all` during quarterly cleanup to find stale
non-archive references.
