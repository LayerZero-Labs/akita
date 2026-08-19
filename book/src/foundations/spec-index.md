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
| [`dyadic-chunk-partition`](../../../specs/dyadic-chunk-partition.md) | active | Defines the current witness chunk partition contract. |
| [`flat-public-matrix-and-exact-ntt-cache`](../../../specs/flat-public-matrix-and-exact-ntt-cache.md) | implemented | Load-bearing setup layout with follow-up provenance and artifact work. |
| [`fold-linf-rejection`](../../../specs/fold-linf-rejection.md) | implemented | Its sizing formula is used by the SIS cap implementation. |
| [`heterogeneous-group-source-contracts`](../../../specs/heterogeneous-group-source-contracts.md) | active | Defines current source-free group and fold-admission rules. |
| [`large-digit-ntt-infrastructure`](../../../specs/large-digit-ntt-infrastructure.md) | active | Current large-digit NTT and terminal verification contract. |
| [`packed-sumcheck`](../../../specs/packed-sumcheck.md) | approved | Ready for implementation, with Stage 1 and Stage 2 gates recorded. |
| [`role-native-projected-digit-layout`](../../../specs/role-native-projected-digit-layout.md) | active | Normative witness and verifier layout source. |
| [`runtime-ring-cutover`](../../../specs/runtime-ring-cutover.md) | implemented | Normative runtime ring contract cited by the architecture chapter. |
| [`selective-l2-fold-security-sizing`](../../../specs/selective-l2-fold-security-sizing.md) | active | Current security sizing review remains open. |
| [`setup-offloading-planner`](../../../specs/setup-offloading-planner.md) | active | Recursive setup selection policy remains under development. |
| [`setup-product-sumcheck`](../../../specs/setup-product-sumcheck.md) | proposed | Unimplemented Stage 3 design. |
| [`sis-quantum128-scalar-n-table`](../../../specs/sis-quantum128-scalar-n-table.md) | implemented | Current 128-bit SIS security policy source. |
| [`structured-e-term`](../../../specs/structured-e-term.md) | active | Current structured verifier E-term contract and acceptance record. |
| [`subring-coefficient-packing`](../../../specs/subring-coefficient-packing.md) | active | Merged implementation still has an unresolved proof blocker. |

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
