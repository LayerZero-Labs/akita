# Spec package: Proof-scoped prover runtime

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-09-01 |
| Status | proposed |
| PR | #461 |
| Supersedes | The unmerged backend-boundary proposal in PR #457 |
| Superseded-by | |
| Book-chapter | |
| Related | `runtime-ring-cutover.md`, `role-native-projected-digit-layout.md`, and `akita-compute-backend-metal.md`; their current contracts remain in force until an implementation cutover updates them |

## Summary

Akita will replace its host-materialized compute-trait hierarchy with a
proof-scoped prover runtime whose externally visible operations align with
Fiat–Shamir message boundaries. The protocol driver will continue to own the
schedule, transcript, challenges, validation, and proof assembly. A selected
backend will receive validated protocol requests and challenges, produce the
next canonical prover message, and mutate arbitrary backend-private state held
for the proof lifetime.

The first cutover is commitment. One semantic backend operation will execute
the complete source-to-message pipeline, including inner commitment, digit
decomposition, outer commitment, and commitment compression. It will return
the canonical terminal `CommittedGroup` and an opaque state handle, not a
protocol-visible `AkitaCommitmentHint` representation.

This package is intentionally a breaking design. Akita makes no backward-
compatibility guarantee, and implementation work must delete or internalize
surfaces that contradict the final ownership model instead of preserving them
through aliases or pass-through wrappers.

## How to read this package

Read the files in this order:

1. [`current-state.md`](current-state.md) compares the Akita and Jolt designs
   pinned for this proposal and identifies what is worth preserving.
2. [`architecture.md`](architecture.md) defines the final ownership, state,
   message, execution, and capability contracts.
3. [`transcript-epochs.md`](transcript-epochs.md) defines a Fiat–Shamir epoch and
   maps Akita's protocol messages to semantic runtime operations.
4. [`commitment-cutover.md`](commitment-cutover.md) specifies the first
   implementation cut, including the surfaces to remove or evolve.
5. [`commitment-implementation-slices.md`](commitment-implementation-slices.md)
   gives the falsifiable implementation order and the deletion-based exit gate
   for the commitment cutover.
6. [`roadmap.md`](roadmap.md) orders the remaining Akita work, the later Jolt
   alignment, and the eventual common-runtime extraction.

The files form one design record. `architecture.md`, `transcript-epochs.md`,
and `commitment-cutover.md` are normative. `current-state.md` is evidence and
rationale. `roadmap.md` is normative about phase gates and dependencies but
allows maintainers to repartition commits or pull requests.

## Normative language

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in
the normative files are to be interpreted as described in BCP 14 when, and
only when, they appear in all capitals.

## Goal

Build a prover runtime in which the protocol-visible backend boundary is the
maximal computation interval between Fiat–Shamir challenge squeezes, while all
live retained prover state remains backend-private and representation-free.

## General principles

1. **Protocol authority stays outside the backend.** The protocol driver owns
   statement validation, schedule selection, message order, transcript labels,
   absorption, challenge sampling, proof assembly, and verifier-visible bytes.
2. **Messages and retained state are different objects.** A canonical prover
   message crosses the protocol boundary. Retained state stays in a
   runtime-owned store or proof session and is referenced only through an
   opaque typed handle.
3. **Fiat–Shamir determines the semantic call boundary.** A remote-capable path
   performs at most one control round trip for all work after one squeeze and
   before the next. Consecutive absorbs without an intervening squeeze form one
   message bundle.
4. **Internal representation is unconstrained.** A backend may retain field
   values, digits, transforms, device buffers, remote object identifiers,
   recomputation recipes, or other state as long as its later operations are
   internally consistent.
5. **Persistence is explicit.** Portable checkpoints are separate from live
   runtime state. Requiring disk persistence must not force every backend to use
   the checkpoint representation while proving.
6. **State ownership is planned.** Mixed backends must not silently move,
   serialize, or recompute state. Any owner change requires an explicit and
   observable transfer or checkpoint operation selected before it is needed.
7. **Canonical messages are backend-invariant.** For fixed public inputs,
   witness, protocol configuration, and prover entropy, backend choice and
   backend configuration must not change proof or transcript bytes.
8. **Reusable forms live below semantic epochs.** A backend should implement a
   small computational vocabulary and may override whole fused epochs. It
   should not have to reimplement every Akita or Jolt protocol relation.
9. **Reference execution remains authoritative.** Optimized and fused paths
   must be differentially checked against a canonical CPU/reference executor
   derived from the same validated protocol plans.
10. **Unsupported execution is decided early.** Capability or state-transfer
    failures must surface during planning or before the next message is
    absorbed. Mid-proof silent fallback is prohibited.

## Scope

This design covers:

- Akita prover setup preparation and proof-scoped state;
- protocol-message and Fiat–Shamir-epoch boundaries;
- source ingress and resident witness reuse;
- commitment, opening, ring-switch, compression, tensor, and sum-check
  execution boundaries;
- runtime capability selection and mixed-backend state ownership;
- live state, portable checkpoints, and setup-prefix persistence;
- CPU reference execution, accelerator or remote execution, and observability;
- the changes Jolt will eventually need before a common runtime can be
  extracted.

## Non-goals

- This design does not merge the Akita and Jolt transcripts, proof formats, or
  protocol drivers.
- This design does not extract a shared Akita/Jolt crate in the first Akita
  cutover.
- This design does not require a particular GPU API, RPC transport, async
  runtime, allocator, or device-memory format.
- This design does not make verifier execution depend on prover runtime code.
- This design does not preserve current backend traits, source capability
  bundles, commitment-hint serialization, or four-cluster stack APIs for
  compatibility.
- This design does not promise that every internal arithmetic kernel is useful
  to every backend. It defines the semantic boundary and a smaller form layer,
  not a universal lowest-common-denominator instruction set.

## Package-level acceptance criteria

- [ ] A maintainer can identify the canonical owner of every protocol message,
      retained-state object, setup artifact, and persistence artifact.
- [ ] Every Akita transcript squeeze has a documented preceding message bundle
      and a proposed semantic epoch boundary.
- [ ] The commitment cutover has a type-level API sketch, deletion/evolution
      inventory, failure policy, checkpoint policy, and test plan.
- [ ] The target permits a backend whose live commitment state has no CPU field
      vector, `RingVec`, clone, default, or serialization representation.
- [ ] The target permits one runtime to retain commitment state across proofs
      and one proof session to reuse resident state across opening and later
      Akita or Jolt work.
- [ ] CPU/reference and remote/mock conformance tests can prove canonical byte
      equality and maximum control-round-trip counts.
- [ ] The roadmap names gates that must pass before Jolt alignment and before
      extraction of a common runtime.
- [ ] Existing live specs that mandate `AkitaCommitmentHint` storage or the old
      backend hierarchy are explicitly reconciled when implementation changes
      current behavior.

## Documentation lifecycle

This package remains in `specs/` while the architecture is proposed, approved,
or being implemented. Implementation pull requests must update the acceptance
criteria and the relationship to affected live specs. Once the runtime model
is stable, durable explanation belongs in `book/src/how/architecture.md`,
`book/src/how/transcript.md`, and `book/src/roadmap/compute-backends.md`; this
package can then be archived according to [`../PRUNING.md`](../PRUNING.md).

## References

- [BCP 14](https://www.rfc-editor.org/info/bcp14)
- Akita PR #457, `feat/backend-commit`, design proposal at
  `e8f34bb6415f20dd5f18f53d390f998d12117c9c`
- Jolt `origin/main` at
  `e789b9f5f418bdc8beac196a11324b949c36f8cf`
- [`../runtime-ring-cutover.md`](../runtime-ring-cutover.md)
- [`../role-native-projected-digit-layout.md`](../role-native-projected-digit-layout.md)
- [`../akita-compute-backend-metal.md`](../akita-compute-backend-metal.md)
- [`../../docs/compute-backends.md`](../../docs/compute-backends.md)
