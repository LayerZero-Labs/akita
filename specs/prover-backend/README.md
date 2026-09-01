# Prover backend design

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-09-01 |
| Status | proposed |
| PR | #461 |
| Supersedes | The unmerged backend-boundary proposal in PR #457 |
| Superseded-by | |
| Book-chapter | |
| Related | `runtime-ring-cutover.md`, `role-native-projected-digit-layout.md`, and `akita-compute-backend-metal.md`; their current contracts remain in force until an implementation replaces them |

## Summary

Akita will replace its compute traits, which pass CPU-shaped values through host
code, with a prover backend that owns private state. Each transcript step
receives the public data and challenges available after the previous challenge
draw and returns every prover message needed before the next draw. Commitment
is one backend call made before transcript work starts because no challenge
draw interrupts it.
The protocol driver will continue to own the schedule, transcript, challenges,
validation, and proof assembly. The backend may keep any private representation
that is consistent with its later calls.

The first replacement is commitment. One backend call will execute
the complete source-to-message pipeline, including inner commitment, digit
decomposition, outer commitment, and commitment compression. It will return
the exact public `CommittedGroup` and a state reference that does not expose
private state, not a
protocol-visible `AkitaCommitmentHint` representation.

This package is intentionally a breaking design. Akita makes no backward-
compatibility guarantee, and implementation work must delete or internalize
surfaces that contradict the final ownership model instead of preserving them
through aliases or pass-through wrappers.

## How to read this package

Read the files in this order:

1. [`current-state.md`](current-state.md) compares the Akita and Jolt designs
   pinned for this proposal and identifies what is worth preserving.
2. [`architecture.md`](architecture.md) defines who owns messages and state,
   which calls a backend exposes, and how backend support is selected.
3. [`transcript-steps.md`](transcript-steps.md) defines a transcript step and
   maps Akita's protocol messages to backend calls.
4. [`commitment-replacement.md`](commitment-replacement.md) specifies the first
   implementation stage, including the surfaces to remove or evolve.
5. [`commitment-implementation-order.md`](commitment-implementation-order.md)
   gives a testable implementation order and the deletion-based exit gate for
   the commitment replacement.
6. [`roadmap.md`](roadmap.md) orders the remaining Akita work, the later Jolt
   alignment, and the eventual extraction of shared backend code.

Together, the files are one design record. `architecture.md`, `transcript-steps.md`,
and `commitment-replacement.md` are normative. `current-state.md` is evidence and
rationale. `roadmap.md` is normative about phase gates and dependencies but
allows maintainers to repartition commits or pull requests.

## Normative language

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in
the normative files are to be interpreted as described in BCP 14 when, and
only when, they appear in all capitals.

## Goal

Build a prover backend in which each transcript step performs all prover work
available between two Fiat–Shamir challenge draws, each operation before the
transcript starts uses the same one-call rule, and all private prover state
remains backend-owned and free to use any representation.

## Preferred terms

Use one name for each concept throughout code and documentation:

| Name | Meaning |
|---|---|
| protocol message | exact prover output consumed by the proof, transcript, or verifier |
| backend call | one request to the prover backend and its response |
| transcript step | all prover work between two challenge draws |
| prover backend | implementation that executes backend calls and owns private state |
| backend state store | state that may be reused across proofs |
| prover session | state and resources for one proof |
| state reference | typed name for backend-owned state that does not expose its representation |
| internal operation | private computation used to implement a backend call |
| checkpoint | explicit versioned file or transfer representation |
| portable checkpoint | checkpoint accepted by different backend implementations |
| backend-specific checkpoint | checkpoint accepted only by a compatible backend |

Do not introduce “epoch,” “runtime,” “artifact,” “handle,” “form,” or
“envelope” as alternate names for these concepts. Existing code symbols with
those words may remain only until the roadmap explicitly deletes or renames
them.

## General principles

1. **Protocol authority stays outside the backend.** The protocol driver owns
   statement validation, schedule selection, message order, transcript labels,
   absorption, challenge sampling, proof assembly, and verifier-visible bytes.
2. **Protocol messages and private state are different objects.** A protocol
   message crosses the backend boundary in the exact representation required
   by the verifier. Private state stays in a backend state store or prover
   session and is named only by a typed state reference.
3. **Fiat–Shamir determines transcript-step boundaries.** A remote-capable path
   performs at most one backend request and response for all work after one squeeze and
   before the next. Consecutive absorbs without an intervening squeeze form one
   ordered set of messages. Commitment is one backend call before transcript
   work starts.
4. **Internal representation is unconstrained.** A backend may retain field
   values, digits, transforms, device buffers, remote object identifiers,
   recomputation recipes, or other state as long as its later operations are
   internally consistent.
5. **Persistence is explicit.** Portable checkpoints are separate from live
   backend state. Requiring disk persistence must not force every backend to use
   the checkpoint representation while proving.
6. **Backend changes are planned.** Mixed backends must not silently move,
   serialize, or recompute state. A transfer, checkpoint, or recomputation path
   must be selected before it is needed.
7. **Protocol messages are backend-invariant.** For fixed public inputs,
   witness, protocol configuration, and prover entropy, backend choice and
   backend configuration must not change proof or transcript bytes.
8. **Reusable operations stay private.** A backend may implement a small set of
   internal operations and may also implement a complete backend call. It
   should not have to reimplement every Akita or Jolt protocol relation.
9. **The CPU backend defines the expected result.** Optimized call
   implementations must be checked against the CPU backend using the same
   checked protocol plans.
10. **Unsupported execution is decided early.** Missing backend support or
    state-transfer failures must surface during planning or before the next
    message is absorbed. Mid-proof silent fallback is prohibited.

## Scope

This design covers:

- Akita prover setup preparation and per-proof state;
- protocol-message and transcript-step boundaries;
- source input and witness data kept by a backend;
- commitment, opening, ring-switch, compression, tensor, and sum-check
  execution boundaries;
- backend selection and state transfer between backends;
- live state, portable checkpoints, and setup-prefix persistence;
- CPU execution, accelerator or remote execution, and observability;
- the changes Jolt will eventually need before a shared backend interface can be
  extracted.

## Non-goals

- This design does not merge the Akita and Jolt transcripts, proof formats, or
  protocol drivers.
- This design does not extract a shared Akita/Jolt crate in the first Akita
  replacement.
- This design does not require a particular GPU API, RPC transport, async
  library, allocator, or device-memory format.
- This design does not make verifier execution depend on prover backend code.
- This design does not preserve current backend traits, source-support
  bundles, commitment-hint serialization, or four-cluster stack APIs for
  compatibility.
- This design does not promise that every internal arithmetic kernel is useful
  to every backend. It defines the backend boundary and leaves low-level
  operations private; it does not define a universal instruction set.

## Package-level acceptance criteria

- [ ] A maintainer can identify who creates and owns every protocol message,
      private state object, setup object, and checkpoint.
- [ ] Every Akita transcript squeeze has documented preceding messages
      and a proposed transcript step boundary.
- [ ] The commitment replacement has a type-level API sketch, deletion/evolution
      inventory, failure policy, checkpoint policy, and test plan.
- [ ] The target permits a backend whose live commitment state has no CPU field
      vector, `RingVec`, clone, default, or serialization representation.
- [ ] The target permits one backend to retain commitment state across proofs
      and one prover session to reuse backend-stored state across opening and later
      Akita or Jolt work.
- [ ] CPU and remote/mock backend tests can prove exact byte
      equality and maximum control-round-trip counts.
- [ ] The roadmap names gates that must pass before Jolt alignment and before
      extraction of a shared backend interface.
- [ ] Existing live specs that mandate `AkitaCommitmentHint` storage or the old
      backend hierarchy are explicitly reconciled when implementation changes
      current behavior.

## Documentation lifecycle

This package remains in `specs/` while the architecture is proposed, approved,
or being implemented. Implementation pull requests must update the acceptance
criteria and the relationship to affected live specs. Once the backend model
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
