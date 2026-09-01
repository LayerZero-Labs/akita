# Prover backend implementation roadmap

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | incremental migration and deletion plan |

## Strategy

The migration starts with one tested backend boundary and expands from there.
It does not begin by designing one trait containing every Akita and Jolt operation.

The sequence is:

```text
record current Akita behavior
    ↓
complete commitment call + state reference
    ↓
commit-to-open state flow + checkpoints
    ↓
remaining Akita transcript steps
    ↓
plan backend use; delete cluster hierarchy
    ↓
make the same changes inside Jolt
    ↓
extract only backend code already shared in practice
```

Each phase MUST leave one coherent source of truth. Transitional adapters may
exist only while an identified caller migrates, and the same phase or a named
follow-up MUST delete them. Backward compatibility is not a goal.

## Success measures

The program is successful when:

- Akita protocol orchestration performs at most one backend call per transcript
  step;
- ordinary backend results contain only protocol messages and state references;
- proof and transcript bytes are independent of backend and backend tuning;
- a remote backend can keep witness-proportional state off the host from
  commitment through opening;
- a state value that cannot be used by its planned backend is rejected before
  the proof starts;
- persistence and migration use explicit checkpoint operations;
- Akita no longer publicly exposes its source × operation × ring-dimension
  trait hierarchy;
- Jolt no longer requires `OpeningHint` or CPU-shaped private ZK state;
- Jolt and Akita can share backend-stored state without sharing a
  transcript or protocol driver;
- the extracted shared code contains only mechanisms already exercised by
  both protocols.

Track at least:

- backend calls and bytes crossing each step boundary;
- device or remote internal work separately from backend calls;
- peak host bytes and bytes stored by the backend;
- checkpoint and transfer bytes;
- recomputation counts;
- transcript-event and proof-byte equivalence;
- stale/wrong-backend reference rejections;
- old public trait, wrapper, and hint counts.

## Phase 0: record current behavior and transcript boundaries

### Deliverables

- A machine-checkable catalog of every Akita absorb/squeeze boundary.
- Protocol message types or temporary named test vectors for every absorbed
  prover value.
- A private-state inventory recording producer, next consumer, current
  representation, and whether it is ever transcript-visible.
- Golden public commitments, proof bytes, verifier outcomes, and
  `LoggingTranscript` event traces for representative schedules.
- A backend call counter around the current CPU backend.
- Failure tests for malformed messages, stale state, and setup mismatch.

### Representative coverage

Test vectors SHOULD cover:

- dense and one-hot sources;
- packed Jolt trace sources;
- recursive and setup-prefix sources;
- every supported backend root ring dimension;
- schedules with and without ring switching;
- commitment compression and relation compression;
- terminal suffix binding;
- disk-persisted setup-prefix state.

### Exit gate

Every value crossing the intended backend boundary is classified as one of:

1. protocol input or verifier-derived checked plan;
2. Fiat–Shamir challenge input;
3. protocol message output;
4. a reference to private prover state;
5. an explicit checkpoint or transfer value.

Anything unclassified blocks the next phase.

## Phase 1: replace the commitment backend

This phase implements [`commitment-replacement.md`](commitment-replacement.md) in the
testable order defined by
[`commitment-implementation-order.md`](commitment-implementation-order.md).

### Minimum coherent PR

Characterization and checked-plan extraction MAY land independently because
they neither expose a second public commitment API nor weaken the current
invariants. The first public replacement PR MUST follow the Stage C3 rule that
only the single-API final state may merge. It SHOULD:

- add backend identity and typed state references;
- add one checked full-commitment plan;
- implement one CPU commitment call covering inner, outer, and all
  compression maps;
- route the ordinary production commitment entry point through it;
- replace parallel public hints and sources with `CommittedGroupWithState`;
- carry that value through the first relation/opening consumer;
- add a representation-independent fake remote backend;
- assert one commitment call and no witness-shaped transfer through
  the first consumer;
- preserve public commitment and proof bytes.

It SHOULD NOT add compatibility blanket implementations for all old backend
traits. Existing leaf algorithms may be called privately from the CPU step.

The implementation MAY be developed as multiple commits or dependent branches,
but only the final version with temporary adapters removed may merge. Main MUST NOT retain
both public APIs between pull requests.

### Follow-up source and producer migrations

- multilinear and packed Jolt trace input;
- sparse-unit and other specialized input;
- grouped and precommitted root producers;
- setup-prefix generation;
- Jolt-to-Akita commitment adapters.

### Exit gate

Every public `CommittedGroup` created before transcript work starts uses the
complete backend call. No caller producing that message sequences inner, outer,
or compression kernels itself. An ordinary `CommittedGroupWithState` reaches
the first relation/opening consumer without witness-sized host readback. Recursive
next-witness messages remain the distinct Phase 4C step.

## Phase 2: extend the opening-state flow

### Deliverables

- Convert the remaining opening and relation computations after the first
  consumer into transcript steps over relation state references.
- Remove protocol access to backend-private B images, `t_hat` materialization,
  compression witnesses, quotients, and source-recomputation recipes.
- Plan one backend to keep each relation state value through its last use, or
  plan an explicit transfer.
- Add explicit source-retention and recomputation policies where needed.

### Deletions

Once the last reader migrates, delete:

- CPU-shaped relation-state values that cross transcript step boundaries;
- protocol-visible compression-witness and quotient accessors;
- operation-cluster routing used only by the migrated opening/relation path;
- implicit backend transfers or host reconstruction fallbacks.

### Exit gate

A fake backend whose state has no `RingVec`, serialized polynomial, or cloneable
hint can continue from the first relation/opening message through the remaining
opening-state flow without witness-sized host readback.

## Phase 3: explicit checkpoints and setup-prefix persistence

### Deliverables

- Versioned commitment checkpoint format.
- CPU export and import implementation.
- Binding to setup, schedule, public commitment, and protocol format.
- Backend-specific checkpoint tags and declared portability.
- Explicit recomputation fallback where the source is available.
- New setup-prefix prover registry and disk-cache namespace.
- Failure before transcript mutation for unsupported or mismatched imports.

### Deletions

- Serialized `AkitaCommitmentHint` persistence.
- Legacy cache compatibility readers and aliases.
- Implicit state migration through host vectors.

### Exit gate

Setup-prefix state survives a process boundary only through the documented
checkpoint API or explicit recomputation. Live references remain non-serializable.

## Phase 4: remaining Akita transcript steps

Convert in protocol order so that each new step consumes challenges derived
since the previous message and returns every protocol message required before
the next challenge draw.

### 4A. Initial opening and relation payloads

- compile checked relation/opening plans;
- consume bound `CommittedGroupWithState` values;
- return the next protocol messages in their exact transcript order;
- retain only a relation state reference.

### 4B. Fold grinding

- keep challenge derivation and grinding policy in the driver;
- move witness-dependent candidate evaluation behind one batch step;
- return the first accepting candidate and a reference to the winning fold state;
- remove protocol-visible cache and transform lifecycle calls.

### 4C. Next-witness binding

- consume the prior challenge and a reference to prior-level state;
- return `OuterPayload` or explicit `TerminalTFieldsMessage` as the exact
  next-witness-binding message;
- retain a reference to next-level state;
- reject unsupported backend changes during planning.

### 4D. Start ring switch and Stage 1

- consume the consecutive ring-switch `alpha`, `tau0`, and `tau1` challenge
  set squeezed after the next-witness binding;
- return the first Stage-1 round message;
- retain relation weights, witness-evaluation tables, and digit-range state.

### 4E. Batched sum-check rounds

Replace per-instance host return values with one group-level transition:

```text
prior round challenge + active group state + checked fold coefficients
    → one already-combined protocol round polynomial
    + updated group state reference
```

The driver validates and absorbs the combined polynomial, squeezes the next
challenge, and invokes the next transition. The final post-challenge call
returns protocol claims and saves or consumes private residues.

### 4F. Final responses

- return exact wire response types;
- validate them before absorption or serialization;
- ensure no final message is extracted by inspecting private prover state.

### Exit gate

Akita's protocol driver can be read as a sequence of:

```text
compile request → invoke step → validate message → absorb → squeeze
```

No witness-proportional intermediate crosses that loop.

## Phase 5: plan backend use and remove the old trait hierarchy

### Planner responsibilities

Before transcript initialization, compile a proof execution plan that records:

- required transcript steps and protocol order;
- where each state value is produced and next consumed;
- the backend that keeps each state value;
- source input requirements;
- setup and transform data each backend must keep;
- explicit checkpoint, transfer, or recomputation edges;
- resource estimates and backend tuning;
- missing backend support.

Fallback MUST apply to an entire state flow unless an explicit transfer edge
exists. Backend tuning MAY change scheduling and resource use but MUST NOT
change protocol bytes.

### Deletions

- `ProverComputeStack` and cluster-specific routing as public protocol APIs;
- `LevelProveStacks` and `TieredProveStacks`;
- `Runtime*Backend*` trait bundles and their macro expansion;
- delegating CPU cluster wrappers;
- cache-release hooks in protocol code;
- public re-exports of fragment-level backend traits from `akita-pcs`;
- source × operation × ring-dimension generic bounds at the PCS entry point.

Private CPU traits MAY survive if multiple transcript steps reuse their
operations and the direct call sites remain clear.

### Exit gate

A complete proof plan either validates before the transcript starts or fails
with a precise unsupported-operation, backend, transfer, or resource error.
There is no proof-time missing-slot fallback.

## Phase 6: Akita backend hardening

### Backend test suite

Every backend MUST pass:

- CPU/optimized message and proof-byte equivalence;
- transcript-state equivalence at every message boundary;
- protocol message shape and encoding tests;
- state-reference tests for backend identity, generation, kind, lifetime, and
  single-use behavior;
- one backend request and response for every backend call;
- explicit transfer/checkpoint tests;
- deterministic cancellation and cleanup tests;
- verifier independence and malformed-proof no-panic tests;
- bounded-memory tests for CPU compression;
- fault-injection tests at every remote boundary.

### Observability

Expose step names, durations, request/message sizes, bytes stored by the backend,
checkpoint bytes, recomputations, and backend-private diagnostic IDs. Logs MUST
not expose witness values, secret randomness, or stored private state.

### Exit gate

At least two materially different implementations pass the suite: the CPU
backend and either a true accelerator/remote implementation or a
test backend with genuinely non-CPU state and transport semantics.

## Phase 7: Jolt alignment

This work belongs in Jolt after Akita demonstrates the message and state-reference
model. It should be staged without waiting for a shared crate.

### 7A. Commit/open committed objects

- Replace `PCS::OpeningHint` in Jolt's live commitment value with a state
  reference that does not expose the committed state.
- Bind statements, public commitments, and state at creation.
- Route Dory commitment/opening through the prover session.
- Make recomputation an explicit backend policy.

### 7B. Packed Akita path

- Pass the prover backend and session through packed stage 0 and stage 8.
- Delete `PackedCommitStub`.
- Remove manual eviction of stored setup data from stage code.
- Stop representing Akita state as a fixed CPU polynomial enum plus cloned
  opening hint.
- Let nested Jolt/Akita execution share backend-stored state while each protocol
  driver retains its own transcript rules.

### 7C. Group-level sum-check transitions

- Lift `ProveRounds` from per-member values to member groups stored on the same backend.
- Return the already-folded polynomial that is actually absorbed.
- Combine final challenge ingestion, claim extraction, validation, and saving
  private residues where they share state.

### 7D. Private ZK state

- Separate wire commitments/proofs from `CommittedSumcheckWitness`.
- Put round coefficients, output rows, and blinds behind state references.
- Make BlindFold consume those references rather than CPU `Vec<Vec<F>>` state.
- Keep prover randomness supplied under protocol policy; do not expose the
  transcript to the backend.

### 7E. Validate the internal operation set

Implement and benchmark the proposed small internal operation set against both
the CPU backend and a genuinely different backend. Do not standardize the
documented six-operation proposal until it replaces a meaningful portion of the
roughly two dozen relation-specific optimized preparations without losing
clarity or performance.

### Exit gate

Jolt's ordinary backend outputs follow the same message-plus-reference rule as
Akita, and its major commitment/opening, sum-check, ZK, and packed-Akita paths
use the prover backend.

## Phase 8: extract shared backend code

Only after both repositories exercise the same invariants should the common
mechanism move to a neutral crate or repository.

### Candidate shared mechanisms

- backend and prover session lifecycle;
- typed state reference identity and validation;
- backend/store identity, generation, single-use rules, and shared-borrow rules;
- backend-selection and state-transfer planning;
- checkpoint/transfer formats and portability declarations;
- cancellation, cleanup, observability, and error categories;
- backend tuning separated from protocol configuration;
- transport support beneath backend calls.

### Protocol-local mechanisms

The common layer MUST NOT own:

- Akita commitment, fold, ring-switch, or response message types;
- Jolt stage or relation declarations;
- either transcript implementation or transcript labels;
- verifier logic;
- PCS wire types as universal backend concepts;
- one flat trait containing every operation from both protocols.

Akita and Jolt define typed backend calls over the shared prover session. One
concrete backend may implement both and co-locate their state.

### Extraction gate

Extract only a mechanism with:

- two real protocol consumers;
- the same tested behavior in both repositories;
- backend tests portable to the neutral crate;
- no dependency cycle between Akita and Jolt;
- a demonstrated second implementation or transport use case.

## Proposed PR series

The exact split may change with implementation evidence, but each PR should be
independently reviewable and delete superseded surface as soon as callers move.

| PR | Scope | Required proof |
|---|---|---|
| A | recorded outputs and step catalog tests | pinned bytes/events and current call counts |
| B | state store and complete CPU commitment call | one backend call; unchanged commitment bytes |
| C | migrate all commitment entry points | no protocol-side inner/outer/compression sequencing |
| D | `CommittedGroupWithState` and first opening consumer | non-CPU state completes the commitment-to-opening flow |
| E | replace setup-prefix checkpoints | explicit cross-process round trip; old format deleted |
| F | remove `AkitaCommitmentHint` and commitment-era public fragments | zero hint readers and no compatibility aliases |
| G | opening/fold/ring-switch transcript steps | transcript-boundary equivalence |
| H | group-level batched sum-check steps | one call per group per round |
| I | backend planner and stack deletion | reject unsupported backend use or transfer before proving |
| J | Akita backend tests and hardening | two materially different implementations |
| K+ | Jolt-local alignment series | Jolt proof-byte and boundary equivalence |
| final | neutral joint-backend extraction | two-protocol consumers, no cycle |

PRs B through F are the immediate commitment program. If a smaller split is
needed, the temporary API MUST be explicitly private or marked for deletion in
the very next PR.

## Review gates for every phase

Every implementation PR MUST answer:

1. What is the exact previous and next transcript event?
2. Is every ordinary output an exact protocol message or state reference?
3. Which backend produces and consumes each state reference?
4. Can the transition be planned before transcript mutation?
5. How many RPCs cross this transcript step?
6. Which recorded proof bytes and transcript events establish equality?
7. Which old trait, wrapper, field, or serialized representation becomes
   unnecessary and is deleted?
8. Can a non-CPU representation implement the new surface without conversion
   to host polynomials?
9. Are malformed messages rejected before absorption?
10. Does backend tuning leave protocol bytes unchanged?

An implementation that only moves an old field behind a wrapper does not pass
these gates.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| step API becomes one overly broad method | keep outputs limited to protocol messages; reuse small private operations inside the backend |
| private state becomes untyped `Any` | expose typed references; keep erasure private |
| commitment reference lifetime is too restrictive | distinguish long-lived commitment store from per-proof transient session |
| fallback silently causes readback | plan whole state flows and require explicit transfer/recompute edges |
| persistence re-freezes CPU layout | separate live state from explicit versioned checkpoint support |
| message validation diverges from verifier | derive plans and encodings from the same protocol types |
| CPU path loses specialized source performance | retain source adapters and optimized private CPU operations |
| shared crate extracted too early | require two protocol consumers with the same tested behavior |
| nested Jolt/Akita backend merges transcripts | share state only; keep protocol drivers and transcripts separate |
| old abstraction survives indefinitely | pair each migration with named deletions and track surface counts |

## Deliberately deferred choices

The roadmap does not yet standardize:

- the neutral crate name or repository;
- a universal wire protocol for remote execution;
- a fixed set of internal arithmetic operations;
- mandatory portable checkpoints for every backend;
- a global scheduler spanning unrelated proofs;
- verifier acceleration;
- distributed trust or attestation for untrusted remote provers.

Those choices require implementation evidence. The message boundary, state
references, transcript authority, and explicit checkpoint rules do not.
