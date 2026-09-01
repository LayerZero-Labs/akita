# Prover-runtime roadmap

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | incremental migration and deletion plan |

## Strategy

The migration proceeds from one proven semantic boundary to a general runtime.
It does not begin by designing a joint Akita/Jolt mega-trait.

The sequence is:

```text
characterize Akita
    ↓
full commitment epoch + opaque state
    ↓
commit-to-open state chain + checkpoints
    ↓
remaining Akita Fiat–Shamir epochs
    ↓
owner-aware planning; delete cluster hierarchy
    ↓
apply the same discipline inside Jolt
    ↓
extract only the demonstrated common runtime substrate
```

Each phase MUST leave one coherent source of truth. Transitional adapters may
exist only while an identified caller migrates, and the same phase or a named
follow-up MUST delete them. Backward compatibility is not a goal.

## Success measures

The program is successful when:

- Akita protocol orchestration performs at most one semantic backend call per
  maximal Fiat–Shamir epoch;
- ordinary backend results contain only canonical prover messages and opaque
  state handles;
- proof and transcript bytes are independent of backend and runtime tuning;
- a remote backend can keep witness-proportional state off the host from
  commitment through opening;
- state-owner incompatibility is rejected during planning, not halfway through
  a proof;
- persistence and migration use explicit checkpoint operations;
- Akita no longer publicly exposes its source × operation × ring-dimension
  capability lattice;
- Jolt no longer requires `OpeningHint` or CPU-shaped ZK retained state;
- Jolt and Akita can share resident state through one runtime without sharing a
  transcript or protocol driver;
- the extracted common layer contains only mechanisms already exercised by
  both protocols.

Track at least:

- semantic calls and bytes crossing each epoch boundary;
- device or remote internal work separately from semantic calls;
- peak host and backend-resident bytes;
- checkpoint and transfer bytes;
- recomputation counts;
- transcript-event and proof-byte equivalence;
- stale/wrong-owner handle rejections;
- old public trait, wrapper, and hint counts.

## Phase 0: characterization and boundary inventory

### Deliverables

- A machine-checkable catalog of every Akita absorb/squeeze boundary.
- Canonical message types or temporary named fixtures for every absorbed
  prover value.
- A retained-state inventory recording producer, next consumer, current
  representation, and whether it is ever transcript-visible.
- Golden public commitments, proof bytes, verifier outcomes, and
  `LoggingTranscript` event traces for representative schedules.
- A semantic call counter around the current CPU implementation.
- Failure fixtures for malformed messages, stale state, and setup mismatch.

### Representative coverage

Fixtures SHOULD cover:

- dense and one-hot sources;
- packed Jolt trace sources;
- recursive and setup-prefix sources;
- every supported runtime root ring dimension;
- schedules with and without ring switching;
- commitment compression and relation compression;
- terminal suffix binding;
- disk-persisted setup-prefix state.

### Exit gate

Every value crossing the intended runtime boundary is classified as one of:

1. protocol input or verifier-derived checked plan;
2. Fiat–Shamir challenge input;
3. canonical prover message output;
4. opaque retained state;
5. explicit checkpoint or transfer artifact.

Anything unclassified blocks the next phase.

## Phase 1: full commitment epoch

This phase implements [`commitment-cutover.md`](commitment-cutover.md) in the
falsifiable order defined by
[`commitment-implementation-slices.md`](commitment-implementation-slices.md).

### Minimum coherent PR

Characterization and checked-plan extraction MAY land independently because
they neither expose a second public commitment API nor weaken the current
invariants. The first public cutover PR MUST follow the C3 cutover-train rules
and SHOULD:

- add runtime owner identity and typed opaque state handles;
- add a validated full-commitment request;
- implement one CPU commitment epoch covering inner, outer, and all
  compression maps;
- route the ordinary production commitment entry point through it;
- replace parallel public hints and sources with bound committed artifacts;
- carry the artifact through the first relation/opening consumer;
- add a representation-independent fake remote backend;
- assert one semantic commitment call and no witness-shaped transfer through
  the first consumer;
- preserve public commitment and proof bytes.

It SHOULD NOT add compatibility blanket implementations for all old backend
traits. Existing leaf algorithms may be called privately from the CPU epoch.

The implementation MAY be developed as multiple commits or a private stack,
but only the adapter-free final tip is a merge candidate. Main MUST NOT retain
both public APIs between pull requests.

### Follow-up source and producer migrations

- multilinear and packed Jolt trace ingress;
- sparse-unit and other specialized ingress;
- grouped and precommitted root producers;
- setup-prefix generation;
- Jolt-to-Akita commitment adapters.

### Exit gate

Every pretranscript/public `CommittedGroup` is created through the full semantic
epoch. No caller producing that message sequences inner, outer, or compression
kernels itself. An ordinary committed artifact reaches the first
relation/opening consumer without witness-sized host readback. Recursive
next-witness messages remain the distinct Phase 4C epoch.

## Phase 2: extend the opening-state chain

### Deliverables

- Convert the remaining opening and relation computations after the first
  consumer into semantic epochs over opaque relation state.
- Remove protocol access to runtime-private B images, `t_hat` materialization,
  compression witnesses, quotients, and source-recomputation recipes.
- Extend owner-affinity planning beyond the first consumer through the last
  relation state user.
- Add explicit source-retention and recomputation policies where needed.

### Deletions

Once the last reader migrates, delete:

- CPU-shaped relation-state carriers that cross semantic epoch boundaries;
- protocol-visible compression-witness and quotient accessors;
- operation-cluster routing used only by the migrated opening/relation path;
- implicit owner transfers or host reconstruction fallbacks.

### Exit gate

A fake backend whose state has no `RingVec`, serialized polynomial, or cloneable
hint can continue from the first relation/opening message through the remaining
opening-state chain without witness-sized host readback.

## Phase 3: explicit checkpoints and setup-prefix persistence

### Deliverables

- Versioned commitment checkpoint envelope.
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
checkpoint API or explicit recomputation. Live handles remain non-serializable.

## Phase 4: remaining Akita message epochs

Convert in protocol order so that each new epoch consumes challenges derived
since the previous message and returns the complete next message bundle.

### 4A. Initial opening and relation payloads

- compile checked relation/opening plans;
- consume committed handles;
- return the exact message bundle next absorbed by the driver;
- retain only opaque relation state.

### 4B. Fold grinding

- keep challenge derivation and grinding policy in the driver;
- move witness-dependent candidate evaluation behind one batch epoch;
- return the first accepting candidate and opaque winning fold state;
- remove protocol-visible cache and transform lifecycle calls.

### 4C. Next-witness binding

- consume prior challenge and opaque prior-level state;
- return `OuterPayload` or explicit `TerminalInnerState` as the canonical
  next-witness binding message;
- retain opaque next-level state;
- reject unsupported owner transitions during planning.

### 4D. Ring-switch and Stage-1 prefix

- consume the consecutive ring-switch `alpha`, `tau0`, and `tau1` challenge
  bundle squeezed after the next-witness binding;
- return the first Stage-1 round message;
- retain relation weights, witness-evaluation tables, and digit-range state.

### 4E. Batched sum-check rounds

Replace per-instance host return values with one group-level transition:

```text
prior round challenge + active group state + checked fold coefficients
    → one already-combined canonical round polynomial
    + updated opaque group state
```

The driver validates and absorbs the combined polynomial, squeezes the next
challenge, and invokes the next transition. The final post-challenge call
returns canonical claims and parks or consumes opaque residues.

### 4F. Final responses

- return exact wire response types;
- validate them before absorption or serialization;
- ensure no final message is extracted by inspecting retained state.

### Exit gate

Akita's protocol driver can be read as a sequence of:

```text
compile request → invoke epoch → validate message → absorb → squeeze
```

No witness-proportional intermediate crosses that loop.

## Phase 5: capability planning and hierarchy removal

### Planner responsibilities

Before transcript initialization, compile a proof execution plan that records:

- required semantic epochs and protocol order;
- state producer/consumer chains;
- selected state owner for each chain;
- source ingress requirements;
- resident setup and transform requirements;
- explicit checkpoint, transfer, or recomputation edges;
- resource estimates and backend tuning;
- unsupported capability failures.

Fallback MUST apply to an entire stateful chain unless an explicit transfer edge
exists. Runtime tuning MAY change scheduling and resource use but MUST NOT
change protocol bytes.

### Deletions

- `ProverComputeStack` and cluster-specific routing as public protocol APIs;
- `LevelProveStacks` and `TieredProveStacks`;
- `Runtime*Backend*` capability bundles and their macro expansion;
- delegating CPU cluster wrappers;
- cache-release hooks in protocol code;
- public re-exports of fragment-level backend traits from `akita-pcs`;
- source × operation × ring-dimension generic bounds at the PCS entry point.

Private CPU form traits MAY survive if multiple semantic epochs reuse them and
they remain a useful implementation vocabulary.

### Exit gate

A complete proof plan either validates before the transcript starts or fails
with a precise capability, owner, transfer, or resource error. There is no
proof-time missing-slot fallback.

## Phase 6: Akita runtime hardening

### Conformance suite

Every backend MUST pass:

- reference/optimized message and proof-byte equivalence;
- transcript-state equivalence at every message boundary;
- canonical message shape and encoding tests;
- state-handle owner, generation, kind, lifetime, and linearity tests;
- one-call-per-epoch RPC tests;
- explicit transfer/checkpoint tests;
- deterministic cancellation and cleanup tests;
- verifier independence and malformed-proof no-panic tests;
- bounded-memory tests for reference compression;
- fault-injection tests at every remote boundary.

### Observability

Expose epoch names, durations, request/message sizes, resident bytes,
checkpoint bytes, recomputations, and backend-private diagnostic IDs. Logs MUST
not expose witness values, secret randomness, or opaque state payloads.

### Exit gate

At least two materially different implementations pass the suite: the CPU
reference/runtime and either a true accelerator/remote implementation or a
test backend with genuinely non-CPU state and transport semantics.

## Phase 7: Jolt alignment

This work belongs in Jolt after Akita demonstrates the handle/message model. It
should be staged without waiting for a shared crate.

### 7A. Commit/open committed objects

- Replace `PCS::OpeningHint` in Jolt's live commitment carrier with an opaque
  committed-state handle.
- Bind statements, public commitments, and state at creation.
- Route Dory commitment/opening through the proof session.
- Make recomputation an explicit runtime policy.

### 7B. Packed Akita path

- Pass the proof runtime/session through packed stage 0 and stage 8.
- Delete `PackedCommitStub`.
- Remove manual setup-residency eviction from stage orchestration.
- Stop representing Akita state as a fixed CPU polynomial enum plus cloned
  opening hint.
- Let nested Jolt/Akita execution share resident state while each protocol
  driver retains its own transcript rules.

### 7C. Group-level sum-check transitions

- Lift `ProveRounds` from per-member values to co-located member groups.
- Return the already-folded polynomial that is actually absorbed.
- Fuse final challenge ingestion, claim extraction, validation, and residue
  parking where they share state.

### 7D. ZK retained state

- Separate wire commitments/proofs from `CommittedSumcheckWitness`.
- Put round coefficients, output rows, and blinds behind opaque state handles.
- Make BlindFold consume those handles rather than CPU `Vec<Vec<F>>` state.
- Keep prover randomness supplied under protocol policy; do not expose the
  transcript to the backend.

### 7E. Validate the form layer

Implement and benchmark the proposed small form vocabulary against both the
reference interpreter and a genuinely different backend. Do not canonize the
documented six-form hypothesis until it replaces a meaningful portion of the
roughly two dozen relation-specific optimized preparations without losing
clarity or performance.

### Exit gate

Jolt's ordinary epoch outputs follow the same message-plus-handle discipline as
Akita, and its major commitment/opening, sum-check, ZK, and packed-Akita paths
use the proof runtime.

## Phase 8: extract the joint substrate

Only after both repositories exercise the same invariants should the common
mechanism move to a neutral crate or repository.

### Candidate shared mechanisms

- runtime and proof-session lifecycle;
- typed opaque handle identity and validation;
- state ownership, generation, linearity, and shared-borrow rules;
- capability and state-chain planning primitives;
- checkpoint/transfer envelopes and portability declarations;
- cancellation, cleanup, observability, and fault taxonomy;
- backend tuning separated from protocol configuration;
- transport support beneath semantic calls.

### Protocol-local mechanisms

The common layer MUST NOT own:

- Akita commitment, fold, ring-switch, or response message types;
- Jolt stage or relation declarations;
- either transcript implementation or transcript labels;
- verifier logic;
- PCS wire types as universal runtime concepts;
- one flat trait containing every operation from both protocols.

Akita and Jolt define typed epoch adapters over the neutral session. One
concrete runtime may implement both and co-locate their state.

### Extraction gate

Extract only a mechanism with:

- two real protocol consumers;
- stable identical semantics in both repositories;
- conformance tests portable to the neutral crate;
- no dependency cycle between Akita and Jolt;
- a demonstrated second implementation or transport use case.

## Proposed PR series

The exact split may change with implementation evidence, but each PR should be
independently reviewable and delete superseded surface as soon as callers move.

| PR | Scope | Required proof |
|---|---|---|
| A | characterization fixtures and epoch catalog tests | pinned bytes/events and current call counts |
| B | handle/store primitives and full CPU commitment epoch | one semantic call; unchanged commitment bytes |
| C | migrate all commitment entry points | no protocol-side inner/outer/compression sequencing |
| D | committed artifacts and first opening consumer | non-CPU state completes commit-to-open chain |
| E | setup-prefix checkpoint cutover | explicit cross-process round trip; old format deleted |
| F | remove `AkitaCommitmentHint` and commitment-era public fragments | zero hint readers and no compatibility aliases |
| G | opening/fold/ring-switch message epochs | transcript-boundary equivalence |
| H | group-level batched sum-check epochs | one call per group per round |
| I | capability planner and stack deletion | preflight owner/transfer rejection |
| J | Akita conformance suite and runtime hardening | two materially different implementations |
| K+ | Jolt-local alignment series | Jolt proof-byte and boundary equivalence |
| final | neutral joint-runtime extraction | two-protocol consumers, no cycle |

PRs B through F are the immediate commitment program. If a smaller split is
needed, the temporary API MUST be explicitly private or marked for deletion in
the very next PR.

## Review gates for every phase

Every implementation PR MUST answer:

1. What is the exact previous and next transcript event?
2. Is every ordinary output an exact canonical message or opaque handle?
3. Which state owner produces and consumes each handle?
4. Can the transition be planned before transcript mutation?
5. What is the semantic RPC count?
6. Which proof-byte and transcript-event fixtures establish equivalence?
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
| epoch API becomes a protocol mega-method | keep protocol messages semantic; reuse small private forms below it |
| dynamic state becomes untyped `Any` | expose typed handles; keep erasure private |
| commitment handle lifetime is too restrictive | distinguish long-lived commitment store from proof-scoped transient session |
| fallback silently causes readback | plan whole state chains and require explicit transfer/recompute edges |
| persistence re-freezes CPU layout | separate live state from versioned checkpoint capabilities |
| message validation diverges from verifier | derive plans and encodings from the same canonical protocol types |
| reference path loses specialized source performance | retain source adapters and optimized CPU forms below ingress |
| shared crate extracted too early | require two protocol consumers and demonstrated common semantics |
| nested Jolt/Akita runtime merges transcripts | share state only; keep protocol drivers and transcripts separate |
| old abstraction survives indefinitely | pair each migration with named deletions and track surface counts |

## Deliberately deferred choices

The roadmap does not yet standardize:

- the neutral crate name or repository;
- a universal wire protocol for remote execution;
- a fixed arithmetic-form vocabulary;
- mandatory portable checkpoints for every backend;
- a global scheduler spanning unrelated proofs;
- verifier acceleration;
- distributed trust or attestation for untrusted remote provers.

Those choices require implementation evidence. The message-boundary, opaque
state, ownership, transcript authority, and explicit checkpoint rules do not.
