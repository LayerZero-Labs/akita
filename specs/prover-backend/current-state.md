# Akita and Jolt today

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | pinned evidence and design rationale |

## Scope and source pins

This comparison records the implementations inspected for the prover-backend
proposal:

- Akita `origin/main`:
  `988fc11b48d0edd77f181e96e9c23f1470a583c8`;
- [Akita PR #457](https://github.com/LayerZero-Labs/akita/pull/457)
  proposal: `e8f34bb6415f20dd5f18f53d390f998d12117c9c`;
- [Jolt `origin/main`](https://github.com/a16z/jolt/tree/e789b9f5f418bdc8beac196a11324b949c36f8cf):
  `e789b9f5f418bdc8beac196a11324b949c36f8cf`.

The Akita sections describe current behavior at the pinned `main` commit, not
the target architecture in this package. The Jolt sections are prior-art
analysis, not requirements imposed on the Jolt repository.

## Executive assessment

Jolt has a better per-proof execution model than Akita and has nearly the
right Fiat–Shamir boundary for sum-check rounds. It has not generalized that
model to commitment, opening, private ZK state, or its packed Akita path.

Akita has strong checked protocol plans, dynamic ring-dimension dispatch,
well-defined setup-cache ownership, and representation-specialized CPU kernels. Its public
backend hierarchy nevertheless describes implementation fragments rather than
complete protocol messages. It also standardizes a CPU-shaped commitment hint
and assumes that state can move freely between independently routed operation
clusters.

The target should combine Jolt's session and round-transition ideas with
Akita's checked plans and CPU kernels, while fixing the private-state and
whole-message boundaries in both designs.

## Akita today

### Current layers

Akita currently has four overlapping abstraction layers:

1. `ComputeBackendSetup<F>` and primitive row-operation traits in
   `crates/akita-prover/src/compute/backend.rs`;
2. source-typed combined-operation traits such as `RootCommitKernel`,
   `OpeningFoldKernel`, and `RingSwitchRelationKernel` in
   `crates/akita-prover/src/compute/kernels.rs`;
3. source/ring-dimension trait bundles in
   `crates/akita-prover/src/compute/poly.rs` and
   `crates/akita-prover/src/compute/runtime_capabilities.rs`;
4. commit/opening/tensor/ring-switch routing through `OperationCtx`,
   `ProverComputeStack`, and `LevelProveStacks` in
   `crates/akita-prover/src/compute/stack.rs`.

These layers solved real earlier problems: CPU NTT caches no longer leak from
setup types, high-level protocol storage supports dynamic ring dimensions, source-specific
commit kernels preserve dense and one-hot structure, and proof execution can
select different prepared contexts by operation cluster and fold level.

They do not define one coherent remote-backend contract.

### Commitment is split at implementation boundaries

The authoritative commitment entry point in
`crates/akita-prover/src/api/commitment.rs` performs:

1. source-value checks and frozen-plan resolution;
2. `compute_inner_outer_commitment`;
3. `compute_commitment_compression`;
4. construction of `CommittedGroup`;
5. construction of `AkitaCommitmentHint` from host-materialized inner rows,
   compression stages, and compression quotients.

The current backend seam can therefore require multiple accelerator calls and
host materialization inside one protocol operation. PR #457 proposes fusing
inner commitment, decomposition, and outer commitment, but still stops before
compression and returns the host rows required by the current hint.

There is no Fiat–Shamir squeeze between outer commitment and commitment
compression. The uncompressed outer result is not the public message. It is an
implementation intermediate and should not define the public backend
boundary.

### `AkitaCommitmentHint` fixes live state representation

`crates/akita-types/src/proof/hints.rs` defines `AkitaCommitmentHint<F>` as:

- one `RingVec<F>` of A-native inner rows per committed polynomial;
- one selected ring dimension;
- packed outer-compression stage bytes; and
- outer-compression quotient `RingVec<F>` values.

It is cloneable, equality-comparable, validated, serialized, and deserialized
as an Akita protocol-adjacent type. Later prover code calls
`inner_rows`, `into_rows`, `outer_compression_witness`, and
`outer_compression_quotients` directly.

This representation works for CPU replay and saved files. It is not a valid
universal live-state contract. A GPU may prefer digits or transforms; a remote
backend may retain an object identifier; another backend may retain the source
and recompute; and a joint Jolt/Akita backend may already own a different
backend-stored witness representation.

The type also mixes two different kinds of data:

- compression stages, quotients, and inner rows used only to continue proving
  are private prover state; but
- the terminal next-witness inner-state binding is read from `inner_rows` and
  absorbed into the transcript in
  `crates/akita-prover/src/protocol/core/fold/mod.rs`.

The latter is a protocol message and must be modeled separately from
private prover state.

### Trait inheritance combines unrelated operations

`DigitRowsComputeBackend<F>` inherits `CompressionComputeBackend<F>`. A backend
cannot advertise generic negacyclic digit-row support without also implementing
commitment/relation compression. `RuntimeCommitBackendFor<F, P>` then combines
that inherited surface with one `RootCommitKernel` bound for every supported
ring dimension selected at run time and every source view.

Consequences include:

- primitive operations imply unrelated protocol duties;
- one backend implementation is coupled to the cross-product of source views
  and backend dimensions;
- full commitment support is inferred from fragment traits rather than
  represented as one operation;
- a missing optimized call can be hidden behind a type-correct trait bundle;
- source input, reusable internal operations, protocol steps, and private state have no
  distinct types.

### The four-part routing stack assumes free state movement

`ProverComputeStack<C, O, T, R>` independently routes commitment, opening,
tensor, and ring-switch work. `TieredProveStacks` additionally changes the stack
by fold range. The stack tracks prepared setup and NTT cache ownership well, but
it does not record which backend owns per-proof state or how that state moves.

A commitment may execute on one backend while opening or ring-switch executes
on another. Protocol code bridges the operations with host values and
`AkitaCommitmentHint`. This makes CPU materialization the implicit universal
transfer format and prevents a planner from rejecting an impossible or
expensive state transition before proving.

### Large host-shaped outputs cross backend boundaries

Several kernel APIs return `Vec`, `Vec<Vec<_>>`, `RingVec`, partial buffers, or
fallback markers intended for host orchestration. These are useful private CPU
interfaces. They become expensive or impossible as the public remote boundary
because they expose witness-proportional intermediate state and require the
host to decide the next computation.

## What Akita should preserve

The replacement should preserve and move beneath the new backend boundary:

- frozen and validated protocol operation plans;
- dynamic ring-dimension dispatch at typed arithmetic leaves;
- exact public-matrix derivation and cache-prefix requirements;
- source-specific dense, one-hot, and recursive CPU algorithms;
- scalar/reference implementations and differential tests;
- setup identity validation and checked resource accounting;
- `CommittedGroup` as the exact public commitment message;
- verifier independence and the no-panic contract.

The replacement should remove or internalize:

- `AkitaCommitmentHint` as a public live-state type;
- the public source/ring-dimension backend supertrait ladders;
- the implication that digit-row support includes compression support;
- host-visible inner/decomposition/compression intermediates between steps;
- proof orchestration's ability to downcast or inspect backend state;
- implicit state transfer through CPU field vectors;
- public routing APIs whose only purpose is to reconstruct the old hierarchy.

## Jolt today

### Jolt's backend registry and `ProofSession`

Jolt's
[`backend.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-kernels/src/backend.rs)
defines `JoltBackend<F, PCS>` as a backend registry of boxed, object-safe
operation slots. It also defines `ProofSession`, a one-proof store keyed by
backend-private Rust types.

This gives Jolt features Akita lacks:

- reference and optimized slots can be composed as values;
- multiple backend configurations can coexist in one binary;
- retained tables and values passed between stages have a one-proof lifetime;
- witness uploads and shared tables have a natural reuse location;
- dynamic dispatch cost is correctly treated as negligible for heavy work.

`ProofSession` is a strong implementation idea but not the desired public
contract. Its `TypeId -> Box<dyn Any>` interface gives protocol and slot code no
typed backend, phase, generation, or transfer guarantee. Missing carries become
proof-time errors, and one concrete Rust type is one global key. Safe type
erasure may remain internal behind typed state references.

### Relation-owned requests and generated drivers

`PrepareKernel<F, R>` receives `ProverInputs<F, R>`, where `R` is the verifier's
concrete sum-check relation and owns the relevant dimensions, claims, points,
and challenges. Generated stage drivers assemble those inputs and hard-check
the resulting claims against verifier-derived relations.

This is an important single-source-of-truth property. Akita step inputs
should likewise contain checked protocol plans, not backend-specific request
structs that repeat geometry.

Jolt's direct CPU evaluation of verifier relations defines the expected result for optimized
paths. The design intends a small set of general internal operations
below per-relation selection, although the current optimized implementation
still contains substantial relation-specific preparation code and has not
fully realized that internal operation set.

### Sum-check already approaches a transcript-step boundary

Jolt's
[`prover.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-sumcheck/src/prover.rs)
defines `ProveRounds` so that `prove_round` receives a member's previous active
challenge and returns its current round polynomial. The backend can combine
binding the previous challenge with evaluating the next polynomial.
`finish_rounds` receives the final challenge.

The host-owned `prove_batch` loop retains protocol authority: it determines
active members, padding, batch coefficients, round-sum checks, transcript
absorption, and challenges.

The remaining boundary is one level too low for a remote backend. The scheduler
invokes each member separately, receives one polynomial per member, and folds
them on the host into the one batched round message that is actually absorbed.
Jolt's
[`clean-slate-prover.md`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/specs/clean-slate-prover.md)
records the intended member-group extension: one group stored on the same backend should return
the pre-folded round polynomial.

### Commitment and opening retain the old hint model

Jolt's
[`commitment.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-kernels/src/commitment.rs)
lets one `CommitWitness` call batch and stream all witness commitments, which
is a better top-level boundary than Akita's current inner/outer split. Its output
still contains `PCS::OpeningHint`.

Jolt's
[`schemes.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-openings/src/schemes.rs)
requires
`OpeningHint: Clone + Send + Sync + Default`. `CommitmentScheme::open` and
`prove_batch` receive a transcript.
The file explicitly records a TODO to replace the hint side channel with a
first-class committed object.

Jolt's
[`opening.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-kernels/src/opening.rs)
returns `MultilinearPoly` references from the joint-opening slot so the separate
PCS can drive lazy folding. The references may be device-backed, but the split
still prevents one backend from owning the whole opening step. Host code also
assembles, clones, and reorders commitment hints before invoking the
transcript-aware PCS.

### Other private-state leaks

The committed/ZK sum-check recorder returns `CommittedSumcheckWitness` beside
the wire proof. That retained witness has a fixed CPU collection shape and
lives outside `ProofSession` even though it exists only for later prover work.

Jolt's current implementation also performs significant ZK commitment work in
the recorder rather than through the general backend seam. This contradicts the
stronger claim that all heavy prover compute is behind `jolt-kernels`.

### Jolt's clean-slate direction is only partially implemented

The clean-slate design is valuable prior art, but it is important not to treat
its target as current implementation:

| Intended property | Current evidence at the pinned SHA | Consequence for Akita |
|---|---|---|
| Heavy transcript-free compute is behind the kernel seam | ZK recording, PCS proving, Dory opening, and packed Akita stage 0/8 still bypass it | Audit actual call paths, not registry coverage |
| A small internal operation set supports many relations | The optimized path has roughly two dozen relation-specific `PrepareKernel` implementations and no implemented general operation/descriptor layer | Treat the internal operation set as a hypothesis to validate with a second backend |
| Full verifier replay is a debug cross-check | [`blindfold.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-prover/src/blindfold.rs) still replays the verifier to map verifier relations into prover work | Keep one authoritative driver/relation source rather than replay-based coupling |
| Jolt `ProofSession` slots can be mixed safely | `TypeId` keys distinguish types, not backend instances, state generations, or several values of one type | Use typed backend/session/generation references |
| Missing backend support fails while constructing a plan | An incompatible value can surface as a missing typed `take` while proving | Plan the whole private-state flow before transcript mutation |
| Round scheduling is transport-ready | The production scheduler is sequential and its interface returns per-member rather than final group messages | Lift scheduling below a group-level step output |

Jolt's current CPU-shaped cross-stage carries make reference and optimized
slots interoperable, but that interoperability comes from fixing the transfer
representation. The Akita target instead makes state compatibility explicit
and rejects unsupported backend changes during planning.

### Packed Akita bypasses the session

Jolt's packed Akita
[`mod.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-prover/src/akita/mod.rs)
constructs one `JoltAkitaBackend` and one `ProofSession`, but packed stage 0
does not receive the session. Instead,
[`stage0.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-prover/src/akita/stage0.rs)
commits directly through Akita's PCS. The base Jolt commitment slot is an
explicit `PackedCommitStub`.

Packed
[`stage8.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-prover/src/akita/stage8.rs)
similarly calls Akita's transcript-aware batch opening directly and does not use
the session. `jolt-akita` retains an
[`AkitaProverHint`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-akita/src/adapters.rs#L641-L658)
that contains a cloned public commitment, Akita's backend commitment and hint,
and a fixed enum of dense, one-hot, trace-one-hot, or sparse-unit CPU polynomial
representations.

This is an adapter around Akita's current representation-bearing API, not a
joint backend.

## What Akita should borrow from Jolt

- one prover session for uploads, private state, memory pools, and
  cross-step carries;
- selecting supported backend operations from values rather than one monomorphic backend
  supertrait at the top-level prover API;
- object-safe top-level operations where arithmetic or RPC dominates dispatch;
- verifier relation or checked protocol plan as the backend input;
- generated/shared protocol drivers where this prevents prover/verifier order
  drift;
- prior-challenge-to-next-message round transitions;
- a scheduler/transport seam below host-owned transcript logic;
- a CPU backend that defines the expected result;
- backend-invariant proof bytes and explicit observability;
- a small internal operation set plus optional optimized call implementations.

## What Akita should not copy

- `OpeningHint` as a generic associated type with clone/default bounds;
- transcript-aware PCS proving;
- `TypeId -> Any` as the protocol-facing state API;
- one host readback per sum-check member per round;
- returning polynomial references when the next PCS work can remain in the
  same opening call;
- mixing a wire proof and retained ZK witness in one returned value;
- a fixed registry field for every relation as the eventual common Akita/Jolt
  backend API;
- claims that a general internal operation set exists before implementation
  and tests actually establish it.

## Comparison rubric

| Property | Akita at the pinned SHA | Jolt at the pinned SHA | Target |
|---|---|---|---|
| Protocol/transcript authority | Host-owned, but compute and transcript orchestration are interleaved | Strong for sum-check; PCS and ZK recording still receive transcripts | Driver only |
| Private prover state | Public serialized CPU-shaped hint | Session for kernels, fixed hints/witnesses elsewhere | Typed references to backend-owned state without a required representation |
| Commitment boundary | Inner, outer, and compression orchestrated on host | Whole witness commit call, fixed hint output | Full source-to-terminal-message step |
| Sum-check boundary | Primitive/stage-specific host calls | Prior challenge to member polynomial | Prior challenge to final batched message |
| Opening boundary | Host rebuilds from hints and source views | Backend returns polynomials, PCS drives opening | Bound `CommittedGroupWithState` to the opening protocol message |
| Backend selection | Static trait ladders and four operation groups | Boxed backend slots with CPU composition | Planned transcript steps over one prover session |
| State movement | Implicit CPU materialization | Session-local for some slots, hint cloning elsewhere | State stays on one backend unless explicitly transferred or checkpointed |
| Expected result | Checked plans and CPU kernels, distributed orchestration | Direct CPU evaluation of verifier relations | Checked protocol plan plus one CPU backend call |
| Jolt/Akita integration | Akita API fixes state shape | Packed stage 0/8 bypass Jolt session | Shared backend later; separate drivers and messages |

## Resulting design decisions

The evidence supports the following decisions:

1. The new Akita boundary is a prover backend that owns state, not another combined method on
   `RootCommitKernel`.
2. A transcript step returns the complete protocol message and state
   references.
3. Commitment compression is inside the commitment call because it precedes
   the first relevant squeeze and produces the actual public commitment.
4. Source adapters and arithmetic operations remain reusable inside the backend.
5. The proof plan names which backend keeps each state value and how any
   transfer occurs.
6. Akita adopts the model first without depending on Jolt.
7. Jolt must remove its hint and transcript-aware opening boundaries before a
   shared backend interface can be extracted cleanly.
