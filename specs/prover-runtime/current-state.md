# Current-state comparison: Akita and Jolt

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | pinned evidence and design rationale |

## Scope and source pins

This comparison records the implementations inspected for the prover-runtime
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

Jolt has a better proof-scoped execution model than Akita and has nearly the
right Fiat–Shamir boundary for sum-check rounds. It has not generalized that
model to commitment, opening, ZK retained state, or its packed Akita path.

Akita has strong checked protocol plans, runtime ring-dimension dispatch, exact
setup-cache ownership, and representation-specialized CPU kernels. Its public
backend hierarchy nevertheless describes implementation fragments rather than
complete protocol messages. It also standardizes a CPU-shaped commitment hint
and assumes that state can move freely between independently routed operation
clusters.

The target should combine Jolt's session and round-transition ideas with
Akita's checked plans and reference kernels, while fixing the retained-state and
whole-message boundaries in both designs.

## Akita today

### Current layers

Akita currently has four overlapping abstraction layers:

1. `ComputeBackendSetup<F>` and primitive row-operation traits in
   `crates/akita-prover/src/compute/backend.rs`;
2. source-typed fused operation traits such as `RootCommitKernel`,
   `OpeningFoldKernel`, and `RingSwitchRelationKernel` in
   `crates/akita-prover/src/compute/kernels.rs`;
3. source/ring-dimension capability bundles in
   `crates/akita-prover/src/compute/poly.rs` and
   `crates/akita-prover/src/compute/runtime_capabilities.rs`;
4. commit/opening/tensor/ring-switch routing through `OperationCtx`,
   `ProverComputeStack`, and `LevelProveStacks` in
   `crates/akita-prover/src/compute/stack.rs`.

These layers solved real earlier problems: CPU NTT caches no longer leak from
setup types, high-level protocol storage is runtime-dimensioned, source-specific
commit kernels preserve dense and one-hot structure, and proof execution can
select different prepared contexts by operation cluster and fold level.

They do not define one coherent remote-backend contract.

### Commitment is split at implementation boundaries

The canonical commitment entry point in
`crates/akita-prover/src/api/commitment.rs` performs:

1. source admission and frozen-plan resolution;
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
implementation intermediate and should not define the semantic backend
boundary.

### `AkitaCommitmentHint` fixes live state representation

`crates/akita-types/src/proof/hints.rs` defines `AkitaCommitmentHint<F>` as:

- one `RingVec<F>` of A-native inner rows per committed polynomial;
- one runtime ring dimension;
- packed outer-compression stage bytes; and
- outer-compression quotient `RingVec<F>` values.

It is cloneable, equality-comparable, validated, serialized, and deserialized
as an Akita protocol-adjacent type. Later prover code calls
`inner_rows`, `into_rows`, `outer_compression_witness`, and
`outer_compression_quotients` directly.

This representation works for CPU replay and disk artifacts. It is not a valid
universal live-state contract. A GPU may prefer digits or transforms; a remote
backend may retain an object identifier; another backend may retain the source
and recompute; and a joint Jolt/Akita runtime may already own a different
resident witness representation.

The type also mixes two semantic categories:

- compression stages, quotients, and inner rows used only to continue proving
  are retained state; but
- the terminal next-witness inner-state binding is read from `inner_rows` and
  absorbed into the transcript in
  `crates/akita-prover/src/protocol/core/fold/mod.rs`.

The latter is a canonical prover message and must be modeled separately from
private retained state.

### Capability inheritance contradicts operation ownership

`DigitRowsComputeBackend<F>` inherits `CompressionComputeBackend<F>`. A backend
cannot advertise generic negacyclic digit-row support without also implementing
commitment/relation compression. `RuntimeCommitBackendFor<F, P>` then combines
that inherited surface with one `RootCommitKernel` bound for every supported
runtime ring dimension and every source view.

Consequences include:

- primitive capabilities imply unrelated protocol duties;
- one backend implementation is coupled to the cross-product of source views
  and runtime dimensions;
- a full commitment capability is inferred from fragment traits rather than
  represented as one operation;
- missing fusion can be hidden behind a type-correct capability bundle;
- source ingress, reusable forms, protocol epochs, and retained state have no
  distinct types.

### The four-cluster stack assumes free state movement

`ProverComputeStack<C, O, T, R>` independently routes commitment, opening,
tensor, and ring-switch work. `TieredProveStacks` additionally changes the stack
by fold range. The stack tracks prepared setup and NTT cache ownership well, but
it has no proof-scoped state owner or transfer contract.

A commitment may execute on one backend while opening or ring-switch executes
on another. Protocol code bridges the operations with host values and
`AkitaCommitmentHint`. This makes CPU materialization the implicit universal
transfer format and prevents a planner from rejecting an impossible or
expensive state transition before proving.

### Large host-shaped outputs cross operation seams

Several kernel APIs return `Vec`, `Vec<Vec<_>>`, `RingVec`, partial buffers, or
fallback markers intended for host orchestration. These are useful CPU form
interfaces. They become expensive or impossible as the public remote boundary
because they expose witness-proportional intermediate state and require the
host to decide the next computation.

## What Akita should preserve

The cutover should preserve and move beneath the new semantic boundary:

- frozen and validated protocol operation plans;
- runtime ring-dimension dispatch at typed arithmetic leaves;
- exact public-matrix derivation and cache-prefix requirements;
- source-specific dense, one-hot, and recursive CPU algorithms;
- scalar/reference implementations and differential tests;
- setup identity validation and checked resource accounting;
- `CommittedGroup` as the canonical public commitment message;
- verifier independence and the no-panic contract.

The cutover should remove or internalize:

- `AkitaCommitmentHint` as a public live-state type;
- the public source/ring-dimension backend supertrait ladders;
- the implication that digit-row support includes compression support;
- host-visible inner/decomposition/compression intermediates between epochs;
- proof orchestration's ability to downcast or inspect backend state;
- implicit state transfer through CPU field vectors;
- public routing APIs whose only purpose is to reconstruct the old hierarchy.

## Jolt today

### Runtime registry and proof session

Jolt's
[`backend.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-kernels/src/backend.rs)
defines `JoltBackend<F, PCS>` as a runtime registry of boxed, object-safe
operation slots. It also defines `ProofSession`, a proof-lifetime store keyed by
backend-private Rust types.

This gives Jolt capabilities Akita lacks:

- reference and optimized slots can be composed as values;
- multiple backend configurations can coexist in one binary;
- retained tables and cross-stage carries have a proof lifetime;
- witness uploads and shared tables have a natural reuse location;
- dynamic dispatch cost is correctly treated as negligible for heavy work.

`ProofSession` is a strong implementation idea but not the desired public
contract. Its `TypeId -> Box<dyn Any>` interface gives protocol and slot code no
typed owner, phase, generation, or transfer guarantee. Missing carries become
proof-time errors, and one concrete Rust type is one global key. Safe type
erasure may remain internal behind typed semantic handles.

### Relation-owned requests and generated drivers

`PrepareKernel<F, R>` receives `ProverInputs<F, R>`, where `R` is the verifier's
concrete sum-check relation and owns the relevant dimensions, claims, points,
and challenges. Generated stage drivers assemble those inputs and hard-check
the resulting claims against verifier-derived relations.

This is an important single-source-of-truth property. Akita epoch requests
should likewise contain canonical checked plans, not backend-specific request
structs that repeat geometry.

Jolt's naive relation interpreter is the reference implementation and semantic
anchor for optimized paths. The design intends a small generic form vocabulary
below per-relation selection, although the current optimized implementation
still contains substantial relation-specific preparation code and has not
fully realized that form layer.

### Sum-check already approaches an epoch transition

Jolt's
[`prover.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-sumcheck/src/prover.rs)
defines `ProveRounds` so that `prove_round` receives a member's previous active
challenge and returns its current round polynomial. The backend can fuse
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
records the intended member-group extension: one co-located group should return
the pre-folded round polynomial.

### Commitment and opening retain the old hint model

Jolt's
[`commitment.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-kernels/src/commitment.rs)
lets one `CommitWitness` call batch and stream all witness commitments, which
is a better macro boundary than Akita's current inner/outer split. Its output
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
returns `MultilinearPoly` handles from the joint-opening slot so the separate
PCS can drive lazy folding. The handles may be device-backed, but the split
still prevents one runtime from owning the whole opening epoch. Host code also
assembles, clones, and reorders commitment hints before invoking the
transcript-aware PCS.

### Other retained-state leaks

The committed/ZK sum-check recorder returns `CommittedSumcheckWitness` beside
the wire proof. That retained witness has a canonical CPU collection shape and
lives outside `ProofSession` even though it exists only for later prover work.

Jolt's current implementation also performs significant ZK commitment work in
the recorder rather than through the general runtime seam. This contradicts the
stronger claim that all heavy prover compute is behind `jolt-kernels`.

### Jolt's clean-slate direction is only partially implemented

The clean-slate design is valuable prior art, but it is important not to treat
its target as current implementation:

| Intended property | Current evidence at the pinned SHA | Consequence for Akita |
|---|---|---|
| Heavy transcript-free compute is behind the kernel seam | ZK recording, PCS proving, Dory opening, and packed Akita stage 0/8 still bypass it | Audit actual call paths, not registry coverage |
| A small form vocabulary supports many relations | The optimized path has roughly two dozen relation-specific `PrepareKernel` implementations and no implemented general form/descriptor layer | Treat the form vocabulary as a hypothesis to validate with a second backend |
| Full verifier replay is a debug cross-check | [`blindfold.rs`](https://github.com/a16z/jolt/blob/e789b9f5f418bdc8beac196a11324b949c36f8cf/crates/jolt-prover/src/blindfold.rs) still replays the verifier to recover protocol lowering | Keep one canonical driver/relation source rather than replay-based coupling |
| Proof-session slots can be mixed safely | `TypeId` keys distinguish types, not backend instances, state generations, or several values of one type | Use typed owner/domain/generation handles |
| Capability misses fail while constructing a plan | An incompatible carry can surface as a missing typed `take` while proving | Plan the whole state chain before transcript mutation |
| Round scheduling is transport-ready | The production scheduler is sequential and its interface returns per-member rather than final group messages | Lift scheduling below a group-level epoch response |

Jolt's current CPU-shaped cross-stage carries make reference and optimized
slots interoperable, but that interoperability comes from fixing the transfer
representation. The Akita target instead makes state compatibility explicit
and rejects unsupported owner changes during planning.

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
joint runtime.

## What Akita should borrow from Jolt

- one proof-scoped session for uploads, retained state, memory pools, and
  cross-epoch carries;
- runtime value-based capability selection rather than one monomorphic backend
  supertrait at the top-level prover API;
- object-safe macro operations where arithmetic or RPC dominates dispatch;
- canonical verifier relation or checked plan as the semantic request;
- generated/shared protocol drivers where this prevents prover/verifier order
  drift;
- prior-challenge-to-next-message round transitions;
- a scheduler/transport seam below host-owned transcript logic;
- reference interpretation as the equivalence anchor;
- backend-invariant proof bytes and explicit observability;
- a small form vocabulary plus optional fused escape hatches.

## What Akita should not copy

- `OpeningHint` as a generic associated type with clone/default bounds;
- transcript-aware PCS proving;
- `TypeId -> Any` as the protocol-facing state API;
- one host readback per sum-check member per round;
- returning polynomial handles when the next PCS step can remain in the same
  semantic opening epoch;
- mixing wire proof and retained ZK witness in one returned carrier;
- a fixed registry field for every relation as the eventual common Akita/Jolt
  runtime API;
- claims that a generic form layer exists before the implementation and tests
  actually establish it.

## Comparison rubric

| Property | Akita at the pinned SHA | Jolt at the pinned SHA | Target |
|---|---|---|---|
| Protocol/transcript authority | Host-owned, but compute and transcript orchestration are interleaved | Strong for sum-check; PCS and ZK recording still receive transcripts | Driver only |
| Live retained state | Public serialized CPU-shaped hint | Session for kernels, fixed hints/witnesses elsewhere | Typed opaque handles into arbitrary runtime-owned state |
| Commitment boundary | Inner, outer, and compression orchestrated on host | Whole witness commit call, fixed hint output | Full source-to-terminal-message epoch |
| Sum-check boundary | Primitive/stage-specific host calls | Prior challenge to member polynomial | Prior challenge to final batched message |
| Opening boundary | Host rebuilds from hints and source views | Backend returns polynomials, PCS drives opening | State handle and claims to canonical opening message |
| Runtime selection | Static capability ladders and four operation clusters | Runtime boxed slots with reference composition | Planned semantic epochs over a common session |
| State movement | Implicit CPU materialization | Session-local for some slots, hint cloning elsewhere | Owner-affine; explicit transfer/checkpoint |
| Reference semantics | Checked plans and CPU kernels, distributed orchestration | Verifier relation plus naive interpreter | Checked protocol plan plus canonical reference epoch executor |
| Jolt/Akita integration | Akita API fixes state shape | Packed stage 0/8 bypass Jolt session | Shared runtime later; separate drivers and messages |

## Resulting design decisions

The evidence supports the following decisions:

1. The new Akita boundary is a proof-scoped runtime, not another fused method on
   `RootCommitKernel`.
2. A semantic epoch returns the complete canonical message and opaque retained
   state.
3. Commitment compression is inside the commitment epoch because it precedes
   the first relevant squeeze and produces the actual public commitment.
4. Source adapters and arithmetic forms remain reusable below the epoch.
5. Runtime composition includes explicit state owner and transfer planning.
6. Akita adopts the model first without depending on Jolt.
7. Jolt must remove its hint and transcript-aware opening boundaries before a
   common runtime can be extracted cleanly.
