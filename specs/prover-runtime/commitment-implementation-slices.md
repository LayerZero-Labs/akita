# Commitment implementation slices

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | falsifiable implementation and deletion order |

## Purpose

The commitment boundary is a hypothesis until a production source can travel
from source ingress through the first retained-state consumer without exposing
CPU-shaped state. Merely wrapping the current `commit` function and storing its
`AkitaCommitmentHint` behind an integer proves almost nothing: the next opening
could immediately read the same hint back to the host.

This plan therefore has two distinct gates:

1. **commitment parity:** one semantic invocation produces the exact existing
   `CommittedGroup` while inner rows, B rows, compression stages, and quotients
   remain runtime-private;
2. **state-chain parity:** the first opening/relation operation consumes the
   resulting artifact without a witness-sized host readback and preserves the
   exact proof and transcript event stream.

The new boundary is accepted only after both gates pass. The old commitment
surface is supplanted only after the deletion gate at the end of this file.

## Current executable seam

The current root path is concentrated in these locations:

| Responsibility | Current source |
|---|---|
| request layout, schedule selection, source admission | `akita-prover/src/api/commitment.rs::resolve_commit_params` |
| source-specialized A commitment | `compute::RootCommitKernel::commit_inner_group` |
| host validation/decomposition and B commitment | `api/commitment/inner_outer.rs` |
| full compression plan and map loop | `api/commitment/compression.rs`, `compute/compression.rs` |
| public message plus CPU-shaped retained value | `api/commitment.rs::CommitOutput` |
| independently supplied hint/source binding | `types/opening_data.rs::ProverOpeningData` |
| first material hint consumer | `protocol/ring_relation.rs::RingRelationProver::new` |
| retained hint propagated into later relation work | `protocol/ring_relation_witness.rs::RingRelationGroupWitness` |

One logical commitment currently invokes at least three backend capability
families: source-specific inner commitment, B digit rows, and one compression
kernel invocation per physical compression map. The characterization test in
`akita-pcs/tests/commitment_contract.rs` records this shape. These are useful
local form calls inside a CPU runtime; they are the wrong remote or protocol
boundary.

## Slice C0: characterize before changing types

This slice MUST NOT add a new public runtime API.

Implement:

- count current inner, outer, and compression capability invocations;
- pin canonical `CommittedGroup`, proof serialization, and
  `LoggingTranscript` events;
- record host-visible bytes returned by each current commitment stage;
- cover dense, one-hot, a downstream/Jolt-like custom source, a multi-slice
  profile, and a grouped opening;
- preserve malformed-profile and out-of-contract source rejection points.

Exit gate:

- the tests distinguish semantic calls from private form/kernel calls;
- base measurements are reproducible in release mode;
- no new abstraction exists merely to make the counters look smaller.

## Slice C1: extract the validated request

Turn `resolve_commit_params` into the one canonical compiler from public
commitment context to a checked, backend-independent request. This is a
protocol-owned value, not a backend trait and not a serializable RPC payload by
default.

The checked request MUST contain or bind:

- setup identity and exact matrix capacity;
- group identity and polynomial layout;
- source class and accepted centered interval;
- A and B geometry;
- outer slice geometry;
- the complete commitment compression plan;
- the canonical terminal message geometry.

Move all checks that can fail without reading source coefficients into this
compiler. Keep coefficient admission in source ingress, because a declaration
alone cannot prove the source satisfies its interval.

Differential tests MUST feed the old execution body and the reference CPU
executor from the same checked request. Delete duplicated sizing or plan
derivation immediately; do not retain `resolve_commit_params` as a forwarding
alias.

Exit gate:

- there is one source of truth for every A/B/compression dimension;
- malformed requests fail before arithmetic;
- source admission still occurs before any public message is returned;
- no proof or commitment byte changes.

## Slice C2: build the walking skeleton through the first consumer

This is the first architectural experiment. It SHOULD be developed on a
stacked branch and MUST NOT be merged as a commitment-only handle API.

A **walking skeleton** is the thinnest executable path that crosses every
proposed architectural layer. For this cutover, it starts at source ingress,
executes the complete commitment epoch, carries opaque state across transcript
absorption and challenge derivation, executes the first relation/opening epoch,
and produces a message accepted by the verifier. It is deliberately narrow in
source and schedule coverage, but it is not a mock of the boundary itself.

The walking skeleton proves that the components compose. It does not establish
production coverage, persistence, optimized placement, or final public API
ergonomics. A test that calls the current `commit`, receives an
`AkitaCommitmentHint`, and then hides that hint in a map is not a walking
skeleton because the witness-shaped value has already crossed the proposed
boundary.

Introduce the minimum lifecycle substrate:

```rust,ignore
ProverRuntimeOwner
ProofSession
StateHandle<CommittedGroupState>
CommittedArtifact<F>
```

`CommittedArtifact` exposes its canonical `CommittedGroup` by borrow and keeps
its handle and binding private. The session owns a generational state store.
The first CPU implementation may store the current source, inner rows, packed
compression stages, and quotients in any convenient private representation.
No bound on that private state may leak into the handle or artifact API.

Implement two semantic operations on the same session:

```text
validated commitment request + source ingress
    -> CommittedArtifact

opening request + bound CommittedArtifact + post-absorb challenges
    -> first canonical relation/opening message + opaque relation state
```

The second operation must cover the work in today's first real hint consumer,
`RingRelationProver::new`: in particular recovery or reuse of the complete B
image, consumption of retained compression stages and quotients, and creation
of relation state. It need not migrate later sum-check rounds in this slice.

Provide two implementations:

1. a CPU reference session that may privately reuse current algorithms;
2. a fake-remote session whose stored state is an uninspectable remote object
   identifier from the host's point of view.

The fake transport records control calls and bytes. Calling a local Rust
function that returns the current hint and then inserting it into a map does
not satisfy this slice.

Exit gate:

- exactly one control invocation covers inner, outer, and every compression
  map;
- the commitment response contains only the canonical message and fixed-size
  opaque metadata;
- the first opening/relation response contains only its canonical message and
  fixed-size opaque metadata;
- no `RingVec`, polynomial, digit block, compression stage, quotient, or hint
  crosses the fake transport;
- CPU and fake-remote paths produce identical commitment bytes, proof bytes,
  and `LoggingTranscript` events;
- swapped, stale, consumed, wrong-owner, wrong-setup, and wrong-plan handles
  fail before transcript mutation;
- source streaming retains dense and one-hot specialization; if a universal
  source stream forces one-hot expansion, revise the ingress design rather
  than accepting the regression.

## Slice C3: run the public cutover train

Once C2 passes, replace the public root API and its first consumer together.
This is intentionally breaking.

“Atomic” applies to the public merge boundary, not to development or review.
C3 MAY use multiple reviewable commits, a draft pull request, or a private
stack. Intermediate tips that expose both public APIs or retain temporary
adapters MUST NOT merge independently. The final merged tree MUST contain one
public commitment path and no compatibility facade.

C3 should be broad in call-site coverage but shallow in new design. If it still
needs to decide handle identity, ownership, source ingress, compression scope,
or the first consumer boundary, C2 has not passed and C3 MUST stop.

### Scope

C3 includes:

- ordinary in-memory root commitments through `akita_prover::commit` and
  `AkitaCommitmentScheme::commit`;
- the proof-session, artifact, and state-binding types proven by C2;
- the CPU runtime and fake-remote conformance harness;
- `SelectedProverOpeningData`, `ProverOpeningData`, root preparation, and the
  work currently performed by `RingRelationProver::new` on commitment hints;
- all in-tree callers of the ordinary root API, including tests, examples, and
  benchmarks;
- dense, one-hot, and downstream custom-source conformance coverage.

C3 excludes:

- setup-prefix checkpoint migration and disk-cache format changes;
- recursive next-witness binding operations;
- later relation, ring-switch, and sum-check epochs;
- extraction of a joint Jolt/Akita runtime;
- unrelated performance rewrites of leaf arithmetic.

These exclusions make one revert sufficient to restore the old in-memory root
API. C3 MUST NOT perform an irreversible state or persistence migration.

### Entry gate

Do not begin the public cutover until C2 has:

- frozen the artifact and session semantics;
- passed CPU/fake-remote commitment and first-consumer parity;
- demonstrated failure-before-transcript-mutation for invalid handles;
- preserved dense and one-hot source specialization;
- recorded golden commitment, proof, and transcript fixtures;
- assigned reviewers for ownership/API safety, protocol/transcript parity, and
  source/performance behavior.

### C3a: promote the proven runtime substrate

Promote the exact C2 session, artifact, state-store, and CPU runtime contracts
into production modules. Keep the fake transport in conformance test support.
Do not redesign either contract during this step. Keep reusable inner,
digit-row, and compression forms private to the CPU runtime.

Any migration-only adapter introduced in an intermediate commit MUST be
crate-private, named in the pull-request checklist, and deleted by C3d. No
public `commit_legacy`, `commit_with_hint`, or parallel artifact constructor is
permitted, even temporarily.

### C3b: migrate the producer and first consumer together

Change one dependency unit inside `akita-prover`:

- `akita_prover::commit` to return a bound committed artifact from a proof
  session;
- `SelectedProverOpeningData::from_committed_claims` and
  `ProverOpeningData` to accept ordered artifacts and source ingress, not
  parallel `Vec<AkitaCommitmentHint<_>>` and polynomial groups;
- root preparation to invoke the first state-consuming semantic operation;
- transcript code to continue borrowing only canonical messages from the
  artifacts.

The producer and first consumer belong in the same review step because neither
is a sound public abstraction alone. This step MUST leave no internal path that
can pair an independently supplied public message with retained state.

### C3c: migrate the workspace callers

Change `AkitaCommitmentScheme::commit`, then migrate `akita-pcs`, integration
tests, examples, benchmarks, and downstream contract fixtures to construct a
proof session, retain committed artifacts, and pass those artifacts into
opening. Mechanical call-site changes SHOULD be separate commits grouped by
crate or test family so reviewers can distinguish API churn from protocol
changes.

All ordinary in-memory source types MUST enter through the one artifact API.
Representation-specific adapters MAY still call existing leaf forms privately;
C4 replaces or optimizes those adapters and covers nonstandard producers.

### C3d: delete the transitional surface and validate the tip

Delete before the cutover is mergeable:

- `CommitOutput`;
- public construction and field access for root `AkitaCommitmentHint` values;
- hint/source count-alignment machinery;
- any temporary adapter that reconstructs a hint from a handle;
- any duplicate public `commit_legacy`, `commit_with_hint`, or
  `commit_artifact` entry point.

Review the complete base-to-tip diff after deletion. Do not approve C3 from an
intermediate commit or from a commit-by-commit summary alone.

### Delivery and rollback rules

- Develop C3 as one coordinated change surface. A draft pull request MAY expose
  intermediate commits for review, but only the adapter-free tip may merge.
- Keep intermediate commits buildable when practical. Temporary private shims
  are acceptable only to make review commits buildable and MUST disappear from
  the final diff.
- Run parity and failure tests after the last deletion, not only before caller
  migration.
- Merge C3 as one public state transition. Do not leave main with both public
  APIs while waiting for a follow-up.
- If the final gate fails after merge, revert the coordinated C3 change. Do not
  restore behavior by adding a public compatibility wrapper.

Exit gate:

- all in-tree ordinary dense and one-hot root prove/verify tests use artifacts;
- downstream custom-source contract tests use the semantic runtime boundary;
- public API consumers cannot construct a message/handle mismatch;
- the fake remote crosses no witness-proportional value through the first
  consumer;
- golden commitment bytes, proof bytes, and transcript events match C0;
- the final diff contains no migration-only adapter or parallel public API;
- Jolt integration compiles without gaining a permanent Akita compatibility
  facade.

## Slice C4: migrate specialized ingress and remaining `CommittedGroup` producers

Move remaining public-root producers one representation at a time:

1. multilinear and packed Jolt trace sources;
2. sparse-unit and other specialized sources;
3. grouped and precommitted roots;
4. setup-prefix generation and import/reuse.

Each representation lands with commitment/proof/transcript parity and the same
two semantic-call gates as C2. A representation adapter is source ingress only;
it must not determine retained state.

Setup-prefix reuse additionally requires the explicit checkpoint API from
`commitment-cutover.md`. Live handles are never serialized. Do not delete the
old persistence format until the new versioned checkpoint can either import
the required live data or explicitly recompute it from an available source.

Exit gate:

- every pretranscript/public `CommittedGroup` producer uses the semantic
  commitment operation;
- no setup-prefix cache relies on a live `AkitaCommitmentHint` encoding;
- checkpoint mismatch is rejected before transcript mutation.

## Slice C5: delete the old commitment backend lattice

Delete only after C3 and C4 make the old path unreachable:

- public `RuntimeCommitSource` and `RuntimeCommitBackendFor` bundles;
- public `RootCommitKernel`, `DigitRowsComputeBackend`, and
  `CompressionComputeBackend` exposure where they serve only root commitment;
- the commit cluster in `ProverComputeStack` and `UniformProverStack`;
- delegating `CommitCluster` scaffolding;
- root-commit reexports from `akita-prover` and `akita-pcs`;
- root `AkitaCommitmentHint` and its serializers;
- dead wrapper functions, tests, examples, and benchmark plumbing for the old
  surface.

Do not delete reusable CPU forms merely because their traits disappear. Move
the inner, digit-row, and bounded compression algorithms under the CPU runtime
as private forms with direct call sites.

Deletion gate (all conditions are mandatory):

```text
rg "CommitOutput|AkitaCommitmentHint|RuntimeCommitBackendFor|RuntimeCommitSource" \
  crates/akita-prover crates/akita-pcs
```

has no live root-commit API or root-opening-state match; any remaining match is
classified as a distinct recursive message, explicit checkpoint migration, or
unrelated reusable form. In addition:

- every production source class and supported root ring dimension passes;
- grouped, precommitted, multi-slice, and setup-prefix cases pass;
- proof bytes and transcript events match the C0 fixtures;
- the fake remote reports one commitment control call and no
  witness-proportional readback through the first consumer;
- release benchmarks report commit time, peak host RSS, backend-resident
  bytes, transferred bytes, and private form-call counts;
- Clippy feature graphs and Jolt compatibility checks pass.

Only this gate justifies the claim that the new runtime fully supplants the
current commitment backend design.

## Kill criteria

Pause the cutover and revise the specification if any experiment shows that:

- a verifier-visible value must be recovered by inspecting retained state;
- an intervening Fiat–Shamir squeeze exists inside the proposed commitment
  call;
- source admission cannot be enforced without prescribing retained state;
- one-hot, packed-trace, or sparse traversal must expand to a dense host
  representation at ingress;
- the first retained-state consumer requires an implicit owner transfer;
- proof bytes depend on runtime tiling, batching, placement, or checkpoint
  policy;
- a fixed-size handle cannot express the required sharing and lifetime rules
  without exposing representation;
- the semantic API merely moves the current capability lattice into a larger
  trait bound.

These are design failures, not reasons to add compatibility wrappers.
