# Commitment implementation order

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | testable implementation and deletion order |

## Purpose

The commitment boundary is a hypothesis until a production source can travel
from source input through the first consumer of private state without exposing
CPU-shaped state. Merely wrapping the current `commit` function and storing its
`AkitaCommitmentHint` behind an integer proves almost nothing: the next opening
could immediately read the same hint back to the host.

This plan therefore has two distinct gates:

1. **same commitment output:** one backend call produces the exact existing
   `CommittedGroup` while inner rows, B rows, compression stages, and quotients
   remain backend-private;
2. **same commitment-to-opening flow:** the first opening/relation call consumes
   the resulting `CommittedGroupWithState` without a witness-sized host readback
   and preserves the exact proof and transcript event stream.

The new boundary is accepted only after both gates pass. The old commitment
surface is supplanted only after the deletion gate at the end of this file.

## Current path

The current root path is concentrated in these locations:

| Responsibility | Current source |
|---|---|
| request layout, schedule selection, source-value checks | `akita-prover/src/api/commitment.rs::resolve_commit_params` |
| source-specialized A commitment | `compute::RootCommitKernel::commit_inner_group` |
| host validation/decomposition and B commitment | `api/commitment/inner_outer.rs` |
| full compression plan and map loop | `api/commitment/compression.rs`, `compute/compression.rs` |
| public message plus CPU-shaped retained value | `api/commitment.rs::CommitOutput` |
| independently supplied hint/source binding | `types/opening_data.rs::ProverOpeningData` |
| first production code that reads the hint | `protocol/ring_relation.rs::RingRelationProver::new` |
| retained hint propagated into later relation work | `protocol/ring_relation_witness.rs::RingRelationGroupWitness` |

One logical commitment currently invokes at least three backend interfaces:
source-specific inner commitment, B digit rows, and one compression
kernel invocation per physical compression map. The characterization test in
`akita-pcs/tests/commitment_contract.rs` records this shape. These are useful
private calls inside a CPU backend; they are the wrong remote or protocol
boundary.

## Stage C0: record current behavior before changing types

This stage MUST NOT add a new public backend API.

Implement:

- count current inner, outer, and compression backend invocations;
- pin exact `CommittedGroup` bytes, proof serialization, and
  `LoggingTranscript` events;
- record host-visible bytes returned by each current commitment stage;
- cover dense, one-hot, a downstream/Jolt-like custom source, a multi-slice
  profile, and a grouped opening;
- preserve malformed-profile and out-of-contract source rejection points.

Exit gate:

- the tests distinguish backend calls from private operation and kernel calls;
- base measurements are reproducible in release mode;
- no new abstraction exists merely to make the counters look smaller.

## Stage C1: create one checked commitment plan

Turn `resolve_commit_params` into the one authoritative function that turns
public commitment context into a checked, backend-independent request. This is a
protocol-owned value, not a backend trait and not a serializable RPC payload by
default.

The checked request MUST contain or bind:

- setup identity and exact matrix capacity;
- group identity and polynomial layout;
- source class and accepted centered interval;
- A and B geometry;
- outer slice geometry;
- the complete commitment compression plan;
- the terminal protocol-message geometry.

Move all checks that can fail without reading source coefficients into this
function. Keep coefficient checks in source input, because a declaration
alone cannot prove the source satisfies its interval.

Differential tests MUST feed the old execution body and the new CPU
implementation from the same checked request. Delete duplicated sizing or plan
derivation immediately; do not retain `resolve_commit_params` as a forwarding
alias.

Exit gate:

- there is one source of truth for every A/B/compression dimension;
- malformed requests fail before arithmetic;
- source values are still checked before any public message is returned;
- no proof or commitment byte changes.

## Stage C2: build an end-to-end prototype through the first consumer

This is the first end-to-end design test. It SHOULD be developed on a
dependent branch and MUST NOT be merged as an API that only changes commitment
to return a state reference.

An **end-to-end prototype** is the smallest executable path that tests every
proposed boundary. It starts at source input, executes the complete commitment
call, carries a state reference across transcript absorption and challenge
derivation, executes the first relation/opening call,
and produces a message accepted by the verifier. It is deliberately narrow in
source and schedule coverage, but it is not a mock of the boundary itself.

The end-to-end prototype proves that the components compose. It does not establish
production coverage, persistence, optimized device selection, or final public API
ergonomics. A test that calls the current `commit`, receives an
`AkitaCommitmentHint`, and then hides that hint in a map is not a valid
prototype because the witness-shaped value has already crossed the proposed
boundary.

Introduce only the state types needed by this path:

```rust,ignore
BackendInstanceId
BackendStateStore
ProverSession
BackendStateRef<CommitmentState>
CommittedGroupWithState<F>
```

`CommittedGroupWithState` lets protocol code borrow its public `CommittedGroup`
and keeps its state reference and link metadata private. The backend owns a
state store that rejects old references; a prover session borrows long-lived
commitment state from that store for one proof.
The first CPU backend may store the current source, inner rows, packed
compression stages, and quotients in any convenient private representation.
No bound on that private state may leak into the `CommittedGroupWithState` API.

Implement two backend calls on the same backend instance and state store:

```text
checked commitment plan + source input
    -> CommittedGroupWithState

prover session + opening request + CommittedGroupWithState + post-absorb challenges
    -> first relation/opening protocol message + relation state reference
```

The second call must cover the work in today's first production hint reader,
`RingRelationProver::new`: in particular recovery or reuse of the complete B
image, consumption of retained compression stages and quotients, and creation
of relation state. It need not migrate later sum-check rounds in this stage.

Provide two implementations:

1. a CPU backend that may privately reuse current algorithms;
2. a fake remote backend whose stored state is an uninspectable remote object
   identifier from the host's point of view.

The fake transport records control calls and bytes. Calling a local Rust
function that returns the current hint and then inserting it into a map does
not satisfy this stage.

Exit gate:

- exactly one backend request and response covers inner, outer, and every
  compression map;
- the commitment response contains only the protocol message and a fixed-size
  state reference;
- the first opening/relation response contains only its protocol message and a
  fixed-size state reference;
- no `RingVec`, polynomial, digit block, compression stage, quotient, or hint
  crosses the fake transport;
- CPU and fake-remote paths produce identical commitment bytes, proof bytes,
  and `LoggingTranscript` events;
- swapped, stale, consumed, wrong-backend, wrong-setup, and wrong-plan references
  fail before transcript mutation;
- source streaming retains dense and one-hot specialization; if a universal
  source stream forces one-hot expansion, revise the input design rather
  than accepting the regression.

## Stage C3: replace the public API

Once C2 passes, replace the public root API and its first consumer together.
This is intentionally breaking.

The replacement must be all-at-once on the main branch, not necessarily during
development or review.
C3 MAY use multiple reviewable commits, a draft pull request, or a series of
dependent branches. Intermediate versions that expose both public APIs or
retain temporary adapters MUST NOT merge independently. The final merged tree
MUST contain one public commitment path and no compatibility layer.

C3 should be broad in call-site coverage but shallow in new design. If it still
needs to decide state-reference identity, backend selection, source input,
compression scope, or the first consumer boundary, C2 has not passed and C3
MUST stop.

### Scope

C3 includes:

- ordinary in-memory root commitments through `akita_prover::commit` and
  `AkitaCommitmentScheme::commit`;
- the backend state store, prover session, state reference, and
  `CommittedGroupWithState` types proven by C2;
- the CPU backend and fake-remote test harness;
- `SelectedProverOpeningData`, `ProverOpeningData`, root preparation, and the
  work currently performed by `RingRelationProver::new` on commitment hints;
- all in-tree callers of the ordinary root API, including tests, examples, and
  benchmarks;
- dense, one-hot, and downstream custom-source tests.

C3 excludes:

- setup-prefix checkpoint migration and disk-cache format changes;
- recursive next-witness binding operations;
- later relation, ring-switch, and sum-check steps;
- extraction of a joint Jolt/Akita backend;
- unrelated performance rewrites of leaf arithmetic.

These exclusions make one revert sufficient to restore the old in-memory root
API. C3 MUST NOT perform an irreversible state or persistence migration.

### Entry gate

Do not begin the public replacement until C2 has:

- frozen the committed-group, state store, and session rules;
- matched CPU and fake-remote commitment and first-consumer outputs;
- demonstrated failure-before-transcript-mutation for invalid references;
- preserved dense and one-hot source specialization;
- recorded commitment, proof, and transcript test vectors;
- assigned reviewers for state/API safety, exact protocol/transcript output, and
  source/performance behavior.

### C3a: move the prototype types into production

Move the exact C2 session, `CommittedGroupWithState`, state store, and CPU
backend contracts into production modules. Keep the fake transport in backend
test support.
Do not redesign either contract during this step. Keep reusable inner,
digit-row, and compression operations private to the CPU backend.

Any migration-only adapter introduced in an intermediate commit MUST be
crate-private, named in the pull-request checklist, and deleted by C3d. No
public `commit_legacy`, `commit_with_hint`, or parallel commitment constructor is
permitted, even temporarily.

### C3b: migrate the producer and first consumer together

Change one dependency unit inside `akita-prover`:

- `akita_prover::commit` to return `CommittedGroupWithState` backed by the
  backend state store;
- `SelectedProverOpeningData::from_committed_claims` and
  `ProverOpeningData` to accept ordered `CommittedGroupWithState` values and
  source input, not parallel `Vec<AkitaCommitmentHint<_>>` and polynomial groups;
- root preparation to invoke the first state-consuming backend call;
- transcript code to continue borrowing only protocol messages from the
  committed groups with state.

The producer and first consumer belong in the same review step because neither
is a sound public abstraction alone. This step MUST leave no internal path that
can pair an independently supplied public message with private prover state.

### C3c: migrate the workspace callers

Change `AkitaCommitmentScheme::commit`, then migrate `akita-pcs`, integration
tests, examples, benchmarks, and downstream contract test vectors to construct
a backend and prover session, retain `CommittedGroupWithState` values, and pass
them into opening. Mechanical call-site changes SHOULD be separate commits
grouped by crate or test family so reviewers can distinguish call-site changes
from protocol changes.

All ordinary in-memory source types MUST enter through the one
`CommittedGroupWithState` API. Representation-specific adapters MAY still call
existing leaf operations privately;
C4 replaces or optimizes those adapters and covers nonstandard producers.

### C3d: delete the transitional surface and validate the tip

Delete before the replacement is mergeable:

- `CommitOutput`;
- public construction and field access for root `AkitaCommitmentHint` values;
- hint/source count-alignment machinery;
- any temporary adapter that reconstructs a hint from a reference;
- any duplicate public `commit_legacy`, `commit_with_hint`, or
  `commit_with_state` entry point.

Review the complete base-to-tip diff after deletion. Do not approve C3 from an
intermediate commit or from a commit-by-commit summary alone.

### Delivery and rollback rules

- Develop C3 as one coordinated change. A draft pull request MAY expose
  intermediate commits for review, but only the final version with temporary
  adapters removed may merge.
- Keep intermediate commits buildable when practical. Temporary private shims
  are acceptable only to make review commits buildable and MUST disappear from
  the final diff.
- Run output-equality and failure tests after the last deletion, not only before caller
  migration.
- Merge C3 as one public API change. Do not leave main with both public
  APIs while waiting for a follow-up.
- If the final gate fails after merge, revert the coordinated C3 change. Do not
  restore behavior by adding a public compatibility wrapper.

Exit gate:

- all in-tree ordinary dense and one-hot root prove/verify tests use
  `CommittedGroupWithState`;
- downstream custom-source contract tests use the new backend boundary;
- public API consumers cannot construct a message/reference mismatch;
- the fake remote crosses no witness-proportional value through the first
  consumer;
- golden commitment bytes, proof bytes, and transcript events match C0;
- the final diff contains no migration-only adapter or parallel public API;
- Jolt integration compiles without gaining a permanent Akita compatibility
  layer.

## Stage C4: migrate specialized inputs and remaining `CommittedGroup` producers

Move remaining public-root producers one representation at a time:

1. multilinear and packed Jolt trace sources;
2. sparse-unit and other specialized sources;
3. grouped and precommitted roots;
4. setup-prefix generation and import/reuse.

Each representation lands with exact commitment/proof/transcript output and the
same two backend-call gates as C2. A representation adapter is source input only;
it must not determine private prover state.

Setup-prefix reuse additionally requires the explicit checkpoint API from
`commitment-replacement.md`. Live references are never serialized. Do not
delete the old persistence format until the new versioned checkpoint can either import
the required live data or explicitly recompute it from an available source.

Exit gate:

- every producer of a public `CommittedGroup` before transcript work starts
  uses the new commitment call;
- no setup-prefix cache relies on a live `AkitaCommitmentHint` encoding;
- checkpoint mismatch is rejected before transcript mutation.

## Stage C5: delete the old commitment trait hierarchy

Delete only after C3 and C4 make the old path unreachable:

- public `RuntimeCommitSource` and `RuntimeCommitBackendFor` bundles;
- public `RootCommitKernel`, `DigitRowsComputeBackend`, and
  `CompressionComputeBackend` exposure where they serve only root commitment;
- the commit cluster in `ProverComputeStack` and `UniformProverStack`;
- delegating `CommitCluster` wrappers;
- root-commit reexports from `akita-prover` and `akita-pcs`;
- root `AkitaCommitmentHint` and its serializers;
- dead wrapper functions, tests, examples, and benchmark support code for the old
  surface.

Do not delete reusable CPU algorithms merely because their traits disappear. Move
the inner, digit-row, and bounded compression algorithms under the CPU backend
as private operations with direct call sites.

Deletion gate (all conditions are mandatory):

```text
rg "CommitOutput|AkitaCommitmentHint|RuntimeCommitBackendFor|RuntimeCommitSource" \
  crates/akita-prover crates/akita-pcs
```

has no live root-commit API or root-opening-state match; any remaining match is
classified as a distinct recursive message, explicit checkpoint migration, or
an unrelated reusable operation. In addition:

- every production source class and supported root ring dimension passes;
- grouped, precommitted, multi-slice, and setup-prefix cases pass;
- proof bytes and transcript events match the C0 test vectors;
- the fake remote reports one commitment control call and no
  witness-proportional readback through the first consumer;
- release benchmarks report commit time, peak host RSS, bytes stored by the backend,
  bytes, transferred bytes, and private operation-call counts;
- Clippy feature graphs and Jolt compatibility checks pass.

Only this gate justifies the claim that the new backend fully supplants the
current commitment backend design.

## Reasons to revise the design

Pause the replacement and revise the specification if any experiment shows that:

- a verifier-visible value must be recovered by inspecting private prover state;
- an intervening Fiat–Shamir squeeze exists inside the proposed commitment
  call;
- source values cannot be checked without prescribing private prover state;
- one-hot, packed-trace, or sparse traversal must expand to a dense host
  representation at input;
- the first private-state consumer requires an implicit backend transfer;
- proof bytes depend on backend tiling, batching, device selection, or checkpoint
  policy;
- a fixed-size reference cannot express the required sharing and lifetime rules
  without exposing representation;
- the new API merely moves the current trait hierarchy into a larger
  trait bound.

These are design failures, not reasons to add compatibility wrappers.
