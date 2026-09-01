# Commitment backend cutover

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | first implementation slice |

## Decision

Akita MUST replace its fragment-level commitment backend surface with one
semantic commitment epoch. The epoch MUST include source admission, inner
commitment, outer decomposition and commitment, and the complete commitment
compression chain.

Compression is inside this boundary because no Fiat–Shamir challenge separates
it from the rest of commitment. The protocol owns the checked compression plan
and the final `CommittedGroup` encoding. The runtime owns every intermediate
representation and all private state retained for opening.

This cutover is intentionally breaking. It does not preserve
`AkitaCommitmentHint`, the four-cluster stack, or the existing public backend
trait hierarchy.

## Current call and materialization path

The current path is approximately:

```text
protocol
  │
  ├─ commit_inner_group
  │    └─ host Vec<CommitInnerWitness>
  ├─ host validation and decomposition
  ├─ digit_rows for B
  │    └─ host outer rows
  ├─ compression_rows_products for map 0
  │    └─ host cyclic and negacyclic images
  ├─ host quotient construction
  ├─ compression_rows_products for map 1
  │    └─ host cyclic and negacyclic images
  └─ host CommittedGroup + AkitaCommitmentHint
```

The normal two-map commitment therefore has at least four independently
observable compute calls and several witness-proportional readbacks. PR #457
fuses the first portion of this path, but deliberately stops before compression
and still returns host rows required by `AkitaCommitmentHint`. That is a useful
local optimization, not the final boundary.

The replacement path is:

```text
protocol-owned ValidatedCommitGroupPlan + source provider
  │
  └─ one semantic commitment epoch
       ├─ arbitrary backend-private intermediates
       ├─ canonical CommittedGroup message
       └─ opaque StateHandle<CommittedGroupState>
```

## Semantic request

The request MUST be produced from the same schedule, geometry, source-class,
magnitude, and setup-identity checks used by the verifier-facing protocol. It
MUST NOT duplicate unchecked dimensions for backend convenience.

The exact Rust decomposition is implementation work, but the semantic shape is:

```rust,ignore
pub struct ValidatedCommitGroupPlan<F> {
    pub group_id: GroupId,
    pub public_geometry: CommitmentGeometry,
    pub inner: InnerCommitmentPlan<F>,
    pub outer: OuterCommitmentPlan<F>,
    pub compression: CompressionChainPlan<F>,
    pub source_contract: ValidatedSourceContract,
}

pub trait CommitmentSource<F> {
    fn metadata(&self) -> CommitmentSourceMetadata;
    fn stream(&self, sink: &mut dyn CommitmentSourceSink<F>)
        -> Result<(), SourceError>;
}
```

`CommitmentSource` is an ingress contract, not retained state. Dense,
multilinear, one-hot, packed-trace, sparse-unit, and setup-prefix
adapters MAY retain their specialized traversal APIs beneath this boundary. A
runtime MAY upload, stream, borrow, or directly traverse the source. It MUST
not require that the source representation become its opening-state
representation.

The validated plan is protocol configuration. Backend tiling, device choice,
stream size, cache policy, thread count, and recomputation policy are runtime
configuration and MUST NOT affect public commitment bytes.

## Semantic result

The ordinary result contains exactly two categories:

```rust,ignore
pub struct CommittedArtifact<F> {
    message: CommittedGroup<F>,
    state: StateHandle<CommittedGroupState>,
    binding: CommitmentArtifactBinding,
}
```

`message` is the canonical value consumed by proof construction, serialization,
and transcript encoding. `state` is a capability to backend-owned private data.

The state handle MUST:

- identify its state owner, state domain, slot, generation, and semantic kind;
- be opaque to protocol code;
- have no field, ring, polynomial, collection, serialization, equality, or
  `Default` bound;
- fail deterministically when stale, consumed, used with the wrong owner, or
  used in the wrong state domain;
- support explicit shared borrowing only where the runtime declares the state
  immutable and shareable.

`CommittedArtifact` MUST have no public constructor and no API that accepts a
replacement message or handle. The runtime constructs it atomically and binds
the canonical commitment, group ID, checked plan identity, setup identity,
state owner, and state generation. Opening and checkpoint operations consume or
borrow the artifact as a unit and MUST reject a binding mismatch before
transcript initialization or mutation. Protocol code may borrow `message` for
serialization and absorption; it cannot pair that message with another
same-kind state handle.

The binding is operational metadata and MUST NOT affect proof or transcript
bytes. It MAY be a canonical digest plus runtime-authenticated state record or
another construction with equivalent swap resistance and deterministic
validation.

The handle MUST NOT provide `inner_rows`, `into_rows`, compression-witness, or
quotient accessors. A CPU runtime MAY store the exact current hint fields
internally during the first migration; that private choice is not a contract.

## Distinct recursive next-witness messages

Current recursive and suffix paths read the final inner row from
`AkitaCommitmentHint` and absorb it. That row is not retained-state metadata; it
is a prover message.

It does not belong to the pretranscript `CommittedGroup` operation defined in
this file. Ordinary recursive `commit_w` produces an `OuterPayload`, while
terminal `commit_terminal_w` has only an inner plan and produces terminal inner
state; neither is a full inner/outer/compression-to-`CommittedGroup` call. They
belong to the later `NextWitnessBinding` epoch in
[`transcript-epochs.md`](transcript-epochs.md).

The cutover MUST introduce a validated canonical message type, provisionally:

```rust,ignore
pub struct TerminalInnerStateMessage<F> {
    t_state: RingVec<F>,
}
```

That later epoch returns this value explicitly in its message bundle. The
driver validates and absorbs it using the same encoding used by the verifier.
No transcript-relevant value may be recovered by inspecting a state handle.

This message wraps exactly one A-native `RingVec<F>` from the terminal inner
plan. Construction MUST validate its ring dimension and coefficient length
against `TerminalFoldParams` and the scheduled terminal-response
`t_field_elems`. Transcript encoding is
`raw_field_segment_bytes(&t_state)`: canonical field coefficients in order,
with no collection length prefix. The identical byte string is absorbed under
`ABSORB_NEXT_LEVEL_WITNESS_BINDING` and later `ABSORB_COMMITMENT`, and the
verifier requires it to equal
`raw_field_segment_bytes(&terminal_response.t_fields)`. The wrapper has no
public unchecked constructor.

## CPU reference implementation

The first implementation SHOULD preserve the current algorithms and move their
composition behind the epoch:

1. while ingesting or traversing the source, enforce the checked source-class
   and accepted-interval contract rather than trusting declared metadata; a
   resident source handle may instead carry a runtime-validated binding to that
   exact contract;
2. run the current source-specialized inner commitment;
3. validate and decompose inner rows;
4. construct the outer commitment;
5. execute every compression map and quotient step;
6. build and validate `CommittedGroup`;
7. retain whatever source, rows, transforms, digit blocks, stages, or quotients
   the CPU opening path chooses;
8. return the public message and opaque handle.

`compute/compression.rs` is suitable as a bounded-memory CPU reference
executor. It SHOULD be internalized rather than deleted merely because the
public boundary moves. The same applies to dense, one-hot, packed-trace, and
runtime-ring-dimension kernels.

The CPU epoch MUST preserve the reference executor's measurable scratch bound.
`MAX_COMPRESSION_RHS_BATCH` remains `8` initially. For each report:

```text
max_expanded_rhs_bytes
    <= max_batch_input_width × map_ring_dimension × 8

executor_peak_scratch_bytes
    = max_expanded_rhs_bytes + max_current_image_bytes
```

The implementation MAY lower the batch cap but MUST NOT raise it without new
profile evidence. `CompressionExecutionReport` or an equivalent private
diagnostic remains the source for this assertion. Run:

```bash
cargo test -p akita-prover --release --no-default-features \
  --features transcript-blake2b \
  mixed_shapes_partition_and_rhs_expansion_is_bounded

cargo test -p akita-prover --release --no-default-features \
  --features transcript-blake2b \
  compression_execution_bench -- --ignored --nocapture
```

End-to-end host RSS is recorded, not given a speculative hard regression
threshold, with the canonical command in
`book/src/usage/profiling.md`. Every implementation PR MUST publish base/head
commit-phase time, peak RSS, the two report terms above, retained-state bytes,
and semantic call/transfer counts for at least `onehot_fp128` at 32 variables
or explain why the profile cannot run on the test host.

## Downstream opening conversion

Commitment is not complete as an abstraction cutover while protocol code can
still pair arbitrary polynomials with arbitrary hints. Replace parallel
`{polynomials, commitment_hints}` inputs with committed artifacts bound when
commitment succeeds.

The next opening-side semantic epoch MUST consume committed artifacts as bound
message/state/plan/setup units and
return only:

- the next canonical opening or relation message bundle; and
- a new opaque state handle for later epochs.

The runtime, not the protocol driver, decides whether to reuse retained rows,
reuse uploaded source state, derive `t_hat`, reconstruct B images, materialize
compression witnesses, or recompute from the source.

During the migration, a private CPU state implementation may expose typed
borrows to private CPU form executors. It MUST NOT recreate a public generic
hint API.

## Persistence and setup-prefix commitments

Setup-prefix commitments are legitimately reused across processes. That does
not make their live state representation a protocol type.

Persistence MUST be explicit:

```rust,ignore
fn export_commitment_checkpoint(
    &mut self,
    artifact: &CommittedArtifact<F>,
    policy: CheckpointPolicy,
) -> Result<CommitmentCheckpoint, BackendError>;

fn import_commitment_checkpoint(
    &mut self,
    checkpoint: &CommitmentCheckpoint,
) -> Result<CommittedArtifact<F>, BackendError>;
```

### Canonical checkpoint envelope

Phase 3 MUST use one Akita-owned envelope even when its payload is
backend-specific. Version 1 has the following canonical field order:

1. four literal content-tag bytes `AKCP`;
2. envelope schema version as an Akita-serialized `u64`, initially `1`;
3. Akita protocol-format version as a `u64`;
4. portability tag as one byte: `0` for protocol-portable and `1` for
   backend-specific;
5. stable 16-byte payload-format identifier followed by its `u64` version;
6. one canonical field-profile tag;
7. a length-prefixed canonical setup-identity descriptor;
8. a length-prefixed canonical commit-plan descriptor, including group and
   schedule identity;
9. a length-prefixed compressed `CommittedGroup` encoding validated against
   that plan;
10. a length-prefixed payload interpreted by the identified format.

All integers and length prefixes use `AkitaSerialize` canonical little-endian
encoding. Lengths are `u64`. The decoder MUST reject an unknown tag, version,
field profile, or payload format; a non-canonical descriptor or commitment;
identity or geometry mismatch; truncation; and any trailing byte.

Before allocating, it MUST convert every length with checked arithmetic and
enforce all of:

- a small versioned `MAX_CHECKPOINT_DESCRIPTOR_BYTES` cap for each header
  descriptor;
- the exact or maximum commitment length derived from the checked plan;
- a format-declared payload bound derived from that plan; and
- a versioned runtime `MAX_CHECKPOINT_PAYLOAD_BYTES` absolute cap selected
  before reading untrusted data.

The implementation PR MUST choose and test the numeric caps. It MUST use the
checked sizing primitives in `akita_error::checked`; it MUST NOT allocate first
and validate later or rely only on the generic `Vec` decoder cap. Each portable
payload format MUST separately specify its canonical internal field order and
validation. A backend-specific payload may be opaque to Akita, but its selected
runtime must validate it fully before publishing a live artifact.

The checkpoint format MUST be versioned and bound to the public commitment,
setup identity, schedule identity, and protocol format version. A checkpoint
MAY be:

- a protocol-defined portable representation;
- an explicitly tagged backend-specific representation; or
- a recipe that asks the runtime to recompute from an available source.

Unsupported import MUST fail before transcript initialization or select an
explicit recomputation path. Ordinary live handles MUST NOT become serializable
to satisfy this use case.

The cutover creates a new versioned setup-prefix registry namespace and MUST NOT
read the legacy serialized-hint registry. On a missing, legacy, truncated,
unknown-version, corrupt, or mismatched setup-prefix checkpoint, `akita-setup`
reconstructs the required slot from the public setup source through the same
semantic commitment operation, validates the resulting public commitment
against the verifier registry, and atomically writes the new envelope. If no
authoritative source is available for an ordinary user commitment, import
returns a typed incompatibility error instead of guessing or accepting a loose
message/handle pair. The public-matrix cache is outside this namespace and is
not rewritten merely because the commitment checkpoint schema changes.

## Ownership planning

Commitment and the first opening consumer form a stateful chain. A runtime MUST
plan that chain before the transcript starts.

The planner MUST choose one of:

1. the same owner executes both epochs;
2. the commitment owner exports and the opening owner imports a supported
   checkpoint or transfer object;
3. the opening owner recomputes from an explicitly retained source; or
4. planning fails.

It MUST NOT silently read back a CPU hint because independently selected
cluster implementations happen to expect it. Backend fallback is therefore a
chain decision, not a per-method retry.

## Public API direction

The intended call site is conceptually:

```rust,ignore
let runtime = AkitaProverRuntime::prepare(setup, runtime_config)?;
let committed = runtime.commit(source, group_plan)?;
let proof = akita_protocol::prove_with_runtime(
    &runtime,
    transcript,
    statement,
    [&committed],
)?;
```

`prove_with_runtime` is the protocol-driver facade: it owns the transcript and
invokes transcript-free runtime epochs. It is not a method on the runtime
layer.

The exact ownership split between a long-lived state store and a proof-scoped
session remains an implementation choice with one constraint: a committed
artifact may outlive one proof, while transient fold and sum-check state should
normally be proof-scoped. The API MUST make that lifetime distinction explicit.

## Deletion and evolution map

| Current surface | Cutover action |
|---|---|
| `GroupContext` | evolve into or compile into a validated commit-group plan |
| `CommitOutput` | replace with canonical message plus opaque state handle |
| `AkitaCommitmentHint` | delete as a public/live type after consumers migrate |
| hint serialization | replace with explicit checkpoint export/import |
| `ProverOpeningData` parallel polynomial/hint vectors | replace with bound committed artifacts |
| `RingRelationGroupWitness::hint` and hint accessors | delete |
| `RootCommitSource` and specialized views | preserve initially as ingress/reference internals |
| `RootCommitKernel` and primitive compute traits | internalize as CPU forms where useful |
| `CompressionChainPlan` and checked formulas | preserve as protocol-owned plan data |
| `compute/compression.rs` | preserve as CPU reference composition, internalize |
| `Runtime*Backend*` bundles and macro | delete after semantic runtime routing lands |
| `ProverComputeStack`, `LevelProveStacks`, `TieredProveStacks` | replace with owner-aware epoch planning |
| delegating cluster wrappers | delete rather than forward to the new runtime |
| protocol cache-release hooks | move to runtime policy and observability |
| `akita-pcs` re-exports of internal backend fragments | delete at public API cutover |

No compatibility aliases or pass-through adapters should remain after the
corresponding caller has migrated.

Before deleting the hint, the implementation MUST prove there are no live
readers in at least this current inventory:

- `api/setup.rs` and `api/setup_prefix.rs`;
- `types/opening_data.rs`;
- `protocol/ring_relation/compression_witness.rs` and
  `protocol/ring_relation_witness.rs`;
- `protocol/ring_switch/commit.rs`;
- `protocol/core/fold/mod.rs` and `protocol/core/suffix.rs`;
- `akita-types/src/proof/hints.rs` and `proof/setup_prefix.rs`; and
- `akita-pcs` public re-exports, tests, benches, examples, and Jolt adapters.

The characterization PR SHOULD encode this as an `rg`-based guard or equivalent
compile-time removal check so a new accessor reader cannot land unnoticed.

## Acceptance tests

The commitment slice is complete only when all of the following pass:

- representative public commitment bytes match the pinned CPU implementation;
- full proof bytes and transcript-event traces match for unchanged protocol
  configuration;
- a counting remote test double observes one semantic commitment call per
  committed group, independent of compression-map count;
- no inner row, B image, digit plane, compression stage, quotient, or source
  polynomial appears in the semantic result type;
- a non-CPU fake backend retains a representation that cannot be converted to
  `RingVec` and still completes the next opening epoch;
- wrong-owner, stale-generation, cross-domain, and wrong-kind handles are
  rejected deterministically;
- malformed backend messages are rejected before transcript absorption;
- the CPU implementation remains bounded-memory for compression;
- setup-prefix checkpoints round-trip through explicit export/import;
- unsupported checkpoint imports fail before transcript mutation;
- checkpoint goldens fix the version-1 envelope bytes, and negative tests cover
  every tag/version, descriptor cap, payload cap, truncation, trailing byte,
  identity/geometry mismatch, and legacy-registry regeneration path;
- dense, one-hot, packed-trace, sparse-unit, and setup-prefix source
  adapters remain differentially tested;
- runtime scheduling and resource configuration do not change proof bytes.

The RPC assertion is about the semantic boundary. A local runtime MAY schedule
many device kernels internally, and a distributed runtime MAY use an internal
streaming protocol. Neither is visible to the Akita driver.

## Implementation sequence

1. Pin commitment bytes, proof bytes, transcript events, and call counts for
   representative schedules and source classes.
2. Add owner IDs, typed state handles, and the long-lived commitment state
   store without exposing payload access.
3. Compile existing commitment inputs into `ValidatedCommitGroupPlan`.
4. Implement the full CPU commitment epoch by composing existing algorithms
   behind one call.
5. Add one intentionally non-CPU-shaped fake backend and enforce the one-call
   contract.
6. Route every `CommittedGroup` entry point, including setup-prefix and Jolt
   source adapters, through the epoch.
7. Introduce explicit next-witness message types where current recursive code
   reads a hint field for absorption; implement their separate epoch in the
   later transcript-boundary phase.
8. Convert the first opening/relation consumers to handles.
9. Add checkpoint export/import and migrate setup-prefix persistence.
10. Delete `AkitaCommitmentHint`, old public capability bundles, cluster
    forwarding wrappers, and obsolete serialized formats once the last reader
    is gone.

Steps 2 through 5 are the minimum first implementation PR. Steps 6 through 10
may be split by coherent caller groups, but each merged step MUST reduce the old
surface and MUST NOT add a second permanent abstraction.

## Failure modes to avoid

- Stopping at outer commitment and calling compression a later backend.
- Returning a new opaque wrapper that still exposes or serializes CPU rows.
- Giving the backend the transcript so it can hide multiple RPCs internally.
- Allowing message validation only after transcript absorption.
- Keeping the old stack through compatibility wrappers.
- Treating `Clone` on a handle as harmless without defining shared-state
  semantics.
- Making every live state portable to solve setup-prefix persistence.
- Planning commitment and opening owners independently.
- Requiring a second backend to implement Akita-specific leaf traits before it
  can implement the semantic commitment epoch.

## Open implementation questions

These questions do not change the boundary decision:

- whether long-lived commitment handles use explicit reference counting or
  store-issued proof borrows;
- whether the first runtime dispatch is an object-safe trait, an enum, or a
  private vtable;
- whether portable checkpoints are always available or only the CPU runtime
  initially provides them;
- whether source ingress is pull-based, push-based, or selected per source;
- which existing primitive traits remain useful as private CPU form traits.

Each choice should be evaluated with both the CPU reference path and an
RPC-counting, representation-independent test backend.
