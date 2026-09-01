# Commitment backend replacement

| Field | Value |
|---|---|
| Package | [`README.md`](README.md) |
| Status | proposed |
| Role | normative commitment-replacement contract |

## Decision

Akita MUST replace its fragment-level commitment interface with one commitment
call. The call MUST ensure that its source satisfies the checked plan and then
perform inner commitment, outer decomposition and commitment, and the complete
commitment compression steps. Establishing the source condition is not a
separate protocol operation and need not be a separate pass over the source.

Compression is inside this boundary because no Fiat–Shamir challenge separates
it from the rest of commitment. The protocol owns the checked compression plan
and the final `CommittedGroup` encoding. The backend owns every intermediate
representation and all private state retained for opening.

This replacement is intentionally breaking. It does not preserve
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
protocol-owned CheckedCommitmentPlan + source provider
  │
  └─ one backend call
       ├─ arbitrary backend-private intermediates
       ├─ CommittedGroup protocol message
       └─ BackendStateRef<CommitmentState>
```

## Backend input

The request MUST be produced from the same schedule, geometry, and setup
identity used by the verifier-facing protocol, plus the source class and
magnitude declarations used by planner sizing. The latter are producer
obligations; the verifier does not inspect the committed source or trust this
admission result. The request carries the required source class and accepted
coefficient interval, but does not certify that one particular source satisfies
them. It MUST NOT duplicate unchecked dimensions for backend convenience.

The exact Rust decomposition is implementation work, but the input is
conceptually:

```rust,ignore
pub struct CheckedCommitmentPlan<F> {
    pub group_id: GroupId,
    pub public_geometry: CommitmentGeometry,
    pub inner: InnerCommitmentPlan<F>,
    pub outer: OuterCommitmentPlan<F>,
    pub compression: CompressionChainPlan<F>,
    pub source_contract: CommittedSourceContract,
}

pub trait CommitmentSource<F> {
    fn metadata(&self) -> CommitmentSourceMetadata;
    fn stream(&self, sink: &mut dyn CommitmentSourceSink<F>)
        -> Result<(), SourceError>;
}
```

`CommitmentSource` describes how the backend reads the input; it does not
prescribe the private state saved for later. Dense,
multilinear, one-hot, packed-trace, sparse-unit, and setup-prefix
adapters MAY retain their specialized traversal APIs beneath this boundary. A
backend MAY upload, stream, borrow, or directly traverse the source. It MUST
not require that the source representation become its opening-state
representation.

The checked plan is protocol configuration. Backend tiling, device choice,
stream size, cache policy, thread count, and recomputation policy are backend
configuration and MUST NOT affect public commitment bytes.

The source requirement constrains the producer result, not the implementation strategy.
The backend MAY rely on a source invariant, reuse a conservative guarantee from
prior backend work, establish the condition while ingesting or decomposing the
source, or scan arbitrary raw coefficients as a fallback. A prior guarantee is
sufficient only when it is bound to the exact source identity and generation
and implies all three parts of the current requirement:

1. when the plan requires `UnitOneHot`, the source has unit one-hot structure at
   the exact scheduled chunk size;
2. every coefficient is representable by the scheduled balanced digits, so no
   high part is discarded; and
3. every coefficient satisfies the possibly tighter bound used to price the
   schedule.

The two magnitude conditions use the plan's decomposition-centering convention
and the canonical `CommittedSourceContract::accepted_bounds` calculation.
Satisfying representability alone is insufficient because rounded digit depth
can represent values wider than the declared source bound.

## Backend output

The ordinary result contains exactly two things:

```rust,ignore
pub struct CommittedGroupWithState<F> {
    public_commitment: CommittedGroup<F>,
    state_ref: BackendStateRef<CommitmentState>,
    // Private backend metadata binds these fields to the checked plan and setup.
}
```

`public_commitment` is the exact value consumed by proof construction,
serialization, and transcript encoding. `state_ref` names private data owned by
the backend.

The state reference MUST:

- identify its backend instance, optional prover session, slot, generation, and
  state kind;
- not expose stored state to protocol code;
- have no field, ring, polynomial, collection, serialization, equality, or
  `Default` bound;
- fail deterministically when stale, consumed, or used with the wrong backend
  instance or prover session;
- support explicit shared borrowing only where the backend declares the state
  immutable and shareable.

`CommittedGroupWithState` MUST have no public constructor and no API that accepts a
replacement message or reference. The backend constructs it as one value and
links the public commitment, group ID, checked plan, setup identity, backend,
and state generation. Opening and checkpoint operations consume or borrow the
committed group with state as a unit and MUST reject a mismatch before transcript
initialization or mutation. Protocol code may borrow `public_commitment` for
serialization and absorption; it cannot pair that message with another
same-kind state reference.

The private binding metadata MUST NOT affect proof or transcript bytes. It MAY
be a digest plus a backend-validated state record or another construction that
deterministically rejects swapped components. It is not a separate public type.

The reference MUST NOT provide `inner_rows`, `into_rows`, compression-witness, or
quotient accessors. A CPU backend MAY store the exact current hint fields
internally during the first migration; that private choice is not a contract.

## Distinct recursive next-witness messages

Current recursive and suffix paths read the final inner row from
`AkitaCommitmentHint` and absorb it. That row is not saved-state metadata; it
is a prover message.

It does not belong to the `CommittedGroup` operation, which occurs before the
transcript starts. Ordinary recursive `commit_w` produces an `OuterPayload`,
while terminal `commit_terminal_w` has only an inner plan and produces terminal
`t_fields`; neither is a complete inner/outer/compression-to-`CommittedGroup`
call. They belong to the later next-witness-binding step in
[`transcript-steps.md`](transcript-steps.md).

The replacement MUST introduce a validated protocol message type, provisionally:

```rust,ignore
pub struct TerminalTFieldsMessage<F> {
    t_fields: RingVec<F>,
}
```

That later step returns this value explicitly as a protocol message. The
driver validates and absorbs it using the same encoding used by the verifier.
No transcript-relevant value may be recovered by inspecting a state reference.

This message wraps exactly one A-native `RingVec<F>` from the terminal inner
plan. Construction MUST validate its ring dimension and coefficient length
against `TerminalFoldParams` and the scheduled terminal-response
`t_field_elems`. Transcript encoding is
`raw_field_segment_bytes(&t_fields)`: canonical field coefficients in order,
with no collection length prefix. The identical byte string is absorbed under
`ABSORB_NEXT_LEVEL_WITNESS_BINDING` and later `ABSORB_COMMITMENT`, and the
verifier requires it to equal
`raw_field_segment_bytes(&terminal_response.t_fields)`. The wrapper has no
public unchecked constructor.

## CPU backend

The first implementation SHOULD preserve the current algorithms and compose
them behind one call:

1. establish that the source satisfies the plan by construction, a reusable
   backend-owned guarantee, work fused with an existing source traversal, or a
   fallback scan for arbitrary raw coefficients;
2. run the current source-specialized inner commitment;
3. validate and decompose inner rows;
4. construct the outer commitment;
5. execute every compression map and quotient step;
6. build and validate `CommittedGroup`;
7. retain whatever source, rows, transforms, digit blocks, stages, or quotients
   the CPU opening path chooses;
8. return the public message and state reference.

The current dense coefficient scan is an acceptable fallback for the first CPU
backend. It is not part of the cross-backend contract. If dense decomposition
or source construction already traverses the same coefficients, the CPU backend
SHOULD compute and retain a conservative bound during that work instead of
performing another full pass. A GPU backend may reduce the bound during upload
or decomposition and keep the resulting guarantee with its stored source.

`compute/compression.rs` is suitable for the bounded-memory CPU backend.
It SHOULD be internalized rather than deleted merely because the
public boundary moves. The same applies to dense, one-hot, packed-trace, and
dynamic-ring-dimension kernels.

The CPU path MUST preserve the current implementation's measurable scratch bound.
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
commit-phase time, peak RSS, the two report terms above, backend-state bytes,
and backend call/transfer counts for at least `onehot_fp128` at 32 variables
or explain why the profile cannot run on the test host.

## Downstream opening conversion

The replacement is incomplete while protocol code can still pair arbitrary
polynomials with arbitrary hints. Replace parallel
`{polynomials, commitment_hints}` inputs with `CommittedGroupWithState` values
created when commitment succeeds.

The next opening call MUST consume each `CommittedGroupWithState` as one linked
public commitment/state/plan/setup value and return only:

- the next opening or relation protocol messages; and
- a new state reference for later steps.

The backend, not the protocol driver, decides whether to reuse retained rows,
reuse uploaded source state, derive `t_hat`, reconstruct B images, materialize
compression witnesses, or recompute from the source.

During the migration, a private CPU state implementation may expose typed
borrows to private CPU operations. It MUST NOT recreate a public generic
hint API.

## Persistence and setup-prefix commitments

Setup-prefix commitments are legitimately reused across processes. That does
not make their live state representation a protocol type.

Persistence MUST be explicit:

```rust,ignore
fn export_commitment_checkpoint(
    &mut self,
    commitment: &CommittedGroupWithState<F>,
    policy: CheckpointPolicy,
) -> Result<CommitmentCheckpoint, ProverBackendError>;

fn import_commitment_checkpoint(
    &mut self,
    checkpoint: &CommitmentCheckpoint,
) -> Result<CommittedGroupWithState<F>, ProverBackendError>;
```

### Checkpoint file format

Phase 3 MUST use one Akita-owned file format even when its payload is
backend-specific. Version 1 has the following field order:

1. four literal content-tag bytes `AKCP`;
2. file-format version as an Akita-serialized `u64`, initially `1`;
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
- a versioned backend `MAX_CHECKPOINT_PAYLOAD_BYTES` absolute cap selected
  before reading untrusted data.

The implementation PR MUST choose and test the numeric caps. It MUST use the
checked sizing primitives in `akita_error::checked`; it MUST NOT allocate first
and validate later or rely only on the generic `Vec` decoder cap. Each portable
payload format MUST separately specify its canonical internal field order and
validation. Akita need not understand a backend-specific payload, but its selected
backend must validate it fully before publishing a live
`CommittedGroupWithState`.

The checkpoint format MUST be versioned and bound to the public commitment,
setup identity, schedule identity, and protocol format version. A checkpoint
MAY be:

- a protocol-defined portable representation;
- an explicitly tagged backend-specific representation.

Recomputation from a retained source is a separate planned fallback, not a
checkpoint. Unsupported import MUST fail before transcript initialization or
the proof plan MUST select that recomputation path. Ordinary live references
MUST NOT become serializable to satisfy this use case.

The replacement creates a new versioned setup-prefix registry namespace and
MUST NOT read the legacy serialized-hint registry. On a missing, legacy, truncated,
unknown-version, corrupt, or mismatched setup-prefix checkpoint, `akita-setup`
reconstructs the required slot from the public setup source through the same
commitment call, validates the resulting public commitment against the verifier
registry, and atomically writes the new checkpoint. If no
authoritative source is available for an ordinary user commitment, import
returns a typed incompatibility error instead of guessing or accepting a loose
message/reference pair. The public-matrix cache is outside this namespace and is
not rewritten merely because the commitment checkpoint schema changes.

## Backend plan

Commitment and the first opening call use the same private state. The proof
planner MUST decide where that state stays before the transcript starts.

The planner MUST choose one of:

1. the same backend executes both calls;
2. the commitment backend exports and the opening backend imports a supported
   checkpoint or transfer value;
3. the opening backend recomputes from an explicitly saved source; or
4. planning fails.

It MUST NOT silently read back a CPU hint because independently selected
cluster implementations happen to expect it. Fallback is therefore planned
for the whole commitment-to-opening flow, not decided per method.

## Public API direction

The intended call site is conceptually:

```rust,ignore
let backend = AkitaProverBackend::prepare(setup, backend_config)?;
let committed = backend.commit(source, group_plan)?;
let proof = akita_protocol::prove_with_backend(
    &backend,
    transcript,
    statement,
    [&committed],
)?;
```

`prove_with_backend` is the protocol-driver entry point: it owns the transcript and
invokes transcript-free backend calls. It is not a method on the backend
layer.

The backend state store owns commitment state that may outlive one proof. A
prover session borrows that state and owns transient fold and sum-check state
for one proof. The exact Rust containment may vary, but this lifetime split is
required.

## Deletion and evolution map

| Current surface | Replacement action |
|---|---|
| `GroupContext` | evolve into or compile into a checked commitment plan |
| `CommitOutput` | replace with protocol message plus state reference |
| `AkitaCommitmentHint` | delete as a public/live type after consumers migrate |
| hint serialization | replace with explicit checkpoint export/import |
| `ProverOpeningData` parallel polynomial/hint vectors | replace with bound `CommittedGroupWithState` values |
| `RingRelationGroupWitness::hint` and hint accessors | delete |
| `RootCommitSource` and specialized views | preserve initially as input/reference internals |
| `RootCommitKernel` and primitive compute traits | keep useful algorithms as private CPU operations |
| `CompressionChainPlan` and checked formulas | preserve as protocol-owned plan data |
| `compute/compression.rs` | preserve as private CPU composition |
| `Runtime*Backend*` bundles and macro | delete after the new backend routing lands |
| `ProverComputeStack`, `LevelProveStacks`, `TieredProveStacks` | replace with a plan that keeps each state value on a named backend |
| delegating cluster wrappers | delete rather than forward to the new backend |
| protocol cache-release hooks | move to backend policy and observability |
| `akita-pcs` re-exports of internal backend fragments | delete when replacing the public API |

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

- representative public commitment bytes match the pinned CPU backend;
- full proof bytes and transcript-event traces match for unchanged protocol
  configuration;
- a fake remote backend that counts calls observes one commitment call per
  committed group, independent of compression-map count;
- no inner row, B image, digit plane, compression stage, quotient, or source
  polynomial appears in the backend output type;
- a non-CPU fake backend retains a representation that cannot be converted to
  `RingVec` and still completes the next opening call;
- wrong-backend, stale-generation, wrong-session, and wrong-kind references are
  rejected deterministically;
- malformed backend messages are rejected before transcript absorption;
- the CPU backend remains bounded-memory for compression;
- setup-prefix checkpoints round-trip through explicit export/import;
- unsupported checkpoint imports fail before transcript mutation;
- checkpoint goldens fix the version-1 file bytes, and negative tests cover
  every tag/version, descriptor cap, payload cap, truncation, trailing byte,
  identity/geometry mismatch, and legacy-registry regeneration path;
- dense, one-hot, packed-trace, sparse-unit, and setup-prefix source
  adapters remain differentially tested;
- backend scheduling and resource configuration do not change proof bytes.

The RPC assertion is about the backend boundary. A local backend MAY schedule
many device kernels internally, and a distributed backend MAY use an internal
streaming protocol. Neither is visible to the Akita driver.

## Implementation sequence

1. Pin commitment bytes, proof bytes, transcript events, and call counts for
   representative schedules and source classes.
2. Add backend IDs, typed state references, and the long-lived commitment state
   store without exposing payload access.
3. Compile existing commitment inputs into `CheckedCommitmentPlan`.
4. Implement the full CPU commitment call by composing existing algorithms
   behind one call.
5. Add one intentionally non-CPU-shaped fake backend and enforce the one-call
   contract.
6. Route every `CommittedGroup` entry point, including setup-prefix and Jolt
   source adapters, through the commitment call.
7. Introduce explicit next-witness message types where current recursive code
   reads a hint field for absorption; implement their separate step in the
   later transcript-boundary phase.
8. Convert the first opening/relation consumers to
   `CommittedGroupWithState`.
9. Add checkpoint export/import and migrate setup-prefix persistence.
10. Delete `AkitaCommitmentHint`, old public support bundles, cluster
    forwarding wrappers, and obsolete serialized formats once the last reader
    is gone.

Steps 2 through 5 are the minimum first implementation PR. Steps 6 through 10
may be split by coherent caller groups, but each merged step MUST reduce the old
surface and MUST NOT add a second permanent abstraction.

## Failure modes to avoid

- Stopping at outer commitment and calling compression a later backend.
- Returning a new wrapper that still exposes or serializes CPU rows.
- Giving the backend the transcript so it can hide multiple RPCs internally.
- Allowing message validation only after transcript absorption.
- Keeping the old stack through compatibility wrappers.
- Treating `Clone` on a reference as harmless without defining shared-state
  semantics.
- Making every live state portable to solve setup-prefix persistence.
- Planning commitment and opening backends independently.
- Requiring a second backend to implement Akita-specific leaf traits before it
  can implement the commitment call.

## Open implementation questions

These questions do not change the boundary decision:

- whether long-lived commitment references use explicit reference counting or
  store-issued proof borrows;
- whether the first backend dispatch is an object-safe trait, an enum, or a
  private vtable;
- whether portable checkpoints are always available or only the CPU backend
  initially provides them;
- whether source input is pull-based, push-based, or selected per source;
- which existing primitive traits remain useful for private CPU operations.

Each choice should be evaluated with both the CPU backend and an
RPC-counting, representation-independent test backend.
