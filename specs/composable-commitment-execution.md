# Composable commitment execution

| Field | Value |
|---|---|
| Author | Omid Bodaghi |
| Revised | 2026-09-10 |
| Status | active |
| PR | [#20](https://github.com/LayerZero-Labs/akita/pull/20) |
| Scope | Prover-side commitment execution in `akita-prover`, direct callers, and downstream Jolt integration |

## Summary

Akita separates commitment into independently replaceable inner, outer, and
compression operations. Inner and outer may instead run as one fused operation.
Callers assemble complete executors and an optional per-round schedule before
execution; Akita resolves sources, validates capabilities, connects stages, owns
type erasure, and assembles the result.

The retained prover state is selected by policy. It may be a portable
`AkitaCommitmentHint`, resident CPU data, a device allocation lease, or another
backend-owned value. It is not required to be a vector of ring elements.

This specification consolidates the original commitment design, its simplified
state model, and the Jolt integration record. It describes the implemented
commitment boundary and the compatibility requirements that remain active.

## Scope and invariants

The implementation covers:

- standard and external commitment sources;
- split inner/outer execution;
- fused inner/outer execution;
- independently selected compression;
- full, uncompressed, and inner-only commitment modes;
- backend-owned retained state and optional portable export;
- independent backend changes from round to round; and
- commitment routing known and inspectable before execution.

Backend routing is prover execution metadata. For identical setup, parameters,
sources, and randomness, changing the route must not change:

- `GroupCommitPhaseParams`, schedule lookup, or `RingRelationMode`;
- the uncompressed outer image `u` or compression chain;
- `Commitment`, `CommittedGroup`, or their canonical encodings;
- transcript events or challenges;
- proof shape, canonical proof bytes, or `proof.size()`; or
- verifier acceptance and rejection behavior.

Prover-private state, backend identities, operation identities, cache owners,
and route names never enter the transcript, schedule identity, commitment, or
proof.

This PR does not provide new backend implementations for opening, folding,
tensor projection, ring switching, or relation proving. Existing proving paths
can consume the new commitment state through their current adapters, but a
backend author implementing only the commitment traits gets commitment support
only. Device-resident proving is deferred.

## Execution model

There are four object-safe operation traits:

| Operation | Input | Output |
|---|---|---|
| `InnerCommitOperation<F>` | checked inner plan and resolved sources | opaque `BackendStateRef<InnerImage>` |
| `OuterCommitOperation<F>` | checked outer plan and resident state or explicit host rows | canonical host `u: RingVec<F>` |
| `FusedInnerOuterOperation<F>` | checked A/B plan and resolved sources | opaque inner state and host `u` |
| `CompressionOperation<F>` | binding, compression plan, relation mode, and `u` | terminal payload and opaque compression state |

A full route has one of these shapes:

```text
split: sources -> inner -> inner state -> outer -> u -> compression -> payload

fused: sources -> fused inner/outer ---------> u -> compression -> payload
                    |
                    +-> inner state
```

Compression receives only `u`; it has no argument through which it can read the
inner witness. Outer and compression are therefore dispatched by the executor,
not by `CommitmentSource`.

`CommitmentExecutionPlan` is the canonical checked arithmetic plan for one
request. Its modes are:

- `Full`: A, B, and compression for root and setup-prefix commitments;
- `Uncompressed`: A and B for recursive commitments whose profile omits
  compression; and
- `InnerOnly`: A for terminal commitment.

Separate constructors derive these plans from root profiles, recursive fold
parameters, terminal parameters, and setup-prefix slot identities. Plans own
protocol geometry; executors own implementation routing.

Stage output constructors validate ring dimensions, coefficient lengths,
source counts, setup identity, relation mode, and state binding before results
can be joined. A malformed backend result is rejected at its boundary.

## Source contract

`CommitmentSource<F>` is ring-dimension-free and object-safe. It reports:

- O(1) shape and source-class metadata through `descriptor`;
- exact centered reach for admission;
- available standard representations without building lazy state;
- the selected representation after request compilation; and
- optional external inner-operation capability and payload.

Capability discovery must be O(1) and side-effect-free. Akita validates the
request first and materializes only the selected representation or external
operation. This avoids building a witness-sized cache for a route that will not
use it.

### Standard sources

A source can translate itself into one or more Akita-owned representations:

| Semantic type | Representation supplied by the source |
|---|---|
| `DenseType` | Canonical field coefficients, or exact-plan balanced digit planes in `[ring][digit][coefficient]` order |
| `ShortNormType` | Packed bounded signed coefficients with exact lengths, bit width, and positive/negative reach |
| `OneHotType` | Complete hot-position slice, chunk size, and variable count |

Dense predecomposed digits remain distinct from packed short-norm coefficients.
One-hot sources preserve their stored index width (`u8`, `u16`, `u32`, or
`usize`) and use `None` for an all-zero chunk; the boundary does not widen or
copy the position buffer.

Sources that can expose one of these representations reuse Akita's inner
implementations. Concrete source types may also implement opening and tensor
traits, but those capabilities are independent of `CommitmentSource`.

### External inner operations

A source with custom layout or traversal can provide its own inner algorithm.
It advertises an `ExternalInnerCommitmentCapability` for a process-local
`BackendKindId`, then returns a checked `PreparedExternalInnerCommitment` only
when that capability is selected.

External capability identity covers payload family, algorithm, context, and
backend kind. Rust `TypeId` values enforce the erased boundary; strings are
diagnostic only. The external operation receives the resolved inner plan,
sources in original group order, and its backend context. It has no transcript
access and cannot implement B or compression through this interface.

When all sources in a group advertise one compatible external capability,
request compilation prefers that homogeneous path. Otherwise it selects one
exact standard representation from the intersection of every source offer and
the backend capabilities. Every source in the group receives that selection.
An empty intersection, or a mixture of external and standard inner execution,
is rejected before materialization or arithmetic.

Each commitment group is therefore representation-homogeneous. Different
groups in one opening batch may use different concrete source types and
representations. Proving erases complete prepared groups through
`ErasedPreparedProverGroup`; it does not erase or dispatch each polynomial.

A fused route accepts an external-only source only when its capability declares
support for the fused command context. The source encoder may append A work to
the fused command, but it may not submit A independently and then call B; fused
execution represents one backend operation.

## Executor construction

Every executor is bound to one expanded setup. Backend preparation produces a
stage object that owns the operation together with its state owner, opaque
backend instance identity, diagnostic name, source and dimension capabilities,
stage resources, and optional export edge. The builder receives these prepared
objects, so correlated configuration cannot be supplied as separate registration
arguments.

A complete split executor registers all three stages:

```rust,ignore
let mut builder = CommitmentExecutorBuilder::new(expanded, ResidentStatePolicy);

builder.register_inner(PreparedInnerCommitment::new(
    inner, inner_owner, inner_context, inner_capabilities, inner_dimensions,
    optional_inner_export,
)?)?;
builder.register_outer(PreparedOuterCommitment::new(
    outer, outer_owner, outer_context, outer_dimensions,
))?;
builder.register_compression(PreparedCompression::new(
    compression, compression_owner, compression_context, compression_capabilities,
    optional_compression_export,
))?;
let executor = builder.build()?;
```

If inner and outer share a `StateOwnerCapability<InnerImage>`, outer borrows the
resident value directly. If they have different owners, executor construction
requires an `InnerImageExportOperation`; execution exports canonical host rows
and passes `InnerImageInput::HostRows` to outer. There is no implicit transfer
or fallback.

A fused-only executor registers no dummy split operations:

```rust,ignore
let mut builder = CommitmentExecutorBuilder::new(expanded, ResidentStatePolicy);

builder.register_fused(PreparedFusedCommitment::new(
    fused, fused_owner, fused_context, fused_capabilities, fused_dimensions,
    optional_inner_export,
)?)?;
builder.register_compression(PreparedCompression::new(
    compression, compression_owner, compression_context, compression_capabilities,
    optional_compression_export,
))?;
let executor = builder.build()?;
```

An executor may contain split operations and a fused operation. Full and
uncompressed plans use fused; inner-only plans use the registered split inner
operation. The builder accepts the components needed by the modes the caller
will use: inner alone supports inner-only, inner plus outer supports
uncompressed, either A/B route plus compression supports full, and fused plus
split inner supports fused nonterminal rounds followed by a terminal inner-only
round. Missing mode dependencies are rejected during preflight before source
materialization.

`CommitmentExecutor::cpu` is the standard CPU constructor. It registers the
existing optimized CPU inner, outer, and compression implementations and
preserves the standard source paths.

## Routing before execution

Applications can build all required executor combinations, then assign the
inner/outer route and compressor independently over round ranges:

```rust,ignore
let mut schedule = CommitmentExecutionScheduleBuilder::new(round_count)?;

schedule
    .register_executor("gpu-fused", "metal", &gpu_fused_metal)?
    .register_executor("gpu-fused", "cpu", &gpu_fused_cpu)?
    .register_executor("cpu-split", "cpu", &cpu_split_cpu)?;

schedule
    .inner_outer(0..i, "gpu-fused", InnerOuterRouteKind::Fused)?
    .inner_outer(i..round_count, "cpu-split", InnerOuterRouteKind::Split)?
    .compression(0..j, "metal")?
    .compression(j..round_count, "cpu")?;

let schedule = schedule.compile()?;
```

When the scheduled range includes the terminal fold, assign that round
`InnerOuterRouteKind::InnerOnly`. An executor that registers both fused and
split-inner operations can therefore run fused nonterminal folds and the
inner-only terminal fold without registering an unused outer operation.

For four rounds with `i = 2` and `j = 1`, this resolves:

| Round | Inner/outer | Compression |
|---|---|---|
| 0 | GPU fused | Metal |
| 1 | GPU fused | CPU |
| 2 | CPU split | CPU |
| 3 | CPU split | CPU |

The schedule is immutable after compilation. `steps()` exposes every decision
for inspection. `preflight` validates the selected round's execution mode,
source admission, capability, ring-dimension, relation-mode, and setup
requirements without invoking backend arithmetic.

Compilation rejects zero rounds, out-of-range or overlapping assignments,
unassigned rounds, missing executor pairs, setup mismatches, and route-kind
mismatches. Empty cutover ranges are valid. Execution uses the resolved
executor and never retries with a fallback.

`TieredProveStacks` remains the fold-level selector for complete prover stacks.
Each selected stack contains its own commitment executor, so applications can
also change commitment routes and the opening/tensor/ring-switch stack from
fold to fold. These decisions are fixed before transcript work begins.
`batched_prove` uses that stack sequence as its only routing authority; the
standalone `CommitmentExecutionSchedule` is for commitment-only callers and is
not a second schedule input to proving.

## Retained state and hints

`BackendStateRef<K>` directly owns one erased backend value. A backend creates
it through its matching `StateOwnerCapability<K>`:

```rust,ignore
let state = owner.bind(binding, retained_bytes, MyDeviceLease { /* ... */ });
```

Only the same owner can recover the concrete value with
`owner.value::<MyDeviceLease>(&state)`. External callers do not downcast it.
The binding records the setup descriptor, inner plan, source count, relation
mode, and an opaque process-local invocation identity. Backends propagate the
binding passed into an operation by cloning it; creating a same-shaped binding
does not identify the same invocation. The executor checks returned state
against its expected binding before export or subsequent arithmetic.

The concrete value needs no `Clone`, serialization, `Debug`, host-row, or
portable-hint implementation. Cloning the state wrapper creates a shared lease;
dropping the last reference drops the value, allowing an ordinary `Drop`
implementation to recycle a GPU allocation or remote object.

`CommitmentStatePolicy<F>` chooses the state returned by `CommitOutput<F, S>`:

- `ResidentStatePolicy` returns `ResidentCommitmentState<F>` containing opaque
  inner and optional compression state references;
- `PortableStatePolicy` explicitly invokes registered exporters and returns the
  existing `AkitaCommitmentHint<F>` representation; and
- `NoRetainedStatePolicy` returns no retained state for flows that do not need
  later witness material.

Portable export is a capability and an explicit transfer boundary. Setup-prefix
persistence uses it because persisted bytes retain their existing format.
Consuming export moves uniquely owned CPU rows and compression buffers; shared
leases use the borrowed copying fallback. Resident routes do not pay that cost
unless a later consumer requests export.

The direct ownership model has no global state slot table, pending deposit,
generation counter, cleanup callback registry, CPU token map, or second
composite dispatcher. CPU state directly owns its witness vectors and
compression retention; resident composite state owns the two opaque references.

### Fused device contract

A fused GPU backend may keep the large inner witness entirely on device:

```text
GPU: A + decomposition + slicing + B
  |-- device inner witness -> BackendStateRef<InnerImage>
  `-- u                    -> host RingVec<F>
```

This version permits transfer of `u`. It does not require or permit an inner-row
download merely to finish commitment, run compression, assemble the result, or
validate the binding. A fused resident route needs no inner exporter. Whether a
future device proving backend consumes the retained image in place is outside
this PR.

## Resources, caches, and performance

Operation traits do not inherit a concrete setup backend. Each registration
instead carries `StageResources`, containing either a
`CommitmentResourceControl<F>` or an explicit no-resources declaration.
Resource control exposes setup identity, exact NTT slot preparation,
cached-versus-streamed policy, cache owner identity, release,
and optional compression-cache accounting.

`CommitmentNttRequirement` identifies the exact key, routing extent, and owning
stage. Proof-wide routed requirements also identify whether the request is
inner-only or A/B. A/B requirements use the fused registration when present;
terminal A requirements use split inner. The same resolved route controls
prewarm, owner selection, and execution. Physical
owners are deduplicated by `NttCacheOwnerId` when stages share prepared state.

Releasing NTT residency does not release live commitment state. A retained
state reference remains valid after cache release; an eligible setup cache may
be rebuilt later.

The CPU route must preserve its current optimized behavior:

- dense coefficients are borrowed without another full-polynomial copy;
- cached digit planes stay cached and plan-key checked;
- one-hot traversal keeps stored index widths and bounded scratch;
- packed short-norm data remains packed until tile decoding;
- B slicing uses its reusable slice buffer and batched digit-row path;
- compression preserves `MAX_COMPRESSION_RHS_BATCH`; and
- `ReducedEvaluation` performs no cyclic quotient work and retains no quotient
  rows, while `QuotientLift` retains its required quotient images.

No route may add a persistent witness-sized copy beyond the inner image needed
by later proving. Representative performance comparisons use identical release
builds, inputs, cache conditions, and at least ten measured repetitions after
warm-up. An unexplained median regression above 3%, peak-memory increase, or
retained-cache increase blocks a production cutover.

## Schedule and validation ownership

`AkitaCommitmentScheme` owns the validated `TrustedScheduleCatalog` used by
setup, commitment, proving, and verification. The executor receives only the
already-resolved commitment profile. It does not decode catalogs, synthesize
rows, run planner search, or cache schedule decisions.

Root commitment preserves this validation order:

1. Validate a nonempty, same-shape group and checked layout arithmetic.
2. Resolve the trusted catalog or explicit profile using ordered
   `PrecommittedGroupProfiles`; reject a missing row without fallback.
3. Check that the complete selected schedule fits the setup.
4. Validate profile geometry and setup capacity.
5. Validate the configured source contract, source class, and centered reach.
6. Compile the immutable executor request and validate route capabilities.
7. Materialize only the selected representation or external operation.
8. Run arithmetic and validate each returned state and output before the next
   stage or transcript-dependent consumer.

All shape, count, product, and range arithmetic uses checked constructors and
`akita_error::checked`. The verifier remains unaware of the producing backend.

## Prover-stack integration

`ProverComputeStack` stores a complete `CommitmentExecutor` alongside the
existing opening, tensor, and ring-switch operation contexts. Its
`commitment()` accessor is the canonical commitment entry point. A uniform CPU
stack constructs the same CPU executor used by commit-only callers.

Root commitment calls `execute_full`; recursive commitment uses full or
uncompressed execution according to its scheduled payload mode; terminal
commitment calls `execute_inner`. Fold-level stack selection remains available
through `LevelProveStacks` and `TieredProveStacks`.

State consumers remain capability-specific. Inner-relation,
outer-compression, terminal-binding, portable-export, and recomputation support
are separate contracts. `batched_prove` checks the existing opening states and
the consumer edges for every scheduled recursive and terminal commitment before
prewarm and transcript mutation. Runtime export still validates the returned
material.

`ProverComputeStack` is generic only over the opening, tensor, and ring-switch
backends it stores as typed contexts. Commitment execution is already fully
owned by `CommitmentExecutor`, so the stack carries no phantom commitment
backend type.

Setup-prefix generation uses a commitment executor and explicitly exports
portable state before building `SetupPrefixSlot`. Slot identities and persisted
registry coverage remain derived from the trusted catalog, and disk namespaces
remain bound to `catalog_digest`.

## Jolt integration contract

Jolt keeps separate dense and one-hot schemes, catalogs, expanded setups, and
prepared resources. One executor never spans both setups. The validated object
flavor selects the matching executor, and the one-hot K=16/K=256 choice remains
part of Jolt's configuration.

Jolt source mapping is:

| Jolt source | Commitment path |
|---|---|
| Akita-ordered dense adapter | standard dense coefficients, preserving Jolt-to-Akita bit reversal |
| ordinary row-major one-hot adapter | standard one-hot positions using the stored `u8` indices |
| `TracePackedOneHot` | checked external inner operation preserving the tuned packed traversal |

The packed trace external operation validates family, algorithm, backend,
context, plan, ring dimension, layout, and source count before calling Jolt's
existing packed commit kernel. It has no implicit dense fallback. A fused
backend can accept it only through an explicitly advertised fused encoder.

Commitment changes must preserve Jolt's committed objects, their group
boundaries, local evaluation points, roles, order, layout digests, and arities.
The canonical final grouped opening order remains:

```text
UntrustedAdvice?
TrustedAdvice?
BytecodeChunk(0..C)
ProgramImageInit
OneHotTrace
```

## Deferred extensions

This version intentionally materializes the outer image `u` as a host
`RingVec<F>` between fused or outer execution and compression. The large inner
witness may remain entirely backend-resident and is never downloaded merely to
complete a fused commitment. A future device-resident `u` carrier requires an
explicit same-owner, cross-owner, and host-export contract and should be added
only if measurement justifies it.

Coordinated production shared by several mathematically independent
commitments is also deferred. Sources may share ordinary `Arc`-owned storage or
backend caches today, but Akita does not yet collect several commitment
requests into one production graph. A future phase-local preparation API must
preserve each commitment's identity, profile, ordering, and opening obligation.

Absent optional groups are omitted, and full-program mode omits the direct
program suffix. For `C` bytecode chunks and `A` present advice kinds, setup
capacity remains `C + 2 + A`, including the existing 260-group maximum at
`C = 256` and `A = 2`.

Jolt's normal adapter stores `ResidentCommitmentState<AkitaField>` alongside
the public committed group and source storage. Cloning the adapter clones shared
leases and source ownership, not witness-sized buffers. Native batching derives
precommitted profiles from stored `CommittedGroup.profile` values and requests
only the state capabilities needed by the selected opening plan.

The commitment redesign does not generalize Jolt's opening kernels.
`TracePackedOneHot` and the heterogeneous grouped source retain their opening,
folding, batching, and coefficient-packing implementations independently.

Downstream validation covers dense advice and program objects, K=16 and K=256
one-hot data, packed traces at supported dimensions, committed/full program
modes, all advice combinations, quotient-lift and reduced-evaluation paths,
single and heterogeneous openings, cache release with live state, malformed
state/capability rejection, and modular/legacy byte parity.

## Acceptance criteria

- [x] Inner, outer, fused, and compression operations are independent traits.
- [x] A split executor can use separate backend instances for all three stages.
- [x] A fused-only executor requires no dummy split operation or inner exporter.
- [x] Compression consumes `u` and cannot access the inner witness.
- [x] Retained state directly owns a backend-defined erased value.
- [x] Portable export is optional and byte-identical when selected.
- [x] Standard dense, short-norm, and one-hot representations remain supported.
- [x] External inner operations support custom source layouts and algorithms.
- [x] Source selection is deterministic and materializes only the selected path.
- [x] Full, uncompressed, inner-only, setup-prefix, recursive, and terminal
  commitment modes use checked plans.
- [x] Independent inner/outer and compression ranges compile into an immutable,
  inspectable round schedule.
- [x] Invalid route configuration, dimensions, relation modes, sources, and
  setup capacity fail before arithmetic; invalid backend-returned state fails
  before the next stage or transcript-dependent consumer.
- [x] Stage-specific NTT routing, planning, release, and owner deduplication are
  preserved.
- [x] Public commitments, portable hints, transcripts, proof bytes, proof size,
  and verifier behavior remain compatible.
- [x] CPU split and fused execution have arithmetic parity tests.
- [x] Jolt can supply an external fused implementation and combine it with CPU
  compression through the public Akita API.
- [ ] New backend implementations for `prove()` operations are specified and
  implemented in a later design.

## Code map and verification

The canonical implementation lives under
`crates/akita-prover/src/commitment/`:

- `source.rs`: source descriptors, standard representations, and request
  compilation;
- `external.rs`: checked external operation erasure;
- `plan.rs`: full, uncompressed, inner-only, and setup-prefix plans;
- `stages.rs`: operation traits and checked stage outputs;
- `prepared.rs`, `builder.rs`, and `executor.rs`: cohesive stage registration,
  preflight, and execution;
- `state_policy.rs`: direct state ownership and portable/resident policies;
- `resources.rs`: prepared resources and NTT routing; and
- `schedule.rs`: immutable per-round orchestration.

Focused tests cover direct ownership and cleanup, foreign-owner and wrong-type
rejection, CPU storage, explicit split transfers, fused parity, absence of
inner export during fused commitment, schedule cutovers and invalid schedules,
resource routing, setup-prefix persistence, and state policies. Repository CI
also covers all supported feature graphs, transcript implementations,
portability targets, Jolt recursion smoke checks, and documentation guardrails.
