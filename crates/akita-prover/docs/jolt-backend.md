# Jolt integration specification for composable commitment execution


| Field          | Value                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Author         | Omid Bodaghi                                                                                                                    |
| Revised        | 2026-09-04                                                                                                                      |
| Status         | Proposed; decision-complete for implementation                                                                                  |
| Akita design   | `design.md`, branch `feat/backend-commit`, rebased onto `origin/codex/trusted-schedule-artifacts` at `c02ed7928`                 |
| Jolt evidence  | `a16z/jolt` `main` at `ab963bb3abe085d23d85e1b345fe9250499bfcd4`                                                                |
| Jolt Akita pin | `4505404b5bd9548970f0753b0118616a373c867e`                                                                                      |
| Scope          | `crates/jolt-akita`, the modular prover under `crates/jolt-prover/src/akita`, and the Akita path in `crates/jolt-prover-legacy` |


## 1. Decision

Jolt adopts the commitment execution API defined by `design.md` without
changing its Akita protocol. Each actual expanded setup receives one
`CommitmentExecutor<F, StatePolicy>`. A per-call `CommitmentExecutionPlan`
selects the inner, outer, and compression operations only after Jolt has
resolved the group profile and precommitted-group context.

Jolt continues to use distinct dense and one-hot Akita configurations,
catalogs, expanded setups, and prepared resources. The executor unifies stage
routing within one expanded setup; it does not merge protocol-incompatible
setups or erase the K=16/K=256 configuration choice.

Each Akita scheme owns its validated `TrustedScheduleCatalog`, decoded from
the selected external artifact. The same catalog is used for setup,
commitment, proving, and verification. It is not stored in a commitment
executor, and a missing row never invokes runtime planner search or a compiled
fallback table.

The standard dense and ordinary one-hot objects use Akita's standard
polynomial representations. `TracePackedOneHot` uses a checked external CPU
inner commitment operation so that its tuned packed-trace traversal remains
intact. The normal Jolt proving path retains
`ProverOpeningState<AkitaField>` rather than requiring a portable
`AkitaCommitmentHint` from every commit.

This is a breaking Rust API migration. No compatibility layer, duplicate
backend abstraction, or deprecated forwarding API is part of the result.
Public commitment bytes, transcript behavior, proof bytes, proof size, and
verification behavior must remain unchanged.

## 2. Protocol contract that must not change



### 2.1 Committed objects

The integration supports these physical commitment groups:


| Object             | Representation                              | Created                                     | Retained until                        |
| ------------------ | ------------------------------------------- | ------------------------------------------- | ------------------------------------- |
| `UntrustedAdvice`  | One singleton dense word polynomial         | Per proof, when present                     | Its Stage 8 opening completes         |
| `TrustedAdvice`    | One singleton dense word polynomial         | Preprocessing or caller setup, when present | All proofs using it complete          |
| `BytecodeChunk(i)` | One singleton bounded-dense polynomial      | Preprocessing                               | All proofs using the program complete |
| `ProgramImageInit` | One singleton bounded-dense polynomial      | Preprocessing                               | All proofs using the program complete |
| `OneHotTrace`      | One `TracePackedOneHot` physical polynomial | Per proof                                   | Its Stage 8 opening completes         |


Advice remains dense word data. Its final evaluations come from the
corresponding Stage 4 `RamValCheck` contributions. The Akita path has no
advice byte decomposition and no advice-specific Stage 6b or Stage 7 claim
reduction.

The trace group preserves Jolt's current semantic column layout, selector
reduction, point permutation, omitted-zero-row rules, layout digest, and
row-major physical order. Direct bytecode and program-image objects preserve
their existing coefficient construction, zero-prefix embedding, arity bounds,
trace-polynomial order, and direct-opening claims.

### 2.2 Final grouped opening

Stage 8 proves one heterogeneous grouped Akita opening. Every group uses its
own local evaluation point. The canonical order is:

```text
UntrustedAdvice?
TrustedAdvice?
BytecodeChunk(0)
...
BytecodeChunk(C - 1)
ProgramImageInit
OneHotTrace
```

Optional advice groups are omitted when absent. Full-program mode omits the
direct-program suffix. A proof with neither advice nor direct-program objects
is a valid single-group `OneHotTrace` opening.

The migration must not change:

- group boundaries, roles, labels, or order;
- any layout, preprocessing, or setup digest;
- logical or physical arity;
- `GroupCommitPhaseParams`, the selected commitment profile, or schedule key;
- the schedule-authenticated `RingRelationMode`, including quotient-free
  reduced suffixes;
- the uncompressed outer image `u` or the compression chain;
- `Commitment`, `CommittedGroup`, or their canonical encodings;
- transcript events, challenges, or proof shape;
- canonical proof bytes or `proof.size()`; or
- the accepted and rejected proof sets.

Prover-private state, backend identifiers, route selection, leases, caches,
and execution diagnostics never enter the protocol or transcript.

The compression operation receives the relation mode explicitly from checked
schedule parameters. `QuotientLift` preserves its quotient images;
`ReducedEvaluation` preserves the current negacyclic-only execution and stores
no quotient image. Jolt must not infer the mode from private state shape.

### 2.3 Setup and schedule capacity

For `C` bytecode chunks and `A` advice kinds with nonzero capacity, Jolt keeps
the current total capacity:

```text
C + 2 + A
```

The terms are the `C` chunks, program image, trace, and optional advice
objects. The largest admitted shape remains 260 total groups/polynomials for
`C = 256` and `A = 2`. Group-local polynomial limits, trace shape, direct
object bounds, and schedule provisioning remain independently enforced.

`PrecommittedGroupProfiles` remains per-call schedule input. Jolt constructs it
in canonical role order from each stored `CommittedGroup.profile`, then passes
it through `GroupContext::scheduler_with_precommitted_groups`. It is not stored
in the executor and is not inferred from private hint internals.

Each dense or one-hot `AkitaCommitmentScheme` continues to own its matching
validated `TrustedScheduleCatalog`. `setup_prover`, root commitment, schedule
selection, proving, and verification use that same instance. Setup-prefix slot
enumeration and persisted-registry coverage remain catalog-derived, and any
disk registry namespace remains bound to `catalog_digest`.

## 3. Execution architecture



### 3.1 Executor ownership

`AkitaProverSetup` continues to distinguish its dense and one-hot halves:

- dense expanded and prepared setup for advice and direct-program objects;
- one-hot expanded and prepared setup for ordinary one-hot data and the packed
trace; and
- the current runtime one-hot configuration choice for K=16 or K=256.

For each present half, setup construction creates the canonical executor for
that expanded setup. Every operation registered in an executor must be bound
to the same setup descriptor and prepared-resource identity. Construction
fails before commitment arithmetic if an operation, state store, setup, or
capability is incompatible.

There is no executor that spans both expanded setups. Jolt dispatches to the
dense or one-hot executor from the already-validated object flavor. Existing
K dispatch selects the corresponding one-hot scheme/catalog and then uses
that setup's executor.

The public integration must not introduce a parallel routing wrapper, another
Jolt-specific commitment stack, or a helper that merely reconstructs the
executor on each call. `backend_stack` may remain only as an
opening/tensor/ring-switch adapter if those paths still need
`UniformProverStack`; it must not remain a second commitment entry point.

### 3.2 Source mapping

Every committed Jolt object is presented through the D-free, object-safe
`CommitmentSource<AkitaField>` contract from `design.md`. Where Rust's orphan
rules prevent an implementation on the original Jolt polynomial type, the
adapter uses the existing Akita-owned converted source or a Jolt-local source
type rather than adding a forwarding backend.


| Jolt source                     | Advertised standard type    | Selected representation             |
| ------------------------------- | --------------------------- | ----------------------------------- |
| Akita-ordered dense adapter     | `DenseType`                 | `DenseRepresentation::Coefficients` |
| Akita row-major one-hot adapter | `OneHotType`                | `UnitPositionSlice::U8`             |
| `TracePackedOneHot`             | None in the initial cutover | External packed-trace operation     |


Dense conversion preserves the current Jolt-to-Akita bit-reversal. The
conversion may allocate the Akita-ordered coefficient buffer once. The
selected representation then borrows that buffer and must not make another
witness-sized copy.

Ordinary one-hot conversion preserves row-major indexing and the source index
width. Jolt's current source uses `u8`, so it exposes
`UnitPositionSlice::U8`; `None` entries continue to represent all-zero rows.
The general Akita representation also supports `u16`, `u32`, and `usize`, but
Jolt must not widen its `u8` indices merely to cross the source boundary.

Capability discovery is O(1) and side-effect-free. It does not build a dense
cache, clone indices, reorder coefficients, or prepare the trace operation.
Only `represent_as` or `prepare_external_inner_commitment`, called after route
selection, may materialize the selected form.

Each source reports descriptor and centered-reach facts from its existing
authoritative shape data. `CommitmentSource` does not inherit `RootPolyMeta`.
Types that also participate in opening continue to implement `RootPolyMeta`
and the relevant opening traits independently.

### 3.3 Packed trace external operation

`TracePackedOneHot` advertises an external inner commitment capability only
for the registered CPU inner-operation backend kind. Preparation returns a
`PreparedExternalInnerCommitment` containing:

- the request compiler's opaque capability token;
- an erased borrow of the trace payload;
- the external operation object;
- the checked source family and algorithm identity; and
- the execution context identity required by that operation.

Preparation and execution validate the capability, source family, algorithm,
backend kind, context, resolved `CommitInnerPlan`, runtime ring dimension,
layout, and source count before arithmetic. Strings are diagnostic only. The
source contract gains no general `as_any` hook.

The operation dispatches the runtime ring dimension inside its method and
calls the existing `commit_packed::<D>` implementation. Its traversal,
decomposition, shift accumulation, omission rules, and output ordering remain
unchanged. On CPU it registers the canonical host
`CommitInnerWitness<AkitaField>` rows as the inner-image state. The standard
outer operation then consumes those rows and the selected compression
operation completes the commitment.

The initial Jolt cutover treats the packed trace as external-operation-only.
There is no implicit dense fallback. Tests that force the standard source path
apply to sources that advertise a standard representation; they do not invent
one for `TracePackedOneHot`.

A future fused inner/outer route may accept this trace only when the source
explicitly advertises the fused encoder capability required by that operation.
The external inner operation does not by itself satisfy that contract, and
running separate inner and outer calls behind a fused interface is forbidden.

### 3.4 Opening-side source types

The commitment redesign does not generalize Jolt's opening kernels.
`TracePackedOneHot` retains its implementations of:

- `RootPolyMeta` and `RootPolyShape`;
- `RootOpeningSource`;
- `OpeningFoldKernel` and `OpeningBatchKernel`; and
- `SubringCoefficientPackingBatchKernel`.

`GroupedRootSource` remains the heterogeneous opening sum type with dense,
ordinary one-hot, and packed-trace variants. It keeps its metadata, shape,
opening-source, fold, batch, and coefficient-packing implementations. It loses
only its commitment responsibilities: `RootCommitSource` and the associated
`RootCommitKernel` dispatch.

After all Akita commitment producers migrate, Jolt contains no implementation
or bound for the deleted Akita surfaces `RootCommitKernel`,
`RootCommitSource`, `CommitBackendFor`, `RuntimeCommitSource`, or
`RuntimeCommitBackendFor`. Opening-only views remain where the opening API
requires them.

## 4. Prover state and lifetime



### 4.1 Selected state representation

The normal Jolt route selects `ProverOpeningState<AkitaField>` as the state
parameter of `CommitOutput`. It may contain resident CPU state, portable state
when explicitly requested, or an explicit recomputation policy. The Jolt
adapter does not inspect its private representation.

`AkitaProverHint` remains Jolt's `CommitmentScheme::OpeningHint` adapter. Its
conceptual contents after migration are:

```rust,ignore
pub struct AkitaProverHint {
    commitment: AkitaCommitment,
    committed_group: Option<CommittedGroup<AkitaField>>,
    state: Option<ProverOpeningState<AkitaField>>,
    polynomials: AkitaHintPolynomials,
}
```

The exact field visibility may remain crate-private. The two `Option` fields
provide the empty value required by Jolt's `OpeningHint: Default` contract;
every proving entry point rejects an empty or incomplete hint before transcript
mutation. A real committed hint always contains both values.

The adapter may implement `Clone` because Jolt's trait requires it. Cloning a
real hint clones `Arc`-owned source storage and backend-state leases; it must
not clone witness-sized buffers. This adapter-local constraint does not add
`Clone`, `Default`, serialization, or portable-hint bounds to Akita's generic
`CommitOutput<F, S>`.

`AkitaCommitmentHint` remains the portable export and import format used by
explicit persistence, compatibility, and differential workflows. Normal
in-memory Jolt proving does not export it. Setup-prefix persistence keeps its
existing portable bytes as required by `design.md`.

### 4.2 Lifetime rules


| Object                | Source storage                           | Commitment-state lease                         |
| --------------------- | ---------------------------------------- | ---------------------------------------------- |
| Direct program object | Owned by preprocessing data              | Reused for the preprocessing object's lifetime |
| Trusted advice        | Owned by its caller/preprocessing object | Reused for every proof that references it      |
| Untrusted advice      | Owned by the proof witness               | Stage 0 through its Stage 8 opening            |
| Main trace            | Owned by the proof witness               | Stage 0 through its Stage 8 opening            |


Dropping a temporary executor view or setup handle must not invalidate a live
state. A `BackendStateRef` lease keeps its state store and physical owner alive
and is checked against owner, setup, plan, public commitment, store, semantic
kind, and generation at each consumer boundary. Dropping the last lease runs
the registered cleanup policy.

`release_post_commit_ntt_residency` releases only releasable transformed setup
caches, with physical-owner deduplication. It does not close a live commitment
state. Opening after cache release must succeed, rebuilding eligible setup
cache entries if necessary.

### 4.3 State consumption in native batching

`AkitaNativeBatching` no longer extracts a raw
`(AkitaBackendCommitment, AkitaBackendHint)` tuple. It performs these steps:

1. Validate statement roles, group order, points, arities, public commitment,
  source storage, and the presence of committed group and state.
2. Build every `PrecommittedGroupProfiles` entry from the stored
  `CommittedGroup.profile`.
3. Select the dense or one-hot opening stack from the validated group flavor.
4. Ask the selected `ProverOpeningState` consumer for the outer-compression,
  inner-relation, terminal-binding, or portable material actually required by
   the opening plan.
5. Build generic `ProverOpeningData`, `SelectedProverOpeningData`, and
  `ProverGroupInput` values over that state and invoke Akita batched proving.

The complete commitment-to-opening route, including any transfer, import,
export, or recomputation, is preflighted before Jolt mutates the proof
transcript. State consumers remain independent capabilities; support for one
does not imply support for another.

Jolt's public `CommitmentScheme`, `BatchOpeningScheme`, and
`TraceOneHotCommitment` boundaries remain. Changing their generic hint
contract is not required for this migration. The adapter absorbs the state
type difference without reintroducing a mandatory portable hint below it.

## 5. Required code changes



### 5.1 `crates/jolt-akita/src/adapters.rs`

- Replace the raw backend commitment/hint tuple in `AkitaProverHint` with the
full committed group and `ProverOpeningState` contract described above.
- Preserve `AkitaHintPolynomials` as retained opening-source storage. Its dense,
one-hot, and trace variants remain the authoritative flavor discriminator.
- Implement adapter-local clone/default behavior without witness-sized copies.
- Store or expose the dense and one-hot executors built from their respective
setup/prepared pairs.
- Keep opening stack construction only for non-commitment operations.
- Preserve all public `AkitaCommitment` serialization and transcript behavior.



### 5.2 `crates/jolt-akita/src/scheme.rs`

- Replace calls that construct a uniform commitment stack with calls to the
appropriate setup-owned `CommitmentExecutor`.
- Adapt dense and ordinary one-hot sources to `CommitmentSource` and pass the
resolved `GroupContext` per call.
- Map `CommitOutput { committed_group, prover_state }` into
`AkitaProverHint` without exporting a portable hint.
- Read precommitted profiles from stored committed groups.
- Preserve current dense/one-hot setup selection, K dispatch, schedule
selection, role validation, and failure ordering.



### 5.3 `crates/jolt-akita/src/trace_onehot`

- In `source.rs`, implement descriptor, centered reach, capability discovery,
and preparation for the packed-trace external operation.
- In `commit.rs` and `kernels.rs`, wrap the current `commit_packed::<D>` body in
the checked object-safe external operation. Do not rewrite its arithmetic.
- In `grouped.rs`, remove commitment-source and commitment-kernel code while
retaining every opening-side implementation.
- Keep `opening.rs`, `decomposition.rs`, and `traversal.rs` behavior unchanged
except for mechanical imports or state plumbing.



### 5.4 `crates/jolt-akita/src/native_batching.rs`

- Generalize construction of Akita opening inputs over
`ProverOpeningState<AkitaField>`.
- Replace private hint-field inspection with selected state-consumer
operations.
- Retain group-local points, source order, selector reduction, native schedule
selection, transcript bridging, proof serialization, and verifier input.
- Reject absent, stale, foreign, wrong-plan, or unsupported state before the
next transcript mutation that could depend on it.



### 5.5 Prover call sites

Update both implementations that exercise the Akita protocol:

- modular code under `crates/jolt-prover/src/akita`; and
- the Akita path under `crates/jolt-prover-legacy`.

Both paths must use the same adapter contract and continue to produce
byte-identical proofs. Do not leave the legacy path on a compatibility wrapper;
the byte-parity tests make it an active migration target.

Setup creation continues to occur during preprocessing. Proof-time Stage 0
creates untrusted advice and trace commitment state; Stage 8 consumes all
groups through the one native grouped opening. Existing stage boundaries and
direct-program commitment ownership remain unchanged.

## 6. Validation and error ordering

Jolt preserves Akita's root validation order:

1. Validate nonempty groups, common `num_vars`, layout arithmetic, group-local
  limits, and total capacity.
2. Resolve the validated trusted catalog or explicit profile using the final
  group and ordered precommitted profiles. A missing row rejects without
  planner search.
3. Validate that the complete selected schedule fits setup.
4. Validate the frozen profile and setup geometry.
5. Validate the configured committed-source contract.
6. Validate source class and centered reach.
7. Compile the immutable execution request and validate operation,
  representation, external capability, setup, and state-store compatibility.
8. Materialize only the selected representation or external operation.
9. Run arithmetic and validate all returned state and output shapes before
compression or transcript-dependent consumption.

For proving, `SelectedProverOpeningData::from_committed_claims` first resolves
the exact grouped profiles to `OpeningScheduleSelection`. `batched_prove` then
resolves that selection through the same trusted catalog, applies the effective
schedule, validates execution and setup, derives NTT requirements, and
preflights every schedule-dependent state transition before transcript
binding. Executor construction validates registrations only; it does not
choose fold transitions before the selected schedule exists.

All counts, products, ranges, and capacities use Akita's existing checked
constructors and `akita_error::checked` helpers. Malformed setup, source,
state, statement, or proof data returns the existing error domains. No new
panic, unchecked index, or unbounded allocation is verifier-reachable.

Verifier code does not know which executor route produced a commitment. Its
protocol validation and arithmetic remain unchanged.

## 7. Migration sequence

The implementation lands in this dependency order:

1. Extend `AkitaProverHint` and Jolt's internal opening-data plumbing to carry
  `ProverOpeningState` and the full committed group.
2. Build distinct setup-owned dense and one-hot executors and switch dense and
  ordinary one-hot commitments to standard `CommitmentSource`
   representations.
3. Add the checked `TracePackedOneHot` external operation and switch the trace
  commitment to it without modifying `commit_packed::<D>` arithmetic.
4. Move precommitted-profile extraction and native batching to generic state
  consumers.
5. Migrate modular and legacy prover call sites together.
6. Remove Jolt's remaining commitment implementations and bounds for the five
  deleted Akita commitment surfaces. Do not replace them with aliases.
7. Run compatibility, negative-path, schedule, performance, and memory gates.

During development only, differential tests may retain the previous commit
path behind test-only code. It must not remain callable by production Jolt
after the cutover.

## 8. Acceptance gates



### 8.1 Functional and protocol gates

The migrated integration must cover:

- dense advice and direct objects;
- ordinary one-hot sources for both K=16 and K=256;
- packed trace commitment for every supported runtime ring dimension;
- full-program and committed-program modes;
- absent, trusted-only, untrusted-only, and both-advice cases;
- one- and two-bytecode-chunk proofs plus the 256-chunk/260-group boundary;
- quotient-lift and reduced-evaluation recursive paths, with no quotient
  allocation or cyclic compression work in reduced mode;
- native single-group and heterogeneous grouped openings;
- exact canonical group order and group-local points;
- cache release while commitment-state leases remain live;
- rejected missing, stale, foreign, wrong-kind, wrong-plan, and wrong-setup
state;
- rejected capability, family, context, layout, K, and source-count mismatch;
- explicit portable export equivalence with the existing
`AkitaCommitmentHint`; and
- modular/legacy commitment, transcript, proof, and serialization byte parity.

Existing tamper, malformed-input, trusted-artifact admission/drift,
direct-role, trace-order, arity, immediate-boundary, and proof-size tests remain
mandatory. Dory clear and ZK suites must remain unaffected.

### 8.2 Jolt CI commands

Run the current Jolt workflow commands at the evidence pin, including:

```bash
rtk cargo nextest run --cargo-profile ci -p jolt-prover-legacy --features akita
rtk cargo nextest run --cargo-profile ci -p jolt-prover --features akita,prover-fixtures
rtk cargo nextest run --cargo-profile ci -p jolt-verifier --features akita,prover-fixtures --test-threads 1
rtk cargo nextest run --cargo-profile ci -p jolt-akita --run-ignored all -E 'test(catalogs_match_planner_regeneration)' --cargo-quiet
```

The regeneration command above is baseline evidence at the pinned Jolt
revision. After Jolt repins to the trusted-artifact Akita base, replace that
gate with the exact artifact admission and drift checks shipped by the new
Jolt workflow; do not reintroduce runtime planner regeneration.

Also run the Jolt workflow's exact Clippy feature graphs for
`jolt-prover-legacy`, `jolt-prover`, and `jolt-verifier`. The workflow file at
the tested revision is the source of truth if these commands change.

### 8.3 Akita gates

Run Akita's repository preflight and all path-specific commitment, opening,
setup-prefix, portability, and Jolt compatibility tests selected by the
implementation diff. The Akita CI workflow is the source of truth for exact
test selectors and feature graphs.

### 8.4 Performance and memory gates

Measure cold and warm packed-trace commitment, dense commitment, native
grouped opening, and representative end-to-end Jolt proofs. Use the existing
`jolt-akita` path benchmark and Jolt Akita profiling workloads.

Compare ten measured repetitions after warmup and report the median. No
representative workload may regress by more than 3% without an explicitly
accepted explanation. Record peak resident memory, Akita planned cache bytes,
retained commitment-state bytes, and cache rebuilds after post-commit release.
The trace route must not add a witness-sized clone, and ordinary one-hot
adaptation must preserve `u8` storage.

## 9. Completion criteria

The Jolt migration is complete only when all of the following are true:

- every Jolt commitment enters Akita through `CommitmentExecutor` and a
per-call `CommitmentExecutionPlan`;
- each Akita scheme retains one validated `TrustedScheduleCatalog` shared by
  setup, commitment, proving, and verification, with no executor-owned or
  runtime-generated fallback;
- dense and one-hot setups remain separate and every executor contains only
operations prepared for its own expanded setup;
- `TracePackedOneHot` uses the checked external CPU operation and its tuned
arithmetic is unchanged;
- Jolt's normal proving path retains `ProverOpeningState`, with portable export
performed only by an explicit consumer;
- recursive compression preserves the schedule's `RingRelationMode` and emits
  no quotient state in `ReducedEvaluation`;
- native batching consumes state through the generic capability boundary and
obtains profiles from `CommittedGroup`;
- `GroupedRootSource` is opening-only and Jolt contains no remaining use of
the five deleted commitment surfaces;
- modular and legacy Jolt are byte-identical for the required fixtures;
- public commitments, transcripts, proof bytes, proof size, schedules, and
verifier behavior match the pre-migration baseline; and
- functional, negative, CI, performance, and memory gates pass.

There are no unresolved design choices in this integration specification.
Implementation spelling may follow the final public names in `design.md`, but
it must preserve the ownership, validation, lifecycle, and protocol contracts
defined here.
