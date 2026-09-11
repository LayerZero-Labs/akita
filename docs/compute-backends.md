# Akita Compute Backends

Akita prover compute is now routed through an explicit backend operation
boundary. The first implementation is `CpuBackend`; Metal and true hybrid
scheduling remain follow-up work.

## Ownership

- `AkitaExpandedSetup<F>` owns setup data shared with verifier/protocol code:
  seed, shared matrix, descriptor digest, and setup shape.
- `AkitaProverSetup<F>` is a D-free prover setup wrapper around expanded setup.
  It stores a flat public-matrix prefix and does not own CPU NTT caches, device
  buffers, command queues, or any backend-prepared state.
- `ComputeBackendSetup<F>` owns backend preparation. Prepared setup slots are
  keyed by field family and ring role at kernel boundaries via `dispatch_for_field!`.
- `RootCommitKernel<S, F, D>` owns source-typed inner commitment. Its single
  group method is the canonical boundary for singleton and batched sources.
- `DigitRowsComputeBackend<F>` and `CyclicRowsComputeBackend<F>` own reusable
  row arithmetic. `RingSwitchRelationKernel<S, F, D>` owns the complete
  source-typed ring-switch relation operation.
- `CpuBackend` prepares `CpuPreparedSetup<F>` from an `AkitaProverSetup<F>` or
  an `Arc<AkitaExpandedSetup<F>>`. Per-dimension NTT caches live inside the
  prepared stack. Matrix-consuming kernels lazily acquire exact prefixes keyed
  by ring dimension and transform domain.

Callers prepare once, then pass both the backend and prepared setup into prover
entrypoints:

```rust
let scheme = AkitaCommitmentScheme::<Cfg>::from_schedule_artifact(&artifact_bytes)?;
let setup = scheme.setup_prover(nv, num_polys)?;
let backend = CpuBackend::DEFAULT;
let prepared = backend.prepare_setup(&setup)?;
let stack = UniformProverStack::uniform(
    &backend,
    &prepared,
    setup.expanded.as_ref(),
)?;
let commit_output = scheme.commit(
    &setup,
    polys,
    &stack,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

Applications may replace the default with
`CpuBackend::with_resource_limits(max_cached_ring_switch_elements,
commit_scratch_bytes_per_worker)`. A zero ring switch limit streams every
operation that has a streamed path. `usize::MAX` retains every supported ring
switch operation. The constructor rejects a zero commitment scratch budget.
Each sparse commitment kernel rejects the operation later if its minimum tile
does not fit. These settings change memory use and CPU work. Cached and streamed
ring-switch routes use the same validated quotient arithmetic, including the
same exact field fallback when one centered term is unsafe in CRT form. The
settings do not change the schedule, transcript, setup bytes, proof bytes, or
verifier.

Ring dimension enters only at kernel boundaries through schedule-derived dispatch,
not as a type parameter on the PCS API.

One hot sources cross this boundary as validated `OneHotView` values. The CPU
kernel derives one flat sparse block tile from those views, selects a private
bucketed or merge sweep, and drops the tile after producing its commitment
rows. `OneHotPoly` does not own block caches. Opening derives its active data
for the lifetime of the operation. Recursive `EvaluationTrace` suffixes derive
their tensor data from the committed recursive witness.

## NTT lifecycle

`NttExecutionRequirements` describes the matrix work for one proof. It does not
choose a cache policy. `prewarm_ntt_requirements` routes each requirement to the
backend that will run it. That backend uses the same retention decision for
prewarming, memory reporting, and runtime execution. The CPU backend skips full
slots for large ring switch operations because those kernels stream transform
chunks from the public matrix.

Each routed requirement keeps the complete operation extent until that
decision is made. Requests from cached and streamed operations are therefore
not joined prematurely. After streamed requests are removed, planned memory
reporting max-joins the retained prefixes by physical cache owner, ring
dimension, and transform domain.

Prepared caches remain resident across proofs by default. This is the normal
choice for shared prepared state. `ReleaseRootNttAfterFold` is an explicit
memory policy for a caller that owns an isolated root cache. It releases each
physical owner once after the root fold.

CPU release removes built keys from the shared matrix NTT cache and returns the
checked sum of the bytes removed. A later request creates its exact extent
unless another populated covering slot exists. Readers that already hold an
`Arc` remain valid. Release does not stop construction already in progress. A
caller that needs the shared matrix cache to be empty after release must prevent
concurrent construction at that boundary.

Compression NTT entries are small and reusable, so generic and root release
leave them resident. This avoids repeated compression setup work while giving
up only a small memory reduction.

`CpuPreparedSetup` reports the shared matrix and compression cache byte counts
separately. `ntt_cache_bytes` is their checked sum and reports the complete
resident CPU NTT footprint. Planned requirement metrics remain specific to the
shared matrix cache because compression entries are created by compression
operations rather than by the proof schedule.

The lifecycle sequence is:

```text
prepare empty state
prewarm retained requirements
stream nonretained operations during the proof
retain slots for another proof, or release at an exclusive boundary
rebuild released shared matrix slots at the next exact request
```

## Boundary Rules

- Protocol code owns transcript order, challenge squeezes, batching order, and
  proof object construction.
- Backends run named operations and return rows or witnesses. They do not
  absorb to or squeeze from transcripts.
- Prepared compute state carries only setup artifact digests for identity
  checks. Prover APIs still take explicit setup metadata and reject a prepared
  context built from a different setup.
- Backend operations return `Result<_, AkitaError>` whenever a future
  accelerator may need to report unsupported shape, device, or submission
  failure.
- Migrated prover code must not accept legacy per-`D` NTT slot caches directly.
  CPU NTT slots stay inside `CpuPreparedSetup` / `ProverComputeStack`.
- Root commit kernels consume borrowed dense and one-hot source views.
  Recursive-witness sources do not cross a public representation-specific
  row-plan boundary.
- The group method is the only root commitment method. A singleton call passes
  one source. Backends cannot replace a fused group operation with an optional
  default loop.
- One-hot compact block storage is private to its source or operation. An
  accelerator integration should implement the source-typed
  kernel for its backend instead of depending on CPU storage plans.
- Dynamic ring-dimension code uses `dispatch_for_field!` and prepares the
  target backend context inside the matched `D` arm.
- Cached and streamed ring-switch relation routes consume the same validated
  source abstraction. Width dispatch selects CRT arithmetic only; route
  selection does not change input acceptance.

## Current Scope

The CPU cutover routes root commit, prove, and ring-switch work through
`CpuBackend`, `ProverComputeStack`, and source-typed kernels. Setup-owned CPU
NTT caches live in `CpuPreparedSetup` only.

Covered operation families:

- dense, one-hot, and recursive-witness commitment through `RootCommitKernel`;
- dense cached digits remain an internal CPU optimization;
- opening fold / decompose-fold, plus suffix-only tensor projection (single +
  batch);
- single-row cyclic and negacyclic digit rows;
- ring-switch relation rows, including the D-domain quotient inputs, via
  `RingSwitchRelationKernel`.

**Prove routing:** `batched_prove` takes `&impl LevelProveStacks`. Each fold
selects a `ProverComputeStack<C, O, TS, R>`; commit / opening / tensor /
ring-switch call the matching `OperationCtx`. `TieredProveStacks` supports
per-fold backend tiers; `UniformProverStack::uniform(cpu)` is the degenerate
single-backend case.

## Deferred Work

Deferred accelerator work should be split into fresh current specs when it
becomes active:

- `akita-metal` device/runtime skeleton with one tiny deterministic dispatch;
- production Metal ring/NTT kernels;
- fused inner-commit witness operations that return decomposed digits and
  recomposed rows together for device backends;
- base-field and MLE kernels tied to concrete prover consumers;
- stage-1/stage-2 sumcheck backend hooks;
- deterministic true CPU/GPU hybrid scheduling;
- Jolt/Akita adapter APIs for opening obligations.
