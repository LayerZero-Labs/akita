# Akita Compute Backends

Akita prover compute is now routed through an explicit backend operation
boundary. The first implementation is `CpuBackend`; Metal and true hybrid
scheduling remain follow-up work.

## Ownership

- `AkitaExpandedSetup<F>` owns setup data shared with verifier/protocol code:
  seed, shared matrix, descriptor digest, and setup shape.
- `AkitaProverSetup<F>` is a D-free prover setup wrapper around expanded setup.
  It stores runtime `gen_ring_dim` and does not own CPU NTT caches, device
  buffers, command queues, or any backend-prepared state.
- `ComputeBackendSetup<F>` owns backend preparation. Prepared setup slots are
  keyed by field family and ring role at kernel boundaries via `dispatch_for_field!`.
- `DigitRowsComputeBackend<F>`, `CyclicRowsComputeBackend<F>`,
  `CommitmentComputeBackend<F>`, and `RingSwitchComputeBackend<F>` own the migrated operation families.
- `CpuBackend` prepares `CpuPreparedSetup<F>` from an `AkitaProverSetup<F>` or
  an `Arc<AkitaExpandedSetup<F>>`. Per-dimension NTT caches live inside the
  prepared stack and are warmed from schedule-derived role dimensions.

Callers prepare once, then pass both the backend and prepared setup into prover
entrypoints:

```rust
let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, num_polys, points)?;
let backend = CpuBackend;
let prepared = backend.prepare_setup(&setup)?;
let (commitment, hint) =
    AkitaCommitmentScheme::<Cfg>::commit(&setup, &backend, &prepared, polys)?;
```

Ring dimension enters only at kernel boundaries through schedule-derived dispatch,
not as a type parameter on the PCS API.

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
- One-hot and sparse-ring plans expose flat entry and offset tables so future
  out-of-crate backends can upload the compact representation without reaching
  into CPU storage.
- Dynamic ring-dimension code uses `dispatch_for_field!` and prepares the
  target backend context inside the matched `D` arm.

## Current Scope

The CPU cutover routes root commit, prove, and ring-switch work through
`CpuBackend`, `ProverComputeStack`, and source-typed kernels. Setup-owned CPU
NTT caches live in `CpuPreparedSetup` only.

Covered operation families:

- dense coefficient and pre-decomposed digit commit rows;
- one-hot and sparse-ring commit rows without dense materialization;
- recursive witness commit rows;
- opening fold / decompose-fold / tensor projection (single + batch);
- single-row cyclic and negacyclic digit rows;
- ring-switch relation and quotient rows via `RingSwitchRelationKernel` /
  `RingSwitchQuotientKernel`.

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
