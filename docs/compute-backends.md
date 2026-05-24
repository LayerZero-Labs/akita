# Akita Compute Backends

Akita prover compute is now routed through an explicit backend operation
boundary. The first implementation is `CpuBackend`; Metal and true hybrid
scheduling remain follow-up work.

## Ownership

- `AkitaExpandedSetup<F>` owns setup data shared with verifier/protocol code:
  seed, shared matrix, descriptor digest, and setup shape.
- `AkitaProverSetup<F, D>` is a prover setup wrapper around expanded setup. It
  does not own CPU NTT caches, device buffers, command queues, or any
  backend-prepared state.
- `ComputeBackendSetup<F>` owns backend preparation. Its associated
  `PreparedSetup<D>` is typed by ring dimension.
- `DigitRowsComputeBackend<F>`, `CyclicRowsComputeBackend<F>`,
  `CommitmentComputeBackend<F>`, and `RingSwitchComputeBackend<F>` own the
  migrated operation families.
- `CpuBackend` prepares `CpuPreparedSetup<F, D>` from an
  `AkitaProverSetup<F, D>` or an `Arc<AkitaExpandedSetup<F>>`. That prepared
  state owns the CPU NTT cache internally.

Callers prepare once, then pass both the backend and prepared setup into prover
entrypoints:

```rust
let setup = AkitaCommitmentScheme::<D, Cfg>::setup_prover(nv, num_polys, points)?;
let backend = CpuBackend;
let prepared = backend.prepare_setup(&setup)?;
let (commitment, hint) =
    AkitaCommitmentScheme::<D, Cfg>::commit(&backend, &prepared, polys)?;
```

## Boundary Rules

- Protocol code owns transcript order, challenge squeezes, batching order, and
  proof object construction.
- Backends run named operations and return rows or witnesses. They do not
  absorb to or squeeze from transcripts.
- Backend operations return `Result<_, AkitaError>` whenever a future
  accelerator may need to report unsupported shape, device, or submission
  failure.
- Migrated prover code must not accept `&NttSlotCache<D>` directly. CPU NTT
  slots stay inside `CpuPreparedSetup`.
- One-hot and sparse-ring plans expose read-only entry views so future
  out-of-crate backends can consume the compact representation without
  reaching into CPU storage.
- Dynamic ring-dimension code uses typed dispatch and prepares the target
  backend context inside the matched `D` arm.

## Current Scope

The current CPU cutover covers root commit/ring-switch work that was previously
wired through setup-owned CPU caches:

- dense coefficient and pre-decomposed digit commit rows;
- one-hot and sparse-ring commit rows without dense materialization;
- recursive witness commit rows;
- single-row cyclic and negacyclic digit rows;
- cyclic and quotient relation rows for ring switch.

## Deferred Work

Deferred accelerator work should be split into fresh current specs when it
becomes active:

- `akita-metal` device/runtime skeleton with one tiny deterministic dispatch;
- production Metal ring/NTT kernels;
- fused inner-commit witness operations that return decomposed digits and
  recomposed rows together for device backends;
- flat one-hot and sparse-ring plan tables for accelerator upload;
- base-field and MLE kernels tied to concrete prover consumers;
- stage-1/stage-2 sumcheck backend hooks;
- deterministic true CPU/GPU hybrid scheduling;
- Jolt/Akita adapter APIs for opening obligations.
