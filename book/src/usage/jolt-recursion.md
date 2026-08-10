# Jolt recursion

> **Status:** stub. Part of the initial Akita Book scaffold.

The standalone `profile/akita-recursion/` sub-workspace (excluded from the main
workspace; Rust 1.95 + RISC-V): the artifact → host → guest flow, the
`AkitaJoltInputs` blob, cycle accounting (Jolt guest pins **`fp128::OneHot`**
at its D256 root envelope; D=32 A-role presets are removed), the
trusted-benchmark vs production-validation distinction, and the nv=32
full-prove trace-length limit. Link the sub-workspace README rather than
duplicating its cycle tables.

**Akita revision pinning:** hosts must pin an exact Akita git tag or commit.
The zk-strip migration used a temporary pinned-proof-byte test
(`transparent_proof_golden`); that tripwire is removed after PR 4f lands so the
protocol can evolve without digest churn. Re-validate your integration on every
Akita upgrade with your host harness, not retired SHA-256 constants.

## Porting the modular Jolt prover

Jolt PR 1732 uses two memory release points with its original optimized Akita
pin. Preserve both when updating it to the current Akita API.

First, release the large shared matrix NTT entries after stage zero finishes
the trace commitment. Propagate any release error through the Jolt setup or
prover error type. The small compression NTT cache remains resident and does
not rebuild at this boundary.

Second, wrap the stack used for Akita opening proofs with
`ReleaseRootNttAfterFold`. This releases the root shared matrix entries before
the recursive suffix. The default stack retains them, so using
`UniformProverStack` alone does not preserve the memory policy from the
original Jolt branch.

```rust
let backend = CpuBackend::DEFAULT;
let prepared = backend.prepare_setup(&akita_setup)?;
let stack = UniformProverStack::uniform(
    &backend,
    &prepared,
    akita_setup.expanded.as_ref(),
)?;

// Stage zero, after the trace commitment.
let _freed_shared_bytes = prepared.drop_built_ntt_slots()?;

// Opening proof.
let releasing_stack = ReleaseRootNttAfterFold::new(stack);
AkitaCommitmentScheme::<Cfg>::batched_prove(
    &akita_setup,
    opening,
    &releasing_stack,
    transcript,
    BasisMode::Lagrange,
)?;
```

Jolt may use `CpuBackend::with_resource_limits` instead of the default. Keep
the chosen backend value alive for as long as the prepared stack borrows it.
The default retains ring switch operations through `2^21` ring elements and
allows 8 MiB of sparse commitment scratch space for each worker. Change these limits only
after measuring the complete Jolt prover. They change CPU work and memory use,
but they do not change Akita proof bytes.

The Jolt adapter must also follow the current source ownership API. A custom
trace one hot source implements `commit_inner_group` and returns witnesses in
source order. It must not restore the removed root storage release hook. Each
operation owns its derived opening and tensor data.

After the dependency update compiles, run the Jolt Akita checks from its own
workflow:

```bash
cargo nextest run --cargo-profile ci -p jolt-prover-legacy --features akita
cargo nextest run --cargo-profile ci -p jolt-prover \
  --features akita,prover-fixtures --test-threads 1
cargo nextest run --cargo-profile ci -p jolt-verifier \
  --features akita,prover-fixtures --test-threads 1
```

Then run the modular benchmark with `--features akita,prover-fixtures`. Record
the proving time and peak RSS from the same scale and host used for the old
pin. Do not compare runtime numbers until the byte parity and verifier suites
pass.

## Sources to fold in

- `profile/akita-recursion/README.md` (canonical runbook)
- `profile/akita-recursion/glue/src/lib.rs`
- `specs/akita-crate-followup-jolt-integration.md`
- `specs/archive/2026-Q3/pr375-prover-streaming-and-onehot-unification.md`
