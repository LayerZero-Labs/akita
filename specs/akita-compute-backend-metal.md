# Spec: Akita Compute Backend Cutover

| Field       | Value                 |
|-------------|-----------------------|
| Author(s)   | @quangvdao, Codex     |
| Created     | 2026-05-24            |
| Status      | current PR            |
| PR          | #105                  |

## Summary

Akita's prover currently has CPU-specific compute state wired through public
prover APIs, polynomial backends, setup construction, and protocol flow. The
most visible symptom is `NttSlotCache<D>`: it lives inside
`AkitaProverSetup`, appears as the default `AkitaPolyOps::CommitCache`, is
threaded through `akita-scheme`, and is lazily rebuilt by dynamic-D dispatch
macros. Adding Metal beside this shape would create a second execution layer
without replacing the CPU-shaped one.

This spec defines the first clean rearchitecture slice: make prover compute an
explicit host-prepared boundary, move the existing ring/commit CPU path behind
that boundary, remove CPU NTT cache ownership from protocol-facing setup and
polynomial APIs, and record the inventory/baselines needed to review the
cutover honestly. The prepared boundary is a small `ComputeBackendSetup` trait
with an associated prepared setup, plus operation-family traits layered on top:
`DigitRowsComputeBackend`, `CyclicRowsComputeBackend`,
`CommitmentComputeBackend`, `RingSwitchComputeBackend`, and the convenience
umbrella `ProverComputeBackend` for call sites that need the full first-PR
operation set. The current PR implements only `CpuBackend`, whose prepared
state is `CpuPreparedSetup<F, D>`. Migrated hot paths call named backend
operations rather than reaching through raw CPU setup matrices or NTT slots.
Metal skeleton work, field kernels, MLE kernels, sumcheck kernels, true hybrid
scheduling, and Jolt adapter work are captured as one remaining-work bucket that
should be expanded into its own detailed spec when that work becomes current.

## Intent

### Goal

Cut over Akita's prover setup and root ring/commit compute path to an explicit
typed compute-backend operation boundary, with `CpuBackend` preserving current
behavior exactly.

The key surfaces modified by this spec are:

- `akita-prover::compute`: new plan/result types, `ComputeBackendSetup`,
  operation-family compute backend traits, `CpuBackend`,
  `CpuPreparedSetup<F, D>`, and the migrated ring/commit operation interface.
- `AkitaProverSetup`: becomes protocol/setup data only; it must not own
  `NttSlotCache`, `MultiDNttCaches`, Metal buffers, command queues, or any
  backend-prepared cache.
- `CommitmentProver`: public prover methods are changed in one pass to require
  an explicit backend plus its typed prepared compute context. There is no
  permanent old API plus optional prepared-context API.
- `AkitaPolyOps`: root polynomial representation logic is cut through rather
  than bypassed. Migrated commit methods must not expose
  `CommitCache = NttSlotCache<D>` or call `mat_vec_mul_ntt_*` directly.
- `akita-prover::protocol` and `akita-scheme`: migrated paths receive a
  backend and its typed prepared setup through the prover call graph; they do
  not thread `&setup.ntt_shared`.
- `akita-verifier` and `akita-types`: unchanged in role. They must remain free
  of prover compute, Metal, and backend cache dependencies.

### Invariants

1. Verifier-visible commitments, proofs, setup descriptors, opening claims,
   transcript labels, and challenge order are unchanged by the CPU
   compute-backend cutover.
2. Backends must not append to or squeeze from the Fiat-Shamir transcript.
   Transcript binding remains in `akita-prover` and `akita-scheme` protocol
   flow.
3. Backends must not choose sumcheck order, batching order, opening points,
   opening IDs, recursive schedule steps, or proof object structure.
4. Backend results are keyed by protocol plan slots, never by backend-invented
   semantic IDs.
5. `akita-verifier` must not depend on `akita-prover`, future accelerator
   crates such as `akita-metal`, runtime crates, device buffers, prover
   polynomial backends, or backend caches.
6. CPU remains the exact reference implementation. The first cutover must keep
   deterministic CPU proof behavior byte-identical where randomness is absent
   and transcript-event-identical everywhere else.
7. `AkitaProverSetup` and `AkitaInstanceDescriptor` must not include device
   placement, command queue state, buffer addresses, runtime timing, or
   backend cache identity.
8. Prepared compute caches are derived from the same `AkitaExpandedSetup`,
   schedule parameters, field family, ring dimension, and CRT/NTT parameter
   family as the current CPU path.
9. Future accelerator availability is a host/backend capability, not a
   protocol assumption. The current CPU backend path must compile and run
   without accelerator crates.
10. There is exactly one migrated compute path. After a path moves to
    `akita-prover::compute`, neither protocol code nor `AkitaPolyOps`
    implementations may call `NttSlotCache`, `MultiDNttCaches`,
    `dispatch_with_ntt!`, or `mat_vec_mul_ntt_*` directly for that path.
11. Existing optimized representations must remain visible to compute
    planning: dense, one-hot, compact digit, sparse ring, and recursive witness
    paths must not be forced through dense `Vec<F>` materialization just to use
    a backend.
12. No compatibility shims, deprecated aliases, or long-lived parallel APIs are
    introduced. This repo makes no backward-compatibility guarantee; update all
    call sites in one pass.
13. Migrated const-generic prepared state must stay typed. Do not introduce
    `Any`/downcast maps, `usize -> Box<dyn ...>` cache registries, or mutexed
    hot-path lookup to recover `D` at runtime. Dynamic-D protocol arms may
    prepare a new typed `CpuPreparedSetup<F, D_LEVEL>` inside the dispatch arm
    and return `AkitaError` on unsupported dimensions.
14. The prepared setup boundary must not become a raw accessor layer. Migrated
    code outside `akita-prover::compute` must not call methods such as
    `shared_matrix()`, expose `&NttSlotCache<D>`, or inspect backend-specific
    storage. It requests named operations from the backend instead.
15. Canonical host setup metadata remains separate from compute. Prover APIs
    receive setup metadata explicitly; the compute backend trait does not become
    the source of `AkitaExpandedSetup` or descriptor information. Prepared
    compute state may expose only cached setup artifact digests to reject
    mismatched setup/prepared-context pairs.

### Non-Goals

1. Changing the Akita PCS protocol, proof layout, schedule policy, SIS
   parameters, Fiat-Shamir labels, verifier equations, or serialized proof
   structures.
2. Adding `crates/akita-metal`, Metal skeleton code, or production Metal field,
   MLE, sumcheck, or ring kernels in this current PR.
3. Implementing true CPU/GPU hybrid split execution. Accelerator fallback and
   deterministic CPU/GPU partitioning are follow-up work.
4. Completing the Jolt adapter in this branch.
5. Implementing a Dory-style homomorphic RLC interface for Akita.
6. Replacing Rayon as the CPU backend. Existing Rust/Rayon kernels become the
   CPU backend implementation.
7. Making Metal mandatory for `cargo test`, default features, verifier builds,
   or non-Apple builds.
8. Hiding Jolt field-boundary issues inside the PCS adapter. A future adapter
   must reject incompatible fields unless a separate transcript-bound
   conversion spec exists.

## Evaluation

### Acceptance Criteria

- [x] `AkitaProverSetup<F, D>` no longer stores `ntt_shared:
      NttSlotCache<D>` or any other backend-prepared cache.
- [x] `AkitaProverSetup::generate_with_capacity` and
      `AkitaProverSetup::from_expanded` do not build CPU NTT caches. CPU caches
      are prepared lazily by `CpuBackend` from `AkitaExpandedSetup`.
- [x] `CommitmentProver::commit`, `CommitmentProver::batched_commit`, and
      `CommitmentProver::batched_prove` require an explicit backend plus that
      backend's typed prepared context after the cutover, and all in-repo
      callers are updated. `CommitmentProver::setup_prover` returns
      `AkitaError` instead of panicking on setup construction failure. No old
      peer API remains.
- [x] `AkitaPolyOps` no longer exposes `CommitCache = NttSlotCache<D>` for
      migrated commit paths. Dense, one-hot, sparse-ring, root-projection, and
      recursive witness implementations route migrated ring mat-vec work
      through representation-aware backend operations.
- [x] `akita-scheme` has no import or direct use of `NttSlotCache`,
      `MultiDNttCaches`, `dispatch_with_ntt!`, or `mat_vec_mul_ntt_*` after the
      migrated compute-backend cutover.
- [x] Migrated protocol functions in `flow.rs`, `ring_switch.rs`, and
      `quadratic_equation.rs` receive an explicit backend plus typed prepared
      setup through the call graph; they do not accept `&NttSlotCache<D>` for
      migrated paths.
- [x] Backend operations that may fail on an accelerator return
      `Result<_, AkitaError>`. This includes single-row digit mat-vec, cyclic
      digit mat-vec, and ring-switch relation rows; future Metal backends must
      not need to panic or silently fall back to CPU to report device or shape
      failure.
- [x] Dynamic-D setup preparation returns `AkitaError` from typed dispatch arms
      instead of panicking through CPU-cache-specific dispatch for migrated
      paths.
- [x] `CpuBackend` implements `ComputeBackendSetup`,
      `DigitRowsComputeBackend`, `CyclicRowsComputeBackend`,
      `CommitmentComputeBackend`, and `RingSwitchComputeBackend`, delegating
      to the existing CPU kernels internally.
- [x] The compute boundary contains no type-erased prepared-cache map, runtime
      downcast, or mutexed hot-path lookup for const-generic prepared state.
- [x] The compute boundary contains no public raw prepared-setup accessor used
      by migrated polynomial/protocol code. In particular, one-hot and
      sparse-ring commit paths must not reach through `CpuPreparedSetup` to
      inspect `shared_matrix` or `NttSlotCache`.
- [x] Prepared compute state is checked against explicit setup metadata by
      setup artifact digest identity; public APIs reject a prepared context
      built from a different setup without making the backend the owner of
      canonical setup metadata.
- [x] Representation-aware plan data needed by future out-of-crate backends is
      readable through public plan variants/accessors. The trait must not
      require `akita-metal` or another accelerator crate to live inside
      `akita-prover` just to inspect one-hot or sparse-ring entries.
- [x] CPU backend proofs for `commit`, `batched_commit`, and `batched_prove`
      pass the existing dense, one-hot, multipoint, recursive, and transcript
      hardening tests.
- [x] Deterministic CPU proof bytes remain unchanged where current tests can
      assert bytes; otherwise `LoggingTranscript` prover/verifier event streams
      remain unchanged.
- [x] The migrated ring/commit plan explicitly covers the current root commit
      and ring-switch NTT work used by `compute_v_rows`, `compute_r_split_eq`,
      `commit_w`, `commit_next_w_with_policy`, dense digit mat-vec, dense
      coefficient mat-vec, single-row cyclic/negacyclic variants, and strided
      recursive variants.
- [x] Ring/CRT parity tests enumerate concrete `(field family, D)` tuples that
      exercise the actual Q32, Q64, and Q128 dispatch branches, including both
      negacyclic and cyclic cache paths.
- [x] `akita-verifier` still has no normal dependency path to `akita-prover`,
      `akita-metal`, Metal runtime crates, examples, benches, or prover
      polynomial backends.
- [x] Documentation explains the new CPU backend selection, prepared setup
      ownership, and which accelerator/kernel families are intentionally
      deferred.

### Testing Strategy

Required baseline checks:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `cargo test -p akita-verifier --no-default-features`
- `scripts/check-crate-deps.sh akita-verifier`
- `scripts/check-crate-deps.sh akita-prover`

CPU cutover tests:

- Existing integration tests under `crates/akita-pcs/tests/`, especially
  `single_poly_e2e.rs`, `multipoint_batched_e2e.rs`,
  `batched_aggregated_e2e.rs`, `ring_switch.rs`, `stage1_roundtrip.rs`,
  `sumcheck_core.rs`, and transcript hardening tests.
- Existing scheme tests under `crates/akita-scheme/src/tests.rs`.
- `cargo test -p akita-sumcheck --test drivers`
- `cargo test -p akita-prover --lib`
- `cargo test -p akita-scheme --lib`

New focused tests:

- `AkitaProverSetup::from_expanded` does not build CPU NTT state.
- `CpuBackend::prepare_setup` builds typed CPU prepared state with the same NTT
  data that `AkitaProverSetup` used to build.
- No `NttSlotCache` import in `akita-scheme`.
- No `dispatch_with_ntt!` use in migrated paths.
- No migrated one-hot/sparse-ring path calls a raw `shared_matrix()` prepared
  setup accessor.
- CPU backend parity for `compute_v_rows`, `compute_r_split_eq`, `commit_w`,
  `commit_next_w_with_policy`, dense digit mat-vec, dense coefficient mat-vec,
  single-row variants, and strided recursive variants.
- Transcript-event equality around prover `v` absorption and the stage
  challenge squeezes that follow it.
- Concrete CRT/NTT branch tests:
  - Q32: small field with `D <= RING_DEGREE`;
  - Q64: small field with larger supported `D`, and native <=64-bit field;
  - Q128: one representative whitelisted 128-bit modulus family that exercises
    the Q128 dispatch branch, plus additional named families only when they
    protect a known regression or parameter-selection bug.

### Performance

The current cutover is judged by CPU non-regression and by making future
accelerator timings possible through explicit prepared-setup ownership.

Required benchmark coverage:

- `cargo bench --bench root_kernels`
- `cargo bench --bench ring_ntt`
- `cargo bench --bench onehot_batched_commit`
- `cargo bench --bench onehot_batched_opening`
- `AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile`

`root_kernels` is a low-level CPU-kernel baseline that intentionally still
constructs CPU NTT slots directly. End-to-end commit/opening/profile commands
exercise the backend operation boundary.

Expected outcomes:

- CPU execution through the new backend boundary is within 2% of the previous CPU
  direct-kernel median on unchanged hardware, or the regression is explained by
  benchmark variance and fixed before the cutover is considered complete.
- CPU prepared-setup time is measured separately from repeated commit/prove
  execution time.

## Design

### Architecture

The first cutover replaces CPU-shaped protocol plumbing with explicitly
prepared typed backend operations.

```mermaid
graph TD
  Types["akita-types<br/>proof/setup/descriptor shapes"]
  Verifier["akita-verifier<br/>verification only"]
  Prover["akita-prover<br/>protocol flow + compute traits"]
  Compute["ComputeBackendSetup<br/>prepared-state boundary"]
  Ops["operation-family traits<br/>linear / commitment / ring-switch"]
  Cpu["CpuBackend + CpuPreparedSetup<br/>current Rust/Rayon kernels"]
  Scheme["akita-scheme<br/>accepts backend + prepared setup"]
  Host["akita-pcs / host / future jolt-akita<br/>prepares backend context"]

  Types --> Verifier
  Types --> Prover
  Prover --> Compute
  Compute --> Ops
  Ops --> Cpu
  Scheme --> Prover
  Host --> Scheme
```

Host layers, examples, tests, or a future adapter construct a backend, prepare
that backend's associated setup from prover setup, and pass both into the
prover API. The current PR wires `CpuBackend` only. Future accelerator crates
must follow the same explicit-preparation shape without making `akita-scheme`
or verifier crates depend on device code. The verifier crate remains outside
this graph.

### Public API Cutover

The public prover API changes in one pass. The existing method names may remain
if their signatures gain explicit setup metadata, an explicit backend, and that
backend's typed prepared-context argument. The old setup-only methods are
removed. The prepared context is owned by the backend that created it, while
canonical setup metadata remains owned by `AkitaProverSetup`. Public entrypoints
compare the prepared context's setup artifact digests with the explicit setup
before using either.

Representative shape:

```rust
pub trait CommitmentProver<F, const D: usize>
where
    F: FieldCore + CanonicalField,
{
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync;
    type Commitment: Clone + Send + Sync;
    type ClaimField: ExtField<F>;
    type CommitHint: Clone + Send + Sync;
    type BatchedProof: Clone + Send + Sync;

    fn commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys: &[P],
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitmentComputeBackend<F>;

    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys_per_point: &[&[P]],
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitmentComputeBackend<F>;

    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        claims: ProverClaims<'a, Self::ClaimField, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F>,
        P: AkitaPolyOps<F, D>,
        B: ProverComputeBackend<F>;
}
```

The exact generic spelling may change, but the design requirements are fixed:

- backend and prepared compute context selection are explicit at every prover
  entrypoint;
- the old `Cache = NttSlotCache<D>` generic disappears from
  `CommitmentProver`;
- migrated hot paths borrow `&B` and `&B::PreparedSetup<D>`; they do not ask a
  mutable runtime to recover const-generic state by downcast;
- migrated hot paths call named backend operations; they do not recover CPU
  internals through raw prepared-setup accessors;
- tests and benches call the new API directly;
- no compatibility method silently constructs CPU prepared setup behind the old
  API.

### `AkitaPolyOps` Cutover

`AkitaPolyOps` currently owns representation-specific work and exposes
CPU-shaped commit cache plumbing. The cutover must preserve representation
knowledge while removing CPU cache ownership from the trait.

Target rules:

- remove or replace `type CommitCache = NttSlotCache<D>` for migrated paths;
- migrated methods that need backend work receive a backend plus prepared
  context rather than a CPU cache;
- dense, one-hot, sparse-ring, root-projection, and recursive witness
  implementations build representation-aware operation requests and submit
  them to the backend;
- one-hot and sparse-ring implementations must preserve their compact
  representation and sparse planning without reading the prepared setup's raw
  shared matrix;
- the representation-aware plans expose read-only views of their entries so an
  out-of-crate backend can implement the trait without CPU internals or crate
  privacy privileges;
- representation-specific operations that are not migrated yet may remain CPU
  local, but they must not participate in the migrated ring/commit path through
  hidden `NttSlotCache` calls;
- the implementation must update `commit_inner`, `commit_inner_witness`,
  `decompose_fold`, `evaluate_and_fold`, and recursive witness call sites
  enough that the migrated commit/prove path has one compute boundary.

This is the main guard against adding a backend layer beside the existing
polynomial abstraction.

### Setup And Cache Ownership

Target ownership:

```text
AkitaExpandedSetup<F>
  shared matrix, seed, descriptor digest, verifier-reachable setup data

AkitaProverSetup<F, D>
  prover setup wrapper around AkitaExpandedSetup, no backend-prepared cache

CpuPreparedSetup<F, D>
  Arc<AkitaExpandedSetup<F>>, NttSlotCache<D>, CPU scratch/cached matrices
  private to CpuBackend operation implementations

FutureAcceleratorPreparedSetup<F, D>
  device buffers/pipelines for supported matrix slices and CRT parameter family
```

Prepared setup is explicitly host-layer owned and backend-typed. It is
constructed by the backend's `prepare_setup` method and then borrowed by
migrated prover hot paths only together with that backend. It is identified by:

- expanded setup descriptor digest;
- ring dimension `D`;
- field modulus family;
- CRT/NTT parameter family;
- backend name and backend cache version.

The current PR must not implement this as a mutable registry of erased cache
objects. If a protocol path needs a different const-generic ring dimension, it
enters the typed dynamic-D dispatch arm and prepares a backend-specific
`PreparedSetup` for `D_LEVEL` there. For the CPU backend, that concrete type is
`CpuPreparedSetup<F, D_LEVEL>`.

The current PR must also avoid a misleading halfway design where prepared setup
is technically explicit but every migrated caller reaches through it to pull
out CPU internals. `CpuPreparedSetup` may own `NttSlotCache<D>` and the expanded
setup, but its raw matrix and NTT slot are not the abstraction. The abstraction
is the backend operation methods that consume representation-aware plan data.

`AkitaProverSetup::generate_with_capacity` and `from_expanded` construct or
wrap expanded setup only. Disk-persistence or setup-cache paths must not
eagerly rebuild CPU NTT state outside the explicit compute-preparation path.

### Compute Operations

Phase-1 compute operations cover the current ring/commit work that is already
tied to `NttSlotCache`:

- shared NTT setup preparation from `AkitaExpandedSetup`;
- dense pre-decomposed digit mat-vec equivalent to
  `mat_vec_mul_ntt_dense_digits_i8`;
- dense ring-coefficient mat-vec equivalent to `mat_vec_mul_ntt_i8_dense`;
- single-row cyclic and negacyclic variants;
- strided recursive witness variants;
- `compute_v_rows` and the transcript-adjacent `v` computation before
  absorption;
- `compute_r_split_eq` and quotient/cyclic rows used by ring switch;
- `commit_w` and `commit_next_w_with_policy`.

Operation inputs contain dimensions, shape slots, representation handles, and
borrowed prepared setup. Fallible backend operations return `AkitaError` so
host/device failures and unsupported shapes are ordinary prover errors, not
panics. They do not contain transcript objects. Protocol code performs
transcript absorption after backend output is returned.

### Backend And Prepared Context Selection

```rust
pub trait ComputeBackendSetup<F>: Send + Sync
where
    F: FieldCore + CanonicalField,
{
    type PreparedSetup<const D: usize>: Send + Sync;

    fn prepare_setup<const D: usize>(
        &self,
        setup: &AkitaProverSetup<F, D>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError>;

    fn prepare_expanded<const D: usize>(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<D>, AkitaError>;

    fn prepared_setup_digests<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
    ) -> SetupArtifactDigests;

    fn validate_prepared_setup<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<(), AkitaError>;
}

pub trait DigitRowsComputeBackend<F>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

pub trait CyclicRowsComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        row_len: usize,
        digits: &[[i8; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
}

pub trait CommitmentComputeBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    fn dense_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: DenseCommitRowsPlan<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;

    fn onehot_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: OneHotCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    fn sparse_ring_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: SparseRingCommitRowsPlan<'_>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;

    fn recursive_witness_commit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RecursiveWitnessCommitRowsPlan<'_, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>;
}

pub trait RingSwitchComputeBackend<F>: CyclicRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup<D>,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField;
}

pub trait ProverComputeBackend<F>:
    CommitmentComputeBackend<F> + RingSwitchComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
}

pub struct CpuBackend;

pub struct CpuPreparedSetup<F, const D: usize> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    ntt_shared: NttSlotCache<D>,
}
```

Full ring-switch relation rows and additional public-row quotient segments use
separate `RingSwitchRelationRowsPlan` / `RingSwitchQuotientRowsPlan` inputs and
separate backend methods, so quotient-only work is not represented with
sentinel zero D/B row counts.

Rules:

- the current PR implements only `CpuBackend`;
- backend and prepared context selection are still explicit so future
  accelerators do not require a second prover API;
- no prover path may construct hidden CPU caches outside prepared setup;
- dynamic ring-dimension dispatch for migrated paths prepares typed
  backend-specific prepared setup inside the dispatch arm and returns
  `AkitaError` rather than panicking;
- migrated parallel code borrows the backend and typed prepared setup
  immutably. It must not require `&mut` runtime access, interior mutability, or
  runtime downcasts in the hot path;
- `CpuPreparedSetup` fields remain implementation details. If a migrated caller
  needs shared setup rows or an NTT slot, that is a sign the compute operation
  boundary is missing a method.

Accelerator fallback policies such as `PreferAccelerator` and
`RequireAccelerator` are remaining work, not part of the current PR.

### Remaining Accelerator Shape

The next accelerator PR is expected to introduce `crates/akita-metal` with
Apple-specific runtime code shaped roughly as follows:

```text
crates/akita-metal/
  src/lib.rs
  src/device.rs
  src/buffer.rs
  src/pipeline.rs
  src/runtime.rs
  src/kernels/
    smoke.metal
```

Implementation rules:

- use safe Rust wrappers around device, queue, pipeline, buffer, and command
  submission objects;
- keep Objective-C/Metal FFI isolated to `device`, `buffer`, and `pipeline`
  modules;
- compile kernels at build time or load embedded source deterministically;
- start with shared-memory buffers for correctness and simpler debugging;
- expose deterministic `is_available()` and `capabilities()`;
- implement one tiny vector/integer dispatch with CPU-checked output;
- do not claim production field, MLE, sumcheck, or ring acceleration in the
  skeleton PR.

### Deferred Kernel Roadmap

These areas are intentionally deferred to follow-up specs after the CPU
compute-backend cutover is complete.

#### Field And MLE Kernels

Follow-up scope:

- elementwise add/sub/mul, FMA, scaling, affine combine;
- dot products and batched reductions;
- signed/unreduced accumulation matching `HasUnreducedOps`;
- `fold_evals_in_place` equivalents;
- `EqPolynomial` table generation;
- `GruenSplitEq` binding and remaining-table state transitions.

The follow-up spec must name the concrete migrated call paths before adding
acceptance criteria. Generic field kernels without a prover consumer are not
enough.

#### Sumcheck Kernels

Stage-1 and stage-2 are state machines, not stateless round-poly callbacks. A
future sumcheck backend spec must either keep the state machine on CPU and
define exact backend hooks, or move backend-owned round state behind a precise
contract.

Required future coverage:

- current-round scan;
- challenge ingestion/fold;
- optional fused next-round scan;
- cached round polynomial behavior;
- two-round-prefix materialization;
- stage-1 tree wrapper;
- stage-2 `live_x_cols`, `m_compact`, `alpha_compact`,
  `prev_norm_claim`, and `prev_norm_poly` transitions.

#### Ring/NTT Metal Kernels

Production Metal ring kernels are a follow-up after the CPU plan boundary and
Metal skeleton exist. The first production target should be dense
pre-decomposed digit mat-vec because it avoids mixing digit extraction, NTT,
and sparse zero-scan policy in the first kernel.

#### True Hybrid Scheduling

True hybrid means deterministic simultaneous CPU/GPU partitioning of one plan.
That is not CPU fallback and not simple asynchronous Metal dispatch. A
follow-up spec must define canonical split reductions and tests before enabling
it.

#### Jolt Adapter Readiness

This spec keeps only the Akita constraints needed for Jolt:

- verifier-only Jolt code must be able to depend on `akita-verifier` and
  `akita-types` without `akita-prover` or `akita-metal`;
- future Jolt host/prover code passes an Akita prepared compute context into
  an adapter;
- Jolt protocol order remains in `jolt-prover`;
- Akita protocol order remains in `akita-prover`;
- Akita should not implement fake Dory-style homomorphic RLC.

The actual opening-obligation API and adapter smoke tests belong in a separate
Jolt/Akita adapter spec.

### Alternatives Considered

1. Metal-only kernel crate without a compute cutover.
   Rejected because it would accelerate a few functions while leaving protocol
   flow coupled to CPU-only caches.

2. Add `*_with_prepared` APIs while preserving old setup-only APIs.
   Rejected because it creates two prover surfaces and lets call sites avoid
   the cutover.

3. Move CPU caches under `AkitaProverSetup::cpu_cache`.
   Rejected because setup would still own backend-prepared state and Metal
   would be bolted on beside it.

4. Include field, MLE, sumcheck, ring, hybrid, and Jolt adapter work in one
   acceptance surface.
   Rejected because that would be a roadmap, not a reviewable implementation
   milestone.

5. Wrap current stage-1/stage-2 provers with stateless backend callbacks.
   Rejected for this spec because the current provers fuse scan/fold work and
   maintain cached state across challenge ingestion.

## Documentation

Required documentation updates:

- Add `docs/compute-backends.md` describing the CPU compute-backend boundary,
  prepared setup ownership, and deferred accelerator roadmap.
- Update `docs/crate-graph.md` for the compute module.
- Update profiling documentation to show how to construct/select explicit
  `CpuBackend` preparation once exposed.
- Add a short note that field/MLE/sumcheck/hybrid/Jolt adapter work is
  intentionally deferred from this cutover.

## Execution

### Rolling Spec Split

At each stage of this project, keep the specs in three buckets:

1. Past PR specs: frozen after merge except for explicit errata.
2. Current PR spec: detailed enough to implement and review now.
3. Remaining work spec: a single coarse placeholder. Expand and split it only
   when that work becomes current.

This prevents the current PR from carrying a long speculative roadmap, while
still preserving the direction of travel.

### Past PR Specs

None yet. This is the first detailed spec in the stack.

After this PR merges, this document should move to the past-PR bucket and stay
unchanged except for errata. The next PR should get a new current spec derived
from the remaining-work placeholder below.

### Current PR: CPU Compute Backend Cutover

This PR combines the earlier PR 0/1/2 ideas because they are not independently
valuable code-review units. The spec/review record, inventory/baselines, and
CPU backend/API/setup/`AkitaPolyOps` cutover should land together as one real
code PR.

Scope:

- add this spec and preserve the design-review conclusions that narrowed the
  first milestone;
- capture CPU benchmark baselines for `root_kernels`, `ring_ntt`, one-hot
  commit/opening, and the release profile example;
- add a checked-in inventory of direct `NttSlotCache`, `MultiDNttCaches`,
  `dispatch_with_ntt!`, `mat_vec_mul_ntt_*`, `commit_w`,
  `commit_next_w_with_policy`, `compute_v_rows`, and `compute_r_split_eq`
  call sites;
- classify each old direct-cache call as in the current cutover, later
  follow-up, or provably outside the migrated root commit/ring-switch/prove
  path;
- inventory `AkitaPolyOps` methods and impls that participate in root commit,
  recursive witness commit, and ring-switch commit paths;
- add `akita-prover::compute` with `ComputeBackendSetup`, operation-family
  compute backend traits, `CpuBackend`, `CpuPreparedSetup<F, D>`, operation
  plan/result helpers, no erased prepared-cache registry, and no raw
  prepared-setup accessor boundary;
- move NTT cache construction from `AkitaProverSetup` into
  `CpuBackend::prepare_setup`;
- make `AkitaProverSetup::generate_with_capacity` and `from_expanded`
  expanded-setup-only with respect to CPU NTT state;
- change `CommitmentProver` entrypoints to require explicit setup metadata,
  an explicit backend, and that backend's typed prepared context;
- update all in-repo tests, benches, examples, `akita-scheme`, and aggregate
  crate call sites;
- remove `CommitCache = NttSlotCache<D>` from migrated `AkitaPolyOps` paths;
- route dense, one-hot, sparse-ring, root-projection, and recursive witness
  migrated commit work through representation-aware backend operations;
- replace migrated `dispatch_with_ntt!` and direct `NttSlotCache` protocol
  plumbing with typed prepared-setup dispatch returning `AkitaError`;
- add focused CPU parity tests and import/dependency guards needed to keep the
  new boundary honest;
- add descriptor/transcript equality tests around prover `v` absorption and
  stage challenge squeezes;
- update documentation for the CPU compute-backend boundary and deferred
  accelerator roadmap.

Done when:

- the workspace compiles and tests through explicit `CpuBackend` preparation
  and the compute-backend operation boundary;
- no old setup-only prover API remains;
- `AkitaProverSetup` does not own backend-prepared caches;
- `akita-scheme` no longer imports `NttSlotCache`, `MultiDNttCaches`,
  `dispatch_with_ntt!`, or `mat_vec_mul_ntt_*`;
- migrated paths have exactly one compute boundary;
- CPU proof behavior and transcript event streams are unchanged;
- CPU performance regression is within the accepted threshold or explained and
  fixed.

This PR is intentionally larger than an inventory PR and intentionally smaller
than a Metal PR. Splitting setup, public API, and `AkitaPolyOps` into separate
mergeable PRs would create the half-cutover state this spec is designed to
avoid.

### Remaining Work: Accelerator And Integration Stack

This is intentionally a single placeholder. Do not expand it into many detailed
subsections until the current CPU compute cutover is merged and this work
becomes current.

Likely contents when expanded:

- typed per-proof prepared schedule/session for the finite supported `D` arms,
  so accelerator backends do not rebuild/upload prepared state at each cross-D
  recursive transition;
- fused inner-commit witness operations that can return decomposed digits and
  recomposed rows together, so a device backend is not forced through an
  A-row host round trip before the B-side rows;
- Metal skeleton: `crates/akita-metal`, target-specific dependencies, device
  discovery, capability reporting, safe buffer wrappers, pipeline loading, one
  tiny deterministic dispatch, Apple-only smoke tests, and non-Apple compile
  tests;
- production Metal ring/NTT kernels, starting with dense pre-decomposed digit
  mat-vec;
- field and MLE backend kernels tied to concrete prover call paths;
- stage-1/stage-2 sumcheck backend state-machine hooks;
- deterministic true hybrid CPU/GPU scheduling;
- Jolt opening adapter and Akita grouped batched-opening interface.

The next PR should take only the first coherent code slice from this bucket and
turn it into a new detailed current spec. The rest stays coarse.

## References

- `specs/akita-pcs-crate-decomposition.md`
- `specs/akita-crate-followup-jolt-integration.md`
- `/Users/quang.dao/Documents/Notes/jolt-prover-model-crate.md`
- `/Users/quang.dao/Documents/Notes/jolt-prover-cpu-backend-port.md`
- `/Users/quang.dao/Documents/Notes/jolt-core-prover-optimization-inventory.md`
- `/Users/quang.dao/Documents/SNARKs/jolt-refactor-crates/crates/jolt-prover/src/stages/stage8.rs`
- `/Users/quang.dao/Documents/SNARKs/jolt-refactor-crates/crates/jolt-openings/src/schemes.rs`
- `crates/akita-prover/src/api/setup.rs`
- `crates/akita-prover/src/api/scheme.rs`
- `crates/akita-prover/src/lib.rs`
- `crates/akita-prover/src/protocol/dispatch.rs`
- `crates/akita-prover/src/protocol/flow.rs`
- `crates/akita-prover/src/protocol/ring_switch.rs`
- `crates/akita-prover/src/protocol/quadratic_equation.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage1.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage1_tree.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2.rs`
- `crates/akita-prover/src/kernels/crt_ntt.rs`
- `crates/akita-prover/src/kernels/linear.rs`
- `crates/akita-field/src/fields/wide.rs`
- `crates/akita-algebra/src/eq_poly.rs`
- `crates/akita-algebra/src/split_eq.rs`
