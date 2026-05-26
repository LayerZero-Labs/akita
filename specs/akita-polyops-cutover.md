# Spec: AkitaPolyOps Cutover To Open Polynomial Representations

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | Quang Dao, Cursor assistant                |
| Created     | 2026-05-26                                 |
| Status      | proposed                                   |
| PR          | stacked after `quang/metal-backend`        |

## Summary

`AkitaPolyOps` is currently the main root-polynomial abstraction in
`akita-prover`, but it mixes several jobs: shape metadata, public polynomial
extension, commitment row construction, opening folds, decompose-fold witness
construction, direct root-witness materialization, and tensor-projection helpers
for extension openings. This spec replaces it with an open representation
boundary: polynomials expose borrowed representation views, and compute
backends run named kernels over those views. The immediate implementation is a
CPU-only architectural cutover with no protocol or proof-format change, but the
resulting boundary is designed to support Metal and heterogeneous backends
without trapping all new polynomial shapes behind a closed Akita-owned enum.

## Intent

### Goal

Remove `AkitaPolyOps` from all crate source, examples, benches, tests, and
public prover APIs by replacing it with view/provider traits plus backend
kernels for commitment, opening/folding, decompose-fold, direct witness, and
tensor projection operations.

The cutover must include the tensor methods that currently live on
`AkitaPolyOps`:

- `tensor_extension_column_partials`
- `tensor_extension_column_partials_batch`
- `tensor_packed_extension_evals`
- `tensor_packed_extension_sparse_evals`
- `tensor_packed_extension_sparse_linear_combination`
- `tensor_packed_extension_poly`
- `tensor_packed_extension_root_poly`

The cutover also covers the non-tensor operations currently attached to
`AkitaPolyOps`:

- `num_ring_elems`
- `num_vars`
- `fold_blocks`
- `fold_blocks_ring`
- `evaluate_and_fold`
- `evaluate_and_fold_ring`
- `evaluate_extension`
- `decompose_fold`
- `decompose_fold_batched`
- `commit_inner`
- `commit_inner_witness`
- `direct_root_witness`

### Invariants

- Proof bytes, transcript order, Fiat-Shamir challenge sampling, and verifier
  replay semantics do not change. This is an internal prover architecture
  refactor.
- Root opening claims stay bit-for-bit equivalent for dense, one-hot,
  multilinear-dispatch, sparse-ring, and root tensor projection polynomials.
- Ring-multiplier openings keep the current distinction between base multiplier
  points and ring multiplier points. Base paths may specialize to scalar folds;
  ring paths must preserve sparse ring-multiplier accumulation.
- Batched decompose-fold at one opening point must preserve the current fused
  one-hot fast path. The implementation must not regress to always computing
  each one-hot polynomial independently and aggregating later.
- Tensor extension-opening reduction must preserve all current dense and sparse
  behavior:
  - dense roots share tail equality-table work across a same-point batch;
  - one-hot roots preserve sparse tensor-packed witnesses when available;
  - sparse linear combinations of tensor-packed one-hot witnesses remain fused;
  - dense fallback is used only when a source/backend cannot provide a sparse
    tensor witness for the whole same-point batch;
  - committed tensor-projection roots still produce the same
    `RootTensorProjectionPoly<F, D>` semantics.
- `RecursiveWitnessView` remains supported for recursive opening and commitment
  work, but it is not modeled as a root polynomial. It gets its own borrowed
  view or uses the same lower-level fold/decompose kernel family.
- The public extension point is open. Downstream users must be able to define a
  custom polynomial representation without modifying an Akita-owned input enum.
  They should either expose one of Akita's standard views or define a local view
  type and implement the relevant backend kernels for that view.
- Backend-prepared setup remains explicit and typed as introduced by
  `ComputeBackendSetup`. The new opening/tensor kernels must borrow the backend
  and prepared context when they need setup-owned work, and must not recover CPU
  internals through hidden downcasts or erased registries.
- Verifier-facing crates remain free of prover-only polynomial representation
  bounds. This cutover must not move `DensePoly`, `OneHotPoly`, new source
  traits, or backend kernels into verifier APIs.
- Existing no-backward-compatibility policy applies. Do not add deprecated
  `AkitaPolyOps` aliases, wrappers, compatibility shims, or partial migration
  layers.

### Non-Goals

- No Metal implementation lands in this PR. The PR prepares an abstraction
  boundary that Metal can implement later.
- No sumcheck protocol backend is introduced here. The scope is root polynomial
  commitment, opening/folding, decompose-fold, recursive witness fold/decompose
  support, direct witness materialization, and tensor projection operations.
- No proof object, serialization format, transcript label, schedule table,
  setup artifact, or verifier algorithm changes are intended.
- No closed `OpeningSource`, `TensorSource`, or `PolynomialSource` enum becomes
  the public extension point. Akita-owned enums may still be used privately for
  Akita-owned sum types such as `MultilinearPolynomial` and
  `RootTensorProjectionPoly`.
- No compatibility methods remain on polynomial structs solely to preserve the
  old trait surface. If tests need helper functions, they should exercise the
  new representation/backend boundary directly.
- No performance rewrite of the current arithmetic kernels is required, except
  where code movement is necessary to preserve the existing fast paths under
  the new boundary.

## Evaluation

### Acceptance Criteria

- [ ] `rg -n "AkitaPolyOps" crates` returns no matches.
- [ ] `akita-prover` no longer exports `AkitaPolyOps`, and `akita-pcs` no
      longer re-exports it.
- [ ] `crates/akita-prover/src/lib.rs` no longer contains a root-polynomial
      mega-trait with algorithm default methods.
- [ ] Commit APIs in `akita-prover`, `akita-scheme`, and `akita-pcs` are generic
      over the new root polynomial representation/provider surface, not
      `P: AkitaPolyOps<F, D>`.
- [ ] Prove APIs and internal flow helpers are generic over the new root
      polynomial representation/provider surface and the backend kernel bounds
      they actually use.
- [ ] All current built-in root representations compile on the new boundary:
      `DensePoly`, `OneHotPoly`, `SparseRingPoly`, `MultilinearPolynomial`, and
      `RootTensorProjectionPoly`.
- [ ] `RecursiveWitnessView` commit, evaluate/fold, and decompose-fold paths
      compile without implementing any root polynomial trait.
- [ ] Tensor extension-opening reduction compiles without calling the former
      operation methods such as `tensor_extension_column_partials` or
      `tensor_packed_extension_sparse_linear_combination` on a polynomial
      object.
- [ ] `crates/akita-pcs/tests/commitment_contract.rs` is updated so its dummy
      downstream-like polynomial uses the new open representation boundary.
      This test remains the canary for out-of-crate custom polynomial support.
- [ ] All existing tests that covered dense, one-hot, sparse-ring,
      root-projection, recursive, zero-knowledge, ring-switch, and extension
      opening flows still pass.
- [ ] The implementation PR includes a short grep/check section in its
      description showing that `AkitaPolyOps` is gone from crate source.

### Testing Strategy

Existing full checks:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

Targeted checks while implementing:

```bash
cargo test -p akita-prover backend::dense
cargo test -p akita-prover backend::onehot
cargo test -p akita-prover protocol::quadratic_equation
cargo test -p akita-pcs --test akita_e2e
cargo test -p akita-pcs --test commitment_contract
cargo test -p akita-pcs --test ring_switch
cargo test -p akita-pcs --test zk
```

New or strengthened tests:

- A contract test for a dummy custom root polynomial whose public type is not
  one of Akita's built-in polynomial structs. It should expose a local view
  type and prove that the CPU backend can run the required commit/opening
  kernels without an Akita-owned source enum.
- Dense tensor projection tests should keep checking that column partials and
  packed extension evals match the straight-line dense tensor helpers.
- One-hot tensor projection tests should keep checking:
  - same-point batch partials match individual partials;
  - sparse tensor-packed witnesses match dense materialization;
  - sparse tensor-packed linear combination matches dense linear combination.
- One-hot decompose-fold batch tests should continue to compare the fused
  batched path against individual decompose-fold plus aggregation.
- Ring-switch tests should continue to compare root and recursive witness
  evaluations at prepared multiplier points.

### Performance

Expected behavior is no meaningful regression for existing CPU paths. The
cutover moves dispatch boundaries but should preserve the same kernels and the
same batched/fused paths.

Performance checks before merging the implementation PR:

```bash
AKITA_MODE=onehot AKITA_NUM_VARS=32 cargo run --release --example profile
AKITA_MODE=dense AKITA_NUM_VARS=26 cargo run --release --example profile
```

If a local field matrix is available from prior profiling work, repeat the
same commands for the field variants already used there. The implementation PR
should report wall-clock deltas and explain any regression above 3% on the
dominant prover spans. For this architectural PR, a regression above 5% in the
canonical one-hot nv32 profile is blocking unless it is tied to an intentional
correctness fix.

Memory expectations:

- Same-point tensor batches may allocate the same `EqPolynomial` tail table
  once per point, as today.
- The cutover must not force dense direct-root materialization for one-hot
  tensor sparse paths.
- The cutover must not clone full polynomial coefficient tables merely to build
  representation views.

Proof-size expectations:

- No proof-size change. Existing proof-size formula tests and e2e verifier
  checks are sufficient because proof objects and transcript flow do not
  change.

## Design

### Architecture

The new architecture separates three roles that `AkitaPolyOps` currently
conflates:

1. Root polynomial objects own data and expose borrowed representation views.
2. Representation views describe data shape without owning backend state.
3. Compute backends execute kernels over representation views and typed
   prepared setup.

Conceptual flow:

```text
DensePoly / OneHotPoly / custom downstream poly
        |
        | exposes borrowed views
        v
RootPoly provider traits
        |
        | source-specific kernel bounds
        v
CpuBackend now, MetalBackend later
        |
        | named commit/open/tensor kernels
        v
current prover protocol code and proof objects
```

The boundary is source-type-based, not enum-based. A backend kernel trait should
take the representation view type as a type parameter, so downstream crates can
write implementations for their own local view type:

```rust
// Sketch only. Exact names may change during implementation.
pub trait RootPoly<F: FieldCore, const D: usize>: Clone + Send + Sync {
    type Commit<'a>
    where
        Self: 'a;
    type Opening<'a>
    where
        Self: 'a;
    type OpeningBatch<'a>
    where
        Self: 'a;
    type Tensor<'a, E>
    where
        Self: 'a,
        E: ExtField<F>;
    type TensorBatch<'a, E>
    where
        Self: 'a,
        E: ExtField<F>;

    fn num_ring_elems(&self) -> usize;
    fn num_vars(&self) -> usize;

    fn commit_view(&self) -> Result<Self::Commit<'_>, AkitaError>;
    fn opening_view(&self) -> Result<Self::Opening<'_>, AkitaError>;
    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatch<'a>, AkitaError>;
    fn tensor_view<E>(&self) -> Result<Self::Tensor<'_, E>, AkitaError>
    where
        E: ExtField<F>;
    fn tensor_batch<'a, E>(polys: &'a [&'a Self]) -> Result<Self::TensorBatch<'a, E>, AkitaError>
    where
        E: ExtField<F>;
}
```

The implementation may split this conceptual trait into capability traits for
Rust ergonomics, for example `RootPolyShape`, `RootCommitSource`,
`RootOpeningSource`, `RootTensorSource`, and `DirectRootWitnessSource`. If it
does, the traits must remain capability-oriented, not protocol-round-oriented.
Avoid traits named after one current call site such as
`Stage2TensorSparseLinearCombinationSource`.

The backend side should use a small number of operation clusters:

```rust
// Sketch only. Exact names may change during implementation.
pub trait RootCommitKernel<S, F: CanonicalField, const D: usize>:
    ComputeBackendSetup<F>
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: CommitInnerPlan,
    ) -> Result<FlatDigitBlocks<D>, AkitaError>;

    fn commit_inner_witness(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>;
}

pub trait OpeningFoldKernel<S, F: FieldCore, const D: usize>:
    ComputeBackendSetup<F>
where
    F: CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError>;

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F, D>, AkitaError>;
}

pub trait OpeningBatchKernel<S, F: CanonicalField, const D: usize>:
    ComputeBackendSetup<F>
{
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError>;
}

pub trait TensorProjectionKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: CanonicalField,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>;

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
    ) -> Result<TensorPackedWitness<E>, AkitaError>;

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
    ) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
    where
        E: RingSubfieldEncoding<F>;
}

pub trait TensorProjectionBatchKernel<S, F, E, const D: usize>:
    ComputeBackendSetup<F>
where
    F: CanonicalField,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>;

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup<D>>,
        source: S,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>;
}
```

The source type parameter `S` is the extensibility hook. For example, a
downstream crate should be able to define `MySparseView<'a>` and implement:

```rust
impl<'a, F, const D: usize> OpeningFoldKernel<MySparseView<'a>, F, D> for CpuBackend
where
    F: CanonicalField,
{
    // Custom CPU path, or reduce to Akita's standard dense/sparse helper plans.
}
```

The exact trait names may change, but the implementation must preserve this
property: source extensibility is open by local source type, not closed by an
Akita enum.

The sketches use `Option<&PreparedSetup>` only to show that not every current
opening or tensor operation needs setup-owned state. The implementation may use
separate no-setup and setup-bound kernel traits if that produces cleaner Rust
bounds, but it must keep setup-dependent work explicitly tied to a backend and
typed prepared context.

### Standard Views

Akita should provide standard borrowed views for built-in representations and
for downstream users that can reduce their representation to an existing shape:

- `DenseRootView<'a, F, D>`: borrowed ring coefficients, optional cached digit
  planes, optional small-i8 coefficients, and validated `num_vars`.
- `OneHotRootView<'a, F, D, I>` or an index-erased internal view: borrowed
  one-hot block tables and shape metadata.
- `SparseRingRootView<'a, F, D>`: borrowed sparse signed ring blocks and shape
  metadata.
- `RootTensorProjectionView<'a, F, D>`: Akita-owned dispatch view for
  `RootTensorProjectionPoly`; this may be an internal enum because
  `RootTensorProjectionPoly` itself is an Akita-owned sum type.
- `MultilinearPolynomialView<'a, F, D>`: Akita-owned dispatch view for the
  existing dense/one-hot `MultilinearPolynomial` sum type.
- `RecursiveWitnessOpeningView<'a, F, D>`: borrowed recursive witness rows and
  shape metadata, used by recursive fold/decompose paths without making
  recursive witnesses root polynomials.

These standard views should live close to the backend representation modules
that already own their invariants:

- dense view and CPU kernels in `crates/akita-prover/src/backend/dense.rs` or a
  small adjacent module;
- one-hot view and CPU kernels in `crates/akita-prover/src/backend/onehot.rs`;
- sparse-ring view and CPU kernels in
  `crates/akita-prover/src/backend/sparse_ring.rs`;
- root tensor projection dispatch in
  `crates/akita-prover/src/backend/field_reduction.rs`;
- multilinear dispatch in
  `crates/akita-prover/src/backend/multilinear_polynomial.rs`;
- recursive witness view and kernels in
  `crates/akita-prover/src/backend/recursive_witness.rs`.

Avoid moving all implementation code into `compute.rs`. `compute.rs` should own
backend traits, shared operation plans, and the CPU backend's setup-dependent
low-level kernels. Representation-specific view construction and algorithms
should stay with the representation modules.

### Operation Mapping

Current `AkitaPolyOps` method to new owner:

| Current method | New owner |
| --- | --- |
| `num_ring_elems`, `num_vars` | root polynomial shape/provider trait |
| `commit_inner`, `commit_inner_witness` | `RootCommitKernel<CommitView, F, D>` over backend and prepared setup |
| `fold_blocks`, `fold_blocks_ring` | private helpers or `OpeningFoldKernel<OpeningView, F, D>` internals |
| `evaluate_and_fold`, `evaluate_and_fold_ring` | `OpeningFoldKernel<OpeningView, F, D>` |
| `evaluate_extension` | tensor/opening kernel over the root view; fallback via explicit direct-witness view, not a trait default on the polynomial |
| `decompose_fold` | `OpeningFoldKernel<OpeningView, F, D>` |
| `decompose_fold_batched` | `OpeningBatchKernel<OpeningBatchView, F, D>` |
| `tensor_extension_column_partials` | `TensorProjectionKernel<TensorView, F, E, D>` |
| `tensor_extension_column_partials_batch` | `TensorProjectionBatchKernel<TensorBatchView, F, E, D>` |
| `tensor_packed_extension_evals` | `TensorProjectionKernel<TensorView, F, E, D>` returning dense packed witness |
| `tensor_packed_extension_sparse_evals` | `TensorProjectionKernel<TensorView, F, E, D>` returning sparse packed witness when available |
| `tensor_packed_extension_sparse_linear_combination` | `TensorProjectionBatchKernel<TensorBatchView, F, E, D>` |
| `tensor_packed_extension_poly`, `tensor_packed_extension_root_poly` | tensor projection root builder/kernel |
| `direct_root_witness` | explicit direct witness provider or kernel, used only by direct-opening paths and fallback tests |

Result enums such as `TensorPackedWitness::Dense(Vec<E>)` versus
`TensorPackedWitness::Sparse(SparseExtensionOpeningWitness<E>)` are acceptable
because the protocol output alternatives are fixed. The prohibited enum is a
closed input-source enum that downstream users cannot extend.

### Public API Cutover

Affected public and semi-public surfaces:

- `crates/akita-prover/src/lib.rs`: delete `AkitaPolyOps`; re-export the new
  root polynomial provider/view traits and backend kernel traits as needed.
- `crates/akita-prover/src/api/commitment.rs`: replace every
  `P: AkitaPolyOps<F, D>` bound with the root commit source and backend kernel
  bounds it needs.
- `crates/akita-prover/src/api/scheme.rs`: update docs and bounds away from
  `impl AkitaPolyOps`.
- `crates/akita-prover/src/protocol/flow.rs`: replace root claim evaluation,
  extension opening reduction, tensor projection, and root tensor projection
  call sites with provider/view plus backend kernel calls.
- `crates/akita-prover/src/protocol/quadratic_equation.rs`: replace
  `P::decompose_fold_batched` and `poly.decompose_fold` with opening batch
  kernels.
- `crates/akita-scheme/src/lib.rs`: replace prover API bounds and tensor root
  projection calls.
- `crates/akita-pcs/src/lib.rs`: remove the `AkitaPolyOps` re-export.
- `crates/akita-pcs/examples/profile/workload.rs`, benches, and tests: update
  helper bounds and direct calls to use the new provider/backend helpers.
- `crates/akita-setup/src/lib.rs`: remove local `AkitaPolyOps` import and call
  the new commit witness path.

The public API may expose an umbrella marker trait such as `AkitaRootPoly` for
readability:

```rust
pub trait AkitaRootPoly<F: FieldCore, const D: usize>:
    RootPolyShape<F, D>
    + RootCommitSource<F, D>
    + RootOpeningSource<F, D>
    + RootTensorSource<F, D>
    + DirectRootWitnessSource<F, D>
{
}
```

This is acceptable if it is the new capability bundle, not a deprecated alias
or algorithm-bearing replacement mega-trait. Internal APIs should prefer the
smallest capability bound that expresses what they use.

### Tensor Cutover Details

Tensor operations are the easiest place to accidentally keep the old design in
spirit. The implementation should treat tensor projection as a first-class
backend operation cluster, not as helper defaults hanging off a polynomial
trait.

Implementation requirements:

- Same-point batch column partials must be represented as a batch view so dense
  roots can share the tail equality table and one-hot roots can keep their
  existing batched sparse handling.
- Sparse tensor-packed linear combination must be a batch kernel. It should not
  ask each polynomial for an optional sparse witness through a public trait
  default and then combine externally unless the source/backend explicitly uses
  that as its fallback.
- Dense tensor packed evals remain available as the universal fallback, but the
  fallback lives in a CPU standard kernel over a direct-witness-capable view.
- `RootTensorProjectionPoly` construction remains an explicit tensor projection
  output. Built-in one-hot roots may still return sparse projection roots, and
  dense roots may still return dense projection roots.
- The extension degree and ring-subfield embedding checks currently performed
  by tensor helper methods must move into tensor plan construction or tensor
  kernels, preserving the same `AkitaError` behavior.

### Relationship To Current Compute Backend

This spec extends the design already present in `compute.rs`.

Similarities:

- Backend setup is still explicit through `ComputeBackendSetup`.
- Prepared setup is typed by backend and ring dimension.
- Hot paths call named backend operations rather than reaching into raw CPU
  setup internals.
- Standard CPU operations can still use existing plan structs such as
  `DenseCommitRowsPlan`, `OneHotCommitRowsPlan`,
  `SparseRingCommitRowsPlan`, and `RecursiveWitnessCommitRowsPlan`.

Differences:

- The current commit backend methods are representation-named methods on
  `CommitmentComputeBackend`. They are good low-level standard plans, but they
  are not enough as the public polynomial extension boundary.
- The new source-type kernel traits add an open layer above those standard
  plans. Built-in sources can reduce to the existing plan methods. Downstream
  sources can either reduce to standard views or implement kernels for their
  own local view types.
- Opening/folding/decompose/tensor work currently still lives directly on
  `AkitaPolyOps`. This spec moves those operations to the same backend-owned
  shape as commitment.

### Alternatives Considered

#### Closed Source Enum

Rejected. An enum such as `OpeningSource::Dense | OneHot | SparseRing` is easy
for Akita internals but impossible for downstream users to extend without
changing Akita. It would reproduce the same design problem under a new name.

Akita-owned enums remain acceptable only for Akita-owned sum types such as
`MultilinearPolynomial` or `RootTensorProjectionPoly`, where the enum is the
actual data model and not the public extension boundary.

#### Keep `AkitaPolyOps` And Add Backend Arguments

Rejected. The current `quang/metal-backend` branch already moved commitment
methods in this direction, but keeping `AkitaPolyOps` as the umbrella makes the
polynomial trait continue to own algorithm dispatch. That blocks a clean
heterogeneous backend design and keeps tensor projection trapped in polynomial
methods.

#### Split Into One Trait Per Protocol Helper

Rejected. A trait for every current helper would reduce file-local coupling but
would produce a fragile API shaped by today's call graph. The split should be
by stable capability and kernel cluster: commit, opening/decompose, tensor
projection, direct witness, recursive witness support.

#### One New Mega-Trait With No Algorithm Defaults

Partially acceptable. A single `RootPoly` provider trait with associated views
is much better than `AkitaPolyOps` because it exposes representation views
instead of algorithms. However, implementation should still use capability
traits or an umbrella marker so internal functions can state precise bounds and
downstream users do not have to implement tensor support for APIs that only
commit.

## Documentation

Required documentation updates in the implementation PR:

- Update `crates/akita-prover/src/api/scheme.rs` docs that currently say
  caller-provided roots are `impl AkitaPolyOps<F, D>`.
- Update crate-level exports and docs in `akita-prover` and `akita-pcs`.
- Update the active compute-backend spec section that currently names
  `AkitaPolyOps` as the remaining cutover target, linking to this spec as the
  successor design.
- Leave historical specs alone unless they describe `AkitaPolyOps` as active
  future guidance. Historical mentions can remain as context.

## Execution

Suggested implementation sequence for one code PR:

1. Add the new provider/view traits and operation plan types without changing
   public APIs yet.
2. Add standard borrowed views for dense, one-hot, sparse-ring, root tensor
   projection, multilinear dispatch, and recursive witness.
3. Implement CPU commit kernels for the new commit views by reducing to the
   existing `CommitmentComputeBackend` low-level plan methods.
4. Implement CPU opening/decompose kernels by moving the current dense,
   one-hot, sparse-ring, root projection, multilinear dispatch, and recursive
   witness logic out of `AkitaPolyOps` impls.
5. Implement CPU tensor projection kernels by moving all tensor helper logic
   out of `AkitaPolyOps` impls, preserving dense same-point sharing and one-hot
   sparse batch paths.
6. Cut over `api/commitment.rs`, `api/scheme.rs`, `protocol/flow.rs`,
   `protocol/quadratic_equation.rs`, `protocol/ring_switch.rs`, `akita-scheme`,
   examples, benches, and tests to the new boundary.
7. Delete `AkitaPolyOps` and its blanket `&P` impl.
8. Remove compatibility imports/re-exports and update docs.
9. Run the full checks and profiling commands named above.

Risks to resolve during implementation:

- Rust generic bounds may become noisy when a flow needs both root tensor
  kernels and opening kernels. Prefer local helper type aliases or marker
  traits for capability bundles over hiding work behind trait-object erasure.
- GAT-heavy traits can create lifetime friction for same-point batches. If this
  happens, prefer explicit batch view structs over cloning polynomial refs into
  temporary vectors.
- `MultilinearPolynomial` and `RootTensorProjectionPoly` dispatch enums should
  remain internal wrappers. Do not let them leak into the public custom
  polynomial extension path.
- Recursive witness support needs careful naming so it reuses fold/decompose
  kernels without implying recursive witnesses are valid root polynomials.
- Error behavior in old default methods must be preserved. Do not replace
  `AkitaError` returns with panics while moving fallback logic.

Expected implementation diff:

- `crates/akita-prover/src/lib.rs`: large deletion of `AkitaPolyOps`, smaller
  re-export additions for new traits.
- `crates/akita-prover/src/compute.rs`: moderate additions for source-kernel
  traits and shared plan/result structs.
- `crates/akita-prover/src/backend/*.rs`: moderate churn moving impl blocks
  from `AkitaPolyOps` to provider/view/kernel impls.
- `crates/akita-prover/src/protocol/flow.rs` and
  `crates/akita-prover/src/protocol/quadratic_equation.rs`: moderate call-site
  churn, no intended protocol logic change.
- `akita-scheme`, `akita-pcs` examples, benches, and tests: mechanical generic
  bound and helper updates.

Rough size estimate for the implementation PR: 20 to 35 files touched, about
1.8k to 3.2k lines added and 1.4k to 2.6k lines removed. The final diff should
feel like moving ownership of existing operations, not adding a parallel layer
beside them.

## References

- [`specs/akita-compute-backend-metal.md`](akita-compute-backend-metal.md):
  current compute backend and Metal roadmap; this spec supersedes the
  `AkitaPolyOps` cutover notes there.
- [`crates/akita-prover/src/lib.rs`](../crates/akita-prover/src/lib.rs):
  current `AkitaPolyOps` definition and blanket reference impl.
- [`crates/akita-prover/src/compute.rs`](../crates/akita-prover/src/compute.rs):
  current typed compute backend setup and low-level commit/ring-switch plans.
- [`crates/akita-prover/src/backend/dense.rs`](../crates/akita-prover/src/backend/dense.rs):
  dense root implementation, tensor dense paths, and dense decompose-fold.
- [`crates/akita-prover/src/backend/onehot.rs`](../crates/akita-prover/src/backend/onehot.rs):
  one-hot root implementation, sparse tensor paths, and fused batched
  decompose-fold.
- [`crates/akita-prover/src/backend/sparse_ring.rs`](../crates/akita-prover/src/backend/sparse_ring.rs):
  sparse signed-ring root implementation.
- [`crates/akita-prover/src/backend/field_reduction.rs`](../crates/akita-prover/src/backend/field_reduction.rs):
  root tensor projection polynomial dispatch.
- [`crates/akita-prover/src/backend/recursive_witness.rs`](../crates/akita-prover/src/backend/recursive_witness.rs):
  recursive witness fold/decompose/commit support.
- [`crates/akita-prover/src/protocol/flow.rs`](../crates/akita-prover/src/protocol/flow.rs):
  root opening evaluation, extension opening reduction, tensor projection, and
  recursive proving flow call sites.
- [`crates/akita-prover/src/protocol/quadratic_equation.rs`](../crates/akita-prover/src/protocol/quadratic_equation.rs):
  decompose-fold and batched decompose-fold call sites.
