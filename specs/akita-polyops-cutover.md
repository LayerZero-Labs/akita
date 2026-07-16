# Spec: AkitaPolyOps Cutover To Open Polynomial Representations

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | Quang Dao, Cursor assistant                |
| Created     | 2026-05-26                                 |
| Status      | implemented                                |
| PR          | [#206](https://github.com/LayerZero-Labs/akita/pull/206) |

## Summary

`AkitaPolyOps` was the main root-polynomial abstraction in
`akita-prover`, but it mixed several jobs: shape metadata, public polynomial
extension, commitment row construction, opening folds, decompose-fold witness
construction, direct root-witness materialization, and tensor-projection helpers
for extension openings. **This cutover is implemented in PR #206:** the trait is
deleted from crate source; commitment and prove dispatch through borrowed
**views** and backend **kernels** (`RootCommitSource`, `RootOpeningSource`,
`RootTensorSource`, `ProverComputeStack`, `LevelProveStacks`, `OperationCtx`).
The current compute backend split improved setup
ownership, but its commitment and ring-switch traits are still closed around
Akita's built-in plan shapes. This spec replaces both layers with an open
representation boundary: protocol inputs expose borrowed views, and operation
backends run source-typed kernels over those views. The immediate
implementation is a CPU-only architectural cutover with no protocol or
proof-format change, but the resulting boundary is designed to support Metal
and heterogeneous backends without trapping all new polynomial or protocol
source shapes behind a closed Akita-owned enum.

## Intent

### Goal

Remove `AkitaPolyOps` from all crate source, examples, benches, tests, and
public prover APIs, and cut over commitment and ring-switch compute traits from
fixed built-in plan methods to source-typed operation kernels. The replacement
is a set of view/provider traits plus backend kernels for commitment,
opening/folding, decompose-fold, direct witness, tensor projection, and
ring-switch relation/quotient operations.

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

The cutover must also include the fixed compute backend methods that currently
make the backend boundary closed over Akita-owned plan shapes:

- `dense_commit_rows`
- `onehot_commit_rows`
- `sparse_ring_commit_rows`
- `recursive_witness_commit_rows`
- `ring_switch_relation_rows`
- `ring_switch_quotient_rows`

These operations may remain as lower-level standard helper kernels, but the
prover should no longer be generic over a monolithic backend trait whose public
surface is exactly this built-in list.

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
- Backend choice is per operation cluster, not globally forced to one
  `B: ProverComputeBackend<F>`. A proof may use one backend/prepared context
  for root commitment, another for opening/decompose-fold, another for tensor
  projection, and another for ring-switch rows, as long as every prepared
  context validates against the same expanded setup digests.
- Prepared setup validation happens at operation-context construction and at
  public prover API entry, before any transcript absorption or backend kernel
  execution. Individual kernel implementations may assume their context has
  already been validated, but public helpers must not let an unvalidated
  `(backend, prepared)` pair reach an operation cluster.
- Operation outputs crossing backend boundaries are canonical Akita-owned data
  structures in this PR: `FlatDigitBlocks`, `CommitInnerWitness`,
  `DecomposeFoldWitness`, `Vec<CyclotomicRing<_, _>>`,
  `RingSwitchRelationRows`, tensor witness structures, and root projection
  polynomials. No operation may require the next operation to understand an
  opaque device buffer owned by a different backend.
- If the same concrete backend handles several operation clusters, it may reuse
  one prepared context for those clusters. If different backends handle
  different clusters, each backend owns and validates its own prepared context.
- Direct root witnesses are an explicit source capability, not a hidden default
  on every root polynomial. APIs that may select a root-direct schedule must
  require `DirectRootWitnessSource` or a folded-only policy that rejects
  root-direct before that path is reached. A source that implements the
  capability may still return `AkitaError` for malformed or unsupported direct
  witness shapes; the serialized `DirectWitnessProof` semantics remain
  verifier-owned and unchanged.
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
  support, direct witness materialization, tensor projection operations, and
  the ring-switch row kernels already present in the compute backend.
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
- No asynchronous job scheduler, cross-device work stealing, or device-resident
  buffer interop protocol is required in this PR. The operation-stack shape must
  leave room for those features, but this PR's interop contract is synchronous
  canonical outputs.

## Evaluation

### Acceptance Criteria

- [x] `rg -n "AkitaPolyOps" crates` returns no matches.
- [x] `akita-prover` no longer exports `AkitaPolyOps`, and `akita-pcs` no
      longer re-exports it.
- [x] `crates/akita-prover/src/lib.rs` no longer contains a root-polynomial
      mega-trait with algorithm default methods.
- [x] Public prover/protocol APIs no longer require one monolithic
      `B: ProverComputeBackend<F>` that implements every operation cluster.
      Heterogeneous `batched_prove` takes `LevelProveStacks` + `ProveStackFor`;
      internal prove paths route `OperationCtx` per cluster. (Uniform PCS entry
      still bounds `RecursiveProveBackend` as a convenience umbrella.)
- [x] The public commitment compute boundary is source-typed. It is no longer
      limited to trait methods named only after Akita's built-in dense,
      one-hot, sparse-ring, and recursive-witness plan shapes.
- [x] The public ring-switch compute boundary is source-typed. Relation/quotient
      protocol code calls `RingSwitchRelationKernel` / `RingSwitchQuotientKernel`;
      row plan helpers remain CPU implementation details behind kernels.
- [x] Existing built-in commit/ring-switch plan structs either become standard
      view/helper types consumed by the CPU implementation or are replaced by
      equivalent source views. They must not remain the only public operation
      boundary.
- [x] Commit APIs in `akita-prover`, `akita-scheme`, and `akita-pcs` are generic
      over the new root polynomial representation/provider surface, not
      `P: AkitaPolyOps<F, D>`.
- [x] Prove APIs and internal flow helpers are generic over the new root
      polynomial representation/provider surface and the backend kernel bounds
      they actually use.
- [x] All current built-in root representations compile on the new boundary:
      `DensePoly`, `OneHotPoly`, `SparseRingPoly`, `MultilinearPolynomial`, and
      `RootTensorProjectionPoly`.
- [x] `RecursiveWitnessFlat` / `SuffixWitnessView` commit, evaluate/fold, and
      decompose-fold paths compile without implementing any root polynomial trait.
- [x] Tensor extension-opening reduction compiles without calling the former
      operation methods such as `tensor_extension_column_partials` or
      `tensor_packed_extension_sparse_linear_combination` on a polynomial
      object.
- [x] `crates/akita-pcs/tests/commitment_contract.rs` is updated so its dummy
      downstream-like polynomial uses the new open representation boundary.
      This test remains the canary for out-of-crate custom polynomial support.
- [x] All existing tests that covered dense, one-hot, sparse-ring,
      root-projection, recursive, zero-knowledge, ring-switch, and extension
      opening flows still pass.
- [x] A mixed-backend contract test proves that at least two different backend
      values can be used for different operation clusters in one prover call.
      (`heterogeneous_prove_e2e` + delegating cluster backends.)
- [x] Lower-level commit APIs compile with a custom source that implements only
      shape plus commit-source capabilities. They must not require opening,
      tensor, or direct-witness capabilities.
- [x] Public proving APIs that can select root-direct require
      `DirectRootWitnessSource`, while folded-only helpers or policies can be
      used without that capability after root-direct is rejected.
- [x] Operation-stack construction or public API validation rejects a prepared
      setup built from a different expanded setup for at least one non-commit
      operation cluster.
- [x] Cross-`D` recursive witness commitment validates the newly prepared target
      dimension context before use and rejects a mismatched prepared context.
- [x] Implementation review checks include forbidden-pattern greps:
      `rg -n "AkitaPolyOps" crates`, public protocol/API bounds on
      `ProverComputeBackend`, and public closed input-source enums used as the
      custom polynomial extension point.
- [x] The implementation PR includes a short grep/check section in its
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
- Add a compile/runtime contract test for a heterogeneous operation stack, for
  example CPU root commitment plus a distinct dummy ring-switch backend, or
  distinct dummy backends for commit/opening/tensor/ring-switch that all
  validate against the same setup digests. This test protects against
  accidentally reintroducing `B: ProverComputeBackend<F>` as a global bound.
- Add operation-boundary tests showing that a custom local commit source view
  can implement the commitment kernel for `CpuBackend` without changing an
  Akita source enum, and that a custom local ring-switch relation view can do
  the same for the ring-switch kernel.
- Strengthen `crates/akita-pcs/tests/commitment_contract.rs` so its local
  custom source view implements an Akita kernel trait for `CpuBackend`. This is
  the orphan-rule canary: the source view is local to the downstream-like test,
  while the backend trait and backend type are Akita-owned.
- Add a root-direct capability test or compile-time helper showing that
  commit-only paths do not require `DirectRootWitnessSource`, and proving paths
  that may choose root-direct do.
- Add prepared-context mismatch tests for a non-commit operation context and
  for recursive witness commitment when the target `D` is prepared through a
  dispatch arm.

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
- Mixed-backend interop may materialize canonical host outputs at operation
  boundaries in this PR, but it must not add extra host copies within the same
  operation when the backend can consume a borrowed view directly.

Proof-size expectations:

- No proof-size change. Existing proof-size formula tests and e2e verifier
  checks are sufficient because proof objects and transcript flow do not
  change.

## Design

### Architecture

The new architecture separates four roles that `AkitaPolyOps` and the current
fixed compute traits conflate:

1. Root polynomial objects own data and expose borrowed representation views.
2. Protocol objects such as ring-switch relation inputs expose borrowed views.
3. Representation views describe data shape without owning backend state.
4. Operation backends execute kernels over representation views and typed
   prepared setup.

Conceptual flow:

```text
DensePoly / OneHotPoly / custom downstream poly
        |
        | exposes root views
        v
RootPoly provider traits
        |
        | selected operation context
        v
Commit/Open/Tensor kernels on CPU or Metal
        |
        | canonical outputs
        v
current prover protocol code and proof objects

Ring-switch witness/protocol state
        |
        | exposes relation/quotient views
        v
Ring-switch kernels on CPU or Metal
        |
        | canonical rows
        v
current prover protocol code and proof objects
```

One prover run uses an operation stack rather than one all-powerful backend:

```text
Akita expanded setup
   |
   +-- prepare for CpuBackend    -> CpuPreparedSetup<D>
   +-- prepare for MetalBackend  -> MetalPreparedSetup<D>

ProverComputeStack<D>
   commit:      MetalBackend + MetalPreparedSetup<D>
   opening:     CpuBackend   + CpuPreparedSetup<D>
   tensor:      MetalBackend + MetalPreparedSetup<D>
   ring_switch: CpuBackend   + CpuPreparedSetup<D>
```

This is the core interop rule: every operation sees a backend plus that
backend's prepared context, but the protocol sees canonical Akita outputs. A
later scheduler may overlap independent CPU and Metal work, but the first
cutover should make the selection and data dependencies explicit before adding
async machinery.

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
    type BatchOpening<'a>
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
    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::BatchOpening<'a>, AkitaError>;
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

Ring-switch sources follow the same rule even though they are not root
polynomials:

```rust
// Sketch only. Exact names may change during implementation.
pub trait RingSwitchSources<const D: usize> {
    type Relation<'a>
    where
        Self: 'a;
    type Quotient<'a>
    where
        Self: 'a;

    fn relation_view(&self) -> Result<Self::Relation<'_>, AkitaError>;
    fn quotient_view(&self) -> Result<Self::Quotient<'_>, AkitaError>;
}
```

The current Akita relation view will carry borrowed `w_hat`, `t_hat`, the
centered `z` segment, and the centered infinity norm. The important point is
that this is a view type, not a mandatory public enum of every possible
ring-switch input layout.

### Operation Contexts

The prover should stop passing one `(backend, prepared)` pair under a single
`B: ProverComputeBackend<F>` bound. It should pass operation contexts:

```rust
// Sketch only. Exact names may change during implementation.
pub struct OperationCtx<'a, F, B, const D: usize>
where
    F: CanonicalField,
    B: ComputeBackendSetup<F>,
{
    pub backend: &'a B,
    pub prepared: &'a B::PreparedSetup<D>,
}

pub struct ProverComputeStack<'a, F, const D: usize, C, O, T, R>
where
    F: CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    pub commit: OperationCtx<'a, F, C, D>,
    pub opening: OperationCtx<'a, F, O, D>,
    pub tensor: OperationCtx<'a, F, T, D>,
    pub ring_switch: OperationCtx<'a, F, R, D>,
}
```

The exact spelling may differ, but these constraints are fixed:

- operation contexts validate their prepared setup against the same expanded
  setup descriptor digests before use;
- an operation that only commits should only require the commit context and
  commit kernel bounds;
- recursive or dynamic-`D` dispatch prepares an operation stack for the concrete
  level dimension in that dispatch arm;
- a `CpuBackend`-only prover stack is just the degenerate case where all four
  contexts use the same backend and prepared setup type;
- a heterogeneous prover stack is the case where one or more contexts use
  different backend types or prepared setup values.

Validation ownership is fixed even if the exact constructors change:

- `OperationCtx::new` or the equivalent constructor takes explicit
  `&AkitaExpandedSetup<F>` metadata and calls
  `backend.validate_prepared_setup::<D>(prepared, expanded)`;
- `ProverComputeStack::new` validates every contained operation context against
  the same expanded setup before the prover starts transcript work;
- lower-level public helpers that accept a single operation context validate
  that context at entry;
- dynamic-`D` recursive witness commitment prepares the target-dimension
  context inside the dispatch arm and validates it before calling the commit
  kernel;
- validation failure is always `AkitaError::InvalidSetup` or a more specific
  existing setup error, never a panic or silent CPU fallback.

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

pub trait RingSwitchRelationKernel<S, F: CanonicalField, const D: usize>:
    ComputeBackendSetup<F>
{
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>;
}

pub trait RingSwitchQuotientKernel<S, F: CanonicalField, const D: usize>:
    ComputeBackendSetup<F>
{
    fn quotient_rows(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: S,
        plan: RingSwitchQuotientPlan,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>;
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

The fixed built-in row methods in today's `CommitmentComputeBackend` and
`RingSwitchComputeBackend` can survive only below this layer as standard helper
kernels. For example, a dense commit view may reduce to a public
`StandardDenseCommitRows` helper, and a custom downstream view may reduce to
canonical digit rows. Protocol code should not be generic over those fixed
standard helper traits; it should be generic over source-typed operation
kernels.

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
- `RingSwitchRelationView<'a, D>`: borrowed decomposed recursive witness rows,
  decomposed inner-commitment rows, one centered quotient segment, and its
  infinity-norm metadata.
- `RingSwitchQuotientView<'a, D>`: borrowed centered quotient segment and
  infinity-norm metadata for additional public rows.

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
- ring-switch relation/quotient source views near
  `crates/akita-prover/src/protocol/ring_switch.rs` or
  `crates/akita-prover/src/protocol/quadratic_equation.rs`, depending on where
  the owning witness state naturally lives.

Avoid moving all implementation code into the compute module tree.
`crates/akita-prover/src/compute/` should own backend traits, shared operation
plans, and the CPU backend's setup-dependent low-level kernels.
Representation-specific view construction and algorithms should stay with the
representation modules.

PO1 landed the tree as sibling modules under `compute/` (all re-exported from
`compute/mod.rs` so `crate::compute::…` paths stay stable):

| Module | Role |
| --- | --- |
| `plans.rs` | Legacy row/commit plan structs and `FlatBlockTable` |
| `backend.rs` | Fixed trait ladder (`ComputeBackendSetup` … `ProverComputeBackend`); removed at PO4 |
| `cpu.rs` | `CpuBackend` / `CpuPreparedSetup` and standard row-kernel impls |
| `operation_plans.rs` | Scalar PO1 operation parameters (`CommitInnerPlan`, `OpeningFoldPlan`, …) |
| `kernels.rs` | Source-typed operation kernel traits generic over view `S` |
| `poly.rs` | Root polynomial capability traits (`RootPolyShape`, `RootCommitSource`, …) |
| `stack.rs` | `OperationCtx` and heterogeneous `ProverComputeStack` |

The split exists to satisfy the repository 1500-line cap without changing
semantics. New compute-boundary work should land in the matching sibling module,
not grow a single file back toward monolith size.

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
| `dense_commit_rows` | standard dense-row helper below `RootCommitKernel`, not the public commit boundary |
| `onehot_commit_rows` | standard one-hot-row helper below `RootCommitKernel`, not the public commit boundary |
| `sparse_ring_commit_rows` | standard sparse-ring-row helper below `RootCommitKernel`, not the public commit boundary |
| `recursive_witness_commit_rows` | recursive witness commit kernel or standard helper below it |
| `ring_switch_relation_rows` | `RingSwitchRelationKernel<RelationView, F, D>` |
| `ring_switch_quotient_rows` | `RingSwitchQuotientKernel<QuotientView, F, D>` |

Result enums such as `TensorPackedWitness::Dense(Vec<E>)` versus
`TensorPackedWitness::Sparse(SparseExtensionOpeningWitness<E>)` are acceptable
because the protocol output alternatives are fixed. The prohibited enum is a
closed input-source enum that downstream users cannot extend.

### Post-cutover naming (views and fields)

After the trait-surface hygiene pass on `quang/po-cutover-v2`:

- **Views:** each representation exposes one single-polynomial view and one batch
  view (`DenseView` / `DenseBatchView`, `OneHotView` / `OneHotBatchView`, and so
  on). The kernel trait the view is passed to selects the operation; per-cluster
  `*CommitView` / `*OpeningView` / `*TensorView` aliases are gone.
- **Fields:** the protocol uses base field `F` and a single extension field `E`
  (`CommitmentConfig::ExtField`). The historical claim-field / challenge-field
  split (`L` vs `C`, `ClaimE` vs `ChallengeE`, separate descriptor degrees) is
  retired. Prove/commit backend bundles, layout gates, and transcript descriptor
  algebra all take one `E: ExtField<F>`.

### Public API Cutover

Affected public and semi-public surfaces:

- `crates/akita-prover/src/lib.rs`: delete `AkitaPolyOps`; re-export the new
  root polynomial provider/view traits and backend kernel traits as needed.
- `crates/akita-prover/src/api/commitment.rs`: replace every
  `P: AkitaPolyOps<F, D>` bound with the root commit source and backend kernel
  bounds it needs. Replace `B: CommitmentComputeBackend<F>` bounds with an
  operation commit context and `RootCommitKernel<_, F, D>` bounds.
- `crates/akita-prover/src/api/scheme.rs`: update docs and bounds away from
  `impl AkitaPolyOps`.
- `crates/akita-prover/src/compute/`: replace the public fixed
  `CommitmentComputeBackend`, `RingSwitchComputeBackend`, and
  `ProverComputeBackend` surfaces (today in `backend.rs`) with operation
  contexts plus source-typed commit/ring-switch/opening/tensor kernels. Low-level
  standard row helpers in `cpu.rs` may remain public if they are useful building
  blocks, but protocol APIs must not depend on them as the main abstraction.
- `crates/akita-prover/src/protocol/flow.rs`: replace root claim evaluation,
  extension opening reduction, tensor projection, and root tensor projection
  call sites with provider/view plus operation-context kernel calls.
- `crates/akita-prover/src/protocol/quadratic_equation.rs`: replace
  `P::decompose_fold_batched` and `poly.decompose_fold` with opening batch
  kernels. Replace ring-switch row calls with relation/quotient source views
  and ring-switch operation contexts.
- `crates/akita-prover/src/protocol/ring_switch.rs`: replace commitment helper
  bounds and recursive witness commit calls with source-typed commit kernels.
- `crates/akita-scheme/src/lib.rs`: replace prover API bounds and tensor root
  projection calls, and thread operation stacks through commit/prove calls.
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

Capability boundaries for the main public and semi-public APIs:

| Surface | Required source capabilities | Required operation context/kernels | Notes |
| --- | --- | --- | --- |
| `prepare_commit_inputs`, `prepare_batched_commit_inputs` | `RootPolyShape` | none | Shape validation only. No commit/opening/tensor/direct bound. |
| `commit_with_params`, `commit_with_policy`, `batched_commit_with_policy` | `RootPolyShape + RootCommitSource` | commit context plus `RootCommitKernel<CommitView, F, D>` and B-side digit rows | Lower-level commit APIs must remain capability-minimal. |
| `AkitaCommitmentScheme::commit`, `AkitaCommitmentScheme::batched_commit` | `RootPolyShape + RootCommitSource`; additionally `RootTensorSource` when the config-generic wrapper may perform root tensor projection before commit | commit context plus root commit kernel; tensor context only if the wrapper performs projection through a backend | If this cannot be expressed ergonomically with conditional bounds, the scheme wrapper may require `RootTensorSource`, but lower-level commit APIs must not. |
| `prove_root_direct` | `DirectRootWitnessSource` | none | This path produces verifier-visible `DirectWitnessProof` values exactly as today. |
| `prove_batched_with_policy` and `AkitaCommitmentScheme::batched_prove` | `RootPolyShape + RootOpeningSource + RootTensorSource + DirectRootWitnessSource` for APIs that may select root-direct | opening, tensor, commit-next, and ring-switch contexts as used by the selected schedule | A custom source without direct-witness support must use a folded-only policy/helper that rejects root-direct before this path, rather than relying on a hidden dense fallback. |
| root extension-opening reduction preparation/proving | `RootTensorSource` and, for dense fallback, explicit direct-witness-capable tensor view support | tensor context plus `TensorProjectionKernel`/`TensorProjectionBatchKernel` | Sparse batch paths must stay batch kernels. Dense fallback is explicit CPU tensor behavior, not a polynomial default. |
| root fold evaluation and decompose-fold | `RootOpeningSource` and matching batch source for batched decompose | opening context plus `OpeningFoldKernel`/`OpeningBatchKernel` | Includes base and ring multiplier points. |
| `QuadraticEquation::new_prover` | root opening/decompose sources for root claims; recursive witness view sources for recursive claims | opening context for decompose-fold and digit rows used by hint construction | It must not require tensor/direct capabilities merely to build quadratic equations. |
| `ring_switch_build_w`, `compute_r_split_eq` | ring-switch relation/quotient source views, not root polynomial sources | ring-switch context plus relation/quotient kernels and cyclic rows for blinding | Relation/quotient views carry the currently validated `w_hat`, `t_hat`, `z` segment, and norm metadata. |
| `commit_w`, `commit_next_w_with_policy` | recursive witness commit source, not root polynomial source | commit context plus recursive witness commit kernel and B-side digit rows | Cross-`D` dispatch prepares and validates a target-dimension commit context inside the dispatch arm. |

The full `AkitaRootPoly` marker is acceptable only on top-level convenience
APIs whose behavior can reach every root capability through config-selected
schedules. Lower-level implementation helpers should use the smallest row in
this table that matches their work.

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

This spec extends the design already present in `crates/akita-prover/src/compute/`.

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
- Ring-switch work currently has a named backend trait, but it is still closed
  around `RingSwitchRelationRowsPlan` and `RingSwitchQuotientRowsPlan`. This
  spec turns those plans into standard views/helpers below a source-typed
  kernel.

### Interoperation Model

The first implementation should use canonical host-owned operation outputs as
the interop boundary:

```text
commit kernel
  input:  root commit view + commit context
  output: CommitInnerWitness / FlatDigitBlocks

opening/decompose kernel
  input:  root opening view + opening context
  output: y_ring, folded rings, DecomposeFoldWitness

tensor kernel
  input:  tensor source view + tensor context
  output: column partials, dense/sparse tensor witnesses, root projection poly

ring-switch kernel
  input:  relation/quotient view + ring-switch context
  output: RingSwitchRelationRows or quotient rows
```

This gives seamless mixed-backend execution because every operation consumes
borrowed source views plus its own prepared context, then returns ordinary
Akita values that the transcript and proof construction already understand.
For example:

```text
one-hot root commit:       Metal commit kernel -> CommitInnerWitness on host
root opening decompose:    CPU opening kernel  -> DecomposeFoldWitness on host
tensor extension witness:  Metal tensor kernel -> sparse tensor witness on host
ring-switch rows:          CPU ring kernel     -> RingSwitchRelationRows on host
```

This is intentionally conservative. Device-resident outputs can be added later
as an optimization layer, but they must be optional and must not become the
semantic contract between operation clusters. If a future Metal backend wants
to fuse commit-to-ring-switch work without host materialization, it can do so
by implementing a larger source-typed fused kernel while still exposing the
canonical output path required for heterogeneous fallback.

### Alternatives Considered

#### Closed Source Enum

Rejected. An enum such as `OpeningSource::Dense | OneHot | SparseRing` is easy
for Akita internals but impossible for downstream users to extend without
changing Akita. It would reproduce the same design problem under a new name.

Akita-owned enums remain acceptable only for Akita-owned sum types such as
`MultilinearPolynomial` or `RootTensorProjectionPoly`, where the enum is the
actual data model and not the public extension boundary.

#### Keep `AkitaPolyOps` And Add Backend Arguments

Rejected. `main` already moved commitment methods in this direction via the
typed `ComputeBackendSetup` compute backend, but keeping `AkitaPolyOps` as the
umbrella makes the polynomial trait continue to own algorithm dispatch. That
blocks a clean heterogeneous backend design and keeps tensor projection trapped
in polynomial methods.

#### Keep Fixed Commitment/Ring-Switch Backend Traits Public

Rejected. The current fixed methods are useful standard CPU/accelerator plans,
but keeping them as the public prover boundary means downstream custom sources
must first pretend to be one of Akita's built-in dense, one-hot, sparse-ring,
recursive-witness, or ring-switch plan shapes. That is exactly the closed-source
problem this cutover is meant to remove.

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

1. Add operation contexts and source-typed kernel traits for commit,
   opening/decompose, tensor projection, and ring-switch relation/quotient
   operations.
2. Add the new root provider/view traits and protocol source views without
   changing public APIs yet.
3. Add standard borrowed views for dense, one-hot, sparse-ring, root tensor
   projection, multilinear dispatch, recursive witness, and ring-switch
   relation/quotient inputs.
4. Implement CPU commit kernels for the new commit views by reducing to
   standard row helpers or directly to existing CPU kernels.
5. Implement CPU ring-switch relation/quotient kernels for the new ring-switch
   views by reducing to the existing fused quotient row kernel.
6. Implement CPU opening/decompose kernels by moving the current dense,
   one-hot, sparse-ring, root projection, multilinear dispatch, and recursive
   witness logic out of `AkitaPolyOps` impls.
7. Implement CPU tensor projection kernels by moving all tensor helper logic
   out of `AkitaPolyOps` impls, preserving dense same-point sharing and one-hot
   sparse batch paths.
8. Cut over `api/commitment.rs`, `api/scheme.rs`, `protocol/flow.rs`,
   `protocol/quadratic_equation.rs`, `protocol/ring_switch.rs`, `akita-scheme`,
   examples, benches, and tests to the operation stack.
9. Delete `AkitaPolyOps`, its blanket `&P` impl, and the old monolithic
   `ProverComputeBackend` public boundary.
10. Remove compatibility imports/re-exports and update docs.
11. Run forbidden-pattern greps for `AkitaPolyOps`, public
    `ProverComputeBackend` protocol/API bounds, and public closed source enums.
12. Run the full checks and profiling commands named above.

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
- Operation-stack generics can get large. Prefer small context structs and
  capability bundle marker traits over passing six unrelated generic type
  parameters through every helper.
- Mixed-backend tests should distinguish backend values at the type level, not
  merely by runtime flags, so they actually protect the heterogeneous design.

Expected implementation diff:

- `crates/akita-prover/src/lib.rs`: large deletion of `AkitaPolyOps`, smaller
  re-export additions for new traits.
- `crates/akita-prover/src/compute/`: substantial replacement of the fixed
  commitment/ring-switch/prover backend surfaces (`backend.rs`) with operation
  contexts, source-kernel traits, and standard helper kernels (`cpu.rs`).
- `crates/akita-prover/src/backend/*.rs`: moderate churn moving impl blocks
  from `AkitaPolyOps` to provider/view/kernel impls.
- `crates/akita-prover/src/protocol/flow.rs` and
  `crates/akita-prover/src/protocol/quadratic_equation.rs`: moderate call-site
  churn, no intended protocol logic change.
- `akita-scheme`, `akita-pcs` examples, benches, and tests: mechanical generic
  bound and helper updates.

Rough size estimate for the implementation PR: 24 to 42 files touched, about
2.4k to 4.0k lines added and 1.8k to 3.2k lines removed. The final diff should
feel like replacing the old abstraction boundaries, not adding a parallel layer
beside them. Before final review, the old layer must be deleted rather than
left as a compatibility path beside the new one.

## References

- [`specs/akita-compute-backend-metal.md`](akita-compute-backend-metal.md):
  current compute backend and Metal roadmap; this spec supersedes the
  `AkitaPolyOps` cutover notes there.
- [`crates/akita-prover/src/lib.rs`](../crates/akita-prover/src/lib.rs):
  current `AkitaPolyOps` definition and blanket reference impl.
- [`crates/akita-prover/src/compute/`](../crates/akita-prover/src/compute/):
  typed compute backend setup, low-level commit/ring-switch plans, and PO1
  source-typed kernel trait skeletons.
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
- [`crates/akita-prover/src/protocol/ring_switch.rs`](../crates/akita-prover/src/protocol/ring_switch.rs):
  recursive witness commitment and ring-switch flow call sites.

## Appendix: Streaming Commitment And The Commit-Source Boundary

This appendix was added after the initial cutover design to record how the
source-typed commit boundary should support a future streaming, low-materialization
commitment path for a Jolt-style integration. It constrains the *shape* of the
commit source, not the kernel algorithms. No streaming is implemented in this PR.

### Motivation

A near-term consumer (Akita as the PCS under a Jolt-style zkVM) wants to commit to
the largest root witnesses directly from an execution trace with little or no
intermediate materialization. The ideal is: from a trace, or a lazy trace over a
program, walk cycles in row-sized chunks, map each chunk to its contribution to
the commitment, and accumulate, without ever holding a full witness polynomial.
Akita's root Ajtai/SIS commitment is linear and block-structured
(`commit = sum_blocks A * decompose(block)`, realized as the column-sweep / NTT
mat-vec kernels in `compute/cpu.rs`), so it is inherently amenable to this
block-incremental accumulation, and only the root (level-0) commitment of the
largest witnesses needs it.

### Durable rule: no separate streaming trait; the source is the seam

The source-typed kernel boundary this spec already defines is the streaming seam.
`RootCommitKernel<S, F, D>::commit_inner(prepared, source: S, plan)` is a pull
model: the kernel receives a source and reads what it needs at its own cadence. A
fully materialized `DensePoly` / `OneHotPoly` is simply the `S` that traverses its
own stored coefficients; a lazy trace-backed source is the `S` that produces rows
on demand. Both use the same kernel and the same single entry point.

Akita must therefore NOT grow a separate `StreamingCommitment`-style trait with a
`begin` / `feed` / `finish` state machine. Streaming is a property of how a kernel
consumes its source, not a distinct public trait. Concretely:

- The commit-source capability must be able to hand back a *traversable*, possibly
  non-materialized source, not only a borrow of an already-built polynomial.
  `RootCommitSource::commit_view` stays the entry, but its associated view should
  expose row/block traversal rather than implying a finished coefficient table.
- The backend kernel owns the *strategy*: block/row accumulation, column-sweep
  parallelism, opening-hint construction, and the transparent-vs-ZK finish. The
  source owns only the *data and traversal shape*. Streaming is not fully
  quarantined inside the source; it is split, with the source describing shape and
  the backend owning the schedule.
- `commit_inner` stays the single entry point. Whether it streams or materializes
  is an internal backend choice, invisible to protocol code and byte-identical in
  output because the commitment is deterministic and linear.

### Lessons this rule encodes (Jolt prior art)

This is a synthesis of three existing designs, recorded so the boundary is not
re-litigated:

- jolt-core today already streams: `StreamingCommitmentScheme`
  (`process_chunk` / `aggregate_chunks`) plus `stream_witness_and_commit_rows`
  drive a lazy trace in row chunks and never materialize a full witness polynomial.
- The modular `refactor/audit-prep` crates kept *two* layers: a row source
  (`MultilinearPoly::for_each_row`, which yields already-widened `&[F]`) and a
  separate push-model `StreamingCommitment` (`begin` / `feed` / `finish`) plus a
  typed `PolynomialChunk` stream. The separate trait exists *because*
  `for_each_row` drops to `&[F]` and loses the compact small-scalar / one-hot
  encoding, so the compact streaming path had to be recovered through a second
  trait.
- The earlier `jolt-openings` cutover (a16z/jolt #1521) went the other way: one
  `CommitmentSource` with a *typed* `for_each_row` row (`SourceRow`) feeding a
  single `commit_batch` / `commit_batch_zk`, with the backend overriding those to
  keep the streaming schedule. No separate streaming trait.

The decisive lesson: a row source must carry the *encoding type*. A `&[F]` source
forces a second streaming trait; a typed-row source does not. Akita's generic `S`
is a strict improvement on both, because the source type is open rather than a
closed `SourceRow` enum, so an encoding palette never has to be the public
extension point.

Two further footguns from the #1521 review, encoded here as constraints:

- Keep fused/batch kernels first-class. In #1521 the fused Stage-8 RLC could not
  express itself as a single row scan and degenerated into a panic stub plus a
  backend-specific override. Akita's `OpeningBatchKernel` and tensor-batch kernels
  must remain the path for fused work; a source must never be forced to fake row
  traversal it cannot provide.
- One canonical lowering. In #1521 the `SourceRow` to evaluations conversion was
  duplicated per backend with divergent one-hot semantics. Akita must have a single
  shared lowering from commit-source rows to ring blocks / digit planes, reused by
  every backend rather than re-derived.

### Provisional: the row-encoding palette (deliberately not frozen)

For the built-in CpuBackend fast paths the kernel needs a concrete row vocabulary.
The natural shape is a small Akita-owned enum of physical encodings, for example:

```rust
// PROVISIONAL: shape illustrative, not a committed API. Start with the encodings
// the built-in sources actually need and expand only when a concrete source does.
pub enum CommitRow<'a, F, const D: usize> {
    Ring(&'a [CyclotomicRing<F, D>]),
    DigitPlanes(&'a [[i8; D]]),
    OneHot(&'a [Option<usize>]),
    Zeros(usize),
}
```

This enum is intentionally left open-ended and is NOT the extensibility contract:

- It is a physical *output-shaped* sum type, the same category as
  `TensorPackedWitness` and `MultilinearPolynomial`, not the closed input-source
  enum this spec rejects elsewhere. A finite set of row encodings is acceptable for
  the same reason those are.
- It is never load-bearing for extension. Because the kernel is generic over `S`, a
  source whose data does not fit the palette can implement the commit kernel
  directly over its own view, exactly as a downstream custom polynomial does for
  the non-streaming path. The palette is a convenience for the standard CpuBackend
  reduction, not a gate.
- Risk acknowledged: a fixed encoding enum can become too restrictive (a packed
  device buffer, a strided layout, a wider small-scalar type). The mitigation is to
  keep the palette minimal at first, let real sources drive additions, and rely on
  the generic `S` escape hatch rather than enumerating speculatively. Whether this
  enum is introduced in the first cutover at all, or deferred until a streaming
  source actually lands, is left open. The durable commitment is the generic source
  seam above, not this enum.

### Deferred decision: opening-hint retention

Akita's commit output retains the full decomposed witness (`CommitInnerWitness`)
because the opening fold consumes it, unlike a Dory-style compact row-commitment
hint. Streaming the commit *input* therefore does not by itself remove the
O(witness) retention. Two models, deferred to when a streaming source is actually
wired:

- Single-materialization: stream the input to avoid a separate `DensePoly` /
  `OneHotPoly` intermediate, but still build `CommitInnerWitness` once for the
  opening. Smaller change; matches the literal "from trace, contribute to the
  commitment" goal.
- Minimal-materialization: `commit_inner` returns only the compact commitment plus
  a small hint, and the opening path re-derives the witness from the source by
  re-traversing it. Requires the opening source and `OpeningFoldKernel` to also
  accept a re-traversable source. Larger change; the asymptote if memory is the
  binding constraint.

### What PO-CUTOVER should preserve now

Without implementing streaming in this PR, the cutover should avoid foreclosing it:

- [ ] `RootCommitSource::commit_view` and its associated view are defined so the
      view can later expose row/block traversal, not only a borrow of a finished
      coefficient table.
- [ ] The commit call-site cutover threads a commit context/driver that consumes a
      source through the kernel, rather than hard-coding a whole-poly `commit_inner`
      shape into every flow site that would then have to be reopened to add a
      streaming source.
- [ ] No `StreamingCommitment`-style begin/feed/finish trait is introduced.
- [ ] Fused/batch kernels remain the path for fused work; no source is required to
      panic-stub row traversal.

### Concrete instantiation: Jolt trace-only commit (lz/integrate-hachi prior art)

This subsection records, comprehensively, how a trace-only commit (no polynomial
materialization) was already made to work against the pre-cutover monolithic
poly-ops trait, and exactly how that maps onto the source-typed boundary. It is
deliberately not distilled: it exists so the design is not rediscovered or
re-litigated. The prior art is the `lz/integrate-hachi` branch of the `jolt` repo
("hachi" is Akita's former name), files
`jolt-core/src/poly/commitment/hachi/{commitment_scheme.rs, packed_poly.rs, packed_layout.rs}`
and `jolt-core/src/zkvm/prover.rs`.

#### The goal restated

Commit the main zkVM witness directly from the execution trace `&[Cycle]`:
(i) every committed polynomial is in one-hot form, including the `RamInc` / `RdInc`
increments (offset-encoded as `inc + 2^64`, then split into `d_inc = ceil(65/8) = 9`
eight-bit one-hot chunks; see `jolt-hachi/HACHI_INTEGRATION.md`), and (ii) all
one-hot polys are concatenated into one mega-polynomial committed with a single
Ajtai commitment. No `OneHotPoly` or `DensePoly` is materialized at any point.

#### Layer A: the trace adapter (trace to one-hot index)

`LazyOneHotSource { trace: &[Cycle], polys: &[CommittedPolynomial], preprocessing,
one_hot_params }` (`jolt-core/src/zkvm/prover.rs:167`). The only essential method is
`onehot_index(cycle_idx, poly_idx) -> Option<u8>` (`prover.rs:180`), which is
`polys[poly_idx].extract_index(&trace[cycle_idx], ...)`. There is no polynomial
array: a "polynomial" is a pure function of the trace. `batch_onehot_indices`
(`prover.rs:189`) decodes a cycle once and fills a whole row of poly indices,
including the increment families computed on the fly via `inc_chunk(base, chunk)`
(`prover.rs:240-265`). This is the seam where the trace becomes one-hot indices,
and it is the only place that knows the `Cycle` type.

#### Layer B: the mega-poly concatenation (a bit-layout, not a buffer)

`PackedBitLayout` (`packed_layout.rs:82`) defines a bijection
`(cycle, poly, hot_index) -> (block_idx, pos_in_block, coeff_idx)` via `locate`
(`packed_layout.rs:278`). It interleaves the one-hot address bits, the cycle bits,
and the poly-selector bits into a single packed multilinear, keeping each block
trace-local (knobs `PACKED_MIN_CYCLE_TILE_BITS` / `PACKED_MAX_POLY_TILE_BITS`,
`packed_layout.rs:9-13`). The concatenation of all polynomials is therefore not a
concatenated buffer; it is a coordinate mapping. The poly-selector is just extra
high coordinates of the same packed polynomial, and the batch-open point is built
by `reorder_packed_point` (`packed_layout.rs:376`).

#### Layer C: the lazy commit (walk blocks, pull trace, accumulate columns)

`JoltPackedPoly { packed_layout, index_fn, batch_fn, num_cycles, num_polys }`
(`packed_poly.rs:159`) holds only closures over the source and implements the old
monolithic poly-ops trait (`packed_poly.rs:334`). `commit_inner`
(`packed_poly.rs:441`) walks output blocks; per block, `for_each_entry_in_block`
(`packed_poly.rs:249`) iterates the block's `(cycle, poly)` rectangle, calls
`index_fn`, `locate`s each hit to `(pos_in_block, coeff_idx)`, and accumulates the
A-matrix column rotated by `coeff_idx` via `shift_accumulate_into_fast`
(`packed_poly.rs:543`). A one-hot value at coefficient `coeff_idx` is the ring
monomial `X^coeff_idx`, so the accumulate is a negacyclic shift of the A-column, not
a plain add. Nothing is materialized: the polynomial is recomputed from the trace on
every pass.

#### Why it is genuinely trace-only, including the opening

- The whole commit is `(&[Cycle], &[CommittedPolynomial]) -> commitment`.
  Materialization is actively forbidden on this path: `poly_to_ring_coeffs`
  `panic!`s on `OneHot` (`commitment_scheme.rs:879`) precisely to ensure the one-hot
  witness only ever flows through the lazy source.
- The opening also re-walks the trace. `fold_blocks`, `evaluate_and_fold`, and
  `decompose_fold` (`packed_poly.rs:374-437`) all re-read `index_fn` / `batch_fn`.
  So across commit -> open the retained state is the trace plus the compact commit
  hint (`CommitInnerWitness { t, t_hat }`, which is O(committed blocks), not O(N)).
  This is the minimal-materialization model from the deferred-decision section
  above, realized: the trace is the retained representation and the witness is
  recomputed, rather than holding a materialized witness across the opening.
- Batching is the mega-poly batch opening. `batch_prove` (`commitment_scheme.rs:501`)
  samples a selector `rho`, forms the combined claim `sum_i eq(rho, i) * v_i`
  (`commitment_scheme.rs:537`), and runs a single Ajtai opening on the reordered
  packed point. There is no homomorphic `combine_commitments`, which is impossible
  for Ajtai because of the gadget decomposition `G^{-1}`.

#### Mapping onto the source-typed boundary

In the monolith the polynomial owned both the data and the algorithm, so
`JoltPackedPoly` simply overrode `commit_inner`. The cutover splits those: the
source owns data and traversal, the backend owns the kernel. The factoring is clean
and load-bearing:

- `JoltPackedPoly` becomes a custom `RootCommitSource` `S`. Its traversal is exactly
  the per-block sparse iterator `for_each_block(block_idx) -> {(pos_in_block,
  coeff_idx)}` plus `live_block_count` / `positions_per_block` and a density hint. The
  `PackedBitLayout` mapping stays inside the source, because the layout is a property
  of how this particular mega-poly is concatenated, not a backend concern.
- The CpuBackend `commit_inner` kernel owns the strategy zoo currently inlined in the
  packed poly: fast-singleton (`packed_poly.rs:985`), column-sweep
  (`packed_poly.rs:805`), and tiled (`packed_poly.rs:1040`). These differ only in how
  they tile A-columns and schedule the accumulate; all consume the same per-block
  entry stream. So traversal = source, accumulate schedule = backend, with no leakage
either way. This is end-to-end confirmation of the "backend owns strategy, no
separate streaming trait" rule.

#### What already exists in Akita today (do not rebuild)

A correction to any impression that the primitives below are future work: the
one-hot commit path the cutover is building already contains them. They are existing
CpuBackend machinery, not things to invent for a streaming consumer.

- The one-hot / monomial-rotation accumulate primitive exists:
  `inner_ajtai_wide_single_chunk` and `inner_ajtai_wide_multi_chunk`
  (`crates/akita-prover/src/backend/onehot/inner_ajtai.rs`) compute
  `t[a] += A[a][col] * X^coeff_idx` via `shift_accumulate_into`, exactly the
  negacyclic monomial rotation of the A-column by the hot coefficient. The same
  rotation drives fold / decompose-fold (`fill_rotated_challenge` in
  `backend/onehot/accumulate.rs`).
- The per-block sparse `(pos, coeff)` entry representation exists:
  `SingleChunkEntry { pos_in_block, coeff_idx }` and
  `MultiChunkEntry { pos_in_block, nonzero_coeffs }` (`backend/onehot/entries.rs`).
- The per-block traversal and the cross-polynomial concatenation exist:
  `FlatBlocks<E>` with `offsets` / `block(i)` and `OneHotBlocks` /
  `commit_plan_blocks()` (`backend/onehot/blocks.rs`), consumed as `&[&[Entry]]` by
  the accumulators, whose batched callers already concatenate slices across
  polynomials (`backend/onehot/accumulate.rs`). That is the mega-poly concatenation
  at the entry level.

So `CommitTraversal` is not a new subsystem; it is a small generalization of the
existing `OneHotView` / `commit_plan_blocks()` boundary
(`backend/onehot/ops.rs`). What is genuinely missing is narrow and splits by
materialization model:

- Single-materialization (build O(nonzeros) entries, not the O(N) dense poly):
  construct an `OneHotBlocks` / commit view from externally produced indices or
  entries and commit it. This needs only a public producer path, no new trait, and
  no orphan-rule problem, because the foreign crate produces Akita-owned types
  (`OneHotBlocks`) and calls an Akita commit entry point. Today the only producer is
  `OneHotPoly::blocks_for` -> `FlatBlocks::from_indices` (`backend/onehot/ops.rs`,
  `backend/onehot/blocks.rs`), which requires a materialized `indices: &[Option<I>]`
  vector; the gap is exposing entry/block construction that does not route through a
  stored `OneHotPoly`.
- Minimal-materialization (re-walk the trace per block, never build an entry
  vector): generalize `onehot_commit_rows` / `inner_ajtai_wide_*` to consume a
  per-block entry *iterator* instead of `&[Entry]`. This pull source is the only
  genuinely new abstraction, and the only place the `CommitTraversal` blanket impl
  (a strategy-reuse convenience, not an orphan-rule requirement) matters.

#### Coherence on the pull path: per-view impls are allowed, the blanket is reuse

For the minimal-materialization path (a foreign source type that drives the
kernel by pull, rather than producing an Akita-owned `OneHotBlocks`), the
question is how a downstream crate (Jolt) plugs a custom source into a CpuBackend
kernel. The single-materialization path above sidesteps it entirely, because the
foreign crate produces Akita-owned types and calls an Akita entry point. Given
the current kernel shape in `crates/akita-prover/src/compute/kernels.rs`:

```rust
pub trait RootCommitKernel<S, F, const D: usize>: ComputeBackendSetup<F> {
    fn commit_inner(&self, prepared: &Self::PreparedSetup<D>, source: S,
        plan: CommitInnerPlan) -> Result<FlatDigitBlocks<D>, AkitaError>;
    // commit_inner_witness ...
}
```

The orphan rule does not forbid the direct per-view impl. The source is the
trait's first type parameter, so in
`impl RootCommitKernel<JoltPackedView, F, D> for CpuBackend` the downstream
crate's local `JoltPackedView` is a local type in the trait reference, appearing
before the only uncovered parameter `F`. That is the same coherence-allowed shape
as `impl From<LocalType> for ForeignType`, and it is exactly the main-body
extension mechanism (`impl OpeningFoldKernel<MySparseView<'a>, ...> for
CpuBackend`). A downstream crate can therefore implement any kernel for
`CpuBackend` over its own local source view without editing Akita, and such an
impl never overlaps Akita's concrete per-view impls because the source type
differs.

What a direct per-view impl does not give downstream is reuse of CpuBackend's
commit strategy zoo: it would have to re-implement `commit_inner` itself. The
optional convenience that fixes that is a traversal trait plus a backend blanket
impl over it:

```rust
// in akita-prover: a downstream source implements `CommitTraversal` and
// inherits CpuBackend's strategy zoo instead of re-implementing commit_inner.
impl<S, F, const D: usize> RootCommitKernel<S, F, D> for CpuBackend
where
    S: CommitTraversal<F, D>,
{ /* fast-singleton / column-sweep / tiled strategy zoo lives here */ }
```

Then Jolt writes `impl CommitTraversal for JoltPackedView` in its own crate (a
foreign trait on a local type, which coherence allows) and reuses
`CpuBackend::commit_inner` unchanged, without re-deriving the accumulate
schedule.

The genuine constraint is coherence between the two forms, which makes the
blanket a now-or-later decision: concrete per-view impls
(`impl RootCommitKernel<DenseView> for CpuBackend`, etc.) and a later
blanket `impl<S: CommitTraversal> ...` overlap and cannot coexist. Enabling the
blanket form later therefore means routing the built-in views through
`CommitTraversal` too. That is an internal akita-prover refactor (it does not
change the public kernel signature or any call site, and there is no downstream
consumer in-repo yet), so it is reversible, but it is not free: it touches every
built-in commit kernel impl. The one thing that would make downstream extension
impossible is leaking a *closed input-source enum* into the kernel signature in
place of the generic `S`; that must not happen.

#### The enum is the backend's accumulate vocabulary, not the extension point

This instantiation is concrete evidence against freezing the `CommitRow` palette.
The packed source does not fit `CommitRow::OneHot(&[Option<usize>])`:

- Its per-block rows are not contiguous slices. They are scattered
  `(pos_in_block, coeff_idx)` pairs produced by the layout walk (`locate`,
  `packed_layout.rs:278`), so an `&[Option<usize>]` indexed by position cannot
  represent them without a dense expansion that defeats the purpose.
- The accumulate is a monomial rotation of the A-column (`shift_accumulate_into_fast`,
  `packed_poly.rs:543`), not a coefficient add. `coeff_idx` is a rotation amount, not
  an index into a value slice.

The correct reading: the palette enumerates the backend's *accumulate primitives*
(dense ring mat-vec vs one-hot-with-monomial-rotation), not input layouts. A
downstream source either maps onto an existing primitive (Jolt's packed one-hot is
exactly "one-hot with monomial rotation", which already exists as
`inner_ajtai_wide_single_chunk` / `_multi_chunk`) or needs a new primitive, which is
an additive Akita-side change. The generic source `S` is the extension point (the
traversal trait is one convenience for reusing CpuBackend's fold); the enum is just
what the CpuBackend knows how to fold. This is why it stays provisional, and
note that "one-hot with monomial rotation" is already the built-in CpuBackend
primitive, not a new palette entry a consumer must add.

### Concrete suggestions and where they land

Precise actions, tagged by the PR that should carry each. The actual trace-only
commit is an explicit non-goal of the cutover; only the shape decisions that keep it
cheap to add later belong in the cutover PRs.

1. (Spec PR #109, polyops-cutover spec) This appendix, including this instantiation.
   Records the boundary so it is not re-derived. Done here.

2. (PO-CUTOVER, MUST) Keep `RootCommitSource::CommitView<'a>` an unconstrained
   associated type (it already is, `crates/akita-prover/src/compute/poly.rs`). Do not
   bind it to a borrow of stored coefficients, to `MultilinearPolynomial`, or to any
   akita-witness type. The view must be allowed to be a computed, non-materialized,
   layout-aware, sparse-per-block traversal.

3. (PO-CUTOVER, MUST) Do the commit call-site rewrite (`api/commitment.rs`,
   `api/scheme.rs`, `protocol/flow*`) by threading commit through an operation
   context/driver that consumes a source via the kernel, never by hard-coding a
   whole-poly `commit_inner` shape into each flow site. This is the natural shape of
   the rewrite the node already does; it costs nothing extra and avoids reopening
   every site to add a streaming source later.

4. (PO-CUTOVER, MUST NOT) Do not put a closed input-source enum in the
   `RootCommitKernel` signature, and do not introduce a separate
   `StreamingCommitment`-style begin/feed/finish trait. Either would foreclose the
   blanket-over-traversal extension and is the one irreversible mistake here.

5. (PO-CUTOVER, SHOULD) Note that the entry/block boundary a streaming consumer
   needs already exists (`OneHotView` / `commit_plan_blocks()` /
   `OneHotBlocks`, see "What already exists in Akita today"); the only coupling is
   that its sole producer is `OneHotPoly::blocks_for` -> `FlatBlocks::from_indices`
   over a materialized index vector. The cheap, real move is to keep the commit
   kernel keyed on that entry/block view rather than re-deriving blocks from a stored
   `OneHotPoly` deep inside the kernel, so a non-`OneHotPoly` producer can later feed
   the same view (single-materialization) with no new trait and no orphan handling.
   Do not build the `CommitTraversal` pull trait or a blanket
   `impl<S: CommitTraversal> RootCommitKernel<S> for CpuBackend` speculatively now:
   there is no in-repo consumer, and the minimal-change rule applies. Record that
   adding the pull path later is an internal akita-prover change, non-breaking for
   any call site and any downstream.

6. (PO-CUTOVER, MUST) Keep fused/batch kernels first-class (`OpeningBatchKernel`,
   tensor-batch). No source is ever required to panic-stub row traversal it cannot
   provide, per the #1521 Stage-8 footgun above.

7. (NOT in any cutover PR; future "Akita-as-PCS / streaming-commit" node, gated on
   V / `akita-witness` #159 plus a Jolt-side adapter) The genuinely new pieces: for
   minimal-materialization, a per-block pull/iterator entry source and its blanket
   backend impl; plus the Jolt-specific packed bit-layout, lazy trace source,
   increment-to-one-hot encoding, and mega-poly selector batch opening. Explicitly
   NOT in this list, because they already exist (see "What already exists in Akita
   today"): the one-hot / monomial-rotation accumulate (`inner_ajtai_wide_*`), the
   sparse-entry block view (`FlatBlocks` / `OneHotBlocks` / `commit_plan_blocks()`),
   and cross-poly concatenation. Bundling the new pieces into the cutover would
   violate the stated non-goal and the minimal-change rule.

8. (NOT PO1 #161, NOT PO-dense #162) PO1 is the trait/context skeleton and is already
   open; PO-dense is the narrow `backend/dense.rs` slice. No streaming or
   trace-source work should be retargeted to either. The compute-boundary edits in
   suggestions 2-6 are owned by PO-CUTOVER under the PO-COMPUTE lock.
