#[cfg(feature = "zk")]
use super::backend::DigitRowsComputeBackend;
use super::backend::{CommitmentComputeBackend, ComputeBackendSetup, RingSwitchComputeBackend};
use super::kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RootCommitKernel, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::backend::RecursiveWitnessFlat;
#[cfg(feature = "zk")]
use crate::DensePoly;
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::RandomSampling;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_types::CleartextWitnessProof;

/// Shape metadata every root polynomial exposes.
///
/// This is the base capability: it carries no view and no backend work, so
/// shape-only APIs can require just `RootPolyShape` without pulling in commit,
/// opening, tensor, or direct-witness capabilities.
pub trait RootPolyShape<F, const D: usize>: Clone + Send + Sync
where
    F: FieldCore,
{
    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// Total number of variables (`log2(num_ring_elems() * D)`).
    ///
    /// # Panics
    ///
    /// Panics if `num_ring_elems() * D` overflows `usize`. This is a prover-only
    /// shape helper and is not reachable from verifier paths.
    fn num_vars(&self) -> usize {
        let total = self
            .num_ring_elems()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        debug_assert!(
            total.is_power_of_two(),
            "total field elements must be a power of 2"
        );
        total.trailing_zeros() as usize
    }

    /// One-hot chunk size for sparse one-hot backends.
    ///
    /// `None` means this backend is not a one-hot root representation.
    fn onehot_chunk_size(&self) -> Option<usize> {
        None
    }
}

/// Capability: expose a borrowed commit source view for a `RootCommitKernel`.
pub trait RootCommitSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Borrowed commit view consumed by `RootCommitKernel`.
    type CommitView<'a>
    where
        Self: 'a;

    /// Borrow a commit view of this polynomial.
    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError>;
}

/// Capability: expose borrowed opening views for the opening fold kernels.
pub trait RootOpeningSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Borrowed single-poly opening view consumed by `OpeningFoldKernel`.
    type OpeningView<'a>
    where
        Self: 'a;

    /// Borrowed same-point batch view consumed by `OpeningBatchKernel`.
    type OpeningBatchView<'a>
    where
        Self: 'a;

    /// Borrow an opening view of this polynomial.
    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError>;

    /// Borrow a same-point batch opening view over several polynomials.
    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError>;
}

/// Capability: expose borrowed tensor views for the tensor projection kernels.
pub trait RootTensorSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Borrowed single-poly tensor view consumed by `TensorProjectionKernel`.
    type TensorView<'a>
    where
        Self: 'a;

    /// Borrowed same-point batch view consumed by `TensorProjectionBatchKernel`.
    type TensorBatchView<'a>
    where
        Self: 'a;

    /// Borrow a tensor view of this polynomial.
    ///
    /// The view is extension-field independent; the opening point type `E`
    /// enters only at kernel evaluation.
    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError>;

    /// Borrow a same-point batch tensor view over several polynomials.
    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError>;
}

/// Capability: materialize a direct root witness for zero-fold openings, and
/// the dense field-element evaluation table derived from it.
///
/// This is an explicit opt-in, not a hidden default on every root polynomial:
/// only proving paths that may select a root-direct schedule (or the
/// extension-opening reduction's dense-term fallback) require it. Both are
/// prove-only capabilities, so bundling them does not widen the commit-path
/// capability bound.
pub trait DirectRootWitnessSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Materialize a direct root witness payload.
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError>;

    /// Dense field-element evaluation table for this polynomial.
    ///
    /// Defaults to the field-element payload of [`Self::direct_root_witness`].
    /// Representations whose direct witness is unavailable, or whose evaluations
    /// have a cheaper derivation, override this.
    ///
    /// # Errors
    ///
    /// Returns an error if the representation cannot materialize its dense
    /// evaluation table.
    fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        let witness = self.direct_root_witness()?;
        let field_elems = witness.as_field_elements().ok_or_else(|| {
            AkitaError::InvalidInput("base evals require field-element witness payload".to_string())
        })?;
        Ok(field_elems.coeffs().to_vec())
    }
}

/// One opening-point polynomial bundle passed to commit entry points.
///
/// The wrapper pins the polynomial type `P` for inference through generic
/// [`crate::api::commitment::commit`] and [`crate::api::CommitmentProver::commit`]. Scheme-level
/// [`crate::api::CommitmentProver::commit`] takes this bundle before `backend` so `P` is known when the
/// compiler checks [`RootCommitBackend`].
#[derive(Clone, Copy, Debug)]
pub struct RootCommitPolys<'a, P> {
    polys: &'a [P],
}

impl<'a, P> RootCommitPolys<'a, P> {
    /// Borrow a slice of root polynomials.
    #[must_use]
    pub fn new(polys: &'a [P]) -> Self {
        Self { polys }
    }

    /// Borrow a singleton polynomial bundle.
    #[must_use]
    pub fn from_ref(poly: &'a P) -> Self {
        Self {
            polys: std::slice::from_ref(poly),
        }
    }

    /// Borrowed polynomial slice.
    #[must_use]
    pub fn as_slice(&self) -> &'a [P] {
        self.polys
    }
}

/// Marker bundle for scheme-level commit entry points that may tensor-project.
///
/// Algorithms live on [`RootCommitKernel`] / [`TensorProjectionKernel`], not here.
/// Lower-level helpers such as [`crate::api::commitment::commit_with_params`]
/// should bound only [`RootCommitSource`].
pub trait RootCommitPoly<F, const D: usize>:
    RootPolyShape<F, D> + RootCommitSource<F, D> + RootTensorSource<F, D>
where
    F: FieldCore,
{
}

impl<F, const D: usize, P> RootCommitPoly<F, D> for P
where
    F: FieldCore,
    P: RootPolyShape<F, D> + RootCommitSource<F, D> + RootTensorSource<F, D>,
{
}

/// Capability: this backend can **commit** a single source `P`.
///
/// This is the uniform "source-typed capability" vocabulary: a bound of the form
/// "backend `Self` can commit source `P`", rather than a hard-coded per-type
/// kernel bundle. It folds together the row-commit surface
/// ([`CommitmentComputeBackend`], which also supplies [`super::DigitRowsComputeBackend`])
/// and the inner-commit kernel over `P`'s borrowed commit view.
///
/// The same alias is applied to the generic input poly and to the internal
/// [`RootTensorProjectionPoly`] (the extension-reduction projection), so both
/// source types are expressed through one symmetric concept.
pub trait CommitBackendFor<F, P, const D: usize>: CommitmentComputeBackend<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, D>,
    Self: for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
}

impl<F, P, const D: usize, B> CommitBackendFor<F, P, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
}

/// Capability: this backend can run **opening fold** kernels over a single
/// source `P` (evaluate/fold and batched decompose-fold).
pub trait OpeningProveBackendFor<F, P, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, D>,
    Self: for<'a> OpeningFoldKernel<<P as RootOpeningSource<F, D>>::OpeningView<'a>, F, D>
        + for<'a> OpeningBatchKernel<<P as RootOpeningSource<F, D>>::OpeningBatchView<'a>, F, D>,
{
}

impl<F, P, const D: usize, B> OpeningProveBackendFor<F, P, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> OpeningFoldKernel<<P as RootOpeningSource<F, D>>::OpeningView<'a>, F, D>
        + for<'a> OpeningBatchKernel<<P as RootOpeningSource<F, D>>::OpeningBatchView<'a>, F, D>,
{
}

/// Capability: this backend can run **tensor projection** kernels (single and
/// batched) over a single source `P` at extension-field opening point `E`.
pub trait TensorBackendFor<F, P, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootTensorSource<F, D>,
    Self: for<'a> TensorProjectionKernel<<P as RootTensorSource<F, D>>::TensorView<'a>, F, E, D>
        + for<'a> TensorProjectionBatchKernel<
            <P as RootTensorSource<F, D>>::TensorBatchView<'a>,
            F,
            E,
            D,
        >,
{
}

impl<F, P, E, const D: usize, B> TensorBackendFor<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionKernel<<P as RootTensorSource<F, D>>::TensorView<'a>, F, E, D>
        + for<'a> TensorProjectionBatchKernel<
            <P as RootTensorSource<F, D>>::TensorBatchView<'a>,
            F,
            E,
            D,
        >,
{
}

/// Capability: this backend can **tensor-project** a single source `P` at an
/// extension-field opening point of type `E`.
///
/// Commit-side alias for single-point tensor projection only. Prove paths use
/// the full [`TensorBackendFor`] bundle (single + batch).
pub trait ProjectBackendFor<F, P, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootTensorSource<F, D>,
    Self: for<'a> TensorProjectionKernel<<P as RootTensorSource<F, D>>::TensorView<'a>, F, E, D>,
{
}

impl<F, P, E, const D: usize, B> ProjectBackendFor<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionKernel<<P as RootTensorSource<F, D>>::TensorView<'a>, F, E, D>,
{
}

/// Capability: this backend can run the full **opening/prove** kernel set over a
/// single source `P` at an extension-field opening point of type `E`.
///
/// Composed from [`OpeningProveBackendFor`] and [`TensorBackendFor`]. Like
/// [`CommitBackendFor`], the same alias is applied to both the generic input poly
/// and the internal [`RootTensorProjectionPoly`].
pub trait ProveBackendFor<F, P, E, const D: usize>:
    OpeningProveBackendFor<F, P, D> + TensorBackendFor<F, P, E, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootOpeningSource<F, D> + RootTensorSource<F, D>,
{
}

impl<F, P, E, const D: usize, B> ProveBackendFor<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootOpeningSource<F, D> + RootTensorSource<F, D>,
    B: OpeningProveBackendFor<F, P, D> + TensorBackendFor<F, P, E, D>,
{
}

/// Backend capability bundle for scheme-level commit with optional tensor transform.
///
/// Use as **`B: RootCommitBackend<F, P, E, D>`** on generic `fn commit<P, B>(backend: &B, …)`.
///
/// Composed from the uniform source-typed capabilities: the backend must
/// [`CommitBackendFor`] the input poly `P`, [`ProjectBackendFor`] it (tensor projection),
/// and [`CommitBackendFor`] the internal [`RootTensorProjectionPoly`] produced by the
/// extension-reduction transform. Read it as "commit `P`, project `P`, commit the
/// projection". A blanket impl covers every backend satisfying those three, so a
/// downstream backend opts in structurally (no explicit marker impl required).
///
/// `F: 'static` is required for the same GAT + `for<'a>` view-kernel reason documented on
/// [`RootProveBackend`]. `E` (tensor extension field) is not bounded `'static` here.
pub trait RootCommitBackend<F, P, E, const D: usize>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootCommitPoly<F, D>,
    Self: CommitBackendFor<F, P, D>
        + ProjectBackendFor<F, P, E, D>
        + CommitBackendFor<F, RootTensorProjectionPoly<F, D>, D>,
{
}

impl<F, P, E, const D: usize, B> RootCommitBackend<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootCommitPoly<F, D>,
    B: CommitBackendFor<F, P, D>
        + ProjectBackendFor<F, P, E, D>
        + CommitBackendFor<F, RootTensorProjectionPoly<F, D>, D>,
{
}

/// Marker bundle for scheme-level prove entry points.
///
/// Algorithms live on [`OpeningFoldKernel`] / [`TensorProjectionKernel`], not here.
pub trait RootProvePoly<F, const D: usize>:
    RootOpeningSource<F, D> + RootTensorSource<F, D> + DirectRootWitnessSource<F, D>
where
    F: FieldCore,
{
}

impl<F, const D: usize, P> RootProvePoly<F, D> for P
where
    F: FieldCore,
    P: RootOpeningSource<F, D> + RootTensorSource<F, D> + DirectRootWitnessSource<F, D>,
{
}

/// Backend capability bundle for scheme-level prove.
///
/// Use as **`B: RootProveBackend<F, P, E, D>`** on generic prove entry points.
/// `E` is the protocol extension field (`CommitmentConfig::ExtField`).
///
/// ## Why `F: 'static`?
///
/// The bundle closes over higher-ranked bounds on borrowed polynomial views, e.g.
/// `for<'a> OpeningFoldKernel<<RootTensorProjectionPoly<F, D> as RootOpeningSource<F, D>>::OpeningView<'a>, …>`.
/// Those GATs carry `where Self: 'a` (see [`RootOpeningSource::OpeningView`]). For the
/// bound to hold for **every** lifetime `'a`, `RootTensorProjectionPoly<F, D>` must be
/// `'static`, which requires `F: 'static`. This is a rustc lifetime solver artifact, not
/// a protocol requirement that base-field types outlive the process.
///
/// `E` does **not** need `'static`; preset extension fields satisfy it vacuously, but the
/// trait does not require it.
///
/// Composed from the uniform [`ProveBackendFor`] capability applied to both the input
/// poly `P` and the internal [`RootTensorProjectionPoly`] (the extension-reduction
/// projection), so both source types are expressed through one symmetric concept.
pub trait RootProveBackend<F, P, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    Self: ProveBackendFor<F, P, E, D> + ProveBackendFor<F, RootTensorProjectionPoly<F, D>, E, D>,
{
}

impl<F, P, E, const D: usize, B> RootProveBackend<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    B: ComputeBackendSetup<F>
        + ProveBackendFor<F, P, E, D>
        + ProveBackendFor<F, RootTensorProjectionPoly<F, D>, E, D>,
{
}

/// Backend capability for ZK hiding witness commitment (`DensePoly` inner commit).
///
/// With `zk` enabled, requires `RootCommitKernel` on [`DensePoly`]. Without `zk`, this is a
/// vacuous marker implemented for every [`ComputeBackendSetup`].
#[cfg(feature = "zk")]
pub trait ZkHidingCommitBackend<F, const D: usize>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField + RandomSampling + 'static,
    Self:
        for<'a> RootCommitKernel<<DensePoly<F, D> as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
}

#[cfg(feature = "zk")]
impl<F, const D: usize, B> ZkHidingCommitBackend<F, D> for B
where
    F: FieldCore + CanonicalField + RandomSampling + 'static,
    B: DigitRowsComputeBackend<F>
        + for<'a> RootCommitKernel<<DensePoly<F, D> as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
}

#[cfg(not(feature = "zk"))]
pub trait ZkHidingCommitBackend<F, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField,
{
}

#[cfg(not(feature = "zk"))]
impl<F, const D: usize, B> ZkHidingCommitBackend<F, D> for B
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
}

/// Ring dimensions the recursive suffix may dispatch besides the config ring `D`.
pub const RECURSIVE_SUFFIX_RING_DIMENSIONS: &[usize] = &[32, 64, 128, 256];

/// Full prove-flow capability at a single ring dimension `RING_D`: opening/tensor
/// prove kernels plus ring-switch, commitment rows, and ZK hiding commit.
pub trait ProveFlowBackendFor<F, P, E, const RING_D: usize>:
    RootProveBackend<F, P, E, RING_D>
    + RingSwitchComputeBackend<F>
    + CommitmentComputeBackend<F>
    + ZkHidingCommitBackend<F, RING_D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, RING_D>,
{
}

impl<F, P, E, const RING_D: usize, B> ProveFlowBackendFor<F, P, E, RING_D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, RING_D>,
    B: RootProveBackend<F, P, E, RING_D>
        + RingSwitchComputeBackend<F>
        + CommitmentComputeBackend<F>
        + ZkHidingCommitBackend<F, RING_D>,
{
}

/// [`ProveFlowBackendFor`] at the config ring degree `D`.
pub trait RootProveFlowBackend<F, P, E, const D: usize>: ProveFlowBackendFor<F, P, E, D>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
{
}

impl<F, P, E, const D: usize, B> RootProveFlowBackend<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    B: ProveFlowBackendFor<F, P, E, D>,
{
}

/// Backend bundle for a full recursive prove run.
///
/// Recursive proving dispatches the suffix witness over [`RECURSIVE_SUFFIX_RING_DIMENSIONS`]
/// plus the config ring `D`, so prove entry points need [`ProveFlowBackendFor`] for
/// the root polynomial `P` and [`RecursiveWitnessFlat`] at every supported dimension.
pub trait RecursiveProveBackend<F, P, E, const D: usize>:
    ProveFlowBackendFor<F, P, E, D>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, D>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
{
}

impl<F, P, E, const D: usize, B> RecursiveProveBackend<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    B: ProveFlowBackendFor<F, P, E, D>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, D>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>,
{
}

impl<F, const D: usize, P> RootPolyShape<F, D> for &P
where
    F: FieldCore,
    P: RootPolyShape<F, D>,
{
    fn num_ring_elems(&self) -> usize {
        RootPolyShape::num_ring_elems(*self)
    }

    fn num_vars(&self) -> usize {
        RootPolyShape::num_vars(*self)
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        RootPolyShape::onehot_chunk_size(*self)
    }
}
