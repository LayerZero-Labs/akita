use super::backend::{ComputeBackendSetup, DigitRowsComputeBackend};
use super::kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RingSwitchRelationKernel, RootCommitKernel,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use super::runtime_capabilities::{
    RootProveFlowBackend, RuntimeOpeningProveBackendFor, RuntimeRecursiveWitnessProveBackend,
    RuntimeRingSwitchProveBackend, RuntimeRootProvePoly, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend,
};
use crate::backend::{RecursiveFoldSource, RingSwitchRelationView};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::RandomSampling;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};

/// D-free shape metadata every root polynomial exposes.
///
/// This is the **PCS/batch-facing** capability bound: it names a polynomial's
/// variable count and ring-element count *without* a const ring dimension `D`,
/// so D-free entry points (e.g. [`crate::ProverOpeningData`]) can require just
/// `RootPolyMeta` while the const-D kernel-entry traits ([`RootPolyShape`] and
/// the commit/opening/tensor/direct-witness family) carry `D`.
///
/// `num_vars` is the polynomial's own (schedule/representation-derived) variable
/// count — **not** `log2(num_ring_elems() * D)`. Every input root polynomial
/// stores it directly, so the count is independent of the ring dimension chosen
/// to commit it.
pub trait RootPolyMeta<F>: Clone + Send + Sync
where
    F: FieldCore,
{
    /// Total number of ring elements in the polynomial.
    fn num_ring_elems(&self) -> usize;

    /// Total number of variables (representation-derived, D-independent).
    fn num_vars(&self) -> usize;

    /// One-hot chunk size `K` when this polynomial is a one-hot root
    /// representation.
    ///
    /// `None` means this backend is not a one-hot root representation.
    fn onehot_chunk_size(&self) -> Option<usize> {
        None
    }
}

/// Shape metadata every root polynomial exposes, keyed on the const ring
/// dimension `D`.
///
/// This is the base **kernel-entry** capability: it carries no view and no
/// backend work, so shape-only kernel APIs can require just `RootPolyShape`
/// without pulling in commit, opening, tensor, or direct-witness capabilities.
/// PCS/batch-facing code should prefer the D-free [`RootPolyMeta`] instead.
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

/// One opening-point polynomial bundle passed to commit entry points.
///
/// The wrapper pins the polynomial type `P` for inference through generic
/// `crate::api::commit` and the scheme-level commit entry point. The scheme
/// method takes this bundle before `backend` so `P` is known when the
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
/// kernel bundle. It folds together the shared outer digit-row surface and the
/// inner-commit kernel over `P`'s borrowed commit view.
///
/// The same alias is applied to the generic input poly and to the internal
/// [`RootTensorProjectionPoly`] (the extension-reduction projection), so both
/// source types are expressed through one symmetric concept.
pub trait CommitBackendFor<F, P, const D: usize>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
    P: RootCommitSource<F, D>,
    Self: for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
}

impl<F, P, const D: usize, B> CommitBackendFor<F, P, D> for B
where
    F: FieldCore + CanonicalField,
    P: RootCommitSource<F, D>,
    B: DigitRowsComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
}

/// Ring-switch cluster capability for the source-typed relation kernel.
pub trait RingSwitchProveBackend<F, const D: usize>:
    for<'a> RingSwitchRelationKernel<RingSwitchRelationView<'a, D>, F, D>
where
    F: FieldCore + CanonicalField,
{
}

impl<F, const D: usize, B> RingSwitchProveBackend<F, D> for B
where
    F: FieldCore + CanonicalField,
    B: for<'a> RingSwitchRelationKernel<RingSwitchRelationView<'a, D>, F, D>,
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
        + CommitBackendFor<F, RootTensorProjectionPoly<F>, D>,
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
        + CommitBackendFor<F, RootTensorProjectionPoly<F>, D>,
{
}

/// Marker bundle for scheme-level prove entry points.
///
/// Algorithms live on [`OpeningFoldKernel`] / [`TensorProjectionKernel`], not here.
pub trait RootProvePoly<F, const D: usize>:
    RootOpeningSource<F, D> + RootTensorSource<F, D>
where
    F: FieldCore,
{
}

impl<F, const D: usize, P> RootProvePoly<F, D> for P
where
    F: FieldCore,
    P: RootOpeningSource<F, D> + RootTensorSource<F, D>,
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
/// `for<'a> OpeningFoldKernel<<RootTensorProjectionPoly<F> as RootOpeningSource<F, D>>::OpeningView<'a>, …>`.
/// Those GATs carry `where Self: 'a` (see [`RootOpeningSource::OpeningView`]). For the
/// bound to hold for **every** lifetime `'a`, `RootTensorProjectionPoly<F>` must be
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
    Self: ProveBackendFor<F, P, E, D> + ProveBackendFor<F, RootTensorProjectionPoly<F>, E, D>,
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
        + ProveBackendFor<F, RootTensorProjectionPoly<F>, E, D>,
{
}

/// Full prove-flow capability at a single root ring dimension `RING_D`:
/// opening/tensor prove kernels plus commitment rows.
pub trait ProveFlowBackendFor<F, P, E, const RING_D: usize>:
    RootProveBackend<F, P, E, RING_D> + DigitRowsComputeBackend<F>
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
    B: RootProveBackend<F, P, E, RING_D> + DigitRowsComputeBackend<F>,
{
}

/// Backend bundle for a full recursive prove run.
///
/// Fold levels take their ring dimension from the schedule (`CommittedGroupParams::role_dims`), so
/// prove entry points need [`RootProveFlowBackend`] for the root polynomial
/// `P`, [`RuntimeRecursiveWitnessProveBackend`] for suffix witness
/// opening/tensor and commitment rows, and [`RuntimeRingSwitchProveBackend`]
/// for ring-switch — each at every runtime-supported ring dimension.
pub trait RecursiveProveBackend<F, P, E>:
    RootProveFlowBackend<F, P, E>
    + RuntimeRecursiveWitnessProveBackend<F, E>
    + RuntimeRingSwitchProveBackend<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
{
}

impl<F, P, E, B> RecursiveProveBackend<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    B: RootProveFlowBackend<F, P, E>
        + RuntimeRecursiveWitnessProveBackend<F, E>
        + RuntimeRingSwitchProveBackend<F>,
{
}

/// Cluster capability bundle for [`crate::batched_prove`] with a heterogeneous
/// [`crate::ProverComputeStack`].
///
/// The uniform case `C = O = TS = R = B` is satisfied automatically when
/// `B: RecursiveProveBackend<F, P, E>`.
pub trait ProveStackFor<F, P, E, C, O, TS, R>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    TS: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
}

impl<F, P, E, C, O, TS, R> ProveStackFor<F, P, E, C, O, TS, R> for ()
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    C: ComputeBackendSetup<F> + DigitRowsComputeBackend<F>,
    O: ComputeBackendSetup<F>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + SuffixOpeningProveBackend<F>
        + DigitRowsComputeBackend<F>,
    TS: ComputeBackendSetup<F>
        + RuntimeTensorBackendFor<F, P, E>
        + RuntimeTensorBackendFor<F, RecursiveFoldSource<F>, E>
        + SuffixTensorProveBackend<F, E>,
    R: ComputeBackendSetup<F> + RuntimeRingSwitchProveBackend<F> + DigitRowsComputeBackend<F>,
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

impl<F, P> RootPolyMeta<F> for &P
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    fn num_ring_elems(&self) -> usize {
        RootPolyMeta::num_ring_elems(*self)
    }

    fn num_vars(&self) -> usize {
        RootPolyMeta::num_vars(*self)
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        RootPolyMeta::onehot_chunk_size(*self)
    }
}
