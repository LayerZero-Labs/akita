use super::backend::{
    CommitmentComputeBackend, ComputeBackendSetup, DigitRowsComputeBackend,
    RingSwitchComputeBackend,
};
use super::kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RingSwitchQuotientKernel, RingSwitchRelationKernel,
    RootCommitKernel, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::backend::{
    RecursiveFoldSource, RecursiveWitnessFlat, RingSwitchQuotientView, RingSwitchRelationView,
};
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

    /// One-hot chunk size for sparse one-hot backends.
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
/// `crate::api::commit` and `CommitmentProver::commit`. Scheme-level
/// `CommitmentProver::commit` takes this bundle before `backend` so `P` is known when the
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
/// Lower-level helpers such as [`crate::api::commitment::batched_commit_with_params`]
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

/// Ring-switch cluster capability: row mat-vecs plus source-typed relation/quotient kernels.
pub trait RingSwitchProveBackend<F, const D: usize>:
    RingSwitchComputeBackend<F>
    + for<'a> RingSwitchRelationKernel<RingSwitchRelationView<'a, D>, F, D>
    + for<'a> RingSwitchQuotientKernel<RingSwitchQuotientView<'a, D>, F, D>
where
    F: FieldCore + CanonicalField,
{
}

impl<F, const D: usize, B> RingSwitchProveBackend<F, D> for B
where
    F: FieldCore + CanonicalField,
    B: RingSwitchComputeBackend<F>
        + for<'a> RingSwitchRelationKernel<RingSwitchRelationView<'a, D>, F, D>
        + for<'a> RingSwitchQuotientKernel<RingSwitchQuotientView<'a, D>, F, D>,
{
}

/// Ring-switch kernels at every runtime-supported fold ring dimension.
pub trait RuntimeRingSwitchProveBackend<F>:
    RingSwitchProveBackend<F, 16>
    + RingSwitchProveBackend<F, 32>
    + RingSwitchProveBackend<F, 64>
    + RingSwitchProveBackend<F, 128>
    + RingSwitchProveBackend<F, 256>
where
    F: FieldCore + CanonicalField,
{
}

impl<F, B> RuntimeRingSwitchProveBackend<F> for B
where
    F: FieldCore + CanonicalField,
    B: RingSwitchProveBackend<F, 16>
        + RingSwitchProveBackend<F, 32>
        + RingSwitchProveBackend<F, 64>
        + RingSwitchProveBackend<F, 128>
        + RingSwitchProveBackend<F, 256>,
{
}

/// Opening kernels for suffix witness and internal root-tensor projection at every
/// supported fold ring dimension.
pub trait SuffixOpeningProveBackend<F>:
    OpeningProveBackendFor<F, RecursiveWitnessFlat, 32>
    + OpeningProveBackendFor<F, RecursiveWitnessFlat, 64>
    + OpeningProveBackendFor<F, RecursiveWitnessFlat, 128>
    + OpeningProveBackendFor<F, RecursiveWitnessFlat, 256>
    + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 32>
    + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 64>
    + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 128>
    + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
{
}

impl<F, B> SuffixOpeningProveBackend<F> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    B: OpeningProveBackendFor<F, RecursiveWitnessFlat, 32>
        + OpeningProveBackendFor<F, RecursiveWitnessFlat, 64>
        + OpeningProveBackendFor<F, RecursiveWitnessFlat, 128>
        + OpeningProveBackendFor<F, RecursiveWitnessFlat, 256>
        + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 32>
        + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 64>
        + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 128>
        + OpeningProveBackendFor<F, RootTensorProjectionPoly<F>, 256>,
{
}

/// Tensor kernels for suffix witness and internal root-tensor projection at every
/// supported fold ring dimension.
pub trait SuffixTensorProveBackend<F, E>:
    TensorBackendFor<F, RecursiveWitnessFlat, E, 32>
    + TensorBackendFor<F, RecursiveWitnessFlat, E, 64>
    + TensorBackendFor<F, RecursiveWitnessFlat, E, 128>
    + TensorBackendFor<F, RecursiveWitnessFlat, E, 256>
    + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 32>
    + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 64>
    + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 128>
    + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
}

impl<F, E, B> SuffixTensorProveBackend<F, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    B: TensorBackendFor<F, RecursiveWitnessFlat, E, 32>
        + TensorBackendFor<F, RecursiveWitnessFlat, E, 64>
        + TensorBackendFor<F, RecursiveWitnessFlat, E, 128>
        + TensorBackendFor<F, RecursiveWitnessFlat, E, 256>
        + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 32>
        + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 64>
        + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 128>
        + TensorBackendFor<F, RootTensorProjectionPoly<F>, E, 256>,
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

/// Root polynomial usable at every runtime-supported ring dimension.
///
/// D-free orchestration bounds on this; operation adapters select a concrete
/// dimension with `dispatch_for_field!` and use the per-D capability
/// inside the arm. Blanket-implemented: the D-free storage types
/// (`DensePoly<F>`, `OneHotPoly<F, I>`, `SparseRingPoly<F>`,
/// `RootTensorProjectionPoly<F>`, `RecursiveWitnessFlat`) satisfy it through
/// their all-D source impls.
pub trait RuntimeRootProvePoly<F>:
    RootPolyMeta<F>
    + RootProvePoly<F, 32>
    + RootProvePoly<F, 64>
    + RootProvePoly<F, 128>
    + RootProvePoly<F, 256>
where
    F: FieldCore,
{
}

impl<F, P> RuntimeRootProvePoly<F> for P
where
    F: FieldCore,
    P: RootPolyMeta<F>
        + RootProvePoly<F, 32>
        + RootProvePoly<F, 64>
        + RootProvePoly<F, 128>
        + RootProvePoly<F, 256>,
{
}

/// Root polynomial committable at every runtime-supported ring dimension.
pub trait RuntimeRootCommitPoly<F>:
    RootPolyMeta<F>
    + RootCommitPoly<F, 32>
    + RootCommitPoly<F, 64>
    + RootCommitPoly<F, 128>
    + RootCommitPoly<F, 256>
where
    F: FieldCore,
{
}

impl<F, P> RuntimeRootCommitPoly<F> for P
where
    F: FieldCore,
    P: RootPolyMeta<F>
        + RootCommitPoly<F, 32>
        + RootCommitPoly<F, 64>
        + RootCommitPoly<F, 128>
        + RootCommitPoly<F, 256>,
{
}

/// Opening-fold backend capability for `P` at every runtime-supported ring
/// dimension (P-generic counterpart of `SuffixOpeningProveBackend`).
pub trait RuntimeOpeningProveBackendFor<F, P>:
    OpeningProveBackendFor<F, P, 32>
    + OpeningProveBackendFor<F, P, 64>
    + OpeningProveBackendFor<F, P, 128>
    + OpeningProveBackendFor<F, P, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, 32>
        + RootOpeningSource<F, 64>
        + RootOpeningSource<F, 128>
        + RootOpeningSource<F, 256>,
{
}

impl<F, P, B> RuntimeOpeningProveBackendFor<F, P> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootOpeningSource<F, 32>
        + RootOpeningSource<F, 64>
        + RootOpeningSource<F, 128>
        + RootOpeningSource<F, 256>,
    B: OpeningProveBackendFor<F, P, 32>
        + OpeningProveBackendFor<F, P, 64>
        + OpeningProveBackendFor<F, P, 128>
        + OpeningProveBackendFor<F, P, 256>,
{
}

/// Tensor-projection capability for `P` at every runtime-supported ring
/// dimension (P-generic counterpart of `SuffixTensorProveBackend`).
pub trait RuntimeTensorBackendFor<F, P, E>:
    TensorBackendFor<F, P, E, 32>
    + TensorBackendFor<F, P, E, 64>
    + TensorBackendFor<F, P, E, 128>
    + TensorBackendFor<F, P, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootTensorSource<F, 32>
        + RootTensorSource<F, 64>
        + RootTensorSource<F, 128>
        + RootTensorSource<F, 256>,
{
}

impl<F, P, E, B> RuntimeTensorBackendFor<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootTensorSource<F, 32>
        + RootTensorSource<F, 64>
        + RootTensorSource<F, 128>
        + RootTensorSource<F, 256>,
    B: TensorBackendFor<F, P, E, 32>
        + TensorBackendFor<F, P, E, 64>
        + TensorBackendFor<F, P, E, 128>
        + TensorBackendFor<F, P, E, 256>,
{
}

/// Commit capability for `P` at every runtime-supported ring dimension.
///
/// Deliberately narrower than [`CommitBackendFor`]: the with-params commit
/// entry points require only digit-row mat-vecs plus the inner-commit kernel
/// over `P`'s borrowed view (the documented downstream contract), not the full
/// [`CommitmentComputeBackend`] surface.
pub trait RuntimeCommitBackendFor<F, P>:
    DigitRowsComputeBackend<F>
    + for<'a> RootCommitKernel<<P as RootCommitSource<F, 32>>::CommitView<'a>, F, 32>
    + for<'a> RootCommitKernel<<P as RootCommitSource<F, 64>>::CommitView<'a>, F, 64>
    + for<'a> RootCommitKernel<<P as RootCommitSource<F, 128>>::CommitView<'a>, F, 128>
    + for<'a> RootCommitKernel<<P as RootCommitSource<F, 256>>::CommitView<'a>, F, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>,
{
}

impl<F, P, B> RuntimeCommitBackendFor<F, P> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>,
    B: DigitRowsComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, 32>>::CommitView<'a>, F, 32>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, 64>>::CommitView<'a>, F, 64>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, 128>>::CommitView<'a>, F, 128>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, 256>>::CommitView<'a>, F, 256>,
{
}

/// Scheme-level commit bundle at every runtime-supported ring dimension
/// (D-free counterpart of the per-D [`RootCommitBackend`]).
pub trait RuntimeRootCommitBackend<F, P, E>:
    RootCommitBackend<F, P, E, 32>
    + RootCommitBackend<F, P, E, 64>
    + RootCommitBackend<F, P, E, 128>
    + RootCommitBackend<F, P, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootCommitPoly<F>,
{
}

impl<F, P, E, B> RuntimeRootCommitBackend<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootCommitPoly<F>,
    B: RootCommitBackend<F, P, E, 32>
        + RootCommitBackend<F, P, E, 64>
        + RootCommitBackend<F, P, E, 128>
        + RootCommitBackend<F, P, E, 256>,
{
}

/// Combined opening + tensor prove capability for `P` at every
/// runtime-supported ring dimension.
pub trait RuntimeProveBackendFor<F, P, E>:
    ProveBackendFor<F, P, E, 32>
    + ProveBackendFor<F, P, E, 64>
    + ProveBackendFor<F, P, E, 128>
    + ProveBackendFor<F, P, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
{
}

impl<F, P, E, B> RuntimeProveBackendFor<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    B: ProveBackendFor<F, P, E, 32>
        + ProveBackendFor<F, P, E, 64>
        + ProveBackendFor<F, P, E, 128>
        + ProveBackendFor<F, P, E, 256>,
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

/// Ring dimensions the recursive suffix may dispatch besides the config ring `D`.
pub const RECURSIVE_SUFFIX_RING_DIMENSIONS: &[usize] = &[32, 64, 128, 256];

/// Full prove-flow capability at a single root ring dimension `RING_D`:
/// opening/tensor prove kernels plus commitment rows.
pub trait ProveFlowBackendFor<F, P, E, const RING_D: usize>:
    RootProveBackend<F, P, E, RING_D> + CommitmentComputeBackend<F>
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
    B: RootProveBackend<F, P, E, RING_D> + CommitmentComputeBackend<F>,
{
}

/// [`ProveFlowBackendFor`] for `P` at every runtime-supported ring dimension.
///
/// Root fold levels take their ring dimension from the schedule
/// (`CommittedGroupParams::role_dims`), so the prove flow must be available at every dimension the
/// dispatcher can select.
pub trait RootProveFlowBackend<F, P, E>:
    ProveFlowBackendFor<F, P, E, 32>
    + ProveFlowBackendFor<F, P, E, 64>
    + ProveFlowBackendFor<F, P, E, 128>
    + ProveFlowBackendFor<F, P, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
{
}

impl<F, P, E, B> RootProveFlowBackend<F, P, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RuntimeRootProvePoly<F>,
    B: ProveFlowBackendFor<F, P, E, 32>
        + ProveFlowBackendFor<F, P, E, 64>
        + ProveFlowBackendFor<F, P, E, 128>
        + ProveFlowBackendFor<F, P, E, 256>,
{
}

/// Recursive witness prove-flow capability over every runtime-supported fold
/// ring dimension.
pub trait RuntimeRecursiveWitnessProveBackend<F, E>:
    ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
    + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 32>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 64>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 128>
    + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 256>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
{
}

impl<F, E, B> RuntimeRecursiveWitnessProveBackend<F, E> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    B: ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
        + ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 32>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 64>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 128>
        + ProveFlowBackendFor<F, RecursiveFoldSource<F>, E, 256>,
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
    C: ComputeBackendSetup<F> + CommitmentComputeBackend<F>,
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
