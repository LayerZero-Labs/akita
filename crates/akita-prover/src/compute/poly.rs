use super::backend::CommitmentComputeBackend;
use super::kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RootCommitKernel, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::RootTensorProjectionPoly;
use akita_field::unreduced::{HasWide, ReduceTo};
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

/// Capability: materialize a direct root witness for zero-fold openings.
///
/// This is an explicit opt-in, not a hidden default on every root polynomial:
/// only proving paths that may select a root-direct schedule require it.
pub trait DirectRootWitnessSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: FieldCore,
{
    /// Materialize a direct root witness payload.
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError>;
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

/// Backend capability bundle for scheme-level commit with optional tensor transform.
///
/// Use as **`B: RootCommitBackend<F, P, E, D>`** on generic `fn commit<P, B>(backend: &B, …)`.
/// Do **not** write `CpuBackend: RootCommitBackend<F, P, E, D>` while `P` is still a type
/// parameter; that fails for the same reason as a bare HRTB on a fixed backend.
///
/// `F: 'static` is required for the same GAT + `for<'a>` view-kernel reason documented on
/// [`RootProveBackend`]. `E` (tensor extension field) is not bounded `'static` here.
pub trait RootCommitBackend<F, P, E, const D: usize>: CommitmentComputeBackend<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootCommitPoly<F, D>,
    Self: for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>
        + for<'a> TensorProjectionKernel<<P as RootTensorSource<F, D>>::TensorView<'a>, F, E, D>
        + for<'a> RootCommitKernel<
            <RootTensorProjectionPoly<F, D> as RootCommitSource<F, D>>::CommitView<'a>,
            F,
            D,
        >,
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
/// Use as **`B: RootProveBackend<F, P, ClaimE, ChallengeE, D>`** on generic prove
/// entry points. `ClaimE` covers extension-opening prepare batch partials at the
/// claim field; `ChallengeE` covers post-transform tensor projection and opening
/// fold paths at the challenge field.
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
/// `ClaimE` and `ChallengeE` do **not** need `'static`; preset extension fields satisfy
/// it vacuously, but the trait does not require it.
pub trait RootProveBackend<F, P, ClaimE, ChallengeE, const D: usize>: CommitmentComputeBackend<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    ClaimE: ExtField<F>,
    ChallengeE: ExtField<F>,
    P: RootProvePoly<F, D>,
    Self: for<'a> OpeningFoldKernel<<P as RootOpeningSource<F, D>>::OpeningView<'a>, F, D>
        + for<'a> OpeningBatchKernel<<P as RootOpeningSource<F, D>>::OpeningBatchView<'a>, F, D>
        + for<'a> TensorProjectionKernel<
            <P as RootTensorSource<F, D>>::TensorView<'a>,
            F,
            ChallengeE,
            D,
        > + for<'a> TensorProjectionBatchKernel<
            <P as RootTensorSource<F, D>>::TensorBatchView<'a>,
            F,
            ClaimE,
            D,
        > + for<'a> TensorProjectionBatchKernel<
            <P as RootTensorSource<F, D>>::TensorBatchView<'a>,
            F,
            ChallengeE,
            D,
        > + for<'a> OpeningFoldKernel<
            <RootTensorProjectionPoly<F, D> as RootOpeningSource<F, D>>::OpeningView<'a>,
            F,
            D,
        > + for<'a> OpeningBatchKernel<
            <RootTensorProjectionPoly<F, D> as RootOpeningSource<F, D>>::OpeningBatchView<'a>,
            F,
            D,
        > + for<'a> TensorProjectionKernel<
            <RootTensorProjectionPoly<F, D> as RootTensorSource<F, D>>::TensorView<'a>,
            F,
            ChallengeE,
            D,
        > + for<'a> TensorProjectionBatchKernel<
            <RootTensorProjectionPoly<F, D> as RootTensorSource<F, D>>::TensorBatchView<'a>,
            F,
            ChallengeE,
            D,
        >,
{
}

/// Umbrella marker for a fully capable Akita root polynomial.
///
/// Acceptable only as a convenience bundle on top-level APIs whose behavior can
/// reach every root capability through config-selected schedules. Lower-level
/// helpers should bound the smallest capability they actually use.
pub trait AkitaRootPoly<F, const D: usize>:
    RootPolyShape<F, D>
    + RootCommitSource<F, D>
    + RootOpeningSource<F, D>
    + RootTensorSource<F, D>
    + DirectRootWitnessSource<F, D>
where
    F: FieldCore,
{
}

impl<F, const D: usize, P> AkitaRootPoly<F, D> for P
where
    F: FieldCore,
    P: RootPolyShape<F, D>
        + RootCommitSource<F, D>
        + RootOpeningSource<F, D>
        + RootTensorSource<F, D>
        + DirectRootWitnessSource<F, D>,
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
}

impl<F, const D: usize, P> RootCommitSource<F, D> for &P
where
    F: FieldCore,
    P: RootCommitSource<F, D>,
{
    type CommitView<'a>
        = P::CommitView<'a>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        (*self).commit_view()
    }
}

impl<F, const D: usize, P> RootTensorSource<F, D> for &P
where
    F: FieldCore,
    P: RootTensorSource<F, D>,
{
    type TensorView<'a>
        = P::TensorView<'a>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = P::TensorBatchView<'a>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        (*self).tensor_view()
    }

    fn tensor_batch<'a>(_polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "tensor_batch through a polynomial reference is not supported; pass by value"
                .to_string(),
        ))
    }
}
