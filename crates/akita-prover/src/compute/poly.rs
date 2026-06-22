use akita_field::{AkitaError, FieldCore};
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
