use super::backend::{ComputeBackendSetup, DigitRowsComputeBackend};
use super::kernels::{
    OpeningBatchKernel, OpeningFoldKernel, RingSwitchRelationKernel, RootCommitKernel,
    SubringCoefficientPackingBatchKernel, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use super::runtime_capabilities::{
    RootProveFlowBackend, RuntimeCoefficientPackingBackendFor, RuntimeOpeningProveBackendFor,
    RuntimeRecursiveWitnessProveBackend, RuntimeRingSwitchProveBackend, RuntimeRootProvePoly,
    RuntimeTensorBackendFor, SuffixOpeningProveBackend, SuffixTensorProveBackend,
};
use crate::backend::{RecursiveFoldSource, RingSwitchRelationView};
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::RandomSampling;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};

/// D-free shape metadata every root polynomial exposes.
///
/// This is the **PCS/batch-facing** capability bound: it names a polynomial's
/// variable count *without* a const ring dimension `D`,
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
    /// Total number of variables (representation-derived, D-independent).
    fn num_vars(&self) -> usize;

    /// One-hot chunk size `K` when this polynomial is a one-hot root
    /// representation.
    ///
    /// `None` means this backend is not a one-hot root representation.
    fn onehot_chunk_size(&self) -> Option<usize> {
        None
    }

    /// Exact squared L2 norm for response-model calibration builds.
    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
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

    /// Exact live ring prefix consumed by coefficient packing.
    ///
    /// Ordinary root sources fill their complete Boolean domain, so this is
    /// identical to [`Self::num_ring_elems`]. Recursive witnesses override it
    /// to exclude their commitment-only zero padding.
    fn num_live_ring_elems(&self) -> usize {
        self.num_ring_elems()
    }

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

/// Capability: this backend can **commit** a single source `P`.
///
/// This is the uniform "source-typed capability" vocabulary: a bound of the form
/// "backend `Self` can commit source `P`", rather than a hard-coded per-type
/// kernel bundle. It folds together the shared outer digit-row surface and the
/// inner-commit kernel over `P`'s borrowed commit view.
///
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

/// Marker bundle for scheme-level prove entry points.
///
/// Algorithms live on [`OpeningFoldKernel`] / [`TensorProjectionKernel`], not here.
pub trait RootProvePoly<F, const D: usize>: RootOpeningSource<F, D>
where
    F: FieldCore,
{
}

impl<F, const D: usize, P> RootProvePoly<F, D> for P
where
    F: FieldCore,
    P: RootOpeningSource<F, D>,
{
}

/// Backend capability bundle for scheme-level prove.
///
/// Use as **`B: RootProveBackend<F, P, E, D>`** on generic prove entry points.
/// `E` is the protocol extension field (`CommitmentConfig::ExtField`).
///
/// ## Why `F: 'static`?
///
/// The bundle closes over higher-ranked bounds on borrowed polynomial views.
///
/// `E` does **not** need `'static`; preset extension fields satisfy it vacuously, but the
/// trait does not require it.
///
/// Root proving evaluates and coefficient-packs the canonical source directly.
pub trait RootProveBackend<F, P, E, const D: usize>: ComputeBackendSetup<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    Self: OpeningProveBackendFor<F, P, D>
        + for<'a> SubringCoefficientPackingBatchKernel<
            <P as RootOpeningSource<F, D>>::OpeningBatchView<'a>,
            F,
            E,
            D,
        >,
{
}

impl<F, P, E, const D: usize, B> RootProveBackend<F, P, E, D> for B
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F>,
    P: RootProvePoly<F, D>,
    B: ComputeBackendSetup<F>
        + OpeningProveBackendFor<F, P, D>
        + for<'a> SubringCoefficientPackingBatchKernel<
            <P as RootOpeningSource<F, D>>::OpeningBatchView<'a>,
            F,
            E,
            D,
        >,
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
        + RuntimeCoefficientPackingBackendFor<F, P, E>
        + RuntimeOpeningProveBackendFor<F, RecursiveFoldSource<F>>
        + SuffixOpeningProveBackend<F>
        + DigitRowsComputeBackend<F>,
    TS: ComputeBackendSetup<F>
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
    fn num_vars(&self) -> usize {
        RootPolyMeta::num_vars(*self)
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        RootPolyMeta::onehot_chunk_size(*self)
    }

    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        RootPolyMeta::exact_integer_coeff_l2_sq(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;

    type F = Fp64<4294967197>;

    #[derive(Clone)]
    struct CanonicalOpeningOnlySource;

    impl RootPolyMeta<F> for CanonicalOpeningOnlySource {
        fn num_vars(&self) -> usize {
            0
        }
    }

    impl<const D: usize> RootPolyShape<F, D> for CanonicalOpeningOnlySource {
        fn num_ring_elems(&self) -> usize {
            1
        }

        fn num_vars(&self) -> usize {
            0
        }
    }

    impl<const D: usize> RootOpeningSource<F, D> for CanonicalOpeningOnlySource {
        type OpeningView<'a>
            = ()
        where
            Self: 'a;
        type OpeningBatchView<'a>
            = ()
        where
            Self: 'a;

        fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
            Ok(())
        }

        fn opening_batch<'a>(
            _polys: &'a [&'a Self],
        ) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
            Ok(())
        }
    }

    #[test]
    fn canonical_root_prove_source_does_not_require_tensor_capability() {
        fn assert_runtime_root_source<P: RuntimeRootProvePoly<F>>() {}
        assert_runtime_root_source::<CanonicalOpeningOnlySource>();
        assert_eq!(RootPolyMeta::num_vars(&CanonicalOpeningOnlySource), 0);
    }
}
