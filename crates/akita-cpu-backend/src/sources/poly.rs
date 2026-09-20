#![allow(private_bounds)]
use crate::opaque::OpeningFoldKernel;

use crate::opaque::ComputeBackendSetup;
use crate::opaque::RingSwitchRelationView;
use crate::opaque::{FoldResponseKernel, RingSwitchRelationKernel};
use akita_error::AkitaError;
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Field, Ring};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// D-free shape metadata every root polynomial exposes.
///
/// This is the **PCS/batch-facing** capability bound: it names a polynomial's
/// variable count *without* a const ring dimension `D`,
/// so D-free entry points (e.g. [`akita_prover::ProverOpeningData`]) can require just
/// `RootPolyMeta` while the const-D kernel-entry traits ([`RootPolyShape`] and
/// the commit/opening/tensor/direct-witness family) carry `D`.
///
/// `num_vars` is the polynomial's own (schedule/representation-derived) variable
/// count — **not** `log2(num_ring_elems() * D)`. Every input root polynomial
/// stores it directly, so the count is independent of the ring dimension chosen
/// to commit it.
pub trait RootPolyMeta<F>: Send + Sync
where
    F: Field,
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
pub trait RootPolyShape<F, const D: usize>: Send + Sync
where
    F: Field,
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

/// Largest centered magnitudes over a flat field slice, as
/// `(negative_abs_max, positive_max)`.
///
/// Shared by dense [`crate::commitment::CommitmentSource`] implementations so the
/// centering convention is written once: a canonical
/// residue at or below `centering_threshold` is the positive side, anything above
/// it is negative with magnitude `modulus - canonical`. That is the same split
/// `akita_algebra::ring::cyclotomic::center_for_decomposition` applies. We track
/// the magnitude directly rather than calling that helper because it is
/// `pub(crate)` to `akita-algebra` and returns an `i128` centered value, which
/// cannot hold the largest negative magnitude a full-width residue reaches
/// (`modulus - canonical` can exceed `i128::MAX`); the range check needs the
/// `u128` magnitude on each side.
///
/// This runs over the whole committed span before any commitment arithmetic, so
/// at `nv = 26` it is ~2^26 canonical reductions. The rest of the commit path is
/// rayon-parallel; keeping this serial would make the guard a visible sequential
/// section in an otherwise parallel phase, so it folds in parallel under the
/// `parallel` feature. Both reaches are commutative-associative maxima, so the
/// reduction order does not affect the result.
pub(crate) fn centered_reach_of_field_coeffs<F: Field + CanonicalEncoding>(
    coeffs: &[F],
    modulus: u128,
    centering_threshold: u128,
) -> (u128, u128) {
    let fold = |(negative_abs_max, positive_max): (u128, u128), coeff: &F| {
        let canonical = coeff
            .to_u128_checked()
            .expect("Akita field element must fit in u128");
        if canonical <= centering_threshold {
            (negative_abs_max, positive_max.max(canonical))
        } else {
            (negative_abs_max.max(modulus - canonical), positive_max)
        }
    };
    let combine =
        |left: (u128, u128), right: (u128, u128)| (left.0.max(right.0), left.1.max(right.1));

    #[cfg(feature = "parallel")]
    {
        coeffs
            .par_iter()
            .fold(|| (0u128, 0u128), fold)
            .reduce(|| (0u128, 0u128), combine)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = combine;
        coeffs.iter().fold((0u128, 0u128), fold)
    }
}

/// Capability: expose borrowed opening views for the opening fold kernels.
pub trait RootOpeningSource<F, const D: usize>: RootPolyShape<F, D>
where
    F: Field,
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

/// Scalar root-polynomial evaluation without exposing opening-fold internals.
pub trait RootPolynomialEvaluator<F, const D: usize>: RootOpeningSource<F, D>
where
    F: Field + CanonicalEncoding,
{
    /// Evaluate this polynomial at `point` under the supplied root layout.
    fn evaluate_root_polynomial(
        &self,
        point: &[F],
        num_positions_per_block: usize,
        num_live_blocks: usize,
        basis: akita_types::BasisMode,
    ) -> Result<F, AkitaError>;
}

fn evaluate_root_polynomial_with_kernel<F, P, const D: usize>(
    poly: &P,
    point: &[F],
    num_positions_per_block: usize,
    num_live_blocks: usize,
    basis: akita_types::BasisMode,
    evaluate: impl FnOnce(&[F], &[F], usize) -> Result<akita_algebra::CyclotomicRing<F, D>, AkitaError>,
) -> Result<F, AkitaError>
where
    F: Field + CanonicalEncoding,
    P: RootPolyShape<F, D>,
{
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = <P as RootPolyShape<F, D>>::num_vars(poly);
    if point.len() > target_num_vars || target_num_vars < alpha_bits {
        return Err(AkitaError::InvalidInput(
            "opening point exceeds source arity".into(),
        ));
    }
    let mut point = point.to_vec();
    point.resize(target_num_vars, F::zero());
    let ring_point = akita_types::ring_opening_point_from_field(
        &point[alpha_bits..],
        num_positions_per_block,
        num_live_blocks,
        basis,
    )?;
    let value = evaluate(
        &ring_point.live_block_weights,
        &ring_point.position_weights,
        num_positions_per_block,
    )?;
    let inner =
        akita_types::reduce_inner_opening_to_ring_element::<F, D>(&point[..alpha_bits], basis)?;
    Ok((value * inner.sigma_m1()).coefficients()[0])
}
impl<F: Field + CanonicalEncoding, const D: usize> RootPolynomialEvaluator<F, D>
    for crate::opaque::DensePoly<F>
{
    fn evaluate_root_polynomial(
        &self,
        point: &[F],
        positions: usize,
        blocks: usize,
        basis: akita_types::BasisMode,
    ) -> Result<F, AkitaError> {
        evaluate_root_polynomial_with_kernel::<F, _, D>(
            self,
            point,
            positions,
            blocks,
            basis,
            |block_weights, position_weights, positions| {
                self.ring_coeffs::<D>()?;
                Ok(self
                    .evaluate_and_fold::<D>(block_weights, position_weights, positions)
                    .0)
            },
        )
    }
}
impl<F: Field + CanonicalEncoding + Unreduced, I: crate::opaque::OneHotIndex, const D: usize>
    RootPolynomialEvaluator<F, D> for crate::opaque::OneHotPoly<F, I>
{
    fn evaluate_root_polynomial(
        &self,
        point: &[F],
        positions: usize,
        blocks: usize,
        basis: akita_types::BasisMode,
    ) -> Result<F, AkitaError> {
        evaluate_root_polynomial_with_kernel::<F, _, D>(
            self,
            point,
            positions,
            blocks,
            basis,
            |block_weights, position_weights, positions| {
                Ok(self
                    .evaluate_and_fold::<D>(block_weights, position_weights, positions)
                    .0)
            },
        )
    }
}

/// Evaluate one root polynomial through the canonical CPU implementation.
pub fn evaluate_root_polynomial<F, P, const D: usize>(
    poly: &P,
    point: &[F],
    num_positions_per_block: usize,
    num_live_blocks: usize,
    basis: akita_types::BasisMode,
) -> Result<F, AkitaError>
where
    F: Field + CanonicalEncoding,
    P: RootPolynomialEvaluator<F, D>,
{
    poly.evaluate_root_polynomial(point, num_positions_per_block, num_live_blocks, basis)
}

/// Ring-switch cluster capability for the source-typed relation kernel.
pub(crate) trait RingSwitchProveBackend<F, const D: usize>:
    for<'a> RingSwitchRelationKernel<RingSwitchRelationView<'a, D>, F, D>
where
    F: Field + CanonicalEncoding,
{
}

impl<F, const D: usize, B> RingSwitchProveBackend<F, D> for B
where
    F: Field + CanonicalEncoding,
    B: for<'a> RingSwitchRelationKernel<RingSwitchRelationView<'a, D>, F, D>,
{
}

/// Capability: this backend can run **opening fold** kernels over a single
/// source `P` (evaluate/fold and opaque accepted responses).
pub(crate) trait OpeningProveBackendFor<F, P, const D: usize>:
    ComputeBackendSetup<F>
    + for<'a> OpeningFoldKernel<<P as RootOpeningSource<F, D>>::OpeningView<'a>, F, D>
    + for<'a> FoldResponseKernel<<P as RootOpeningSource<F, D>>::OpeningBatchView<'a>, F, D>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F>,
    P: RootOpeningSource<F, D>,
{
}

impl<F, P, const D: usize, B> OpeningProveBackendFor<F, P, D> for B
where
    F: Field + CanonicalEncoding + Ring + Unreduced + 'static,
    <F as Unreduced>::Wide: From<F>,
    P: RootOpeningSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> OpeningFoldKernel<<P as RootOpeningSource<F, D>>::OpeningView<'a>, F, D>
        + for<'a> FoldResponseKernel<<P as RootOpeningSource<F, D>>::OpeningBatchView<'a>, F, D>,
{
}

impl<F, const D: usize, P> RootPolyShape<F, D> for &P
where
    F: Field,
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
    F: Field,
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
    use jolt_field::Fp64;

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
        fn assert_runtime_root_source<P: crate::opaque::capabilities::RuntimeRootProvePoly<F>>() {}
        assert_runtime_root_source::<CanonicalOpeningOnlySource>();
        assert_eq!(RootPolyMeta::num_vars(&CanonicalOpeningOnlySource), 0);
    }
}
