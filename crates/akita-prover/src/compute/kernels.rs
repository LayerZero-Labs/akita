use crate::compute::backend::ComputeBackendSetup;
use crate::compute::operation_plans::{
    DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldOutput, OpeningFoldPlan,
    RingSwitchRelationPlan, SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::compute::plans::RingSwitchRelationRows;
use crate::DecomposeFoldWitness;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

/// Checked aggregation for batch kernels that fold their sources individually.
pub fn aggregate_decompose_fold_witnesses<F: Field, const D: usize>(
    witnesses: impl IntoIterator<Item = Result<DecomposeFoldWitness<F>, AkitaError>>,
) -> Result<DecomposeFoldWitness<F>, AkitaError> {
    let mut witnesses = witnesses.into_iter();
    let Some(first) = witnesses.next() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    let first = first?;
    first.ensure_ring_dim::<D>()?;
    let row_count = first.row_count();
    let (z_folded_rings, mut centered_coeffs) = first.into_owned_flat_parts();
    let mut z_folded_coeffs = z_folded_rings.into_coeffs();

    for witness in witnesses {
        let witness = witness?;
        witness.ensure_ring_dim::<D>()?;
        if witness.row_count() != row_count {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_folded_coeffs
            .iter_mut()
            .zip(witness.z_folded_rings.coeffs())
        {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs_flat())
        {
            *dst = dst.checked_add(*src).ok_or_else(|| {
                AkitaError::InvalidInput(
                    "batched decompose_fold centered coefficient overflow".to_string(),
                )
            })?;
        }
    }

    DecomposeFoldWitness::from_owned_flat_parts::<D>(
        akita_types::RingVec::from_coeffs_with_ring_dim(z_folded_coeffs, D)?,
        centered_coeffs,
    )
}

/// Fused ring-switch relation-rows kernel over a borrowed relation view `S`.
pub trait RingSwitchRelationKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Fused D rows in both domains, B cyclic rows, and A-side quotient rows.
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: S,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>;
}

/// Opening fold / decompose-fold kernel over a borrowed opening view `S`.
///
/// `prepared` is optional because some opening folds do not need setup-owned
/// state; setup-dependent work stays explicitly tied to the backend context.
pub trait OpeningFoldKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Fused fold + evaluation in one pass over the source.
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError>;

    /// Decompose + challenge-fold step.
    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError>;
}

/// Batched decompose-fold kernel over a borrowed opening-batch view `S`.
///
/// Implementations return one aggregate witness per requested block window. A
/// backend without a fused path folds its source polynomials individually and
/// aggregates them within each window before returning.
pub trait OpeningBatchKernel<S, F, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    /// Batched decompose-fold at one opening point, in chunk order.
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Vec<DecomposeFoldWitness<F>>, AkitaError>;
}

/// Tensor projection kernel over a borrowed tensor view `S` for opening at an
/// extension-field point of type `E`.
pub trait TensorProjectionKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    /// Tensor-column partials at one logical point.
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>;

    /// Tensor-packed EOR witness.
    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
    ) -> Result<Vec<E>, AkitaError>;
}

/// Batched tensor projection kernel over a borrowed tensor-batch view `S`.
pub trait TensorProjectionBatchKernel<S, F, E, const D: usize>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    /// Tensor-column partials for a same-point batch.
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>;
}

/// Coefficient-packing projection over a borrowed same-shape source batch.
pub trait SubringCoefficientPackingBatchKernel<S, F, E, const D: usize>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    /// Return one canonical base-field partial buffer per claim.
    ///
    /// Every returned buffer uses
    /// `[block][extension coordinate][subring coefficient]` order.
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: S,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError>;
}
