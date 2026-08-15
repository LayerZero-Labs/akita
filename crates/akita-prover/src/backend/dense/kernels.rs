//! CpuBackend kernels over dense polynomial views.

use super::poly::DensePoly;
use super::views::{DenseBatchView, DenseView};
use crate::backend::coefficient_packing::partials_from_indexed_source;
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, RootPolyMeta, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::FpExtEncoding;

impl<F, const D: usize> RootCommitKernel<DenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<DenseView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        cfg_into_iter!(sources)
            .map(|source| {
                source
                    .poly
                    .commit_rows::<D>(
                        self,
                        prepared,
                        plan.n_a,
                        plan.num_positions_per_block,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                    )
                    .map(CommitInnerWitness::from_rows)
            })
            .collect()
    }
}

impl<F, const D: usize> OpeningFoldKernel<DenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let num_positions_per_block = plan.num_positions_per_block();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source
            .poly
            .ring_coeffs::<D>()?
            .len()
            .div_ceil(num_positions_per_block);
        plan.validate::<D>(num_live_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.poly.evaluate_and_fold::<D>(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            ),
            OpeningFoldPlan::Subfield {
                multipliers,
                num_positions_per_block,
            } => source
                .poly
                .evaluate_and_fold_subfield(multipliers, num_positions_per_block)?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        Ok(source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, const D: usize> OpeningBatchKernel<DenseBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: DenseBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

impl<F, E, const D: usize> TensorProjectionKernel<DenseView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source
            .poly
            .tensor_extension_column_partials::<E, D>(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(TensorPackedWitness::Dense(
            source.poly.tensor_packed_extension_evals::<E, D>()?,
        ))
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        source.poly.tensor_packed_extension_root_poly::<E, D>()
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        DensePoly::tensor_extension_column_partials_batch::<E, D>(source.polys, logical_point)
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        DensePoly::tensor_packed_extension_sparse_linear_combination(source.polys, coeffs)
    }
}

impl<F, E, const D: usize> SubringCoefficientPackingBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + akita_types::FpExtEncoding<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        source
            .polys
            .iter()
            .map(|poly| {
                let rings = poly.ring_coeffs::<D>()?;
                let source_len =
                    plan.point
                        .num_live_positions()
                        .checked_mul(D)
                        .ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "coefficient-packing dense source length overflow".into(),
                            )
                        })?;
                // Dense roots authenticate the complete Boolean hypercube, so
                // every stored ring is live. Exact-prefix storage is reserved
                // for recursive witness views.
                if rings.len() != plan.point.num_live_positions() {
                    return Err(AkitaError::InvalidSize {
                        expected: plan.point.num_live_positions(),
                        actual: rings.len(),
                    });
                }
                let coordinates = partials_from_indexed_source::<F, E, D>(
                    plan,
                    RootPolyMeta::<F>::num_vars(*poly),
                    source_len,
                    |index| {
                        let ring_index = index / D;
                        let coefficient_index = index % D;
                        rings
                            .get(ring_index)
                            .and_then(|ring| ring.coefficients().get(coefficient_index))
                            .copied()
                            .ok_or(AkitaError::InvalidProof)
                    },
                )?;
                SubringCoefficientPackingPartials::new(
                    plan.point.geometry(),
                    plan.point.num_live_blocks(),
                    coordinates,
                )
            })
            .collect()
    }
}
