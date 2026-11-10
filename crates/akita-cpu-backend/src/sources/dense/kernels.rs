//! CpuBackend kernels over dense polynomial views.

use super::poly::DensePoly;
use super::views::{DenseBatchView, DenseView};
use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_position_source, FusedPackingWeights,
};
use crate::opaque::DecomposeFoldWitness;
use crate::opaque::{aggregate_decompose_fold_witnesses, CpuFoldResponses};
use crate::opaque::{
    CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldPlan, RootOpeningSource,
    RootPolyMeta, RootPolyShape, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::opaque::{OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput};
use crate::sources::block_digits::{RingValueSource, RingValues};
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

impl<F, E, const D: usize> OpeningFoldKernel<DenseView<'_, F, D>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
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
            .num_ring_elems_at(D)
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
            )?,
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
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        if plan.num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source
            .poly
            .num_ring_elems_at(D)
            .div_ceil(plan.num_positions_per_block);
        if plan.challenges.len() != num_live_blocks {
            return Err(AkitaError::InvalidSize {
                expected: num_live_blocks,
                actual: plan.challenges.len(),
            });
        }
        source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        )
    }
}

impl<F, E, const D: usize> OpeningBatchKernel<DenseBatchView<'_, F, D>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<Vec<OpeningFoldOutput<F, D>>, AkitaError> {
        source
            .polys
            .iter()
            .map(|poly| {
                self.evaluate_and_fold(
                    prepared,
                    <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(*poly)?,
                    plan,
                )
            })
            .collect()
    }

    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        let (num_positions_per_block, num_digits, log_basis) = plan.scalar_params();
        let challenges_per_poly = plan.validate_uniform_batch(source.polys.iter().map(|poly| {
            RootPolyShape::<F, D>::num_live_ring_elems(*poly).div_ceil(num_positions_per_block)
        }))?;
        match plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => Ok(CpuFoldResponses::sparse(
                aggregate_decompose_fold_witnesses::<D>(
                    source
                        .polys
                        .iter()
                        .zip(challenges.chunks_exact(challenges_per_poly))
                        .map(|(poly, poly_challenges)| {
                            poly.decompose_fold::<D>(
                                poly_challenges,
                                num_positions_per_block,
                                num_digits,
                                log_basis,
                            )
                        }),
                )?,
            )),
            DecomposeFoldBatchPlan::SparseChunked {
                challenges,
                chunk_ranges,
                ..
            } => {
                let mut by_chunk = (0..chunk_ranges.len())
                    .map(|_| Vec::with_capacity(source.polys.len()))
                    .collect::<Vec<_>>();
                for (poly, poly_challenges) in source
                    .polys
                    .iter()
                    .zip(challenges.as_slice().chunks_exact(challenges_per_poly))
                {
                    for (chunk, witness) in
                        by_chunk.iter_mut().zip(poly.decompose_fold_chunked::<D>(
                            poly_challenges,
                            chunk_ranges,
                            num_positions_per_block,
                            num_digits,
                            log_basis,
                        )?)
                    {
                        chunk.push(Ok(witness));
                    }
                }
                CpuFoldResponses::chunked::<D>(
                    by_chunk
                        .into_iter()
                        .map(aggregate_decompose_fold_witnesses::<D>)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        }
    }
}

pub(crate) fn dense_coefficient_packing_partials<F, E, const D: usize>(
    source: DenseBatchView<'_, F, D>,
    fused_weights: &FusedPackingWeights<'_, E>,
) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let point = fused_weights.tables().point();
    source
        .polys
        .iter()
        .map(|poly| {
            let rings = poly.nonzero_ring_coeffs::<D>()?;
            // Stored zeros may be omitted, but the authenticated domain is full.
            let logical_rings = poly.num_ring_elems_at(D);
            if logical_rings != point.num_live_positions() {
                return Err(AkitaError::InvalidSize {
                    expected: point.num_live_positions(),
                    actual: logical_rings,
                });
            }
            let live_coeff_len = poly.nonzero_coeff_len();
            let coordinates = seam_packing_partials(
                &rings,
                live_coeff_len,
                RootPolyMeta::<F>::num_vars(*poly),
                fused_weights,
            )?;
            SubringCoefficientPackingPartials::new(
                point.geometry(),
                point.num_live_blocks(),
                coordinates,
            )
        })
        .collect()
}

pub(crate) fn seam_packing_partials<F, E, S, const D: usize>(
    source: &S,
    live_coeff_len: usize,
    source_num_vars: usize,
    fused_weights: &FusedPackingWeights<'_, E>,
) -> Result<Vec<F>, AkitaError>
where
    F: Field,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
    S: RingValueSource<F, D>,
{
    let full_len = checked::product([source.num_rings(), D]).ok_or_else(|| {
        AkitaError::InvalidInput("coefficient-packing source extent overflow".into())
    })?;
    if live_coeff_len == full_len {
        coefficient_packing_partials_from_position_source::<F, E, _, D>(
            fused_weights,
            source_num_vars,
            live_coeff_len.div_ceil(D),
            |position| source.values(position),
            |_, coefficient, values| values.coeff(coefficient),
        )
    } else {
        coefficient_packing_partials_from_position_source::<F, E, _, D>(
            fused_weights,
            source_num_vars,
            live_coeff_len.div_ceil(D),
            |position| source.values(position),
            |position, coefficient, values| {
                let flat_index = position * D + coefficient;
                if flat_index < live_coeff_len {
                    values.coeff(coefficient)
                } else {
                    F::zero()
                }
            },
        )
    }
}

impl<F, E, const D: usize> SubringCoefficientPackingBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_weights = FusedPackingWeights::new(plan.tables)?;
        dense_coefficient_packing_partials(source, &fused_weights)
    }
}
