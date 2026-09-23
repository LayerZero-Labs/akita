//! CpuBackend kernels over dense polynomial views.

use super::views::{DenseBatchView, DenseView};
use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_position_source, FusedPackingWeights,
};
use crate::opaque::DecomposeFoldWitness;
use crate::opaque::{aggregate_decompose_fold_witnesses, CpuFoldResponses};
use crate::opaque::{
    CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldPlan, RootPolyMeta,
    RootPolyShape, SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use crate::opaque::{OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

impl<F, Cfg, const D: usize> OpeningFoldKernel<DenseView<'_, F, D>, F, D> for CpuBackend<Cfg>
where
    Cfg: akita_config::CommitmentConfig,
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
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        if plan.num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source
            .poly
            .ring_coeffs::<D>()?
            .len()
            .div_ceil(plan.num_positions_per_block);
        if plan.challenges.len() != num_live_blocks {
            return Err(AkitaError::InvalidSize {
                expected: num_live_blocks,
                actual: plan.challenges.len(),
            });
        }
        Ok(source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, Cfg, const D: usize> OpeningBatchKernel<DenseBatchView<'_, F, D>, F, D> for CpuBackend<Cfg>
where
    Cfg: akita_config::CommitmentConfig,
    F: Field + CanonicalEncoding,
{
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
                            Ok(poly.decompose_fold::<D>(
                                poly_challenges,
                                num_positions_per_block,
                                num_digits,
                                log_basis,
                            ))
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
                        ))
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
    let point = fused_weights.point();
    source
        .polys
        .iter()
        .map(|poly| {
            let rings = poly.ring_coeffs::<D>()?;
            // Dense roots authenticate the complete Boolean hypercube, so
            // every stored ring is live. Exact-prefix storage is reserved
            // for recursive witness views.
            if rings.len() != point.num_live_positions() {
                return Err(AkitaError::InvalidSize {
                    expected: point.num_live_positions(),
                    actual: rings.len(),
                });
            }
            let coordinates = coefficient_packing_partials_from_position_source::<F, E, _, D>(
                fused_weights,
                RootPolyMeta::<F>::num_vars(*poly),
                |position| {
                    rings
                        .get(position)
                        .map(|ring| ring.coefficients())
                        .ok_or(AkitaError::InvalidProof)
                },
                |_, coefficient, source| source[coefficient],
            )?;
            SubringCoefficientPackingPartials::new(
                point.geometry(),
                point.num_live_blocks(),
                coordinates,
            )
        })
        .collect()
}

impl<F, E, Cfg, const D: usize>
    SubringCoefficientPackingBatchKernel<DenseBatchView<'_, F, D>, F, E, D> for CpuBackend<Cfg>
where
    Cfg: akita_config::CommitmentConfig,
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_weights = FusedPackingWeights::new(plan.point)?;
        dense_coefficient_packing_partials(source, &fused_weights)
    }
}
