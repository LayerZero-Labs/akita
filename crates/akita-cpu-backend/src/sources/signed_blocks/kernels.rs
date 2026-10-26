use super::{Decoded, I16Slice, I8Slice, SignedBlockDigits, SignedStorage};
use crate::arithmetic::coefficient_packing::FusedPackingWeights;
use crate::commitment::{SignedBlocks, SignedValues};
use crate::opaque::{
    aggregate_decompose_fold_witnesses, CpuBackend, CpuFoldResponses, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, DecomposeFoldWitness, OpeningBatchKernel, OpeningFoldKernel,
    OpeningFoldOutput, OpeningFoldPlan, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::sources::block_digits::evaluate::{evaluate_and_fold_base, evaluate_and_fold_ring};
use crate::sources::block_digits::{DigitShape, RingValues};
use crate::sources::dense::seam_packing_partials;
use crate::sources::poly_helpers::{element_partitioned_decompose_fold, SignedDigitBasis};
use akita_challenges::SparseChallenge;
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};
use std::ops::Range;

macro_rules! with_source {
    ($blocks:expr, $source:ident => $body:expr) => {{
        let blocks = $blocks;
        match blocks.values {
            SignedValues::I8(values) => {
                let $source = SignedBlockDigits::<F, _, D>::new(
                    I8Slice(values),
                    blocks.num_vars(),
                    blocks.len(),
                    None,
                    None,
                )?;
                $body
            }
            SignedValues::I16(values) => {
                let $source = SignedBlockDigits::<F, _, D>::new(
                    I16Slice(values),
                    blocks.num_vars(),
                    blocks.len(),
                    None,
                    None,
                )?;
                $body
            }
            SignedValues::DecodedI8(decoder) => {
                let $source = SignedBlockDigits::<F, _, D>::new(
                    Decoded(decoder),
                    blocks.num_vars(),
                    blocks.len(),
                    None,
                    None,
                )?;
                $body
            }
        }
    }};
}

fn num_rings<const D: usize>(blocks: SignedBlocks<'_>) -> Result<usize, AkitaError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "signed block ring dimension must be a power of two".into(),
        ));
    }
    let domain = checked::pow2(blocks.num_vars())
        .ok_or_else(|| AkitaError::InvalidInput("signed block extent overflow".into()))?;
    Ok(domain.div_ceil(D))
}

fn fold_chunks<F, St, const D: usize>(
    source: &SignedBlockDigits<'_, F, St, D>,
    challenges: &[SparseChallenge],
    ranges: &[Range<usize>],
    positions: usize,
    shape: DigitShape,
) -> Result<Vec<DecomposeFoldWitness>, AkitaError>
where
    F: Field + CanonicalEncoding,
    St: SignedStorage,
    [St::Value; D]: RingValues<F, D>,
{
    SignedDigitBasis::new(shape.log_basis)?;
    if shape.num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "signed block fold digit count must be nonzero".into(),
        ));
    }
    Ok(
        element_partitioned_decompose_fold::<_, D>(source, challenges, ranges, positions, shape)?
            .into_iter()
            .map(DecomposeFoldWitness::from_centered_rows)
            .collect(),
    )
}

impl<F, E, const D: usize> OpeningFoldKernel<SignedBlocks<'_>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        blocks: SignedBlocks<'_>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let positions = plan.num_positions_per_block();
        if positions == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".into(),
            ));
        }
        let live_blocks = num_rings::<D>(blocks)?.div_ceil(positions);
        plan.validate::<D>(live_blocks)?;
        with_source!(blocks, source => {
            let (eval, folded) = match plan {
                OpeningFoldPlan::Base { live_block_weights, position_weights, num_positions_per_block } => evaluate_and_fold_base::<F, _, D>(&source, live_block_weights, position_weights, num_positions_per_block, live_blocks)?,
                OpeningFoldPlan::Subfield { multipliers, num_positions_per_block } => {
                    let block_weights = multipliers.materialize_fold_rings::<D>()?;
                    let position_weights = multipliers.materialize_position_rings::<D>()?;
                    evaluate_and_fold_ring::<F, _, D>(&source, &block_weights, &position_weights, num_positions_per_block, live_blocks)?
                },
            };
            Ok(OpeningFoldOutput { eval, folded })
        })
    }
    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        blocks: SignedBlocks<'_>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        if plan.num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".into(),
            ));
        }
        let live_blocks = num_rings::<D>(blocks)?.div_ceil(plan.num_positions_per_block);
        if plan.challenges.len() != live_blocks {
            return Err(AkitaError::InvalidSize {
                expected: live_blocks,
                actual: plan.challenges.len(),
            });
        }
        with_source!(blocks, source => {
            let rows = fold_chunks::<F, _, D>(&source, plan.challenges, std::slice::from_ref(&(0..live_blocks)), plan.num_positions_per_block, DigitShape { num_digits: plan.num_digits, log_basis: plan.log_basis })?;
            rows.into_iter().next().ok_or_else(|| AkitaError::Internal("signed block fold omitted its output".into()))
        })
    }
}

impl<F, E, const D: usize> OpeningBatchKernel<Vec<SignedBlocks<'_>>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: Vec<SignedBlocks<'_>>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<Vec<OpeningFoldOutput<F, D>>, AkitaError> {
        source
            .into_iter()
            .map(|blocks| {
                <Self as OpeningFoldKernel<SignedBlocks<'_>, F, D>>::evaluate_and_fold(
                    self, prepared, blocks, plan,
                )
            })
            .collect()
    }
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: Vec<SignedBlocks<'_>>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        let (positions, num_digits, log_basis) = plan.scalar_params();
        let ring_counts = source
            .iter()
            .map(|&blocks| num_rings::<D>(blocks))
            .collect::<Result<Vec<_>, _>>()?;
        let per_poly = plan.validate_uniform_batch(
            ring_counts
                .into_iter()
                .map(|rings| rings.div_ceil(positions)),
        )?;
        let shape = DigitShape {
            num_digits,
            log_basis,
        };
        match plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => Ok(CpuFoldResponses::sparse(aggregate_decompose_fold_witnesses::<D>(source.into_iter().zip(challenges.chunks_exact(per_poly)).map(|(blocks, challenges)| {
                with_source!(blocks, storage => {
                    let rows = fold_chunks::<F, _, D>(&storage, challenges, std::slice::from_ref(&(0..per_poly)), positions, shape)?;
                    rows.into_iter().next().ok_or_else(|| AkitaError::Internal("signed block fold omitted its output".into()))
                })
            }))?)),
            DecomposeFoldBatchPlan::SparseChunked { challenges, chunk_ranges, .. } => {
                let mut by_chunk = (0..chunk_ranges.len()).map(|_| Vec::with_capacity(source.len())).collect::<Vec<_>>();
                for (blocks, challenges) in source.into_iter().zip(challenges.as_slice().chunks_exact(per_poly)) {
                    let chunks = with_source!(blocks, storage => fold_chunks::<F, _, D>(&storage, challenges, chunk_ranges, positions, shape))?;
                    for (chunk, witness) in by_chunk.iter_mut().zip(chunks) { chunk.push(Ok(witness)); }
                }
                CpuFoldResponses::chunked::<D>(by_chunk.into_iter().map(aggregate_decompose_fold_witnesses::<D>).collect::<Result<Vec<_>, _>>()?)
            }
        }
    }
}

impl<F, E, const D: usize> SubringCoefficientPackingBatchKernel<Vec<SignedBlocks<'_>>, F, E, D>
    for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: Vec<SignedBlocks<'_>>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_weights = FusedPackingWeights::new(plan.tables)?;
        let point = plan.tables.point();
        source.into_iter().map(|blocks| {
            if num_rings::<D>(blocks)? != point.num_live_positions() { return Err(AkitaError::InvalidSize { expected: point.num_live_positions(), actual: num_rings::<D>(blocks)? }); }
            let coordinates = with_source!(blocks, storage => seam_packing_partials::<F, E, _, D>(&storage, blocks.len(), blocks.num_vars(), &fused_weights))?;
            SubringCoefficientPackingPartials::new(point.geometry(), point.num_live_blocks(), coordinates)
        }).collect()
    }
}
