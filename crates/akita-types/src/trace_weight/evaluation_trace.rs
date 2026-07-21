//! Checked evaluation-trace inputs and group geometry shared by prover and verifier.

use std::ops::Range;
use std::sync::Arc;

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

use crate::field_reduction::trace_open_ring_row;
use crate::{
    gadget_row_scalars, BasisMode, CommitmentRingDims, FlatBooleanDomain, FpExtEncoding,
    LevelParams, OpeningClaimsLayout, PreparedOpeningPoint, WitnessLayout,
};

/// Reject extension degrees with no evaluation-trace implementation.
pub fn ensure_trace_stage2_supported(extension_degree: usize) -> Result<(), AkitaError> {
    if matches!(extension_degree, 1 | 2 | 4 | 8) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "Stage-2 evaluation trace has no implementation for claim-field extension degree {extension_degree}"
        )))
    }
}

/// Slice the fold-block coordinates out of one prepared evaluation-trace point.
fn evaluation_trace_block_point<X: FieldCore>(
    opening_point: &[X],
    num_positions_per_block: usize,
    num_live_blocks: usize,
    alpha_bits: usize,
) -> Result<Vec<X>, AkitaError> {
    if !num_positions_per_block.is_power_of_two() || num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "trace opening requires power-of-two positions and a live block".into(),
        ));
    }
    let position_bits = num_positions_per_block.trailing_zeros() as usize;
    let block_bits = num_live_blocks
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("trace block domain overflow".into()))?
        .trailing_zeros() as usize;
    let target_len = alpha_bits
        .checked_add(position_bits)
        .and_then(|len| len.checked_add(block_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("trace opening width overflow".into()))?;
    if opening_point.len() > target_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: target_len,
            actual: opening_point.len(),
        });
    }
    let mut padded = opening_point.to_vec();
    padded.resize(target_len, X::zero());
    let block_start = alpha_bits + position_bits;
    Ok(padded[block_start..block_start + block_bits].to_vec())
}

/// Checked short trace parameters shared by the prover and verifier builders.
///
/// This contains protocol facts, not either side's runtime representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationTraceGroupParameters<E: FieldCore> {
    group_index: usize,
    claim_range: Range<usize>,
    block_opening_point: Arc<[E]>,
    basis: BasisMode,
    group_block_count: usize,
    source_ring_dimension: usize,
    opening_digit_weights: Arc<[E]>,
    inner_trace: Arc<[E]>,
}

impl<E: FieldCore> EvaluationTraceGroupParameters<E> {
    #[must_use]
    pub fn group_index(&self) -> usize {
        self.group_index
    }

    #[must_use]
    pub fn claim_range(&self) -> Range<usize> {
        self.claim_range.clone()
    }

    #[must_use]
    pub fn block_opening_point(&self) -> &[E] {
        &self.block_opening_point
    }

    #[must_use]
    pub fn shared_block_opening_point(&self) -> Arc<[E]> {
        Arc::clone(&self.block_opening_point)
    }

    #[must_use]
    pub fn basis(&self) -> BasisMode {
        self.basis
    }

    #[must_use]
    pub fn group_block_count(&self) -> usize {
        self.group_block_count
    }

    #[must_use]
    pub fn source_ring_dimension(&self) -> usize {
        self.source_ring_dimension
    }

    #[must_use]
    pub fn opening_digit_weights(&self) -> &[E] {
        &self.opening_digit_weights
    }

    #[must_use]
    pub fn shared_opening_digit_weights(&self) -> Arc<[E]> {
        Arc::clone(&self.opening_digit_weights)
    }

    #[must_use]
    pub fn inner_trace(&self) -> &[E] {
        &self.inner_trace
    }

    #[must_use]
    pub fn shared_inner_trace(&self) -> Arc<[E]> {
        Arc::clone(&self.inner_trace)
    }
}

/// Apply one uniform reduction scale to normalized evaluation-trace coefficients.
pub fn scale_evaluation_trace_claim_coefficients<E: FieldCore>(
    claim_coefficients: &[E],
    uniform_scale: E,
) -> Result<Vec<E>, AkitaError> {
    if claim_coefficients.is_empty() {
        return Err(AkitaError::InvalidInput(
            "evaluation trace requires a claim coefficient".into(),
        ));
    }
    Ok(claim_coefficients
        .iter()
        .map(|&coefficient| coefficient * uniform_scale)
        .collect())
}

/// Checked common inputs from which prover and verifier build separate
/// evaluation-trace representations.
pub struct EvaluationTraceInputs<'a, F: FieldCore, E: FieldCore> {
    pub digit_witness_domain: FlatBooleanDomain,
    pub witness_layout: &'a WitnessLayout,
    pub role_dims: CommitmentRingDims,
    pub level_params: &'a LevelParams,
    pub opening_batch: &'a OpeningClaimsLayout,
    pub prepared_points: &'a [PreparedOpeningPoint<F, E>],
    pub claim_coefficients: &'a [E],
    pub basis: BasisMode,
}

/// Prepare the checked, short per-group parameters from which prover and
/// verifier build their separate trace representations.
pub fn prepare_evaluation_trace_group_parameters<F, E, const D: usize>(
    inputs: &EvaluationTraceInputs<'_, F, E>,
) -> Result<Vec<EvaluationTraceGroupParameters<E>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    if inputs.role_dims.d_a() != D
        || inputs.prepared_points.len() != inputs.opening_batch.num_groups()
        || inputs.claim_coefficients.len() != inputs.opening_batch.num_total_polynomials()
    {
        return Err(AkitaError::InvalidProof);
    }
    let expected_live_len = inputs
        .witness_layout
        .total_len()
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("trace witness size overflow".into()))?;
    if inputs.digit_witness_domain.live_len() != expected_live_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_live_len,
            actual: inputs.digit_witness_domain.live_len(),
        });
    }
    let alpha_bits = D.trailing_zeros() as usize;
    inputs
        .opening_batch
        .root_group_order()?
        .into_iter()
        .map(|group_index| {
            let group_params = inputs
                .level_params
                .group_params(inputs.opening_batch, group_index)?;
            let units = inputs.witness_layout.units_for_group(group_index)?;
            let covered_blocks = units.iter().enumerate().try_fold(
                0usize,
                |expected_start, (expected_chunk, unit)| {
                    if unit.chunk_index() != expected_chunk
                        || unit.global_block_start() != expected_start
                        || unit.num_live_blocks() == 0
                    {
                        return Err(AkitaError::InvalidSetup(
                            "trace witness chunks do not form one ordered block partition".into(),
                        ));
                    }
                    expected_start
                        .checked_add(unit.num_live_blocks())
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace witness block coverage overflow".into())
                        })
                },
            )?;
            if covered_blocks != group_params.num_live_blocks() {
                return Err(AkitaError::InvalidProof);
            }
            let prepared = inputs
                .prepared_points
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            prepared.ensure_ring_dim::<D>()?;
            let block_opening_point: Arc<[E]> = evaluation_trace_block_point(
                &prepared.padded_point,
                group_params.num_positions_per_block(),
                group_params.num_live_blocks(),
                alpha_bits,
            )?
            .into();
            let packed_inner = prepared.packed_inner_trusted::<D>()?;
            let inner_trace: Arc<[E]> = if E::EXT_DEGREE == 1 {
                packed_inner
                    .coefficients()
                    .iter()
                    .copied()
                    .map(E::lift_base)
                    .collect::<Vec<_>>()
                    .into()
            } else {
                trace_open_ring_row::<F, E, D>(
                    &CyclotomicRing::<F, D>::one(),
                    packed_inner,
                    alpha_bits,
                )?
                .into()
            };
            if inner_trace.len() != D {
                return Err(AkitaError::InvalidProof);
            }
            let opening_digit_weights: Arc<[E]> = gadget_row_scalars::<F>(
                group_params.num_digits_open(),
                group_params.log_basis_open(),
            )
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>()
            .into();
            Ok(EvaluationTraceGroupParameters {
                group_index,
                claim_range: inputs.opening_batch.root_group_claim_range(group_index)?,
                block_opening_point,
                basis: inputs.basis,
                group_block_count: group_params.num_live_blocks(),
                source_ring_dimension: D,
                opening_digit_weights,
                inner_trace,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{basis_weights, block_rings_at_opening, embed_ring_subfield_vector};
    use akita_field::{Ext2, Fp32, FpExt4, FpExt8, RandomSampling};
    use rand::{rngs::StdRng, SeedableRng};

    type BaseField = Fp32<251>;
    type Extension2 = Ext2<BaseField>;
    type Extension4 = FpExt4<BaseField>;
    type Extension8 = FpExt8<BaseField>;

    fn assert_extension_trace_factorization<E, const D: usize>(seed: u64)
    where
        E: FpExtEncoding<BaseField> + ExtField<BaseField> + FromPrimitiveInt + RandomSampling,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let inner_point_len = (D / E::EXT_DEGREE).trailing_zeros() as usize;
        for _ in 0..8 {
            let inner_point: Vec<E> = (0..inner_point_len).map(|_| E::random(&mut rng)).collect();
            let inner_weights = basis_weights(&inner_point, BasisMode::Lagrange).unwrap();
            let packed_inner = embed_ring_subfield_vector::<BaseField, E, D>(
                &inner_weights,
                AkitaError::InvalidInput("test inner point does not embed".into()),
            )
            .unwrap();
            let block_point: Vec<E> = (0..2).map(|_| E::random(&mut rng)).collect();
            let block_weights = basis_weights(&block_point, BasisMode::Lagrange).unwrap();
            let block_rings =
                block_rings_at_opening::<BaseField, E, D>(&block_point, block_weights.len())
                    .unwrap();
            let inner_trace = trace_open_ring_row::<BaseField, E, D>(
                &CyclotomicRing::one(),
                &packed_inner,
                D.trailing_zeros() as usize,
            )
            .unwrap();
            for (&block_weight, block_ring) in block_weights.iter().zip(&block_rings) {
                let row = trace_open_ring_row::<BaseField, E, D>(
                    block_ring,
                    &packed_inner,
                    D.trailing_zeros() as usize,
                )
                .unwrap();
                assert_eq!(
                    row,
                    inner_trace
                        .iter()
                        .map(|&inner| block_weight * inner)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn extension_inner_trace_factorization_is_exact() {
        assert_extension_trace_factorization::<Extension2, 8>(0x2008);
        assert_extension_trace_factorization::<Extension4, 8>(0x4008);
        assert_extension_trace_factorization::<Extension8, 16>(0x8010);
    }
}
