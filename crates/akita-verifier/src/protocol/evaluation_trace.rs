//! Verifier-owned evaluation-trace contraction.
//!
//! The prover materializes foldable trace storage. The verifier instead keeps
//! one compact descriptor per group and witness chunk, then contracts the
//! rank-one trace factors directly at the final Stage 2 point.

use std::sync::Arc;

use akita_algebra::offset_eq::eval_affine_digit_interval;
use akita_algebra::poly::multilinear_eval;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use akita_types::{
    basis_weights, prepare_evaluation_trace_group_parameters, BasisMode,
    EvaluationTraceWeightInputs, FpExtEncoding,
};

/// One chunk's compact E-segment geometry, shared by every claim in its group.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedEvaluationTraceUnit {
    first_claim_column: usize,
    claim_stride: usize,
    global_block_start: usize,
    block_count: usize,
}

/// Verifier state for one opening group.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedEvaluationTraceGroup<E: FieldCore> {
    block_opening_point: Arc<[E]>,
    basis: BasisMode,
    source_ring_dimension: usize,
    opening_digit_weights: Arc<[E]>,
    inner_trace: Arc<[E]>,
    claim_coefficients: Vec<E>,
    units: Vec<PreparedEvaluationTraceUnit>,
}

/// Succinct verifier representation of the complete evaluation-trace weight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedEvaluationTrace<E: FieldCore> {
    groups: Vec<PreparedEvaluationTraceGroup<E>>,
    num_variables: usize,
}

impl<E: FieldCore> PreparedEvaluationTrace<E> {
    /// Evaluate the trace-weight MLE without constructing prover terms or
    /// scanning physical coefficient support.
    pub(crate) fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.num_variables {
            return Err(AkitaError::InvalidSize {
                expected: self.num_variables,
                actual: point.len(),
            });
        }

        let mut evaluation = E::zero();
        for group in &self.groups {
            let source_ring_dimension = group.source_ring_dimension;
            if source_ring_dimension == 0 || !source_ring_dimension.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "trace source ring dimension must be a power of two".into(),
                ));
            }
            let coefficient_variables = source_ring_dimension.trailing_zeros() as usize;
            let (coefficient_point, column_point) = point
                .split_at_checked(coefficient_variables)
                .ok_or(AkitaError::InvalidProof)?;
            let inner_trace_evaluation = multilinear_eval(&group.inner_trace, coefficient_point)?;
            let block_point = &group.block_opening_point;
            let low_variables = block_point.len() / 2;
            let (low_block_point, high_block_point) = block_point
                .split_at_checked(low_variables)
                .ok_or(AkitaError::InvalidProof)?;
            let low_block_weights = basis_weights(low_block_point, group.basis)?;
            let high_block_weights = basis_weights(high_block_point, group.basis)?;
            let digit_weights = &group.opening_digit_weights;

            let mut group_evaluation = E::zero();
            for (claim, &claim_coefficient) in group.claim_coefficients.iter().enumerate() {
                let mut claim_evaluation = E::zero();
                for unit in &group.units {
                    let claim_column = claim
                        .checked_mul(unit.claim_stride)
                        .and_then(|offset| unit.first_claim_column.checked_add(offset))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace claim address overflow".into())
                        })?;
                    claim_evaluation += eval_affine_digit_interval(
                        column_point,
                        claim_column,
                        unit.global_block_start,
                        unit.block_count,
                        digit_weights.len(),
                        digit_weights,
                        &high_block_weights,
                        &low_block_weights,
                    )?;
                }
                group_evaluation += claim_coefficient * claim_evaluation;
            }
            evaluation += inner_trace_evaluation * group_evaluation;
        }
        Ok(evaluation)
    }
}

/// Prepare the verifier's compact group/chunk descriptors from checked common
/// trace parameters.
pub(crate) fn prepare_evaluation_trace<F, E, const D: usize>(
    inputs: &EvaluationTraceWeightInputs<'_, F, E>,
) -> Result<PreparedEvaluationTrace<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let group_parameters = prepare_evaluation_trace_group_parameters::<F, E, D>(inputs)?;
    let mut groups = Vec::with_capacity(group_parameters.len());
    for parameters in group_parameters {
        let group_layout = inputs
            .opening_batch
            .group_layout(parameters.group_index())?;
        let num_claims = group_layout.num_polynomials();
        if num_claims == 0 || parameters.claim_range().len() != num_claims {
            return Err(AkitaError::InvalidProof);
        }
        let units = inputs
            .witness_layout
            .units_for_group(parameters.group_index())?;
        let digit_count = parameters.opening_digit_weights().len();
        let mut prepared_units = Vec::with_capacity(units.len());
        for unit in units {
            let first_claim_column =
                unit.e_index(num_claims, digit_count, 0, unit.global_block_start(), 0)?;
            let claim_stride = unit
                .num_live_blocks()
                .checked_mul(digit_count)
                .ok_or_else(|| AkitaError::InvalidSetup("trace claim stride overflow".into()))?;
            let final_claim_start = (num_claims - 1)
                .checked_mul(claim_stride)
                .and_then(|offset| first_claim_column.checked_add(offset))
                .ok_or_else(|| AkitaError::InvalidSetup("trace claim address overflow".into()))?;
            let physical_end = final_claim_start
                .checked_add(claim_stride)
                .and_then(|column_end| column_end.checked_mul(D))
                .ok_or_else(|| AkitaError::InvalidSetup("trace segment end overflow".into()))?;
            if physical_end > inputs.digit_witness_domain.live_len() {
                return Err(AkitaError::InvalidProof);
            }
            prepared_units.push(PreparedEvaluationTraceUnit {
                first_claim_column,
                claim_stride,
                global_block_start: unit.global_block_start(),
                block_count: unit.num_live_blocks(),
            });
        }
        let claim_coefficients = inputs
            .claim_coefficients
            .get(parameters.claim_range())
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        groups.push(PreparedEvaluationTraceGroup {
            block_opening_point: parameters.shared_block_opening_point(),
            basis: parameters.basis(),
            source_ring_dimension: parameters.source_ring_dimension(),
            opening_digit_weights: parameters.shared_opening_digit_weights(),
            inner_trace: parameters.shared_inner_trace(),
            claim_coefficients,
            units: prepared_units,
        });
    }
    if groups.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(PreparedEvaluationTrace {
        groups,
        num_variables: inputs.digit_witness_domain.num_vars(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_config::proof_optimized::fp128;
    use akita_config::CommitmentConfig;
    use akita_types::{
        build_evaluation_trace_weights, r_decomp_levels, relation_rhs_layout_for,
        relation_rhs_row_count, ring_opening_point_from_field, BasisMode, DigitRangePlan,
        FlatBooleanDomain, OpeningClaimsLayout, PreparedOpeningPoint, RelationMatrixRowLayout,
        RelationRangeImagePlan, RingMultiplierOpeningPoint, WitnessLayout,
    };

    #[test]
    fn compact_trace_matches_prover_weight_function() {
        type Cfg = fp128::D128Full;
        type F = fp128::Field;
        type E = F;
        const D: usize = Cfg::D;
        const NUM_VARIABLES: usize = 20;

        let opening_batch =
            OpeningClaimsLayout::new(NUM_VARIABLES, 2).expect("two-claim opening group");
        let level_params =
            Cfg::get_params_for_batched_commitment(&opening_batch).expect("level parameters");
        let rhs_layout = relation_rhs_layout_for(
            &level_params,
            &opening_batch,
            RelationMatrixRowLayout::WithDBlock,
        )
        .expect("relation RHS layout");
        let witness_layout = WitnessLayout::new(
            &level_params,
            &opening_batch,
            2,
            relation_rhs_row_count(&rhs_layout),
            r_decomp_levels::<F>(level_params.log_basis),
        )
        .expect("two-chunk witness layout");
        let live_len = witness_layout.total_len() * D;
        let digit_witness_domain = FlatBooleanDomain::new(
            live_len,
            live_len.next_power_of_two().trailing_zeros() as usize,
        )
        .expect("flat trace domain");
        let plan = RelationRangeImagePlan::new(
            digit_witness_domain,
            DigitRangePlan::new(1usize << level_params.log_basis).expect("range basis"),
            witness_layout,
            &opening_batch,
            level_params.role_dims(),
            D,
        )
        .expect("relation/range-image plan");

        let group_params = level_params
            .group_params(&opening_batch, 0)
            .expect("group parameters");
        let alpha_variables = D.trailing_zeros() as usize;
        let base_outer_point = vec![F::zero(); NUM_VARIABLES - alpha_variables];
        let ring_opening_point = ring_opening_point_from_field(
            &base_outer_point,
            group_params.num_positions_per_block(),
            group_params.num_live_blocks(),
            BasisMode::Lagrange,
        )
        .expect("ring opening point");
        let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let prepared_points = vec![PreparedOpeningPoint::from_parts(
            (0..NUM_VARIABLES)
                .map(|index| E::from_u64(17 + 2 * index as u64))
                .collect(),
            ring_opening_point,
            ring_multiplier_point,
            CyclotomicRing::<F, D>::one(),
        )];
        let claim_coefficients = vec![E::from_u64(41), E::from_u64(43)];
        let inputs = || EvaluationTraceWeightInputs {
            digit_witness_domain: plan.digit_witness_domain(),
            witness_layout: plan.witness_layout(),
            role_dims: plan.role_dims(),
            level_params: &level_params,
            opening_batch: &opening_batch,
            prepared_points: &prepared_points,
            claim_coefficients: &claim_coefficients,
            basis: BasisMode::Lagrange,
        };
        let verifier_trace =
            prepare_evaluation_trace::<F, E, D>(&inputs()).expect("compact verifier trace");
        let prover_trace = build_evaluation_trace_weights::<F, E, D>(inputs())
            .expect("canonical prover trace weights");
        let point = (0..digit_witness_domain.num_vars())
            .map(|index| E::from_u64(47 + 2 * index as u64))
            .collect::<Vec<_>>();
        assert_eq!(
            verifier_trace.evaluate_at_point(&point).unwrap(),
            prover_trace.evaluate_at_point(&point).unwrap()
        );
    }
}
