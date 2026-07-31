//! Verifier-owned evaluation-trace contraction.
//!
//! The prover materializes foldable trace storage. The verifier instead keeps
//! one compact descriptor per group and witness chunk, then contracts the
//! rank-one trace factors directly at the final Stage 2 point.

use std::sync::Arc;

use akita_algebra::offset_eq::eval_affine_digit_intervals;
use akita_algebra::poly::multilinear_eval;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};
use akita_types::{
    basis_weights, prepare_evaluation_trace_group_parameters, BasisMode, EvaluationTraceInputs,
    FpExtEncoding,
};

/// One chunk's compact E-segment geometry, shared by every claim in its group.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedEvaluationTraceUnit {
    first_claim_coefficient: usize,
    claim_stride_coefficients: usize,
    global_block_start: usize,
    block_count: usize,
}

/// Verifier state for one opening group.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedEvaluationTraceGroup<E: FieldCore> {
    block_opening_point: Arc<[E]>,
    basis: BasisMode,
    source_ring_dimension: usize,
    opening_ring_dimension: usize,
    coefficient_block_len: usize,
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
            let coefficient_variables = group.coefficient_block_len.trailing_zeros() as usize;
            let (coefficient_point, column_point) = point
                .split_at_checked(coefficient_variables)
                .ok_or(AkitaError::InvalidProof)?;
            let block_point = &group.block_opening_point;
            let low_variables = block_point.len() / 2;
            let (low_block_point, high_block_point) = block_point
                .split_at_checked(low_variables)
                .ok_or(AkitaError::InvalidProof)?;
            let low_block_weights = basis_weights(low_block_point, group.basis)?;
            let high_block_weights = basis_weights(high_block_point, group.basis)?;
            let digit_weights = &group.opening_digit_weights;

            let role_subcolumns = source_ring_dimension / group.opening_ring_dimension;
            if role_subcolumns != 1 || source_ring_dimension != group.coefficient_block_len {
                let mut role_digit_weights = Vec::with_capacity(role_subcolumns);
                for role_subcolumn in 0..role_subcolumns {
                    let role_start = role_subcolumn
                        .checked_mul(group.opening_ring_dimension)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace role offset overflow".into())
                        })?;
                    let role_end = role_start
                        .checked_add(group.opening_ring_dimension)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace role end overflow".into())
                        })?;
                    let inner_trace = group
                        .inner_trace
                        .get(role_start..role_end)
                        .ok_or(AkitaError::InvalidProof)?;
                    let mut weights = Vec::with_capacity(
                        digit_weights
                            .len()
                            .checked_mul(inner_trace.len())
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("trace tensor size overflow".into())
                            })?,
                    );
                    for &digit_weight in digit_weights.iter() {
                        weights.extend(inner_trace.iter().map(|&inner| digit_weight * inner));
                    }
                    role_digit_weights.push(weights);
                }
                for (claim, &claim_coefficient) in group.claim_coefficients.iter().enumerate() {
                    for unit in &group.units {
                        let claim_start = claim
                            .checked_mul(unit.claim_stride_coefficients)
                            .and_then(|offset| unit.first_claim_coefficient.checked_add(offset))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("trace claim address overflow".into())
                            })?;
                        let block_stride = digit_weights
                            .len()
                            .checked_mul(source_ring_dimension)
                            .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace block stride overflow".into())
                        })?;
                        for (role_subcolumn, projected_weights) in
                            role_digit_weights.iter().enumerate()
                        {
                            let subcolumn_base = claim_start
                                .checked_add(
                                    role_subcolumn
                                        .checked_mul(digit_weights.len())
                                        .and_then(|offset| {
                                            offset.checked_mul(group.opening_ring_dimension)
                                        })
                                        .ok_or_else(|| {
                                            AkitaError::InvalidSetup(
                                                "trace subcolumn offset overflow".into(),
                                            )
                                        })?,
                                )
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("trace base address overflow".into())
                                })?;
                            evaluation += claim_coefficient
                                * eval_affine_digit_intervals(
                                    point,
                                    &[subcolumn_base],
                                    unit.global_block_start,
                                    unit.block_count,
                                    block_stride,
                                    projected_weights,
                                    &high_block_weights,
                                    &low_block_weights,
                                )?;
                        }
                    }
                }
                continue;
            }

            let inner_trace_evaluation = multilinear_eval(&group.inner_trace, coefficient_point)?;
            let mut group_evaluation = E::zero();
            for (claim, &claim_coefficient) in group.claim_coefficients.iter().enumerate() {
                let mut claim_evaluation = E::zero();
                for unit in &group.units {
                    let claim_column = claim
                        .checked_mul(unit.claim_stride_coefficients)
                        .and_then(|offset| unit.first_claim_coefficient.checked_add(offset))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace claim address overflow".into())
                        })?;
                    claim_evaluation += eval_affine_digit_intervals(
                        column_point,
                        &[claim_column / group.coefficient_block_len],
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
pub(crate) fn prepare_evaluation_trace<F, E>(
    inputs: &EvaluationTraceInputs<'_, F, E>,
) -> Result<PreparedEvaluationTrace<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let group_parameters = prepare_evaluation_trace_group_parameters::<F, E>(inputs)?;
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
        let group_dims = inputs
            .level_params
            .group_role_dims(inputs.opening_batch, parameters.group_index())?;
        let mut prepared_units = Vec::with_capacity(units.len());
        for unit in units {
            let first_claim_coefficient = unit.e_coefficient_index(
                group_dims.d_a(),
                group_dims.d_d(),
                num_claims,
                digit_count,
                0,
                unit.global_block_start(),
                0,
                0,
                0,
            )?;
            let claim_stride_coefficients = unit
                .num_live_blocks()
                .checked_mul(digit_count)
                .and_then(|count| count.checked_mul(group_dims.d_a()))
                .ok_or_else(|| AkitaError::InvalidSetup("trace claim stride overflow".into()))?;
            let final_claim_start = (num_claims - 1)
                .checked_mul(claim_stride_coefficients)
                .and_then(|offset| first_claim_coefficient.checked_add(offset))
                .ok_or_else(|| AkitaError::InvalidSetup("trace claim address overflow".into()))?;
            let physical_end = final_claim_start
                .checked_add(claim_stride_coefficients)
                .ok_or_else(|| AkitaError::InvalidSetup("trace segment end overflow".into()))?;
            if physical_end > inputs.digit_witness_domain.live_len() {
                return Err(AkitaError::InvalidProof);
            }
            prepared_units.push(PreparedEvaluationTraceUnit {
                first_claim_coefficient,
                claim_stride_coefficients,
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
            opening_ring_dimension: group_dims.d_d(),
            coefficient_block_len: inputs.relation_coefficient_block_len,
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
        basis_weights_prefix, r_decomp_levels, ring_opening_point_from_field, BasisMode,
        DigitRangePlan, OpeningClaimsLayout, PreparedOpeningPoint, RelationAddressGeometry,
        RelationRangeImagePlan, RingMultiplierOpeningPoint, WitnessLayout,
    };

    #[test]
    fn projected_subcolumn_trace_matches_dense_definition() {
        type F = fp128::Field;
        let block_point = Arc::<[F]>::from(vec![F::from_u64(3), F::from_u64(5)]);
        let digit_weights = Arc::<[F]>::from(vec![F::from_u64(7), F::from_u64(11)]);
        let inner_trace = Arc::<[F]>::from(
            (0..8)
                .map(|index| F::from_u64(13 + index as u64))
                .collect::<Vec<_>>(),
        );
        let claim_coefficient = F::from_u64(23);
        let unit = PreparedEvaluationTraceUnit {
            first_claim_coefficient: 3,
            claim_stride_coefficients: 48,
            global_block_start: 1,
            block_count: 2,
        };
        let trace = PreparedEvaluationTrace {
            groups: vec![PreparedEvaluationTraceGroup {
                block_opening_point: Arc::clone(&block_point),
                basis: BasisMode::Lagrange,
                source_ring_dimension: 8,
                opening_ring_dimension: 4,
                coefficient_block_len: 2,
                opening_digit_weights: Arc::clone(&digit_weights),
                inner_trace: Arc::clone(&inner_trace),
                claim_coefficients: vec![claim_coefficient],
                units: vec![unit.clone()],
            }],
            num_variables: 6,
        };
        let block_weights = basis_weights(&block_point, BasisMode::Lagrange).unwrap();
        let mut dense = vec![F::zero(); 1 << trace.num_variables];
        for local_block in 0..unit.block_count {
            let global_block = unit.global_block_start + local_block;
            for role_subcolumn in 0..2 {
                for (digit, &digit_weight) in digit_weights.iter().enumerate() {
                    for role_coefficient in 0..4 {
                        let address = unit.first_claim_coefficient
                            + local_block * digit_weights.len() * 8
                            + role_subcolumn * digit_weights.len() * 4
                            + digit * 4
                            + role_coefficient;
                        dense[address] += claim_coefficient
                            * block_weights[global_block]
                            * digit_weight
                            * inner_trace[role_subcolumn * 4 + role_coefficient];
                    }
                }
            }
        }
        let point = (0..trace.num_variables)
            .map(|index| F::from_u64(29 + index as u64))
            .collect::<Vec<_>>();

        assert_eq!(
            trace.evaluate_at_point(&point).unwrap(),
            multilinear_eval(&dense, &point).unwrap()
        );
    }

    #[test]
    fn compact_trace_matches_dense_definition_across_coefficient_blocks() {
        type Cfg = fp128::D128Dense;
        type F = fp128::Field;
        type E = F;
        const D: usize = Cfg::D;
        const NUM_VARIABLES: usize = 20;

        let opening_batch =
            OpeningClaimsLayout::new(NUM_VARIABLES, 2).expect("two-claim opening group");
        let level_params =
            Cfg::get_params_for_batched_commitment(&opening_batch).expect("level parameters");
        let witness_layout = WitnessLayout::new(
            &level_params,
            &opening_batch,
            2,
            r_decomp_levels::<F>(level_params.log_basis_open),
        )
        .expect("two-chunk witness layout");
        let live_len = witness_layout.live_coeff_len();
        let relation_address_geometry =
            RelationAddressGeometry::new(level_params.role_dims(), D / 2, live_len)
                .expect("flat trace domain");
        let plan = RelationRangeImagePlan::new(
            relation_address_geometry,
            DigitRangePlan::new(1usize << level_params.log_basis_open).expect("range basis"),
            witness_layout,
            &opening_batch,
        )
        .expect("relation/range-image plan");
        let digit_witness_domain = plan.digit_witness_domain();

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
        let inputs = || EvaluationTraceInputs {
            digit_witness_domain: plan.digit_witness_domain(),
            witness_layout: plan.witness_layout(),
            relation_coefficient_block_len: plan
                .relation_address_geometry()
                .relation_coefficient_block_len(),
            level_params: &level_params,
            opening_batch: &opening_batch,
            prepared_points: &prepared_points,
            claim_coefficients: &claim_coefficients,
            basis: BasisMode::Lagrange,
        };
        let verifier_trace =
            prepare_evaluation_trace::<F, E>(&inputs()).expect("compact verifier trace");
        let parameters = prepare_evaluation_trace_group_parameters::<F, E>(&inputs())
            .expect("checked trace geometry");
        let mut dense = vec![E::zero(); digit_witness_domain.live_len()];
        for parameters in parameters {
            let group_dims = level_params
                .group_role_dims(&opening_batch, parameters.group_index())
                .expect("group dimensions");
            let group_layout = opening_batch
                .group_layout(parameters.group_index())
                .expect("group layout");
            let units = plan
                .witness_layout()
                .units_for_group(parameters.group_index())
                .expect("group witness units");
            let block_weights = basis_weights_prefix(
                parameters.block_opening_point(),
                parameters.basis(),
                parameters.group_block_count(),
            )
            .expect("block weights");
            for (local_claim, claim_index) in parameters.claim_range().enumerate() {
                for unit in &units {
                    for local_block in 0..unit.num_live_blocks() {
                        let block = unit.global_block_start() + local_block;
                        let role_subcolumns = group_dims.d_a() / group_dims.d_d();
                        for (digit, &digit_weight) in
                            parameters.opening_digit_weights().iter().enumerate()
                        {
                            let factor = claim_coefficients[claim_index]
                                * block_weights[block]
                                * digit_weight;
                            for role_subcolumn in 0..role_subcolumns {
                                for role_coefficient in 0..group_dims.d_d() {
                                    let address = unit
                                        .e_coefficient_index(
                                            group_dims.d_a(),
                                            group_dims.d_d(),
                                            group_layout.num_polynomials(),
                                            parameters.opening_digit_weights().len(),
                                            local_claim,
                                            block,
                                            role_subcolumn,
                                            digit,
                                            role_coefficient,
                                        )
                                        .expect("trace coefficient");
                                    let source =
                                        role_subcolumn * group_dims.d_d() + role_coefficient;
                                    dense[address] += factor * parameters.inner_trace()[source];
                                }
                            }
                        }
                    }
                }
            }
        }
        let point = (0..digit_witness_domain.num_vars())
            .map(|index| E::from_u64(47 + 2 * index as u64))
            .collect::<Vec<_>>();
        dense.resize(1usize << point.len(), E::zero());
        assert_eq!(
            verifier_trace.evaluate_at_point(&point).unwrap(),
            multilinear_eval(&dense, &point).unwrap()
        );
    }
}
