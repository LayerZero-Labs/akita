//! Verifier-owned evaluation-trace contraction.
//!
//! The prover materializes foldable trace storage. The verifier instead keeps
//! one compact descriptor per group and witness chunk, then contracts the
//! rank-one trace factors directly at the final Stage 2 point.

use std::sync::Arc;

use akita_algebra::offset_eq::{
    eval_affine_digit_intervals, eval_boolean_pair_tensor_families, EqPairTensorAxis,
    EqPairTensorFamily, MAX_COMPACT_STRIDE_TERMS,
};
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

/// Below this per-unit block count the affine scan wins despite its linear
/// term; at and above it the logarithmic paired recurrence wins decisively.
const TRACE_TENSOR_MIN_BLOCKS_PER_UNIT: usize = 64;

#[derive(Clone, Copy)]
struct TraceUnitTensorAxis {
    len: usize,
    block_stride: usize,
    lane_stride: usize,
}

#[derive(Clone, Copy)]
struct TraceUnitTensorSegment {
    first_unit: usize,
    axis: Option<TraceUnitTensorAxis>,
}

fn trace_unit_tensor_axis(
    units: &[PreparedEvaluationTraceUnit],
    coefficient_block_len: usize,
) -> Option<TraceUnitTensorAxis> {
    if units.len() < 2 || !units.len().is_power_of_two() {
        return None;
    }
    let first = units.first()?;
    let second = units.get(1)?;
    let block_stride = second
        .global_block_start
        .checked_sub(first.global_block_start)
        .filter(|&stride| stride != 0)?;
    let coefficient_stride = second
        .first_claim_coefficient
        .checked_sub(first.first_claim_coefficient)
        .filter(|&stride| stride != 0 && stride.is_multiple_of(coefficient_block_len))?;
    let lane_stride = coefficient_stride / coefficient_block_len;

    for (index, unit) in units.iter().enumerate() {
        let expected_block_start = index
            .checked_mul(block_stride)
            .and_then(|offset| first.global_block_start.checked_add(offset))?;
        let expected_coefficient = index
            .checked_mul(coefficient_stride)
            .and_then(|offset| first.first_claim_coefficient.checked_add(offset))?;
        if unit.global_block_start != expected_block_start
            || unit.first_claim_coefficient != expected_coefficient
            || unit.block_count != first.block_count
            || unit.claim_stride_coefficients != first.claim_stride_coefficients
        {
            return None;
        }
    }
    Some(TraceUnitTensorAxis {
        len: units.len(),
        block_stride,
        lane_stride,
    })
}

fn trace_unit_tensor_segments(
    units: &[PreparedEvaluationTraceUnit],
    coefficient_block_len: usize,
) -> Result<Vec<TraceUnitTensorSegment>, AkitaError> {
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(units.len())
        .map_err(|_| AkitaError::InvalidSetup("trace unit segments are too large".into()))?;
    let mut run_start = 0usize;
    while run_start < units.len() {
        let first = units.get(run_start).ok_or(AkitaError::InvalidProof)?;
        let mut run_end = run_start + 1;
        let strides = units.get(run_end).and_then(|second| {
            let block_stride = second
                .global_block_start
                .checked_sub(first.global_block_start)?;
            let coefficient_stride = second
                .first_claim_coefficient
                .checked_sub(first.first_claim_coefficient)?;
            (block_stride != 0
                && coefficient_stride != 0
                && coefficient_stride.is_multiple_of(coefficient_block_len)
                && second.block_count == first.block_count
                && second.claim_stride_coefficients == first.claim_stride_coefficients)
                .then_some((block_stride, coefficient_stride))
        });
        if let Some((block_stride, coefficient_stride)) = strides {
            run_end += 1;
            while let (Some(previous), Some(next)) = (units.get(run_end - 1), units.get(run_end)) {
                let matches_run = next.block_count == first.block_count
                    && next.claim_stride_coefficients == first.claim_stride_coefficients
                    && previous
                        .global_block_start
                        .checked_add(block_stride)
                        .is_some_and(|expected| expected == next.global_block_start)
                    && previous
                        .first_claim_coefficient
                        .checked_add(coefficient_stride)
                        .is_some_and(|expected| expected == next.first_claim_coefficient);
                if !matches_run {
                    break;
                }
                run_end += 1;
            }
        }

        let mut segment_start = run_start;
        while segment_start < run_end {
            let remaining = run_end - segment_start;
            let segment_len = 1usize << (usize::BITS - remaining.leading_zeros() - 1);
            let segment_end = segment_start.checked_add(segment_len).ok_or_else(|| {
                AkitaError::InvalidSetup("trace unit segment range overflow".into())
            })?;
            let segment_units = units
                .get(segment_start..segment_end)
                .ok_or(AkitaError::InvalidProof)?;
            segments.push(TraceUnitTensorSegment {
                first_unit: segment_start,
                axis: trace_unit_tensor_axis(segment_units, coefficient_block_len),
            });
            segment_start = segment_end;
        }
        run_start = run_end;
    }
    Ok(segments)
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
            let coefficient_block_len = group.coefficient_block_len;
            if coefficient_block_len == 0
                || !coefficient_block_len.is_power_of_two()
                || !source_ring_dimension.is_multiple_of(coefficient_block_len)
                || !group
                    .opening_ring_dimension
                    .is_multiple_of(coefficient_block_len)
                || !source_ring_dimension.is_multiple_of(group.opening_ring_dimension)
            {
                return Err(AkitaError::InvalidSetup(
                    "trace dimensions do not decompose over the common coefficient block".into(),
                ));
            }
            let coefficient_variables = coefficient_block_len.trailing_zeros() as usize;
            let (coefficient_point, column_point) = point
                .split_at_checked(coefficient_variables)
                .ok_or(AkitaError::InvalidProof)?;
            let block_point = &group.block_opening_point;
            let max_unit_blocks = group
                .units
                .iter()
                .map(|unit| unit.block_count)
                .max()
                .ok_or(AkitaError::InvalidProof)?;
            let use_tensor_recurrence =
                max_unit_blocks == 1 || max_unit_blocks >= TRACE_TENSOR_MIN_BLOCKS_PER_UNIT;
            let linear_block_weights = if use_tensor_recurrence {
                None
            } else {
                let low_variables = block_point.len() / 2;
                let (low_block_point, high_block_point) = block_point
                    .split_at_checked(low_variables)
                    .ok_or(AkitaError::InvalidProof)?;
                Some((
                    basis_weights(low_block_point, group.basis)?,
                    basis_weights(high_block_point, group.basis)?,
                ))
            };
            let digit_weights = &group.opening_digit_weights;
            let source_lane_count = source_ring_dimension / coefficient_block_len;
            let role_lane_count = group.opening_ring_dimension / coefficient_block_len;
            let role_subcolumns = source_ring_dimension / group.opening_ring_dimension;
            let expected_source_lanes = role_subcolumns
                .checked_mul(role_lane_count)
                .ok_or_else(|| AkitaError::InvalidSetup("trace lane count overflow".into()))?;
            if source_lane_count != expected_source_lanes
                || group.inner_trace.len() != source_ring_dimension
            {
                return Err(AkitaError::InvalidProof);
            }
            let inner_trace_evaluations = group
                .inner_trace
                .chunks_exact(coefficient_block_len)
                .map(|trace| multilinear_eval(trace, coefficient_point))
                .collect::<Result<Vec<_>, _>>()?;
            if inner_trace_evaluations.len() != source_lane_count {
                return Err(AkitaError::InvalidProof);
            }
            let block_stride = digit_weights
                .len()
                .checked_mul(source_lane_count)
                .ok_or_else(|| AkitaError::InvalidSetup("trace block stride overflow".into()))?;
            let unit_tensor_segments = if use_tensor_recurrence {
                trace_unit_tensor_segments(&group.units, coefficient_block_len)?
            } else {
                Vec::new()
            };
            let mut families = Vec::new();
            if use_tensor_recurrence {
                let unit_family_count = unit_tensor_segments.len();
                let family_count = group
                    .claim_coefficients
                    .len()
                    .checked_mul(unit_family_count)
                    .and_then(|count| count.checked_mul(role_subcolumns))
                    .and_then(|count| count.checked_mul(role_lane_count))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("trace family count overflow".into())
                    })?;
                let family_work = family_count
                    .checked_mul(digit_weights.len())
                    .ok_or_else(|| AkitaError::InvalidSetup("trace family work overflow".into()))?;
                if family_work > MAX_COMPACT_STRIDE_TERMS {
                    return Err(AkitaError::InvalidSize {
                        expected: MAX_COMPACT_STRIDE_TERMS,
                        actual: family_work,
                    });
                }
                families.try_reserve_exact(family_count).map_err(|_| {
                    AkitaError::InvalidSetup("trace tensor families are too large".into())
                })?;
            }
            let mut contract_unit = |unit: &PreparedEvaluationTraceUnit,
                                     tensor_axis: Option<TraceUnitTensorAxis>|
             -> Result<(), AkitaError> {
                for (claim, &claim_coefficient) in group.claim_coefficients.iter().enumerate() {
                    let claim_start = claim
                        .checked_mul(unit.claim_stride_coefficients)
                        .and_then(|offset| unit.first_claim_coefficient.checked_add(offset))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace claim address overflow".into())
                        })?;
                    let claim_lane_start = claim_start
                        .checked_div(coefficient_block_len)
                        .filter(|_| claim_start.is_multiple_of(coefficient_block_len))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "trace claim is not coefficient-block aligned".into(),
                            )
                        })?;
                    for role_subcolumn in 0..role_subcolumns {
                        let subcolumn_offset = role_subcolumn
                            .checked_mul(digit_weights.len())
                            .and_then(|offset| offset.checked_mul(role_lane_count))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("trace subcolumn offset overflow".into())
                            })?;
                        for role_lane in 0..role_lane_count {
                            let source_lane = role_subcolumn
                                .checked_mul(role_lane_count)
                                .and_then(|lane| lane.checked_add(role_lane))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("trace source lane overflow".into())
                                })?;
                            let base = claim_lane_start
                                .checked_add(subcolumn_offset)
                                .and_then(|offset| offset.checked_add(role_lane))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("trace base address overflow".into())
                                })?;
                            let inner_trace_evaluation = *inner_trace_evaluations
                                .get(source_lane)
                                .ok_or(AkitaError::InvalidProof)?;
                            if let Some((low_block_weights, high_block_weights)) =
                                &linear_block_weights
                            {
                                evaluation += claim_coefficient
                                    * inner_trace_evaluation
                                    * eval_affine_digit_intervals(
                                        column_point,
                                        &[base],
                                        unit.global_block_start,
                                        unit.block_count,
                                        block_stride,
                                        role_lane_count,
                                        digit_weights,
                                        high_block_weights,
                                        low_block_weights,
                                        &[],
                                    )?;
                                continue;
                            }
                            let mut axes = vec![
                                EqPairTensorAxis::dense(0, role_lane_count, digit_weights.to_vec()),
                                EqPairTensorAxis::unit(unit.block_count, 1, block_stride),
                            ];
                            if let Some(axis) = tensor_axis {
                                axes.push(EqPairTensorAxis::unit(
                                    axis.len,
                                    axis.block_stride,
                                    axis.lane_stride,
                                ));
                            }
                            families.push(EqPairTensorFamily::new(
                                unit.global_block_start,
                                base,
                                claim_coefficient * inner_trace_evaluation,
                                axes,
                            )?);
                        }
                    }
                }
                Ok(())
            };
            if use_tensor_recurrence {
                for segment in unit_tensor_segments {
                    let first_unit = group
                        .units
                        .get(segment.first_unit)
                        .ok_or(AkitaError::InvalidProof)?;
                    contract_unit(first_unit, segment.axis)?;
                }
            } else {
                for unit in &group.units {
                    contract_unit(unit, None)?;
                }
            }
            if linear_block_weights.is_none() {
                evaluation += match group.basis {
                    BasisMode::Lagrange => eval_boolean_pair_tensor_families::<_, false, false>(
                        block_point,
                        column_point,
                        &families,
                    )?,
                    BasisMode::Monomial => eval_boolean_pair_tensor_families::<_, true, false>(
                        block_point,
                        column_point,
                        &families,
                    )?,
                };
            }
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
        let mut prepared_units = Vec::with_capacity(units.clone().count());
        for unit in units {
            if unit.num_live_blocks() == 0 {
                continue;
            }
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

/// Exact synthetic trace fixture for production-kernel benchmarks.
#[cfg(any(test, feature = "benchmark-support"))]
pub struct EvaluationTraceBenchmarkCase {
    trace: PreparedEvaluationTrace<akita_field::Prime128OffsetA7F7>,
    point: Vec<akita_field::Prime128OffsetA7F7>,
}

#[cfg(any(test, feature = "benchmark-support"))]
impl EvaluationTraceBenchmarkCase {
    /// Evaluate the prepared trace at its fixed benchmark point.
    pub fn evaluate(&self) -> Result<akita_field::Prime128OffsetA7F7, AkitaError> {
        self.trace.evaluate_at_point(&self.point)
    }
}

/// Build a checked trace benchmark with two claims, two opening digits, and a
/// D128 source split into D64 coefficient blocks.
#[cfg(any(test, feature = "benchmark-support"))]
pub fn evaluation_trace_benchmark_case(
    num_live_blocks: usize,
    witness_chunks: usize,
    basis: BasisMode,
) -> Result<EvaluationTraceBenchmarkCase, AkitaError> {
    use akita_field::Prime128OffsetA7F7 as F;
    use akita_types::dyadic_block_ranges;

    const SOURCE_RING_DIMENSION: usize = 128;
    const OPENING_RING_DIMENSION: usize = 128;
    const COEFFICIENT_BLOCK_LEN: usize = 64;
    const NUM_CLAIMS: usize = 2;
    const DIGIT_COUNT: usize = 2;

    if num_live_blocks == 0 || !witness_chunks.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "trace benchmark requires nonempty blocks and a dyadic chunk count".into(),
        ));
    }
    let mut coefficient_cursor = 0usize;
    let mut units = Vec::with_capacity(witness_chunks.min(num_live_blocks));
    for range in dyadic_block_ranges(num_live_blocks, witness_chunks)? {
        let block_count = range.len();
        if block_count == 0 {
            continue;
        }
        let claim_stride_coefficients = block_count
            .checked_mul(DIGIT_COUNT)
            .and_then(|count| count.checked_mul(SOURCE_RING_DIMENSION))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("trace benchmark claim stride overflow".into())
            })?;
        units.push(PreparedEvaluationTraceUnit {
            first_claim_coefficient: coefficient_cursor,
            claim_stride_coefficients,
            global_block_start: range.start,
            block_count,
        });
        coefficient_cursor = claim_stride_coefficients
            .checked_mul(NUM_CLAIMS)
            .and_then(|count| coefficient_cursor.checked_add(count))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("trace benchmark unit offset overflow".into())
            })?;
    }
    let block_variables = num_live_blocks
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("trace benchmark block domain overflow".into()))?
        .trailing_zeros() as usize;
    let column_len = num_live_blocks
        .checked_mul(NUM_CLAIMS * DIGIT_COUNT * (SOURCE_RING_DIMENSION / COEFFICIENT_BLOCK_LEN))
        .ok_or_else(|| AkitaError::InvalidSetup("trace benchmark column span overflow".into()))?;
    let column_variables = column_len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("trace benchmark column domain overflow".into()))?
        .trailing_zeros() as usize;
    let coefficient_variables = COEFFICIENT_BLOCK_LEN.trailing_zeros() as usize;
    let num_variables = coefficient_variables
        .checked_add(column_variables)
        .ok_or_else(|| AkitaError::InvalidSetup("trace benchmark point width overflow".into()))?;
    let trace = PreparedEvaluationTrace {
        groups: vec![PreparedEvaluationTraceGroup {
            block_opening_point: (0..block_variables)
                .map(|index| F::from_u64(101 + index as u64))
                .collect::<Vec<_>>()
                .into(),
            basis,
            source_ring_dimension: SOURCE_RING_DIMENSION,
            opening_ring_dimension: OPENING_RING_DIMENSION,
            coefficient_block_len: COEFFICIENT_BLOCK_LEN,
            opening_digit_weights: vec![F::from_u64(211), F::from_u64(223)].into(),
            inner_trace: (0..SOURCE_RING_DIMENSION)
                .map(|index| F::from_u64(307 + index as u64))
                .collect::<Vec<_>>()
                .into(),
            claim_coefficients: vec![F::from_u64(401), F::from_u64(409)],
            units,
        }],
        num_variables,
    };
    let point = (0..num_variables)
        .map(|index| F::from_u64(503 + index as u64))
        .collect();
    Ok(EvaluationTraceBenchmarkCase { trace, point })
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
    fn unit_tensor_segments_match_separate_unit_contractions() {
        for (blocks, chunks) in [(256, 4), (253, 64), (61, 64)] {
            for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
                let case = evaluation_trace_benchmark_case(blocks, chunks, basis).unwrap();
                let expected = case.evaluate().unwrap();
                let group = case.trace.groups.first().unwrap();
                let groups = group
                    .units
                    .iter()
                    .map(|unit| {
                        let mut separate = group.clone();
                        separate.units = vec![unit.clone()];
                        separate
                    })
                    .collect();
                let separate = PreparedEvaluationTrace {
                    groups,
                    num_variables: case.trace.num_variables,
                };

                assert_eq!(separate.evaluate_at_point(&case.point).unwrap(), expected);
            }
        }
    }

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
            first_claim_coefficient: 4,
            claim_stride_coefficients: 48,
            global_block_start: 1,
            block_count: 2,
        };
        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let trace = PreparedEvaluationTrace {
                groups: vec![PreparedEvaluationTraceGroup {
                    block_opening_point: Arc::clone(&block_point),
                    basis,
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
            let block_weights = basis_weights(&block_point, basis).unwrap();
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
                multilinear_eval(&dense, &point).unwrap(),
                "basis={basis:?}"
            );
        }
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
                for unit in units.clone() {
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
