//! Exact synthetic evaluation-trace fixture for the production contraction kernel.

use std::sync::Arc;

use crate::relation::evaluation_trace::{
    prepare_trace_contraction, PreparedEvaluationTrace, PreparedEvaluationTraceGroup,
    PreparedEvaluationTraceUnit,
};
use akita_error::AkitaError;
use akita_types::BasisMode;
use jolt_field::Ring;

/// Exact synthetic trace fixture for production-kernel benchmarks.
pub struct EvaluationTraceBenchmarkCase {
    pub(crate) trace: PreparedEvaluationTrace<jolt_field::Prime128OffsetA7F7>,
    pub(crate) point: Vec<jolt_field::Prime128OffsetA7F7>,
}

impl EvaluationTraceBenchmarkCase {
    /// Evaluate the prepared trace at its fixed benchmark point.
    pub fn evaluate(&self) -> Result<jolt_field::Prime128OffsetA7F7, AkitaError> {
        self.trace.evaluate_at_point(&self.point)
    }
}

/// Build a checked trace benchmark with two claims, two opening digits, and a
/// D128 source split into D64 coefficient blocks.
pub fn evaluation_trace_benchmark_case(
    num_live_blocks: usize,
    witness_chunks: usize,
    basis: BasisMode,
) -> Result<EvaluationTraceBenchmarkCase, AkitaError> {
    use akita_types::dyadic_block_ranges;
    use jolt_field::Prime128OffsetA7F7 as F;

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
    let block_opening_point: Arc<[F]> = (0..block_variables)
        .map(|index| F::from_u64(101 + index as u64))
        .collect::<Vec<_>>()
        .into();
    let contraction =
        prepare_trace_contraction(&block_opening_point, basis, &units, COEFFICIENT_BLOCK_LEN)?;
    let trace = PreparedEvaluationTrace {
        groups: vec![PreparedEvaluationTraceGroup {
            block_opening_point,
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
            contraction,
        }],
        num_variables,
    };
    let point = (0..num_variables)
        .map(|index| F::from_u64(503 + index as u64))
        .collect();
    Ok(EvaluationTraceBenchmarkCase { trace, point })
}
