//! Prover-owned evaluation-trace support prepared for Stage 2.

use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::{basis_weights_prefix, EvaluationTraceWeights, TraceSparseColumn, TraceTable};

/// One opening block/digit contribution over contiguous common-coordinate columns.
struct PreparedOpeningSupport<E: FieldCore> {
    first_column: usize,
    factor: E,
    inner_trace_index: usize,
}

/// Canonical prover preparation of the evaluation trace's exact live E support.
///
/// Block, claim, and digit scalars are compiled once. The source-coordinate trace stays
/// factored so the Stage 2 cutover can contract it directly after initial challenges.
/// Until that cutover, `into_stage2_fold_table` is the sole bridge to the existing
/// foldable trace storage.
pub(crate) struct PreparedProverEvaluationTrace<E: FieldCore> {
    opening_support: Vec<PreparedOpeningSupport<E>>,
    source_inner_traces: Vec<std::sync::Arc<[E]>>,
    live_column_count: usize,
    common_relation_witness_coeff_count: usize,
    all_source_rings_match_common: bool,
}

impl<E: FieldCore> PreparedProverEvaluationTrace<E> {
    /// Compile checked semantic trace terms into exact opening support.
    #[tracing::instrument(
        skip_all,
        name = "PreparedProverEvaluationTrace::new",
        fields(
            terms = weights.terms().len(),
            common_relation_witness_coeff_count,
            physical_field_len = weights.physical_field_len()
        )
    )]
    pub(crate) fn new(
        weights: &EvaluationTraceWeights<E>,
        common_relation_witness_coeff_count: usize,
        output_scale: E,
    ) -> Result<Self, AkitaError> {
        if common_relation_witness_coeff_count == 0
            || !common_relation_witness_coeff_count.is_power_of_two()
            || !weights
                .physical_field_len()
                .is_multiple_of(common_relation_witness_coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "evaluation-trace common-coordinate geometry is malformed".into(),
            ));
        }
        let live_column_count = weights.physical_field_len() / common_relation_witness_coeff_count;
        let opening_support_count =
            weights
                .terms()
                .iter()
                .try_fold(0usize, |term_count, term| {
                    term.segments()
                        .iter()
                        .try_fold(term_count, |count, segment| {
                            segment
                                .block_count()
                                .checked_mul(term.opening_digit_weights().len())
                                .and_then(|segment_count| count.checked_add(segment_count))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "evaluation-trace support count overflow".into(),
                                    )
                                })
                        })
                })?;
        let mut opening_support = Vec::new();
        opening_support
            .try_reserve_exact(opening_support_count)
            .map_err(|_| {
                AkitaError::InvalidInput("evaluation-trace support allocation failed".into())
            })?;
        let mut source_inner_traces = Vec::with_capacity(weights.terms().len());
        let mut all_source_rings_match_common = true;
        for term in weights.terms() {
            let source_ring_dimension = term.source_ring_dimension();
            if source_ring_dimension == 0
                || !source_ring_dimension.is_power_of_two()
                || !source_ring_dimension.is_multiple_of(common_relation_witness_coeff_count)
                || term.inner_trace().len() != source_ring_dimension
            {
                return Err(AkitaError::InvalidSetup(
                    "evaluation-trace source ring is incompatible with Stage 2".into(),
                ));
            }
            all_source_rings_match_common &=
                source_ring_dimension == common_relation_witness_coeff_count;
            let block_weights = basis_weights_prefix(
                term.block_opening_point(),
                term.block_opening_basis(),
                term.group_block_count(),
            )?;
            let block_stride = term
                .opening_digit_weights()
                .len()
                .checked_mul(source_ring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("evaluation-trace block stride overflow".into())
                })?;
            let inner_trace_index = source_inner_traces.len();
            source_inner_traces.push(term.shared_inner_trace());
            for segment in term.segments() {
                for local_block in 0..segment.block_count() {
                    let global_block = segment
                        .global_block_start()
                        .checked_add(local_block)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace global block overflow".into(),
                            )
                        })?;
                    let block_weight = *block_weights
                        .get(global_block)
                        .ok_or(AkitaError::InvalidProof)?;
                    let local_block_offset =
                        block_stride.checked_mul(local_block).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace block offset overflow".into(),
                            )
                        })?;
                    let block_start = segment
                        .physical_coefficient_start()
                        .checked_add(local_block_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace block address overflow".into(),
                            )
                        })?;
                    for (digit, &digit_weight) in term.opening_digit_weights().iter().enumerate() {
                        let digit_offset =
                            source_ring_dimension.checked_mul(digit).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace digit offset overflow".into(),
                                )
                            })?;
                        let coefficient_start =
                            block_start.checked_add(digit_offset).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace digit address overflow".into(),
                                )
                            })?;
                        if !coefficient_start.is_multiple_of(common_relation_witness_coeff_count) {
                            return Err(AkitaError::InvalidSetup(
                                "evaluation-trace support is not common-coordinate aligned".into(),
                            ));
                        }
                        let first_column = coefficient_start / common_relation_witness_coeff_count;
                        let column_count =
                            source_ring_dimension / common_relation_witness_coeff_count;
                        let support_end =
                            first_column.checked_add(column_count).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace support range overflow".into(),
                                )
                            })?;
                        if support_end > live_column_count {
                            return Err(AkitaError::InvalidProof);
                        }
                        opening_support.push(PreparedOpeningSupport {
                            first_column,
                            factor: output_scale * term.coefficient() * block_weight * digit_weight,
                            inner_trace_index,
                        });
                    }
                }
            }
        }
        if opening_support.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        if opening_support.len() != opening_support_count {
            return Err(AkitaError::InvalidProof);
        }
        opening_support.sort_unstable_by_key(|support| support.first_column);
        Ok(Self {
            opening_support,
            source_inner_traces,
            live_column_count,
            common_relation_witness_coeff_count,
            all_source_rings_match_common,
        })
    }

    /// Bridge the prepared support to the current Stage 2 state machine.
    ///
    /// The bridge preserves the existing scalar/same-ring sparse policy and dense policy
    /// for extension or mixed source rings. It disappears when Stage 2 consumes prepared
    /// support directly.
    #[tracing::instrument(
        skip_all,
        name = "PreparedProverEvaluationTrace::into_stage2_fold_table",
        fields(
            opening_support = self.opening_support.len(),
            common_relation_witness_coeff_count = self.common_relation_witness_coeff_count,
            live_column_count = self.live_column_count
        )
    )]
    pub(crate) fn into_stage2_fold_table<F>(self) -> Result<TraceTable<E>, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let sparse = E::EXT_DEGREE == 1 && self.all_source_rings_match_common;
        if !sparse {
            let physical_field_len = self
                .live_column_count
                .checked_mul(self.common_relation_witness_coeff_count)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("evaluation-trace fold table length overflow".into())
                })?;
            let mut dense = vec![E::zero(); physical_field_len];
            for support in self.opening_support {
                let source_inner_trace = self
                    .source_inner_traces
                    .get(support.inner_trace_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let coefficient_start = support
                    .first_column
                    .checked_mul(self.common_relation_witness_coeff_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "evaluation-trace fold table address overflow".into(),
                        )
                    })?;
                let coefficient_end = coefficient_start
                    .checked_add(source_inner_trace.len())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "evaluation-trace fold table range overflow".into(),
                        )
                    })?;
                let destination = dense
                    .get_mut(coefficient_start..coefficient_end)
                    .ok_or(AkitaError::InvalidProof)?;
                for (slot, &inner) in destination.iter_mut().zip(source_inner_trace.iter()) {
                    *slot += support.factor * inner;
                }
            }
            return Ok(TraceTable::ring_dense(dense));
        }

        let mut columns = Vec::new();
        for support in self.opening_support {
            let source_inner_trace = self
                .source_inner_traces
                .get(support.inner_trace_index)
                .ok_or(AkitaError::InvalidProof)?;
            for (lane, inner_trace) in source_inner_trace
                .chunks_exact(self.common_relation_witness_coeff_count)
                .enumerate()
            {
                columns.push(TraceSparseColumn {
                    col: support.first_column + lane,
                    values: inner_trace
                        .iter()
                        .map(|&inner| support.factor * inner)
                        .collect(),
                });
            }
        }
        Ok(TraceTable::field_sparse(
            columns,
            self.live_column_count,
            self.common_relation_witness_coeff_count,
        ))
    }
}

#[cfg(test)]
mod tests;
