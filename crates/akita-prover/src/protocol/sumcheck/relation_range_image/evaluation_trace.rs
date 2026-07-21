//! Prover-owned evaluation-trace support prepared for Stage 2.

use super::fold_two_round_quad;
use std::{mem, sync::Arc};

#[cfg(test)]
use akita_algebra::offset_eq::eval_affine_digit_interval;
#[cfg(test)]
use akita_algebra::poly::multilinear_eval;
use akita_field::{AkitaError, FieldCore};
use akita_field::{CanonicalField, ExtField, FromPrimitiveInt, Invertible};
#[cfg(test)]
use akita_types::basis_weights;
use akita_types::{
    basis_weights_prefix, prepare_evaluation_trace_group_parameters, BasisMode,
    EvaluationTraceInputs, FpExtEncoding,
};

/// One contiguous physical opening-digit run for a claim inside one witness chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
struct EvaluationTraceSegment {
    physical_coefficient_start: usize,
    global_block_start: usize,
    block_count: usize,
}

/// One opening claim's rank-one evaluation-trace factors and physical support.
#[derive(Clone, Debug, Eq, PartialEq)]
struct EvaluationTraceTerm<E: FieldCore> {
    coefficient: E,
    block_opening_point: Arc<[E]>,
    basis: BasisMode,
    group_block_count: usize,
    source_ring_dimension: usize,
    opening_digit_weights: Arc<[E]>,
    inner_trace: Arc<[E]>,
    segments: Vec<EvaluationTraceSegment>,
}

/// Complete nonempty evaluation-trace weight function over one flat witness domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EvaluationTraceWeights<E: FieldCore> {
    terms: Vec<EvaluationTraceTerm<E>>,
    physical_field_len: usize,
    #[cfg(test)]
    num_vars: usize,
}

impl<E: FieldCore> EvaluationTraceWeights<E> {
    #[cfg(test)]
    fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.num_vars {
            return Err(AkitaError::InvalidSize {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let mut evaluation = E::zero();
        for term in &self.terms {
            if term.source_ring_dimension == 0 || !term.source_ring_dimension.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "trace source ring dimension must be a power of two".into(),
                ));
            }
            let coefficient_bits = term.source_ring_dimension.trailing_zeros() as usize;
            let (coefficient_point, column_point) = point
                .split_at_checked(coefficient_bits)
                .ok_or(AkitaError::InvalidProof)?;
            let inner_trace_evaluation = multilinear_eval(&term.inner_trace, coefficient_point)?;
            let low_bits = term.block_opening_point.len() / 2;
            let low_weights = basis_weights(&term.block_opening_point[..low_bits], term.basis)?;
            let high_weights = basis_weights(&term.block_opening_point[low_bits..], term.basis)?;
            let mut term_evaluation = E::zero();
            for segment in &term.segments {
                if !segment
                    .physical_coefficient_start
                    .is_multiple_of(term.source_ring_dimension)
                {
                    return Err(AkitaError::InvalidSetup(
                        "trace segment is not source-ring aligned".into(),
                    ));
                }
                term_evaluation += eval_affine_digit_interval(
                    column_point,
                    segment.physical_coefficient_start / term.source_ring_dimension,
                    segment.global_block_start,
                    segment.block_count,
                    term.opening_digit_weights.len(),
                    &term.opening_digit_weights,
                    &high_weights,
                    &low_weights,
                )?;
            }
            evaluation += term.coefficient * inner_trace_evaluation * term_evaluation;
        }
        Ok(evaluation)
    }
}

/// Build one canonical prover term per opening claim and witness chunk.
pub(crate) fn build_evaluation_trace_weights<F, E, const D: usize>(
    inputs: EvaluationTraceInputs<'_, F, E>,
) -> Result<EvaluationTraceWeights<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let group_parameters = prepare_evaluation_trace_group_parameters::<F, E, D>(&inputs)?;
    let mut terms = Vec::with_capacity(inputs.claim_coefficients.len());
    for parameters in group_parameters {
        let group_index = parameters.group_index();
        let group_layout = inputs.opening_batch.group_layout(group_index)?;
        let units = inputs.witness_layout.units_for_group(group_index)?;
        for (local_claim, claim_index) in parameters.claim_range().enumerate() {
            let mut segments = Vec::with_capacity(units.len());
            for &unit in &units {
                let physical_column_start = unit.e_index(
                    group_layout.num_polynomials(),
                    parameters.opening_digit_weights().len(),
                    local_claim,
                    unit.global_block_start(),
                    0,
                )?;
                let physical_coefficient_start = physical_column_start
                    .checked_mul(D)
                    .ok_or_else(|| AkitaError::InvalidSetup("trace address overflow".into()))?;
                let coeff_count = unit
                    .num_live_blocks()
                    .checked_mul(parameters.opening_digit_weights().len())
                    .and_then(|count| count.checked_mul(D))
                    .ok_or_else(|| AkitaError::InvalidSetup("trace segment overflow".into()))?;
                let end = physical_coefficient_start
                    .checked_add(coeff_count)
                    .ok_or_else(|| AkitaError::InvalidSetup("trace segment end overflow".into()))?;
                if end > inputs.digit_witness_domain.live_len() {
                    return Err(AkitaError::InvalidProof);
                }
                segments.push(EvaluationTraceSegment {
                    physical_coefficient_start,
                    global_block_start: unit.global_block_start(),
                    block_count: unit.num_live_blocks(),
                });
            }
            terms.push(EvaluationTraceTerm {
                coefficient: *inputs
                    .claim_coefficients
                    .get(claim_index)
                    .ok_or(AkitaError::InvalidProof)?,
                block_opening_point: parameters.shared_block_opening_point(),
                basis: parameters.basis(),
                group_block_count: parameters.group_block_count(),
                source_ring_dimension: parameters.source_ring_dimension(),
                opening_digit_weights: parameters.shared_opening_digit_weights(),
                inner_trace: parameters.shared_inner_trace(),
                segments,
            });
        }
    }
    if terms.len() != inputs.claim_coefficients.len() || terms.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EvaluationTraceWeights {
        terms,
        physical_field_len: inputs.digit_witness_domain.live_len(),
        #[cfg(test)]
        num_vars: inputs.digit_witness_domain.num_vars(),
    })
}

/// One opening block/digit contribution over contiguous common-coordinate lanes.
struct PreparedOpeningSupport<E: FieldCore> {
    first_lane: usize,
    factor: E,
    inner_trace_index: usize,
}

#[derive(Clone)]
struct PreparedLaneTerm<E: FieldCore> {
    factor: E,
    source_index: usize,
    lane: usize,
}

struct PreparedTraceSource<E: FieldCore> {
    values: Vec<E>,
    lane_count: usize,
}

/// Canonical prover preparation of the evaluation trace's exact live E support.
///
/// Block, claim, and digit scalars are compiled once. The source-coordinate trace stays
/// factored while coefficient coordinates are folded; lane challenges then merge the
/// prepared support directly. No full coefficient-domain trace table is materialized.
pub(crate) struct PreparedProverEvaluationTrace<E: FieldCore> {
    lane_terms: Vec<Vec<PreparedLaneTerm<E>>>,
    sources: Vec<PreparedTraceSource<E>>,
    live_lane_count: usize,
    coeff_count: usize,
}

impl<E: FieldCore> PreparedProverEvaluationTrace<E> {
    #[cfg(test)]
    pub(crate) fn from_dense(dense: Vec<E>, live_lane_count: usize, coeff_count: usize) -> Self {
        assert_eq!(dense.len(), live_lane_count * coeff_count);
        let mut lane_terms = vec![Vec::new(); live_lane_count];
        let sources = dense
            .chunks_exact(coeff_count)
            .enumerate()
            .map(|(lane, values)| {
                lane_terms[lane].push(PreparedLaneTerm {
                    factor: E::one(),
                    source_index: lane,
                    lane: 0,
                });
                PreparedTraceSource {
                    values: values.to_vec(),
                    lane_count: 1,
                }
            })
            .collect();
        Self {
            lane_terms,
            sources,
            live_lane_count,
            coeff_count,
        }
    }

    /// Compile checked semantic trace terms into exact opening support.
    #[tracing::instrument(
        skip_all,
        name = "PreparedProverEvaluationTrace::new",
        fields(
            terms = weights.terms.len(),
            coeff_count,
            physical_field_len = weights.physical_field_len
        )
    )]
    pub(crate) fn new(
        weights: &EvaluationTraceWeights<E>,
        coeff_count: usize,
        output_scale: E,
    ) -> Result<Self, AkitaError> {
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || !weights.physical_field_len.is_multiple_of(coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "evaluation-trace common-coordinate geometry is malformed".into(),
            ));
        }
        let live_lane_count = weights.physical_field_len / coeff_count;
        let opening_support_count = weights.terms.iter().try_fold(0usize, |term_count, term| {
            term.segments.iter().try_fold(term_count, |count, segment| {
                segment
                    .block_count
                    .checked_mul(term.opening_digit_weights.len())
                    .and_then(|segment_count| count.checked_add(segment_count))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("evaluation-trace support count overflow".into())
                    })
            })
        })?;
        let mut opening_support = Vec::new();
        opening_support
            .try_reserve_exact(opening_support_count)
            .map_err(|_| {
                AkitaError::InvalidInput("evaluation-trace support allocation failed".into())
            })?;
        let mut source_inner_traces = Vec::with_capacity(weights.terms.len());
        for term in &weights.terms {
            let source_ring_dimension = term.source_ring_dimension;
            if source_ring_dimension == 0
                || !source_ring_dimension.is_power_of_two()
                || !source_ring_dimension.is_multiple_of(coeff_count)
                || term.inner_trace.len() != source_ring_dimension
            {
                return Err(AkitaError::InvalidSetup(
                    "evaluation-trace source ring is incompatible with Stage 2".into(),
                ));
            }
            let block_weights = basis_weights_prefix(
                &term.block_opening_point,
                term.basis,
                term.group_block_count,
            )?;
            let block_stride = term
                .opening_digit_weights
                .len()
                .checked_mul(source_ring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("evaluation-trace block stride overflow".into())
                })?;
            let inner_trace_index = source_inner_traces.len();
            source_inner_traces.push(Arc::clone(&term.inner_trace));
            for segment in &term.segments {
                for local_block in 0..segment.block_count {
                    let global_block = segment
                        .global_block_start
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
                        .physical_coefficient_start
                        .checked_add(local_block_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace block address overflow".into(),
                            )
                        })?;
                    for (digit, &digit_weight) in term.opening_digit_weights.iter().enumerate() {
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
                        if !coefficient_start.is_multiple_of(coeff_count) {
                            return Err(AkitaError::InvalidSetup(
                                "evaluation-trace support is not common-coordinate aligned".into(),
                            ));
                        }
                        let first_lane = coefficient_start / coeff_count;
                        let column_count = source_ring_dimension / coeff_count;
                        let support_end =
                            first_lane.checked_add(column_count).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace support range overflow".into(),
                                )
                            })?;
                        if support_end > live_lane_count {
                            return Err(AkitaError::InvalidProof);
                        }
                        opening_support.push(PreparedOpeningSupport {
                            first_lane,
                            factor: output_scale * term.coefficient * block_weight * digit_weight,
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
        let sources = source_inner_traces
            .into_iter()
            .map(|values| PreparedTraceSource {
                lane_count: values.len() / coeff_count,
                values: values.as_ref().to_vec(),
            })
            .collect::<Vec<_>>();
        let mut lane_terms = vec![Vec::new(); live_lane_count];
        for support in opening_support {
            let source = sources
                .get(support.inner_trace_index)
                .ok_or(AkitaError::InvalidProof)?;
            for source_lane in 0..source.lane_count {
                let target_lane = support.first_lane.checked_add(source_lane).ok_or_else(|| {
                    AkitaError::InvalidSetup("evaluation-trace lane overflow".into())
                })?;
                lane_terms
                    .get_mut(target_lane)
                    .ok_or(AkitaError::InvalidProof)?
                    .push(PreparedLaneTerm {
                        factor: support.factor,
                        source_index: support.inner_trace_index,
                        lane: source_lane,
                    });
            }
        }
        Ok(Self {
            lane_terms,
            sources,
            live_lane_count,
            coeff_count,
        })
    }

    #[inline]
    fn values_in_lane<const N: usize>(&self, lane: usize, coefficients: [usize; N]) -> [E; N] {
        let mut values = [E::zero(); N];
        let Some(terms) = self.lane_terms.get(lane) else {
            return values;
        };
        if let [term] = terms.as_slice() {
            let Some(source) = self.sources.get(term.source_index) else {
                return values;
            };
            let Some(source_lane_start) = term.lane.checked_mul(self.coeff_count) else {
                return values;
            };
            for (value, coefficient) in values.iter_mut().zip(coefficients) {
                if let Some(source_value) = source.values.get(source_lane_start + coefficient) {
                    *value = term.factor * *source_value;
                }
            }
            return values;
        }
        for term in terms {
            let Some(source) = self.sources.get(term.source_index) else {
                continue;
            };
            let source_lane_start = term.lane * self.coeff_count;
            for (value, coefficient) in values.iter_mut().zip(coefficients) {
                if let Some(source_value) = source.values.get(source_lane_start + coefficient) {
                    *value += term.factor * *source_value;
                }
            }
        }
        values
    }

    #[inline]
    pub(crate) fn get(&self, lane: usize, coefficient: usize, coeff_count: usize) -> E {
        debug_assert_eq!(self.coeff_count, coeff_count);
        let [value] = self.values_in_lane(lane, [coefficient]);
        value
    }

    #[inline]
    pub(crate) fn pair_at_lanes(
        &self,
        lane0: usize,
        lane1: usize,
        coefficient: usize,
        coeff_count: usize,
    ) -> (E, E) {
        (
            self.get(lane0, coefficient, coeff_count),
            self.get(lane1, coefficient, coeff_count),
        )
    }

    #[inline]
    pub(crate) fn pair_from_flat_index(&self, index0: usize, coeff_count: usize) -> (E, E) {
        debug_assert_eq!(self.coeff_count, coeff_count);
        debug_assert!(coeff_count.is_power_of_two());
        let coefficient0 = index0 & (coeff_count - 1);
        let lane0 = index0 >> coeff_count.trailing_zeros();
        if coefficient0 + 1 < coeff_count {
            let [value0, value1] = self.values_in_lane(lane0, [coefficient0, coefficient0 + 1]);
            (value0, value1)
        } else {
            (
                self.get(lane0, coefficient0, coeff_count),
                self.get(lane0 + 1, 0, coeff_count),
            )
        }
    }

    pub(crate) fn quad_at(&self, lane: usize, base: usize, coeff_count: usize) -> [E; 4] {
        debug_assert_eq!(self.coeff_count, coeff_count);
        self.values_in_lane(lane, [base, base + 1, base + 2, base + 3])
    }

    pub(crate) fn validate_len(&self, witness_len: usize) -> Result<(), AkitaError> {
        let actual = self
            .live_lane_count
            .checked_mul(self.coeff_count)
            .ok_or_else(|| AkitaError::InvalidSetup("evaluation-trace length overflow".into()))?;
        if actual != witness_len {
            return Err(AkitaError::InvalidSize {
                expected: witness_len,
                actual,
            });
        }
        if self.lane_terms.len() != self.live_lane_count
            || self.sources.iter().any(|source| {
                source.values.len() != source.lane_count.saturating_mul(self.coeff_count)
            })
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    pub(crate) fn fold_coefficients(&mut self, challenge: E) {
        let coeff_count = self.coeff_count;
        debug_assert!(coeff_count.is_power_of_two() && coeff_count >= 2);
        let next_coeff_count = coeff_count / 2;
        for source in &mut self.sources {
            for lane in 0..source.lane_count {
                let source_start = lane * coeff_count;
                let target_start = lane * next_coeff_count;
                for coefficient in 0..next_coeff_count {
                    let left = source.values[source_start + 2 * coefficient];
                    let right = source.values[source_start + 2 * coefficient + 1];
                    source.values[target_start + coefficient] = left + challenge * (right - left);
                }
            }
            source.values.truncate(source.lane_count * next_coeff_count);
        }
        self.coeff_count = next_coeff_count;
    }

    pub(crate) fn fold_two_coefficients(&mut self, r0: E, r1: E) {
        let coeff_count = self.coeff_count;
        debug_assert!(coeff_count.is_power_of_two() && coeff_count >= 4);
        let next_coeff_count = coeff_count / 4;
        for source in &mut self.sources {
            for lane in 0..source.lane_count {
                let source_start = lane * coeff_count;
                let target_start = lane * next_coeff_count;
                for coefficient in 0..next_coeff_count {
                    let base = source_start + 4 * coefficient;
                    source.values[target_start + coefficient] = fold_two_round_quad(
                        source.values[base],
                        source.values[base + 1],
                        source.values[base + 2],
                        source.values[base + 3],
                        r0,
                        r1,
                    );
                }
            }
            source.values.truncate(source.lane_count * next_coeff_count);
        }
        self.coeff_count = next_coeff_count;
    }

    pub(crate) fn fold_lanes(&mut self, challenge: E) {
        let next_live_lane_count = self.live_lane_count.div_ceil(2);
        let even_scale = E::one() - challenge;
        let mut source_lanes = mem::take(&mut self.lane_terms).into_iter();
        let mut folded = Vec::with_capacity(next_live_lane_count);
        while let Some(mut even_terms) = source_lanes.next() {
            for term in &mut even_terms {
                term.factor *= even_scale;
            }
            if let Some(mut odd_terms) = source_lanes.next() {
                for term in &mut odd_terms {
                    term.factor *= challenge;
                }
                even_terms.reserve(odd_terms.len());
                even_terms.append(&mut odd_terms);
            }
            folded.push(even_terms);
        }
        debug_assert_eq!(folded.len(), next_live_lane_count);
        self.lane_terms = folded;
        self.live_lane_count = next_live_lane_count;
    }

    #[cfg(test)]
    pub(crate) fn materialize_dense(&self) -> Vec<E> {
        (0..self.live_lane_count)
            .flat_map(|lane| {
                (0..self.coeff_count)
                    .map(move |coefficient| self.get(lane, coefficient, self.coeff_count))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
