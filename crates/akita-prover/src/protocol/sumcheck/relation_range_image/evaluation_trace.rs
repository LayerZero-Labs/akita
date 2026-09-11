//! Prover-owned evaluation-trace support prepared for Stage 2.

use super::fold_two_round_quad;
use std::num::NonZeroUsize;
#[cfg(test)]
use std::ops::Range;
use std::sync::Arc;

use akita_error::AkitaError;

use akita_types::{
    basis_weights_prefix, prepare_evaluation_trace_group_parameters, BasisMode,
    CoefficientPackingStage2Source, CoefficientPackingStage2Terms, EvaluationTraceInputs,
    FpExtEncoding,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

/// One contiguous physical opening-digit run for a claim inside one witness chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
struct EvaluationTraceSegment {
    physical_coefficient_start: usize,
    global_block_start: usize,
    block_count: usize,
}

/// One opening claim's rank-one evaluation-trace factors and physical support.
#[derive(Clone, Debug, Eq, PartialEq)]
struct EvaluationTraceTerm<E: Field> {
    coefficient: E,
    block_opening_point: Arc<[E]>,
    basis: BasisMode,
    group_block_count: usize,
    source_ring_dimension: usize,
    opening_ring_dimension: usize,
    coefficient_block_len: usize,
    opening_digit_weights: Arc<[E]>,
    inner_trace: Arc<[E]>,
    segments: Vec<EvaluationTraceSegment>,
}

/// Complete nonempty evaluation-trace weight function over one flat witness domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EvaluationTraceWeights<E: Field> {
    terms: Vec<EvaluationTraceTerm<E>>,
    physical_field_len: usize,
    #[cfg(test)]
    num_vars: usize,
}

/// Build one canonical prover term per opening claim and witness chunk.
pub(crate) fn build_evaluation_trace_weights<F, E>(
    inputs: EvaluationTraceInputs<'_, F, E>,
) -> Result<EvaluationTraceWeights<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring,
{
    let group_parameters = prepare_evaluation_trace_group_parameters::<F, E>(&inputs)?;
    let mut terms = Vec::with_capacity(inputs.claim_coefficients.len());
    for parameters in group_parameters {
        let group_index = parameters.group_index();
        let group_dims = inputs
            .level_params
            .group_role_dims(inputs.opening_batch, group_index)?;
        let group_layout = inputs.opening_batch.group_layout(group_index)?;
        let units = inputs.witness_layout.units_for_group(group_index)?;
        for (local_claim, claim_index) in parameters.claim_range().enumerate() {
            let mut segments = Vec::with_capacity(units.clone().count());
            for unit in units.clone() {
                if unit.num_live_blocks() == 0 {
                    continue;
                }
                let physical_coefficient_start = unit.e_coefficient_index(
                    group_dims.d_d(),
                    group_layout.num_polynomials(),
                    parameters.opening_digit_weights().len(),
                    local_claim,
                    unit.global_block_start(),
                    0,
                    0,
                    0,
                )?;
                let coeff_count = unit
                    .num_live_blocks()
                    .checked_mul(parameters.opening_digit_weights().len())
                    .and_then(|count| count.checked_mul(group_dims.d_a()))
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
                opening_ring_dimension: group_dims.d_d(),
                coefficient_block_len: inputs.relation_coefficient_block_len,
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
struct PreparedOpeningSupport<E: Field> {
    first_lane: usize,
    source_lane_start: usize,
    lane_count: usize,
    factor: E,
    inner_trace_index: usize,
}

#[derive(Clone)]
pub(super) struct PreparedLaneTerm<E: Field> {
    pub(super) factor: E,
    pub(super) source_index: usize,
    pub(super) lane: usize,
}

#[derive(Clone)]
pub(super) struct PreparedPackingSegment<E: Field> {
    pub(super) factor: E,
    pub(super) source_index: usize,
    pub(super) target_lane_start: usize,
    pub(super) source_lane_start: usize,
    pub(super) lane_count: usize,
}

pub(super) struct PreparedPackingLaneMap<E: Field> {
    pub(super) segments: Vec<PreparedPackingSegment<E>>,
    pub(super) lane_to_segment: Vec<Option<NonZeroUsize>>,
}

impl<E: Field> PreparedPackingLaneMap<E> {
    fn add_segment(&mut self, lane: usize, segment: usize) -> Result<(), AkitaError> {
        let slot = self
            .lane_to_segment
            .get_mut(lane)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.is_some() {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing segments overlap".into(),
            ));
        }
        let encoded = segment
            .checked_add(1)
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| AkitaError::InvalidSetup("packing segment index overflow".into()))?;
        *slot = Some(encoded);
        Ok(())
    }
}

pub(super) enum PreparedLaneWeights<E: Field> {
    Sparse(Vec<Vec<PreparedLaneTerm<E>>>),
    Packing(PreparedPackingLaneMap<E>),
    Dense(Vec<E>),
}

pub(super) struct PreparedTraceSource<E: Field> {
    pub(super) values: Vec<E>,
    pub(super) lane_count: usize,
}

/// One contiguous source-to-witness contribution to a structured linear term.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StructuredLinearSegment {
    pub(crate) physical_coefficient_start: usize,
    pub(crate) source_coefficient_start: usize,
    pub(crate) coefficient_count: usize,
}

/// One factored linear term supported on selected witness coefficient ranges.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StructuredLinearTerm<E: Field> {
    pub(crate) factor: E,
    pub(crate) source_index: usize,
    pub(crate) segment_range: Range<usize>,
}

/// Method-neutral structured linear weights over one flat witness domain.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StructuredLinearWeights<E: Field> {
    pub(crate) sources: Vec<Arc<[E]>>,
    pub(crate) segments: Vec<StructuredLinearSegment>,
    pub(crate) terms: Vec<StructuredLinearTerm<E>>,
    pub(crate) physical_field_len: usize,
}

/// Canonical prover preparation of exact structured linear support.
///
/// Scalar factors are compiled once. Source-coordinate vectors stay factored while
/// coefficient coordinates are folded; lane challenges then merge the prepared support
/// directly. No full coefficient-domain weight table is materialized.
pub(crate) struct PreparedProverLinearTerms<E: Field> {
    pub(super) lane_weights: PreparedLaneWeights<E>,
    pub(super) sources: Vec<PreparedTraceSource<E>>,
    pub(super) live_lane_count: usize,
    pub(super) coeff_count: usize,
}

impl<E: Field> PreparedProverLinearTerms<E> {
    #[cfg(test)]
    pub(crate) fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub(crate) fn final_value(&self) -> Result<E, AkitaError> {
        if self.live_lane_count != 1 || self.coeff_count != 1 {
            return Err(AkitaError::InvalidProof);
        }
        Ok(self.get(0, 0, 1))
    }

    /// A trace of the given geometry whose weight function is identically zero.
    ///
    /// Used by virtual-only stage-2 instances that carry no committed
    /// evaluation-trace term: every lane has empty support, so `get` returns
    /// zero everywhere, coefficient/lane folds are no-ops over the empty
    /// source set, and [`Self::final_value`] resolves to zero once folding
    /// completes.
    pub(crate) fn zero(live_lane_count: usize, coeff_count: usize) -> Self {
        Self {
            lane_weights: PreparedLaneWeights::Sparse(vec![Vec::new(); live_lane_count]),
            sources: Vec::new(),
            live_lane_count,
            coeff_count,
        }
    }

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
            lane_weights: PreparedLaneWeights::Sparse(lane_terms),
            sources,
            live_lane_count,
            coeff_count,
        }
    }

    /// Compile checked semantic trace terms into exact opening support.
    #[tracing::instrument(
        skip_all,
        name = "PreparedProverLinearTerms::from_evaluation_trace",
        fields(
            terms = weights.terms.len(),
            coeff_count,
            physical_field_len = weights.physical_field_len
        )
    )]
    pub(crate) fn from_evaluation_trace(
        weights: &EvaluationTraceWeights<E>,
        coeff_count: usize,
        output_scale: E,
    ) -> Result<Self, AkitaError> {
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || weights.physical_field_len == 0
            || !weights.physical_field_len.is_multiple_of(coeff_count)
            || weights.terms.is_empty()
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
                    .and_then(|count| {
                        count.checked_mul(term.source_ring_dimension / term.opening_ring_dimension)
                    })
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
                || term.opening_ring_dimension == 0
                || !source_ring_dimension.is_multiple_of(term.opening_ring_dimension)
                || !term.opening_ring_dimension.is_multiple_of(coeff_count)
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
                .checked_mul(term.source_ring_dimension)
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
                    let role_subcolumns = source_ring_dimension / term.opening_ring_dimension;
                    for role_subcolumn in 0..role_subcolumns {
                        for (digit, &digit_weight) in term.opening_digit_weights.iter().enumerate()
                        {
                            let digit_offset = role_subcolumn
                                .checked_mul(term.opening_digit_weights.len())
                                .and_then(|offset| offset.checked_add(digit))
                                .and_then(|offset| offset.checked_mul(term.opening_ring_dimension))
                                .ok_or_else(|| {
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
                                    "evaluation-trace support is not common-coordinate aligned"
                                        .into(),
                                ));
                            }
                            let first_lane = coefficient_start / coeff_count;
                            let column_count = term.opening_ring_dimension / coeff_count;
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
                                source_lane_start: role_subcolumn * term.opening_ring_dimension
                                    / coeff_count,
                                lane_count: column_count,
                                factor: output_scale
                                    * term.coefficient
                                    * block_weight
                                    * digit_weight,
                                inner_trace_index,
                            });
                        }
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
            if support.source_lane_start + support.lane_count > source.lane_count {
                return Err(AkitaError::InvalidProof);
            }
            for lane_offset in 0..support.lane_count {
                let source_lane = support.source_lane_start + lane_offset;
                let target_lane = support.first_lane.checked_add(lane_offset).ok_or_else(|| {
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
            lane_weights: PreparedLaneWeights::Sparse(lane_terms),
            sources,
            live_lane_count,
            coeff_count,
        })
    }

    /// Consume canonical coefficient-packing terms into the Stage 2 engine.
    pub(crate) fn from_coefficient_packing(
        weights: CoefficientPackingStage2Terms<E>,
    ) -> Result<Self, AkitaError> {
        let physical_field_len = weights.physical_field_len();
        let coeff_count = weights.relation_coefficient_block_len();
        let (source_values, segments, terms) = weights.into_linear_parts();
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || physical_field_len == 0
            || !physical_field_len.is_multiple_of(coeff_count)
            || terms.is_empty()
        {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing linear geometry is malformed".into(),
            ));
        }
        let sources = source_values
            .into_iter()
            .map(|values| {
                if values.is_empty() || !values.len().is_multiple_of(coeff_count) {
                    return Err(AkitaError::InvalidSetup(
                        "coefficient-packing source geometry is malformed".into(),
                    ));
                }
                Ok(PreparedTraceSource {
                    lane_count: values.len() / coeff_count,
                    values,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let live_lane_count = physical_field_len / coeff_count;
        let mut packing = PreparedPackingLaneMap {
            segments: Vec::new(),
            lane_to_segment: vec![None; live_lane_count],
        };
        for term in terms {
            let source_index = match term.source() {
                CoefficientPackingStage2Source::DirectOpening => 0,
                CoefficientPackingStage2Source::PackingZ => 1,
            };
            let source = sources.get(source_index).ok_or(AkitaError::InvalidProof)?;
            let term_segments = segments
                .get(term.segments())
                .ok_or(AkitaError::InvalidProof)?;
            if term_segments.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "coefficient-packing term has no support".into(),
                ));
            }
            for segment in term_segments {
                let physical = segment.physical_coefficients();
                let source_range = segment.source_coefficients();
                if physical.len() != source_range.len()
                    || physical.is_empty()
                    || !physical.start.is_multiple_of(coeff_count)
                    || !physical.len().is_multiple_of(coeff_count)
                    || !source_range.start.is_multiple_of(coeff_count)
                    || physical.end > physical_field_len
                    || source_range.end > source.values.len()
                {
                    return Err(AkitaError::InvalidSetup(
                        "coefficient-packing segment is unaligned or out of bounds".into(),
                    ));
                }
                let target_lane_start = physical.start / coeff_count;
                let source_lane_start = source_range.start / coeff_count;
                let lane_count = physical.len() / coeff_count;
                let segment_index = packing.segments.len();
                packing.segments.push(PreparedPackingSegment {
                    factor: term.factor(),
                    source_index,
                    target_lane_start,
                    source_lane_start,
                    lane_count,
                });
                for lane_offset in 0..lane_count {
                    let target_lane =
                        target_lane_start.checked_add(lane_offset).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "coefficient-packing target lane overflow".into(),
                            )
                        })?;
                    let source_lane =
                        source_lane_start.checked_add(lane_offset).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "coefficient-packing source lane overflow".into(),
                            )
                        })?;
                    if source_lane >= source.lane_count {
                        return Err(AkitaError::InvalidProof);
                    }
                    packing.add_segment(target_lane, segment_index)?;
                }
            }
        }
        Ok(Self {
            lane_weights: PreparedLaneWeights::Packing(packing),
            sources,
            live_lane_count,
            coeff_count,
        })
    }

    /// Compile arbitrary checked source segments into the shared Stage 2 engine.
    #[cfg(test)]
    pub(crate) fn from_structured_weights(
        weights: &StructuredLinearWeights<E>,
        coeff_count: usize,
    ) -> Result<Self, AkitaError> {
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || weights.physical_field_len == 0
            || !weights.physical_field_len.is_multiple_of(coeff_count)
            || weights.sources.is_empty()
            || weights.terms.is_empty()
        {
            return Err(AkitaError::InvalidSetup(
                "structured linear common-coordinate geometry is malformed".into(),
            ));
        }
        let live_lane_count = weights.physical_field_len / coeff_count;
        let sources = weights
            .sources
            .iter()
            .map(|source| {
                if source.is_empty() || !source.len().is_multiple_of(coeff_count) {
                    return Err(AkitaError::InvalidSetup(
                        "structured linear source geometry is malformed".into(),
                    ));
                }
                Ok(PreparedTraceSource {
                    values: source.as_ref().to_vec(),
                    lane_count: source.len() / coeff_count,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut lane_terms = vec![Vec::new(); live_lane_count];
        for term in &weights.terms {
            let source = weights
                .sources
                .get(term.source_index)
                .ok_or(AkitaError::InvalidProof)?;
            let segments = weights
                .segments
                .get(term.segment_range.clone())
                .ok_or(AkitaError::InvalidProof)?;
            if segments.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "structured linear source geometry is malformed".into(),
                ));
            }
            let source_lane_count = source.len() / coeff_count;
            for segment in segments {
                let target_end = segment
                    .physical_coefficient_start
                    .checked_add(segment.coefficient_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("structured linear target range overflow".into())
                    })?;
                let source_end = segment
                    .source_coefficient_start
                    .checked_add(segment.coefficient_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("structured linear source range overflow".into())
                    })?;
                if segment.coefficient_count == 0
                    || !segment.coefficient_count.is_multiple_of(coeff_count)
                    || !segment
                        .physical_coefficient_start
                        .is_multiple_of(coeff_count)
                    || !segment.source_coefficient_start.is_multiple_of(coeff_count)
                    || target_end > weights.physical_field_len
                    || source_end > source.len()
                {
                    return Err(AkitaError::InvalidSetup(
                        "structured linear segment is unaligned or out of bounds".into(),
                    ));
                }
                let target_lane_start = segment.physical_coefficient_start / coeff_count;
                let source_lane_start = segment.source_coefficient_start / coeff_count;
                let lane_count = segment.coefficient_count / coeff_count;
                for lane_offset in 0..lane_count {
                    let target_lane =
                        target_lane_start.checked_add(lane_offset).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "structured linear target lane overflow".into(),
                            )
                        })?;
                    let source_lane =
                        source_lane_start.checked_add(lane_offset).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "structured linear source lane overflow".into(),
                            )
                        })?;
                    if source_lane >= source_lane_count {
                        return Err(AkitaError::InvalidProof);
                    }
                    lane_terms
                        .get_mut(target_lane)
                        .ok_or(AkitaError::InvalidProof)?
                        .push(PreparedLaneTerm {
                            factor: term.factor,
                            source_index: term.source_index,
                            lane: source_lane,
                        });
                }
            }
        }
        Ok(Self {
            lane_weights: PreparedLaneWeights::Sparse(lane_terms),
            sources,
            live_lane_count,
            coeff_count,
        })
    }

    /// Add another checked structured term set over the same witness domain.
    pub(crate) fn merge(&mut self, other: Self) -> Result<(), AkitaError> {
        if self.live_lane_count != other.live_lane_count || self.coeff_count != other.coeff_count {
            return Err(AkitaError::InvalidSize {
                expected: self.live_lane_count * self.coeff_count,
                actual: other.live_lane_count * other.coeff_count,
            });
        }
        let Self {
            lane_weights: source_weights,
            sources,
            ..
        } = other;
        self.sources
            .len()
            .checked_add(sources.len())
            .ok_or_else(|| AkitaError::InvalidSetup("packing source index overflow".into()))?;
        match (&self.lane_weights, &source_weights) {
            (PreparedLaneWeights::Sparse(_), PreparedLaneWeights::Sparse(_)) => {}
            (PreparedLaneWeights::Packing(target), PreparedLaneWeights::Packing(source)) => {
                target
                    .segments
                    .len()
                    .checked_add(source.segments.len())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("packing segment index overflow".into())
                    })?;
                if target
                    .lane_to_segment
                    .iter()
                    .zip(&source.lane_to_segment)
                    .any(|(target, source)| target.is_some() && source.is_some())
                {
                    return Err(AkitaError::InvalidSetup(
                        "coefficient-packing segments overlap".into(),
                    ));
                }
            }
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "cannot merge different Stage 2 weight representations".into(),
                ));
            }
        }

        let source_offset = self.sources.len();
        self.sources.extend(sources);
        match (&mut self.lane_weights, source_weights) {
            (PreparedLaneWeights::Sparse(target), PreparedLaneWeights::Sparse(source)) => {
                for (target, terms) in target.iter_mut().zip(source) {
                    target.extend(terms.into_iter().map(|mut term| {
                        term.source_index += source_offset;
                        term
                    }));
                }
            }
            (PreparedLaneWeights::Packing(target), PreparedLaneWeights::Packing(mut source)) => {
                let segment_offset = target.segments.len();
                for segment in &mut source.segments {
                    segment.source_index += source_offset;
                }
                target.segments.extend(source.segments);
                for (target, source) in target
                    .lane_to_segment
                    .iter_mut()
                    .zip(source.lane_to_segment)
                {
                    if let Some(source) = source {
                        *target = NonZeroUsize::new(source.get() + segment_offset);
                    }
                }
            }
            _ => unreachable!("weight representations were preflighted above"),
        }
        Ok(())
    }

    #[inline]
    fn values_in_lane<const N: usize>(&self, lane: usize, coefficients: [usize; N]) -> [E; N] {
        self.resolve_lane(lane).evaluated_values(coefficients)
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
    pub(crate) fn pair_from_flat_index(&self, index0: usize) -> (E, E) {
        let coeff_count = self.coeff_count;
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
        let lane_shape_is_valid = match &self.lane_weights {
            PreparedLaneWeights::Sparse(terms) => terms.len() == self.live_lane_count,
            PreparedLaneWeights::Packing(packing) => {
                packing.lane_to_segment.len() == self.live_lane_count
            }
            PreparedLaneWeights::Dense(values) => {
                self.coeff_count == 1 && values.len() == self.live_lane_count
            }
        };
        if !lane_shape_is_valid
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
        if !matches!(self.lane_weights, PreparedLaneWeights::Dense(_)) {
            debug_assert_eq!(self.coeff_count, 1);
            let dense = (0..self.live_lane_count)
                .map(|lane| self.get(lane, 0, 1))
                .collect();
            self.lane_weights = PreparedLaneWeights::Dense(dense);
            self.sources.clear();
        }
        let next_live_lane_count = self.live_lane_count.div_ceil(2);
        let PreparedLaneWeights::Dense(values) = &mut self.lane_weights else {
            unreachable!("lane weights were materialized above");
        };
        let even_scale = E::one() - challenge;
        for target in 0..next_live_lane_count {
            let source = 2 * target;
            let left = values[source];
            values[target] = if let Some(&right) = values.get(source + 1) {
                left + challenge * (right - left)
            } else {
                even_scale * left
            }
        }
        values.truncate(next_live_lane_count);
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
