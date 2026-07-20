//! Canonical evaluation-trace terms shared by the prover and verifier.

use std::sync::Arc;

use akita_algebra::offset_eq::eval_affine_digit_interval;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

use super::{TraceSparseColumn, TraceTable};
use crate::field_reduction::trace_open_ring_row;
use crate::{
    basis_weights, basis_weights_prefix, gadget_row_scalars, BasisMode, FpExtEncoding, LevelParams,
    OpeningClaimsLayout, PreparedOpeningPoint, RelationRangeImagePlan,
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

/// One contiguous physical opening-digit run for a claim inside one witness chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationTraceSegment {
    physical_coefficient_start: usize,
    global_block_start: usize,
    block_count: usize,
}

impl EvaluationTraceSegment {
    /// Flat field-coefficient address of the segment's first opening digit.
    #[must_use]
    pub fn physical_coefficient_start(&self) -> usize {
        self.physical_coefficient_start
    }

    /// First group-global block represented by this chunk segment.
    #[must_use]
    pub fn global_block_start(&self) -> usize {
        self.global_block_start
    }

    /// Exact number of live blocks represented by this chunk segment.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

/// One opening claim's rank-one evaluation-trace factors and physical support.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationTraceTerm<E: FieldCore> {
    coefficient: E,
    block_opening_point: Arc<[E]>,
    basis: BasisMode,
    group_block_count: usize,
    source_ring_dimension: usize,
    opening_digit_weights: Arc<[E]>,
    inner_trace: Arc<[E]>,
    segments: Vec<EvaluationTraceSegment>,
}

impl<E: FieldCore> EvaluationTraceTerm<E> {
    /// Public claim coefficient including any extension-opening-reduction scale.
    #[must_use]
    pub fn coefficient(&self) -> E {
        self.coefficient
    }

    /// Source ring dimension whose coordinates are represented by `inner_trace`.
    #[must_use]
    pub fn source_ring_dimension(&self) -> usize {
        self.source_ring_dimension
    }

    /// Fixed trace values across the source ring coordinates.
    #[must_use]
    pub fn inner_trace(&self) -> &[E] {
        &self.inner_trace
    }

    /// Exact physical opening-digit segments owned by this claim.
    #[must_use]
    pub fn segments(&self) -> &[EvaluationTraceSegment] {
        &self.segments
    }

    fn digit_inner_weights(&self) -> Result<Vec<E>, AkitaError> {
        let len = self
            .opening_digit_weights
            .len()
            .checked_mul(self.inner_trace.len())
            .ok_or_else(|| AkitaError::InvalidSetup("trace digit-inner length overflow".into()))?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(len)
            .map_err(|_| AkitaError::InvalidInput("trace digit-inner allocation failed".into()))?;
        for &digit_weight in self.opening_digit_weights.iter() {
            weights.extend(self.inner_trace.iter().map(|&inner| digit_weight * inner));
        }
        Ok(weights)
    }
}

/// Complete nonempty evaluation-trace weight function over one flat witness domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationTraceWeights<E: FieldCore> {
    terms: Vec<EvaluationTraceTerm<E>>,
    physical_field_len: usize,
    num_vars: usize,
}

impl<E: FieldCore> EvaluationTraceWeights<E> {
    /// Claim terms in authenticated group/claim order.
    #[must_use]
    pub fn terms(&self) -> &[EvaluationTraceTerm<E>] {
        &self.terms
    }

    /// Evaluate the trace-weight MLE directly at one flat coefficient point.
    pub fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        if point.len() != self.num_vars {
            return Err(AkitaError::InvalidSize {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let mut evaluation = E::zero();
        for term in &self.terms {
            let low_bits = term.block_opening_point.len() / 2;
            let low_weights = basis_weights(&term.block_opening_point[..low_bits], term.basis)?;
            let high_weights = basis_weights(&term.block_opening_point[low_bits..], term.basis)?;
            let digit_inner_weights = term.digit_inner_weights()?;
            let block_stride = digit_inner_weights.len();
            let mut term_evaluation = E::zero();
            for segment in &term.segments {
                term_evaluation += eval_affine_digit_interval(
                    point,
                    segment.physical_coefficient_start,
                    segment.global_block_start,
                    segment.block_count,
                    block_stride,
                    &digit_inner_weights,
                    &high_weights,
                    &low_weights,
                )?;
            }
            evaluation += term.coefficient * term_evaluation;
        }
        Ok(evaluation)
    }

    /// Materialize the temporary foldable table consumed by the current Stage-2 prover.
    ///
    /// The semantic source remains this term list. Uniform scalar openings keep the
    /// existing sparse-column storage; extension or mixed-ring openings materialize one
    /// exact live flat prefix directly in the destination ring geometry.
    pub fn materialize_prover_table<F>(
        &self,
        destination_ring_dimension: usize,
        output_scale: E,
    ) -> Result<TraceTable<E>, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        if destination_ring_dimension == 0
            || !destination_ring_dimension.is_power_of_two()
            || !self
                .physical_field_len
                .is_multiple_of(destination_ring_dimension)
        {
            return Err(AkitaError::InvalidSetup(
                "trace destination ring geometry is malformed".into(),
            ));
        }
        let live_destination_columns = self.physical_field_len / destination_ring_dimension;
        let sparse = E::EXT_DEGREE == 1
            && self
                .terms
                .iter()
                .all(|term| term.source_ring_dimension == destination_ring_dimension);
        if sparse {
            let mut columns = Vec::new();
            for term in &self.terms {
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
                        AkitaError::InvalidSetup("trace block stride overflow".into())
                    })?;
                for segment in &term.segments {
                    for local_block in 0..segment.block_count {
                        let global_block = segment.global_block_start + local_block;
                        let block_weight = *block_weights
                            .get(global_block)
                            .ok_or(AkitaError::InvalidProof)?;
                        let block_start = segment
                            .physical_coefficient_start
                            .checked_add(block_stride * local_block)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("trace block address overflow".into())
                            })?;
                        for (digit, &digit_weight) in term.opening_digit_weights.iter().enumerate()
                        {
                            let coefficient_start = block_start
                                .checked_add(digit * term.source_ring_dimension)
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("trace digit address overflow".into())
                                })?;
                            if !coefficient_start.is_multiple_of(destination_ring_dimension) {
                                return Err(AkitaError::InvalidSetup(
                                    "trace sparse column is not ring aligned".into(),
                                ));
                            }
                            let col = coefficient_start / destination_ring_dimension;
                            if col >= live_destination_columns {
                                return Err(AkitaError::InvalidProof);
                            }
                            let factor =
                                output_scale * term.coefficient * block_weight * digit_weight;
                            columns.push(TraceSparseColumn {
                                col,
                                values: term
                                    .inner_trace
                                    .iter()
                                    .map(|&inner| factor * inner)
                                    .collect(),
                            });
                        }
                    }
                }
            }
            return Ok(TraceTable::field_sparse(
                columns,
                live_destination_columns,
                destination_ring_dimension,
            ));
        }

        let mut table = vec![E::zero(); self.physical_field_len];
        for term in &self.terms {
            let block_weights = basis_weights_prefix(
                &term.block_opening_point,
                term.basis,
                term.group_block_count,
            )?;
            let digit_inner_weights = term.digit_inner_weights()?;
            let block_stride = digit_inner_weights.len();
            for segment in &term.segments {
                for local_block in 0..segment.block_count {
                    let global_block = segment.global_block_start + local_block;
                    let factor = output_scale
                        * term.coefficient
                        * *block_weights
                            .get(global_block)
                            .ok_or(AkitaError::InvalidProof)?;
                    let block_start = segment
                        .physical_coefficient_start
                        .checked_add(block_stride * local_block)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("trace block address overflow".into())
                        })?;
                    let block_end = block_start.checked_add(block_stride).ok_or_else(|| {
                        AkitaError::InvalidSetup("trace block end overflow".into())
                    })?;
                    let destination = table
                        .get_mut(block_start..block_end)
                        .ok_or(AkitaError::InvalidProof)?;
                    for (slot, &weight) in destination.iter_mut().zip(&digit_inner_weights) {
                        *slot += factor * weight;
                    }
                }
            }
        }
        Ok(TraceTable::ring_dense(table))
    }
}

/// Apply optional per-claim reduction scales to already normalized coefficients.
pub fn scale_evaluation_trace_claim_coefficients<E: FieldCore>(
    claim_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<Vec<E>, AkitaError> {
    if claim_coefficients.is_empty() {
        return Err(AkitaError::InvalidInput(
            "evaluation trace requires a claim coefficient".into(),
        ));
    }
    if let Some(scales) = claim_scales {
        if scales.len() != claim_coefficients.len() {
            return Err(AkitaError::InvalidSize {
                expected: claim_coefficients.len(),
                actual: scales.len(),
            });
        }
    }
    Ok(claim_coefficients
        .iter()
        .enumerate()
        .map(|(claim, &coefficient)| {
            coefficient
                * claim_scales
                    .and_then(|scales| scales.get(claim).copied())
                    .unwrap_or_else(E::one)
        })
        .collect())
}

/// Inputs to the one evaluation-trace term builder.
pub struct EvaluationTraceWeightInputs<'a, F: FieldCore, E: FieldCore> {
    pub plan: &'a RelationRangeImagePlan,
    pub level_params: &'a LevelParams,
    pub opening_batch: &'a OpeningClaimsLayout,
    pub prepared_points: &'a [PreparedOpeningPoint<F, E>],
    pub claim_coefficients: &'a [E],
    pub basis: BasisMode,
}

/// Build one canonical term per opening claim across every group and witness chunk.
pub fn build_evaluation_trace_weights<F, E, const D: usize>(
    inputs: EvaluationTraceWeightInputs<'_, F, E>,
) -> Result<EvaluationTraceWeights<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let plan = inputs.plan;
    if plan.role_dims().d_a() != D
        || inputs.prepared_points.len() != inputs.opening_batch.num_groups()
        || inputs.claim_coefficients.len() != inputs.opening_batch.num_total_polynomials()
    {
        return Err(AkitaError::InvalidProof);
    }
    let alpha_bits = D.trailing_zeros() as usize;
    let mut terms = Vec::with_capacity(inputs.claim_coefficients.len());
    for group in plan.groups() {
        let group_index = group.group_index();
        let group_params = inputs
            .level_params
            .group_params(inputs.opening_batch, group_index)?;
        let group_layout = inputs.opening_batch.group_layout(group_index)?;
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
        let opening_digit_weights: Arc<[E]> =
            gadget_row_scalars::<F>(group_params.num_digits_open(), group_params.log_basis())
                .into_iter()
                .map(E::lift_base)
                .collect::<Vec<_>>()
                .into();
        let units = plan
            .witness_layout()
            .units()
            .get(group.unit_range())
            .ok_or(AkitaError::InvalidProof)?;
        let claim_range = group.claim_range();
        for (local_claim, claim_index) in claim_range.enumerate() {
            let mut segments = Vec::with_capacity(units.len());
            for unit in units {
                let physical_column_start = unit.e_index(
                    group_layout.num_polynomials(),
                    group_params.num_digits_open(),
                    local_claim,
                    unit.global_block_start(),
                    0,
                )?;
                let physical_coefficient_start = physical_column_start
                    .checked_mul(D)
                    .ok_or_else(|| AkitaError::InvalidSetup("trace address overflow".into()))?;
                let coefficient_count = unit
                    .num_live_blocks()
                    .checked_mul(opening_digit_weights.len())
                    .and_then(|count| count.checked_mul(D))
                    .ok_or_else(|| AkitaError::InvalidSetup("trace segment overflow".into()))?;
                let end = physical_coefficient_start
                    .checked_add(coefficient_count)
                    .ok_or_else(|| AkitaError::InvalidSetup("trace segment end overflow".into()))?;
                if end > plan.digit_witness_domain().live_len() {
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
                block_opening_point: Arc::clone(&block_opening_point),
                basis: inputs.basis,
                group_block_count: group_params.num_live_blocks(),
                source_ring_dimension: D,
                opening_digit_weights: Arc::clone(&opening_digit_weights),
                inner_trace: Arc::clone(&inner_trace),
                segments,
            });
        }
    }
    if terms.len() != inputs.claim_coefficients.len() || terms.is_empty() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EvaluationTraceWeights {
        terms,
        physical_field_len: plan.digit_witness_domain().live_len(),
        num_vars: plan.digit_witness_domain().num_vars(),
    })
}

#[cfg(test)]
#[path = "evaluation_trace_tests.rs"]
mod tests;
