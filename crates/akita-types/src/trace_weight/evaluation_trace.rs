//! Checked evaluation-trace inputs and canonical prover semantic terms.

use std::ops::Range;
use std::sync::Arc;

use akita_algebra::offset_eq::eval_affine_digit_interval;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

#[cfg(test)]
use crate::basis_weights_prefix;
use crate::field_reduction::trace_open_ring_row;
use crate::{
    basis_weights, gadget_row_scalars, BasisMode, CommitmentRingDims, FlatBooleanDomain,
    FpExtEncoding, LevelParams, OpeningClaimsLayout, PreparedOpeningPoint, WitnessLayout,
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

    /// Opening-point coordinates whose basis weights index group-global blocks.
    #[must_use]
    pub fn block_opening_point(&self) -> &[E] {
        &self.block_opening_point
    }

    /// Basis used to evaluate `block_opening_point`.
    #[must_use]
    pub fn block_opening_basis(&self) -> BasisMode {
        self.basis
    }

    /// Exact number of live blocks in this claim's group.
    #[must_use]
    pub fn group_block_count(&self) -> usize {
        self.group_block_count
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

    /// Share the immutable source-coordinate trace across prepared support entries.
    #[must_use]
    pub fn shared_inner_trace(&self) -> Arc<[E]> {
        Arc::clone(&self.inner_trace)
    }

    /// Opening-digit gadget weights in semantic digit order.
    #[must_use]
    pub fn opening_digit_weights(&self) -> &[E] {
        &self.opening_digit_weights
    }

    /// Exact physical opening-digit segments owned by this claim.
    #[must_use]
    pub fn segments(&self) -> &[EvaluationTraceSegment] {
        &self.segments
    }

    #[cfg(test)]
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

    /// Exact live flat field-coefficient length of the witness domain.
    #[must_use]
    pub fn physical_field_len(&self) -> usize {
        self.physical_field_len
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
pub struct EvaluationTraceWeightInputs<'a, F: FieldCore, E: FieldCore> {
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
    inputs: &EvaluationTraceWeightInputs<'_, F, E>,
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

/// Build one canonical term per opening claim across every group and witness chunk.
pub fn build_evaluation_trace_weights<F, E, const D: usize>(
    inputs: EvaluationTraceWeightInputs<'_, F, E>,
) -> Result<EvaluationTraceWeights<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let group_parameters = prepare_evaluation_trace_group_parameters::<F, E, D>(&inputs)?;
    let mut terms = Vec::with_capacity(inputs.claim_coefficients.len());
    for parameters in group_parameters {
        let group_index = parameters.group_index;
        let group_layout = inputs.opening_batch.group_layout(group_index)?;
        let units = inputs.witness_layout.units_for_group(group_index)?;
        for (local_claim, claim_index) in parameters.claim_range.clone().enumerate() {
            let mut segments = Vec::with_capacity(units.len());
            for &unit in &units {
                let physical_column_start = unit.e_index(
                    group_layout.num_polynomials(),
                    parameters.opening_digit_weights.len(),
                    local_claim,
                    unit.global_block_start(),
                    0,
                )?;
                let physical_coefficient_start = physical_column_start
                    .checked_mul(D)
                    .ok_or_else(|| AkitaError::InvalidSetup("trace address overflow".into()))?;
                let coefficient_count = unit
                    .num_live_blocks()
                    .checked_mul(parameters.opening_digit_weights.len())
                    .and_then(|count| count.checked_mul(D))
                    .ok_or_else(|| AkitaError::InvalidSetup("trace segment overflow".into()))?;
                let end = physical_coefficient_start
                    .checked_add(coefficient_count)
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
                block_opening_point: Arc::clone(&parameters.block_opening_point),
                basis: parameters.basis,
                group_block_count: parameters.group_block_count,
                source_ring_dimension: parameters.source_ring_dimension,
                opening_digit_weights: Arc::clone(&parameters.opening_digit_weights),
                inner_trace: Arc::clone(&parameters.inner_trace),
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
        num_vars: inputs.digit_witness_domain.num_vars(),
    })
}

#[cfg(test)]
#[path = "evaluation_trace_tests.rs"]
mod tests;
