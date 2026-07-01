//! Stage-2 wiring helpers for the fused trace term.

use std::marker::PhantomData;

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

use super::build::{
    build_trace_weight_compact_field_sparse_scaled, build_trace_weight_compact_ring_terms_scaled,
};
use super::trace_table::TraceTable;
use crate::{
    embed_ring_subfield_scalar, BasisMode, FpExtEncoding, LevelParams, OpeningBatchShape,
    PreparedOpeningPoint, RingRelationSegmentLayout, TraceFieldBlockOpening, TraceRingBlockOpening,
    TraceTerm, TraceWeightLayout,
};

/// Owned public trace-weight factors used by the fused stage-2 trace term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TracePublicWeights<F: FieldCore, E: FieldCore, const D: usize> {
    /// Degree-one path: scalar block-weight terms with their packed inner openings.
    Field {
        terms: Vec<TraceFieldBlockOpening<F, D>>,
    },
    /// Extension path: ring block weights and psi-packed inner point.
    Ring {
        terms: Vec<TraceRingBlockOpening<F, D>>,
        _ext: PhantomData<E>,
    },
}

/// Verifier-side trace claim inputs for the stage-2 sumcheck final check.
///
/// The verifier reconstructs the fused trace term in its short closed form
/// ([`TraceTerm`]): one term per claim opening carrying the block-axis opening
/// `b_open`, the ψ-packed inner point, and a public coefficient. This is the
/// succinct counterpart of the prover's materialized [`TracePublicWeights`]
/// table; the two are kept distinct because the prover folds every block while
/// the verifier collapses each claim to a single `Tr_H`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceClaim<F: FieldCore, E: FieldCore, const D: usize> {
    pub layout: TraceWeightLayout,
    pub trace_terms: Vec<TraceTerm<F, E, D>>,
    /// Batching weight applied to the fused trace term. This is the `γ²` power
    /// of the stage-2 batching challenge (`CHALLENGE_SUMCHECK_BATCH`); the trace
    /// term reuses that challenge rather than sampling a dedicated one, so it is
    /// sampled after the next-level witness is bound to the transcript.
    pub trace_coeff: E,
    pub trace_opening_claim: E,
}

/// Whether the trace-weight dispatcher has an algebraic implementation for this
/// claim-field extension degree.
#[inline]
fn trace_stage2_supported(extension_degree: usize) -> bool {
    matches!(extension_degree, 1 | 2 | 4 | 8)
}

/// Reject extension degrees with no trace-weight implementation.
///
/// The fused trace term is mandatory: it is what binds the fold opening to the
/// committed witness in place of the dropped on-wire `y_ring`. A degree with no
/// algebraic implementation must therefore be rejected rather than silently
/// skipped, which would leave the opening unbound (a soundness footgun). This
/// is verifier-reachable, so it returns an error instead of panicking.
#[inline]
pub fn ensure_trace_stage2_supported(extension_degree: usize) -> Result<(), AkitaError> {
    if trace_stage2_supported(extension_degree) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "fused stage-2 trace term has no implementation for claim-field extension degree {extension_degree}; cannot bind the fold opening"
        )))
    }
}

/// Derive the trace-weight layout for the `e_hat` digit segment.
///
/// `num_trace_blocks` is the logical number of folded opening blocks addressed
/// by the trace term. Recursive singleton folds use `lp.num_blocks`; batched
/// root folds can use a wider claim-weighted block row.
pub fn trace_weight_layout_from_segment(
    lp: &LevelParams,
    segment: &RingRelationSegmentLayout,
    col_bits: usize,
    ring_bits: usize,
    num_trace_blocks: usize,
) -> Result<TraceWeightLayout, AkitaError> {
    if num_trace_blocks == 0 {
        return Err(AkitaError::InvalidInput(
            "trace-weight block count must be non-zero".to_string(),
        ));
    }
    if ring_bits != lp.ring_dimension.trailing_zeros() as usize {
        return Err(AkitaError::InvalidInput(
            "trace-weight ring bits do not match level ring dimension".to_string(),
        ));
    }
    let r_vars = num_trace_blocks.next_power_of_two().trailing_zeros() as usize;
    let layout = TraceWeightLayout {
        ring_bits,
        col_bits,
        opening_digit_offset: segment.offset_e,
        num_blocks: num_trace_blocks,
        num_digits_open: lp.num_digits_open,
        r_vars,
        log_basis: lp.log_basis,
    };
    layout.validate_opening_digit_segment()?;
    Ok(layout)
}

/// Build degree-one public trace weights from explicit block-offset terms.
pub(crate) fn trace_public_weights_field_terms<F, E, const D: usize>(
    terms: &[TraceFieldBlockOpening<F, D>],
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "field trace terms must be non-empty".to_string(),
        ));
    }
    Ok(TracePublicWeights::Field {
        terms: terms.to_vec(),
    })
}

/// Build extension-valued public trace weights from explicit block-offset terms.
pub(crate) fn trace_public_weights_ring_terms<F, E, const D: usize>(
    terms: &[TraceRingBlockOpening<F, D>],
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    if terms.is_empty() {
        return Err(AkitaError::InvalidInput(
            "ring trace terms must be non-empty".to_string(),
        ));
    }
    Ok(TracePublicWeights::Ring {
        terms: terms.to_vec(),
        _ext: PhantomData,
    })
}

fn scaled_base_weights<F, E>(weights: &[F], scale: E) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore,
    E: FpExtEncoding<F> + FieldCore,
{
    let scale = scale.degree_one_base().ok_or_else(|| {
        AkitaError::InvalidInput("trace field scale had no base coordinate".to_string())
    })?;
    Ok(weights.iter().map(|&weight| scale * weight).collect())
}

fn scaled_ring_weights<F, E, const D: usize>(
    weights: &[CyclotomicRing<F, D>],
    scale: E,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + FieldCore,
{
    let scale = embed_ring_subfield_scalar::<F, E, D>(
        scale,
        AkitaError::InvalidInput("trace ring scale had no ring-subfield encoding".to_string()),
    )?;
    Ok(weights.iter().map(|&weight| weight * scale).collect())
}

struct RootTraceClaimInputs<'a, F: FieldCore, E: FieldCore, const D: usize> {
    lp: &'a LevelParams,
    opening_batch: &'a OpeningBatchShape,
    prepared_point: &'a PreparedOpeningPoint<F, E, D>,
    row_coefficients: &'a [E],
    claim_scales: Option<&'a [E]>,
}

struct RootTraceClaimItem<'a, F: FieldCore, E: FieldCore, const D: usize> {
    prepared: &'a PreparedOpeningPoint<F, E, D>,
    scaled_coefficient: E,
    block_offset: usize,
}

fn validate_root_trace_claim_inputs<F: FieldCore, E: FieldCore, const D: usize>(
    inputs: &RootTraceClaimInputs<'_, F, E, D>,
) -> Result<(), AkitaError> {
    if inputs.row_coefficients.len() != inputs.opening_batch.num_polynomials() {
        return Err(AkitaError::InvalidSize {
            expected: inputs.opening_batch.num_polynomials(),
            actual: inputs.row_coefficients.len(),
        });
    }
    if let Some(scales) = inputs.claim_scales {
        if scales.len() != inputs.opening_batch.num_polynomials() {
            return Err(AkitaError::InvalidSize {
                expected: inputs.opening_batch.num_polynomials(),
                actual: scales.len(),
            });
        }
    }
    Ok(())
}

fn collect_root_trace_claim_items<'a, F: FieldCore, E: FieldCore, const D: usize>(
    inputs: &'a RootTraceClaimInputs<'a, F, E, D>,
) -> Result<Vec<RootTraceClaimItem<'a, F, E, D>>, AkitaError> {
    validate_root_trace_claim_inputs(inputs)?;
    let mut items = Vec::with_capacity(inputs.opening_batch.num_polynomials());
    for (claim_idx, &coefficient) in inputs.row_coefficients.iter().enumerate() {
        let scale = inputs
            .claim_scales
            .and_then(|scales| scales.get(claim_idx).copied())
            .unwrap_or_else(E::one);
        let block_offset = claim_idx
            .checked_mul(inputs.lp.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block offset overflow".to_string()))?;
        items.push(RootTraceClaimItem {
            prepared: inputs.prepared_point,
            scaled_coefficient: coefficient * scale,
            block_offset,
        });
    }
    Ok(items)
}

/// Fused trace coefficient: `γ²` on terminal folds, otherwise `batching_coeff²`.
#[inline]
pub fn stage2_trace_coeff<L: FieldCore>(batching_coeff: L, trace_gamma: L, is_terminal: bool) -> L {
    if is_terminal {
        trace_gamma * trace_gamma
    } else {
        batching_coeff * batching_coeff
    }
}

/// Build public trace weights for a root opening_batch, optionally scaling each
/// claim term by an extra public factor such as the EOR final tensor factor.
pub fn trace_public_weights_root_terms<F, E, const D: usize>(
    lp: &LevelParams,
    opening_batch: &OpeningBatchShape,
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    let inputs = RootTraceClaimInputs {
        lp,
        opening_batch,
        prepared_point,
        row_coefficients,
        claim_scales,
    };
    let items = collect_root_trace_claim_items(&inputs)?;
    if E::EXT_DEGREE == 1 {
        let mut terms = Vec::with_capacity(items.len());
        for item in items {
            terms.push(TraceFieldBlockOpening {
                block_offset: item.block_offset,
                block_weights: scaled_base_weights(
                    &item.prepared.ring_opening_point.b,
                    item.scaled_coefficient,
                )?,
                inner_opening_ring: item.prepared.packed_inner_point,
            });
        }
        trace_public_weights_field_terms(&terms)
    } else {
        let terms = items
            .into_iter()
            .map(|item| {
                let block_rings = item
                    .prepared
                    .ring_multiplier_point
                    .b_rings_trusted::<D>()?
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "extension trace opening point is missing ring block weights"
                                .to_string(),
                        )
                    })?;
                Ok(TraceRingBlockOpening {
                    block_offset: item.block_offset,
                    block_rings: scaled_ring_weights(block_rings, item.scaled_coefficient)?,
                    packed_inner_point: item.prepared.packed_inner_point,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        trace_public_weights_ring_terms(&terms)
    }
}

/// Build public trace weights for a recursive singleton fold, optionally
/// scaling it by an EOR final tensor factor.
pub fn trace_public_weights_recursive<F, E, const D: usize>(
    prepared: &PreparedOpeningPoint<F, E, D>,
    scale: E,
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    if E::EXT_DEGREE == 1 {
        trace_public_weights_field_terms(&[TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: scaled_base_weights(&prepared.ring_opening_point.b, scale)?,
            inner_opening_ring: prepared.packed_inner_point,
        }])
    } else {
        let block_rings = prepared
            .ring_multiplier_point
            .b_rings_trusted::<D>()?
            .ok_or_else(|| {
            AkitaError::InvalidInput(
                "extension trace opening point is missing ring block weights".to_string(),
            )
        })?;
        trace_public_weights_ring_terms(&[TraceRingBlockOpening {
            block_offset: 0,
            block_rings: scaled_ring_weights(block_rings, scale)?,
            packed_inner_point: prepared.packed_inner_point,
        }])
    }
}

/// Slice the block-axis opening `b_open` out of a root opening point.
///
/// The root uses [`crate::BlockOrder::RowMajor`]: after padding `opening_point`
/// to `m_vars + r_vars + alpha_bits` coordinates, the outer coordinates are
/// `padded[alpha_bits..]` and the block coordinates are their tail
/// `outer[m_vars..m_vars + r_vars]`. This reproduces the `b_open` that
/// `prepare_opening_point` consumes (and then discards) when it
/// materializes the ring block multipliers.
pub fn root_trace_block_opening<X: FieldCore>(
    opening_point: &[X],
    lp: &LevelParams,
    alpha_bits: usize,
) -> Result<Vec<X>, AkitaError> {
    let target = lp
        .m_vars
        .checked_add(lp.r_vars)
        .and_then(|n| n.checked_add(alpha_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() > target {
        return Err(AkitaError::InvalidPointDimension {
            expected: target,
            actual: opening_point.len(),
        });
    }
    let mut padded = opening_point.to_vec();
    padded.resize(target, X::zero());
    let start = alpha_bits + lp.m_vars;
    Ok(padded[start..start + lp.r_vars].to_vec())
}

/// Build the verifier's short closed-form trace terms for a root opening_batch.
///
/// `b_open` holds the block-axis opening (in the evaluation field `E`) for the
/// shared opening point; the verifier obtains it via
/// [`root_trace_block_opening`] (lifting claim-field coordinates into `E` as
/// needed). Mirrors [`trace_public_weights_root_terms`] but emits the succinct
/// per-claim terms instead of materialized block weights.
#[allow(clippy::too_many_arguments)]
pub fn trace_terms_root<F, E, const D: usize>(
    lp: &LevelParams,
    opening_batch: &OpeningBatchShape,
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    b_open: &[E],
    basis: BasisMode,
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<Vec<TraceTerm<F, E, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    let inputs = RootTraceClaimInputs {
        lp,
        opening_batch,
        prepared_point,
        row_coefficients,
        claim_scales,
    };
    let items = collect_root_trace_claim_items(&inputs)?;
    let mut terms = Vec::with_capacity(items.len());
    for item in items {
        terms.push(TraceTerm {
            block_offset: item.block_offset,
            b_open: b_open.to_vec(),
            basis,
            packed_inner_point: item.prepared.packed_inner_point,
            coefficient: item.scaled_coefficient,
        });
    }
    Ok(terms)
}

/// Build the verifier's short closed-form root trace claim.
#[allow(clippy::too_many_arguments)]
pub fn build_trace_claim_root<F, E, const D: usize>(
    layout: TraceWeightLayout,
    lp: &LevelParams,
    opening_batch: &OpeningBatchShape,
    prepared_point: &PreparedOpeningPoint<F, E, D>,
    b_open: &[E],
    basis: BasisMode,
    row_coefficients: &[E],
    trace_coeff: E,
    trace_eval_target: E,
    claim_scales: Option<&[E]>,
) -> Result<TraceClaim<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    Ok(TraceClaim {
        layout,
        trace_coeff,
        trace_opening_claim: trace_coeff * trace_eval_target,
        trace_terms: trace_terms_root(
            lp,
            opening_batch,
            prepared_point,
            b_open,
            basis,
            row_coefficients,
            claim_scales,
        )?,
    })
}

/// Build the verifier's short closed-form trace term for a recursive singleton
/// fold. The block-axis opening is sliced from the retained padded point under
/// the [`crate::BlockOrder::ColumnMajor`] convention.
pub fn trace_terms_recursive<F, E, const D: usize>(
    prepared: &PreparedOpeningPoint<F, E, D>,
    lp: &LevelParams,
    basis: BasisMode,
    scale: E,
) -> Result<Vec<TraceTerm<F, E, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    let outer_len = lp
        .m_vars
        .checked_add(lp.r_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let alpha_bits = prepared
        .padded_point
        .len()
        .checked_sub(outer_len)
        .ok_or(AkitaError::InvalidProof)?;
    // Column-major: the block coordinates are the head of the outer point.
    let b_open = prepared.padded_point[alpha_bits..alpha_bits + lp.r_vars].to_vec();
    Ok(vec![TraceTerm {
        block_offset: 0,
        b_open,
        basis,
        packed_inner_point: prepared.packed_inner_point,
        coefficient: scale,
    }])
}

/// Materialize the trace-weight table and keep only live witness columns.
#[cfg(test)]
pub(crate) fn trace_weight_evals_for_witness<E: FieldCore>(
    layout: &TraceWeightLayout,
    table: &[E],
    live_x_cols: usize,
) -> Result<Vec<E>, AkitaError> {
    let x_len = 1usize
        .checked_shl(layout.col_bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput("trace-weight x length overflow".to_string()))?;
    if live_x_cols > x_len {
        return Err(AkitaError::InvalidSize {
            expected: x_len,
            actual: live_x_cols,
        });
    }
    let expected = layout.table_len()?;
    if table.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: table.len(),
        });
    }

    let ring_len = layout.ring_len();
    let out_len = live_x_cols.checked_mul(ring_len).ok_or_else(|| {
        AkitaError::InvalidInput("trace-weight compact table length overflow".to_string())
    })?;
    let mut out = Vec::with_capacity(out_len);
    for col in 0..live_x_cols {
        for ring_coord in 0..ring_len {
            out.push(table[layout.witness_index(col, ring_coord)]);
        }
    }
    Ok(out)
}

/// Build the typed trace table used by the stage-2 prover.
///
/// `K = 1` field weights use sparse active columns; `K > 1` ring weights use a dense flat table.
pub fn build_trace_table_scaled<F, E, const D: usize>(
    layout: &TraceWeightLayout,
    public_weights: &TracePublicWeights<F, E, D>,
    live_x_cols: usize,
    output_scale: E,
) -> Result<TraceTable<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    match public_weights {
        TracePublicWeights::Field { terms } => {
            let ring_len = layout.ring_len();
            let columns = build_trace_weight_compact_field_sparse_scaled::<F, E, D>(
                layout,
                terms,
                live_x_cols,
                output_scale,
            )?;
            Ok(TraceTable::field_sparse(columns, live_x_cols, ring_len))
        }
        TracePublicWeights::Ring { terms, .. } => Ok(TraceTable::ring_dense(
            build_trace_weight_compact_ring_terms_scaled::<F, E, D>(
                layout,
                terms,
                live_x_cols,
                output_scale,
            )?,
        )),
    }
}
