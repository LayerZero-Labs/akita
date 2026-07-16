//! Stage-2 wiring helpers for the fused trace term.

use std::marker::PhantomData;

use akita_algebra::eq_poly::{EqPolynomial, SplitEqEvals};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

use super::build::{
    build_trace_weight_compact_field_sparse_scaled, build_trace_weight_compact_ring_terms_scaled,
};
use super::trace_table::TraceTable;
use crate::{
    dispatch_for_field, embed_ring_subfield_scalar, BasisMode, FpExtEncoding, LevelParams,
    OpeningClaimsLayout, PreparedOpeningPoint, TraceFieldBlockOpening, TraceRingBlockOpening,
    TraceTerm, TraceWeightLayout, WitnessLayout,
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

/// One closed-form trace batch evaluated with its own column geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceTermBatch<F: FieldCore, E: FieldCore, const D: usize> {
    pub layout: TraceWeightLayout,
    pub terms: Vec<TraceTerm<F, E, D>>,
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
    /// Dense multi-group-root trace-weight table (`col ⊗ ring`, `output_scale = 1`).
    /// When present the stage-2 verifier evaluates its multilinear extension at
    /// the witness point instead of the closed-form [`Self::trace_terms`]. This
    /// is the multi-group-root counterpart of the succinct per-claim terms: multi-group
    /// roots decompose each group with per-group `num_blocks`/`num_digits_open`
    /// and a group-major e-hat offset, which the closed form cannot express.
    pub dense_evals: Option<Vec<E>>,
    /// Optional closed-form batches with independent layouts.
    pub trace_term_batches: Vec<TraceTermBatch<F, E, D>>,
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
    witness_layout: &WitnessLayout,
    opening_source_len: usize,
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
    let group_id = witness_layout.first_group_index()?;
    let group_live_block_count = witness_layout.group_live_block_count(group_id)?;
    if !num_trace_blocks.is_multiple_of(group_live_block_count) {
        return Err(AkitaError::InvalidSetup(
            "trace block count disagrees with witness layout".to_string(),
        ));
    }
    let block_bits = num_trace_blocks.next_power_of_two().trailing_zeros() as usize;
    let layout = TraceWeightLayout {
        ring_bits,
        col_bits,
        num_blocks: num_trace_blocks,
        num_digits_open: lp.num_digits_open,
        block_bits,
        log_basis: lp.log_basis,
        witness_layout: witness_layout.clone(),
        opening_source_len,
        group_id,
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

struct RootTraceClaimInputs<'a, F: FieldCore, E: FieldCore> {
    /// M-matrix block count per claim (`LevelParams::num_blocks`, extracted by
    /// the caller — trace-weight construction must not read schedule types).
    num_blocks: usize,
    opening_batch: &'a OpeningClaimsLayout,
    prepared_point: &'a PreparedOpeningPoint<F, E>,
    row_coefficients: &'a [E],
    claim_scales: Option<&'a [E]>,
}

struct RootTraceClaimItem<'a, F: FieldCore, E: FieldCore> {
    prepared: &'a PreparedOpeningPoint<F, E>,
    scaled_coefficient: E,
    block_offset: usize,
}

fn validate_root_trace_claim_inputs<F: FieldCore, E: FieldCore>(
    inputs: &RootTraceClaimInputs<'_, F, E>,
) -> Result<(), AkitaError> {
    if inputs.row_coefficients.len() != inputs.opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidSize {
            expected: inputs.opening_batch.num_total_polynomials(),
            actual: inputs.row_coefficients.len(),
        });
    }
    if let Some(scales) = inputs.claim_scales {
        if scales.len() != inputs.opening_batch.num_total_polynomials() {
            return Err(AkitaError::InvalidSize {
                expected: inputs.opening_batch.num_total_polynomials(),
                actual: scales.len(),
            });
        }
    }
    Ok(())
}

fn collect_root_trace_claim_items<'a, F: FieldCore, E: FieldCore>(
    inputs: &'a RootTraceClaimInputs<'a, F, E>,
) -> Result<Vec<RootTraceClaimItem<'a, F, E>>, AkitaError> {
    validate_root_trace_claim_inputs(inputs)?;
    let mut items = Vec::with_capacity(inputs.opening_batch.num_total_polynomials());
    for (claim_idx, &coefficient) in inputs.row_coefficients.iter().enumerate() {
        let scale = inputs
            .claim_scales
            .and_then(|scales| scales.get(claim_idx).copied())
            .unwrap_or_else(E::one);
        let block_offset = claim_idx
            .checked_mul(inputs.num_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block offset overflow".to_string()))?;
        items.push(RootTraceClaimItem {
            prepared: inputs.prepared_point,
            scaled_coefficient: coefficient * scale,
            block_offset,
        });
    }
    Ok(items)
}

/// Build public trace weights for a root opening_batch, optionally scaling each
/// claim term by an extra public factor such as the EOR final tensor factor.
pub fn trace_public_weights_root_terms<F, E, const D: usize>(
    num_blocks: usize,
    opening_batch: &OpeningClaimsLayout,
    prepared_point: &PreparedOpeningPoint<F, E>,
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    let inputs = RootTraceClaimInputs {
        num_blocks,
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
                    &item.prepared.ring_opening_point.block_weights,
                    item.scaled_coefficient,
                )?,
                inner_opening_ring: item.prepared.packed_inner_owned::<D>()?,
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
                    .fold_rings_trusted::<D>()?
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "extension trace opening point is missing ring block weights"
                                .to_string(),
                        )
                    })?;
                Ok(TraceRingBlockOpening {
                    block_offset: item.block_offset,
                    block_rings: scaled_ring_weights(block_rings, item.scaled_coefficient)?,
                    packed_inner_point: item.prepared.packed_inner_owned::<D>()?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        trace_public_weights_ring_terms(&terms)
    }
}

/// Build public trace weights for a recursive singleton fold, optionally
/// scaling it by an EOR final tensor factor.
pub fn trace_public_weights_recursive<F, E, const D: usize>(
    prepared: &PreparedOpeningPoint<F, E>,
    scale: E,
) -> Result<TracePublicWeights<F, E, D>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    if E::EXT_DEGREE == 1 {
        trace_public_weights_field_terms(&[TraceFieldBlockOpening {
            block_offset: 0,
            block_weights: scaled_base_weights(&prepared.ring_opening_point.block_weights, scale)?,
            inner_opening_ring: prepared.packed_inner_owned::<D>()?,
        }])
    } else {
        let block_rings = prepared
            .ring_multiplier_point
            .fold_rings_trusted::<D>()?
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension trace opening point is missing ring block weights".to_string(),
                )
            })?;
        trace_public_weights_ring_terms(&[TraceRingBlockOpening {
            block_offset: 0,
            block_rings: scaled_ring_weights(block_rings, scale)?,
            packed_inner_point: prepared.packed_inner_owned::<D>()?,
        }])
    }
}

/// Slice the fold-axis opening out of a root opening point.
pub fn root_trace_block_opening<X: FieldCore>(
    opening_point: &[X],
    block_len: usize,
    num_blocks: usize,
    alpha_bits: usize,
) -> Result<Vec<X>, AkitaError> {
    if !block_len.is_power_of_two() || num_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "trace opening requires power-of-two L and positive F".to_string(),
        ));
    }
    let position_bits = block_len.trailing_zeros() as usize;
    let block_bits = num_blocks
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("block capacity overflow".to_string()))?
        .trailing_zeros() as usize;
    let target = position_bits
        .checked_add(block_bits)
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
    let start = alpha_bits + position_bits;
    Ok(padded[start..start + block_bits].to_vec())
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
    opening_batch: &OpeningClaimsLayout,
    prepared_point: &PreparedOpeningPoint<F, E>,
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
        num_blocks: lp.num_blocks,
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
            packed_inner_point: item.prepared.packed_inner_owned::<D>()?,
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
    opening_batch: &OpeningClaimsLayout,
    prepared_point: &PreparedOpeningPoint<F, E>,
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
        dense_evals: None,
        trace_term_batches: Vec::new(),
    })
}

/// Build the multi-group-root verifier trace claim from one short closed-form
/// batch per group.
#[allow(clippy::too_many_arguments)]
pub fn build_trace_claim_multi_group_root<F, E, const D: usize>(
    layout: TraceWeightLayout,
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    prepared_points: &[PreparedOpeningPoint<F, E>],
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
    basis: BasisMode,
    trace_coeff: E,
    trace_eval_target: E,
    live_x_cols: usize,
) -> Result<TraceClaim<F, E, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    if prepared_points.len() != opening_batch.num_groups()
        || row_coefficients.len() != opening_batch.num_total_polynomials()
        || live_x_cols != crate::opening_domain_len(layout.opening_source_len)?
    {
        return Err(AkitaError::InvalidProof);
    }
    if let Some(scales) = claim_scales {
        if scales.len() != row_coefficients.len() {
            return Err(AkitaError::InvalidProof);
        }
    }
    let alpha_bits = lp.role_dims().d_a().trailing_zeros() as usize;
    let mut batches = Vec::with_capacity(opening_batch.num_groups());
    let mut claim_offset = 0usize;
    for (group_index, prepared) in prepared_points.iter().enumerate() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_claims = opening_batch.group_layout(group_index)?.num_polynomials();
        let group_id = group_index;
        let num_blocks = group_claims
            .checked_mul(group_lp.num_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if layout.witness_layout.group_live_block_count(group_id)? != group_lp.num_blocks() {
            return Err(AkitaError::InvalidSetup(
                "trace group geometry disagrees with witness layout".to_string(),
            ));
        }
        let b_open = root_trace_block_opening(
            &prepared.padded_point,
            group_lp.block_len(),
            group_lp.num_blocks(),
            alpha_bits,
        )?;
        let group_layout = TraceWeightLayout {
            ring_bits: layout.ring_bits,
            col_bits: layout.col_bits,
            num_blocks,
            num_digits_open: group_lp.num_digits_open(),
            block_bits: num_blocks.next_power_of_two().trailing_zeros() as usize,
            log_basis: group_lp.log_basis(),
            witness_layout: layout.witness_layout.clone(),
            opening_source_len: layout.opening_source_len,
            group_id,
        };
        group_layout.validate_opening_digit_segment()?;
        let packed_inner_point = prepared.packed_inner_owned::<D>()?;
        let mut terms = Vec::with_capacity(group_claims);
        for local_claim in 0..group_claims {
            let claim_idx = claim_offset + local_claim;
            let scale = claim_scales
                .and_then(|scales| scales.get(claim_idx).copied())
                .unwrap_or_else(E::one);
            terms.push(TraceTerm {
                block_offset: local_claim * group_lp.num_blocks(),
                b_open: b_open.clone(),
                basis,
                packed_inner_point,
                coefficient: row_coefficients[claim_idx] * scale,
            });
        }
        batches.push(TraceTermBatch {
            layout: group_layout,
            terms,
        });
        claim_offset += group_claims;
    }
    Ok(TraceClaim {
        layout,
        trace_coeff,
        trace_opening_claim: trace_coeff * trace_eval_target,
        trace_terms: Vec::new(),
        dense_evals: None,
        trace_term_batches: batches,
    })
}

/// Evaluate the multilinear extension of a dense `col ⊗ ring` trace table at the
/// stage-2 witness point.
///
/// `challenges = [y_challenges (ring), x_challenges (col)]` matches the stage-2
/// witness variable order (`table[col * ring_len + coord]`, `col` outer, ring
/// coordinate inner). Live columns below the padded width evaluate as implicit
/// zeros, mirroring the prover's compact table.
pub fn eval_dense_trace_table<E>(
    dense_evals: &[E],
    y_challenges: &[E],
    x_challenges: &[E],
) -> Result<E, AkitaError>
where
    E: FieldCore,
{
    let ring_len = 1usize
        .checked_shl(u32::try_from(y_challenges.len()).map_err(|_| AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    if ring_len == 0 || !dense_evals.len().is_multiple_of(ring_len) {
        return Err(AkitaError::InvalidProof);
    }
    let live_x_cols = dense_evals.len() / ring_len;
    let eq_x = SplitEqEvals::new(x_challenges)?;
    let eq_y = EqPolynomial::evals(y_challenges)?;
    if live_x_cols > eq_x.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut acc = E::zero();
    for col in 0..live_x_cols {
        let x_weight = eq_x.eval_at(col)?;
        let base = col * ring_len;
        let mut ring_acc = E::zero();
        for (coord, &y_weight) in eq_y.iter().enumerate() {
            ring_acc += y_weight * dense_evals[base + coord];
        }
        acc += x_weight * ring_acc;
    }
    Ok(acc)
}

/// Build the dense multi-group-root stage-2 trace table (`col ⊗ ring`).
///
/// Group `g`'s e-hat block sits inside its contiguous `[z_g ‖ e_g ‖ t_g]` stride
/// at `base_g + z_g`, where `base_g` is the cumulative `z+e+t` width of the
/// groups emitted before it in `root_group_order()`; this matches the prover's
/// `ring_switch_build_w` / `segment_layout` witness emission. `output_scale` is
/// the stage-2 `trace_coeff` on the prover and `1` on the verifier (which keeps
/// `trace_coeff` separate for `expected_output_claim`).
///
/// # Errors
///
/// Returns an error for a non-degree-one opening field, mismatched group counts,
/// or any segment-width arithmetic overflow.
#[allow(clippy::too_many_arguments)]
pub fn build_multi_group_root_stage2_trace_table<F, E>(
    ring_d: usize,
    witness_layout: &WitnessLayout,
    opening_source_len: usize,
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    prepared_points: &[PreparedOpeningPoint<F, E>],
    row_coefficients: &[E],
    trace_claim_scales: Option<&[E]>,
    output_scale: E,
    live_x_cols: usize,
) -> Result<TraceTable<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    if E::EXT_DEGREE != 1 {
        return Err(AkitaError::InvalidSetup(
            "multi-group root trace table currently requires degree-one openings".to_string(),
        ));
    }
    if live_x_cols != crate::opening_domain_len(opening_source_len)?
        || witness_layout.total_len() > opening_source_len
    {
        return Err(AkitaError::InvalidProof);
    }
    if prepared_points.len() != opening_batch.num_groups()
        || row_coefficients.len() != opening_batch.num_total_polynomials()
        || witness_layout.num_groups() != opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    if let Some(scales) = trace_claim_scales {
        if scales.len() != opening_batch.num_total_polynomials() {
            return Err(AkitaError::InvalidProof);
        }
    }
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        ring_d,
        |D| {
            let ring_len = D;
            let table_len = live_x_cols.checked_mul(ring_len).ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group trace table length overflow".to_string())
            })?;
            let mut table = vec![E::zero(); table_len];
            let mut claim_offset = 0usize;
            for (group_index, prepared) in prepared_points.iter().enumerate() {
                let group_lp = lp.group_params(opening_batch, group_index)?;
                let group_layout = opening_batch.group_layout(group_index)?;
                let group_id = group_index;
                let inner = prepared.packed_inner_owned::<D>()?;
                let inner_coeffs = inner.coefficients();
                let gadget = crate::gadget_row_scalars::<F>(
                    group_lp.num_digits_open(),
                    group_lp.log_basis(),
                );
                for local_claim in 0..group_layout.num_polynomials() {
                    let claim_idx = claim_offset + local_claim;
                    let scale = trace_claim_scales
                        .and_then(|scales| scales.get(claim_idx).copied())
                        .unwrap_or_else(E::one);
                    let coefficient = output_scale * row_coefficients[claim_idx] * scale;
                    for block in 0..group_lp.num_blocks() {
                        let block_weight = prepared
                            .ring_opening_point
                            .block_weights
                            .get(block)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?;
                        let block_weight = E::lift_base(block_weight);
                        for (plane, gadget_scalar) in gadget.iter().enumerate() {
                            if witness_layout.group_live_block_count(group_id)?
                                != group_lp.num_blocks()
                            {
                                return Err(AkitaError::InvalidSetup(
                                    "trace group geometry disagrees with witness layout"
                                        .to_string(),
                                ));
                            }
                            let unit = witness_layout.unit_for_block(group_id, block)?;
                            let physical_col = witness_layout.e_index(
                                unit,
                                group_layout.num_polynomials(),
                                group_lp.num_digits_open(),
                                local_claim,
                                block,
                                plane,
                            )?;
                            let col = crate::checked_opening_source_index(
                                opening_source_len,
                                physical_col,
                            )?;
                            if col >= live_x_cols {
                                continue;
                            }
                            let dst_base = col.checked_mul(ring_len).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "multi-group trace row offset overflow".to_string(),
                                )
                            })?;
                            let dst_end = dst_base.checked_add(ring_len).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "multi-group trace row overflow".to_string(),
                                )
                            })?;
                            let factor = coefficient * block_weight * E::lift_base(*gadget_scalar);
                            let dst_row = table
                                .get_mut(dst_base..dst_end)
                                .ok_or(AkitaError::InvalidProof)?;
                            for (dst, coeff) in dst_row.iter_mut().zip(inner_coeffs.iter()) {
                                *dst += factor * E::lift_base(*coeff);
                            }
                        }
                    }
                }
                claim_offset += group_layout.num_polynomials();
            }
            Ok::<_, AkitaError>(TraceTable::ring_dense(table))
        }
    )
}

/// Build the verifier's short closed-form trace term for a recursive singleton
/// fold. The fold-axis opening follows the physical digit-innermost order.
pub fn trace_terms_recursive<F, E, const D: usize>(
    prepared: &PreparedOpeningPoint<F, E>,
    lp: &LevelParams,
    basis: BasisMode,
    scale: E,
) -> Result<Vec<TraceTerm<F, E, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    let outer_len = lp
        .position_bits()
        .checked_add(lp.block_bits())
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    let alpha_bits = prepared
        .padded_point
        .len()
        .checked_sub(outer_len)
        .ok_or(AkitaError::InvalidProof)?;
    let block_start = alpha_bits
        .checked_add(lp.position_bits())
        .ok_or_else(|| AkitaError::InvalidSetup("block opening offset overflow".to_string()))?;
    let block_end = block_start
        .checked_add(lp.block_bits())
        .ok_or_else(|| AkitaError::InvalidSetup("block opening end overflow".to_string()))?;
    let b_open = prepared
        .padded_point
        .get(block_start..block_end)
        .ok_or(AkitaError::InvalidProof)?
        .to_vec();
    Ok(vec![TraceTerm {
        block_offset: 0,
        b_open,
        basis,
        packed_inner_point: prepared.packed_inner_owned::<D>()?,
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
