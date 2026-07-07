//! Stage-2 wiring helpers for the fused trace term.

use std::marker::PhantomData;

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, Invertible};

use super::build::{
    build_trace_weight_compact_field_sparse_scaled, build_trace_weight_compact_ring_terms_scaled,
};
use super::layout::TraceChunkLayout;
use super::trace_table::TraceTable;
use crate::{
    embed_ring_subfield_scalar, BasisMode, FpExtEncoding, LevelParams, OpeningClaimsLayout,
    PreparedOpeningPoint, TraceFieldBlockOpening, TraceRingBlockOpening, TraceTerm,
    TraceWeightLayout, WitnessLayout,
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
    /// Dense grouped-root trace-weight table (`col ⊗ ring`, `output_scale = 1`).
    /// When present the stage-2 verifier evaluates its multilinear extension at
    /// the witness point instead of the closed-form [`Self::trace_terms`]. This
    /// is the grouped-root counterpart of the succinct per-claim terms: grouped
    /// roots decompose each group with per-group `num_blocks`/`num_digits_open`
    /// and a group-major e-hat offset, which the closed form cannot express.
    pub dense_evals: Option<Vec<E>>,
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
    let chunk0 = witness_layout.chunks.first().ok_or_else(|| {
        AkitaError::InvalidSetup("trace-weight witness layout has no chunks".to_string())
    })?;
    let num_blocks_global = lp.num_blocks;
    if num_blocks_global == 0 || !num_trace_blocks.is_multiple_of(num_blocks_global) {
        return Err(AkitaError::InvalidSetup(
            "trace block count is not a multiple of the level block count".to_string(),
        ));
    }
    // Chunk geometry: `chunk[c].offset_e = c·chunk_stride + chunk0.offset_e`.
    let chunk = TraceChunkLayout {
        num_chunks: witness_layout.num_chunks(),
        blocks_per_chunk: witness_layout.blocks_per_chunk,
        num_claims: num_trace_blocks / num_blocks_global,
        num_blocks_global,
        chunk_stride: witness_layout
            .chunks
            .get(1)
            .map(|c| c.offset_z)
            .unwrap_or(0),
    };
    let r_vars = num_trace_blocks.next_power_of_two().trailing_zeros() as usize;
    let layout = TraceWeightLayout {
        ring_bits,
        col_bits,
        opening_digit_offset: chunk0.offset_e,
        num_blocks: num_trace_blocks,
        num_digits_open: lp.num_digits_open,
        r_vars,
        log_basis: lp.log_basis,
        chunk,
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

/// Fused trace coefficient: `γ²` on terminal folds, otherwise `batching_coeff²`.
#[inline]
pub fn stage2_trace_coeff<E: FieldCore>(batching_coeff: E, trace_gamma: E, is_terminal: bool) -> E {
    if is_terminal {
        trace_gamma * trace_gamma
    } else {
        batching_coeff * batching_coeff
    }
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
                    &item.prepared.ring_opening_point.b,
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
            block_weights: scaled_base_weights(&prepared.ring_opening_point.b, scale)?,
            inner_opening_ring: prepared.packed_inner_owned::<D>()?,
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
            packed_inner_point: prepared.packed_inner_owned::<D>()?,
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
    })
}

/// Build the grouped-root verifier trace claim from a dense trace-weight table.
///
/// The table is [`build_grouped_root_stage2_trace_table`] with `output_scale =
/// 1`; the stage-2 `trace_coeff` factor stays separate so `expected_output_claim`
/// applies it once. `trace_terms` are left empty because the closed form cannot
/// express per-group block geometry; the dense table is evaluated directly.
#[allow(clippy::too_many_arguments)]
pub fn build_trace_claim_grouped_root<F, E, const D: usize>(
    layout: TraceWeightLayout,
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    prepared_points: &[PreparedOpeningPoint<F, E>],
    row_coefficients: &[E],
    claim_scales: Option<&[E]>,
    trace_coeff: E,
    trace_eval_target: E,
    live_x_cols: usize,
) -> Result<TraceClaim<F, E, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let table = build_grouped_root_stage2_trace_table::<F, E>(
        lp.role_dims().d_a(),
        lp,
        opening_batch,
        prepared_points,
        row_coefficients,
        claim_scales,
        E::one(),
        live_x_cols,
    )?;
    Ok(TraceClaim {
        layout,
        trace_coeff,
        trace_opening_claim: trace_coeff * trace_eval_target,
        trace_terms: Vec::new(),
        dense_evals: Some(table.into_ring_dense()?),
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
    let eq_x = EqPolynomial::evals(x_challenges)?;
    let eq_y = EqPolynomial::evals(y_challenges)?;
    if live_x_cols > eq_x.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut acc = E::zero();
    for (col, &x_weight) in eq_x.iter().take(live_x_cols).enumerate() {
        let base = col * ring_len;
        let mut ring_acc = E::zero();
        for (coord, &y_weight) in eq_y.iter().enumerate() {
            ring_acc += y_weight * dense_evals[base + coord];
        }
        acc += x_weight * ring_acc;
    }
    Ok(acc)
}

/// Build the dense grouped-root stage-2 trace table (`col ⊗ ring`).
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
pub fn build_grouped_root_stage2_trace_table<F, E>(
    ring_d: usize,
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
            "grouped root trace table currently requires degree-one openings".to_string(),
        ));
    }
    if prepared_points.len() != opening_batch.num_groups()
        || row_coefficients.len() != opening_batch.num_total_polynomials()
    {
        return Err(AkitaError::InvalidProof);
    }
    if let Some(scales) = trace_claim_scales {
        if scales.len() != opening_batch.num_total_polynomials() {
            return Err(AkitaError::InvalidProof);
        }
    }
    crate::dispatch_ring_dim_result!(ring_d, |D| {
        let ring_len = D;
        let order = opening_batch.root_group_order()?;
        let mut e_offsets = vec![0usize; opening_batch.num_groups()];
        let mut base = 0usize;
        for &group_index in &order {
            let group_lp = lp.root_group_params(opening_batch, group_index)?;
            let group_layout = opening_batch.group_layout(group_index)?;
            let depth_fold = lp.num_digits_fold_for_params(
                group_lp,
                group_layout.num_polynomials(),
                lp.field_bits_for_cache(),
            )?;
            let overflow =
                || AkitaError::InvalidSetup("grouped trace segment width overflow".to_string());
            let z_g = group_lp
                .block_len()
                .checked_mul(group_lp.num_digits_commit())
                .and_then(|n| n.checked_mul(depth_fold))
                .ok_or_else(overflow)?;
            let e_g = group_layout
                .num_polynomials()
                .checked_mul(group_lp.num_blocks())
                .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
                .ok_or_else(overflow)?;
            let t_g = group_layout
                .num_polynomials()
                .checked_mul(group_lp.num_blocks())
                .and_then(|n| n.checked_mul(group_lp.a_rows_len()))
                .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
                .ok_or_else(overflow)?;
            e_offsets[group_index] = base.checked_add(z_g).ok_or_else(overflow)?;
            base = base
                .checked_add(z_g)
                .and_then(|n| n.checked_add(e_g))
                .and_then(|n| n.checked_add(t_g))
                .ok_or_else(overflow)?;
        }

        let mut table = vec![E::zero(); live_x_cols * ring_len];
        let mut claim_offset = 0usize;
        for group_index in 0..opening_batch.num_groups() {
            let group_lp = lp.root_group_params(opening_batch, group_index)?;
            let group_layout = opening_batch.group_layout(group_index)?;
            let prepared = &prepared_points[group_index];
            let inner = prepared.packed_inner_owned::<D>()?;
            let inner_coeffs = inner.coefficients();
            let gadget =
                crate::gadget_row_scalars::<F>(group_lp.num_digits_open(), group_lp.log_basis());
            for local_claim in 0..group_layout.num_polynomials() {
                let claim_idx = claim_offset + local_claim;
                let scale = trace_claim_scales
                    .and_then(|scales| scales.get(claim_idx).copied())
                    .unwrap_or_else(E::one);
                let coefficient = output_scale * row_coefficients[claim_idx] * scale;
                for block in 0..group_lp.num_blocks() {
                    let block_weight = prepared
                        .ring_opening_point
                        .b
                        .get(block)
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    let block_weight = E::lift_base(block_weight);
                    for (plane, gadget_scalar) in gadget.iter().enumerate() {
                        let col = e_offsets[group_index]
                            + plane * (group_layout.num_polynomials() * group_lp.num_blocks())
                            + local_claim * group_lp.num_blocks()
                            + block;
                        if col >= live_x_cols {
                            continue;
                        }
                        let dst_base = col * ring_len;
                        let factor = coefficient * block_weight * E::lift_base(*gadget_scalar);
                        for (dst, coeff) in table[dst_base..dst_base + ring_len]
                            .iter_mut()
                            .zip(inner_coeffs.iter())
                        {
                            *dst += factor * E::lift_base(*coeff);
                        }
                    }
                }
            }
            claim_offset += group_layout.num_polynomials();
        }
        Ok::<_, AkitaError>(TraceTable::ring_dense(table))
    })
}

/// Build the verifier's short closed-form trace term for a recursive singleton
/// fold. The block-axis opening is sliced from the retained padded point under
/// the [`crate::BlockOrder::ColumnMajor`] convention.
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
