//! Pure fold kernels and operation adapters for the prover core.
//!
//! Everything here passes the spec's kernel discriminator: no function reads a
//! schedule type (`ExecutionSchedule`, `LevelParams`). Const-D functions receive extracted numbers and typed
//! buffers; the D-free functions are operation adapters that dispatch exactly
//! once on a schedule-derived ring dimension supplied by the caller.

use super::*;
use crate::compute::{
    ComputeBackendSetup, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource,
};

/// Batched trace-target data derived from folded claim openings.
pub(in crate::protocol::core) struct TraceTarget<E: FieldCore> {
    pub(in crate::protocol::core) trace_eval_target: E,
    pub(in crate::protocol::core) trace_claim_scales: Option<Vec<E>>,
    pub(in crate::protocol::core) trace_scale: E,
}

/// Extract the typed `b`/`a` ring-weight slices from a ring multiplier point.
pub(in crate::protocol::core) fn multiplier_ring_weights<F: FieldCore, const D: usize>(
    point: &RingMultiplierOpeningPoint<F>,
) -> Result<MultiplierWeightSlices<'_, F, D>, AkitaError> {
    let b = point.b_rings_trusted::<D>()?.ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring b weights".to_string())
    })?;
    let a = point.a_rings_trusted::<D>()?.ok_or_else(|| {
        AkitaError::InvalidInput("ring multiplier must carry ring a weights".to_string())
    })?;
    Ok((b, a))
}

fn evaluate_poly_at_multiplier_point<F, Q, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    poly: &Q,
    point: &RingMultiplierOpeningPoint<F>,
    block_len: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField,
    Q: RootOpeningSource<F, D>,
    B: ComputeBackendSetup<F> + for<'a> OpeningFoldKernel<Q::OpeningView<'a>, F, D>,
{
    let plan = if let Some(base_point) = point.as_base() {
        OpeningFoldPlan::Base {
            eval_outer_scalars: &base_point.b,
            fold_scalars: &base_point.a,
            block_len,
        }
    } else {
        let (b, a) = multiplier_ring_weights(point)?;
        OpeningFoldPlan::Ring {
            eval_outer_scalars: b,
            fold_scalars: a,
            block_len,
        }
    };
    let OpeningFoldOutput { eval, folded } =
        OpeningFoldKernel::evaluate_and_fold(backend, prepared, poly.opening_view()?, plan)?;
    Ok((eval, folded))
}

pub(in crate::protocol::core) fn evaluate_claims_at_prepared_point<F, E, Q, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    polys: &[&Q],
    prepared_point: &PreparedOpeningPoint<F, E>,
    block_len: usize,
) -> Result<FoldedClaimEvals<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
    Q: RootOpeningSource<F, D>,
    B: ComputeBackendSetup<F> + for<'a> OpeningFoldKernel<Q::OpeningView<'a>, F, D>,
{
    let _span = tracing::info_span!("fold_evaluate_claims", num_claims = polys.len()).entered();
    let mut folded_rings = Vec::with_capacity(polys.len());
    let mut folded_blocks = Vec::with_capacity(polys.len());
    for poly in polys {
        let (folded_ring, folded_block) = evaluate_poly_at_multiplier_point(
            backend,
            prepared,
            *poly,
            &prepared_point.ring_multiplier_point,
            block_len,
        )?;
        folded_rings.push(folded_ring);
        folded_blocks.push(folded_block);
    }
    Ok((folded_rings, folded_blocks))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn compute_trace_target<F, E, T, const D: usize>(
    reduction: &Option<ExtensionOpeningReduction<E>>,
    folded_rings: &[CyclotomicRing<F, D>],
    prepared_points: &[PreparedOpeningPoint<F, E>],
    protocol_point: &[E],
    alpha_bits: usize,
    basis: BasisMode,
    opening_batch: &OpeningClaimsLayout,
    row_coefficients: Option<Vec<E>>,
    transcript: &mut T,
) -> Result<(TraceTarget<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F>,
    T: Transcript<F>,
{
    if prepared_points.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidSize {
            expected: opening_batch.num_groups(),
            actual: prepared_points.len(),
        });
    }
    if folded_rings.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidSize {
            expected: opening_batch.num_total_polynomials(),
            actual: folded_rings.len(),
        });
    }
    let inner_claim_point = &protocol_point[..protocol_point.len().min(alpha_bits)];
    let mut openings = Vec::with_capacity(opening_batch.num_total_polynomials());
    let mut claim_offset = 0usize;
    for (group_index, prepared_point) in prepared_points.iter().enumerate() {
        let group_layout = opening_batch.group_layout(group_index)?;
        let end = claim_offset
            .checked_add(group_layout.num_polynomials())
            .ok_or(AkitaError::InvalidProof)?;
        let group_folded_rings = folded_rings
            .get(claim_offset..end)
            .ok_or(AkitaError::InvalidProof)?;
        for folded_ring in group_folded_rings {
            openings.push(scalar_opening_from_folded_ring::<F, E, D>(
                folded_ring,
                prepared_point,
                inner_claim_point,
                basis,
            )?);
        }
        claim_offset = end;
    }
    let row_coefficients = if let Some(row_coefficients) = row_coefficients {
        row_coefficients
    } else {
        append_claim_values_to_transcript::<F, E, T>(&openings, transcript);
        if opening_batch.num_total_polynomials() == 1 {
            vec![E::one()]
        } else {
            sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?
        }
    };
    let ordinary_trace_eval_target =
        opening_batch.batched_eval_target(&row_coefficients, &openings)?;
    let trace_eval_target =
        reduction
            .as_ref()
            .map_or(Ok(ordinary_trace_eval_target), |reduction| {
                check_extension_opening_reduction_output(
                    reduction.final_claim,
                    ordinary_trace_eval_target,
                    reduction.final_factor,
                )?;
                Ok(reduction.final_claim)
            })?;
    let trace_claim_scales = reduction
        .as_ref()
        .map(|reduction| vec![reduction.final_factor; opening_batch.num_total_polynomials()]);
    let trace_scale = reduction
        .as_ref()
        .map_or(E::one(), |reduction| reduction.final_factor);

    Ok((
        TraceTarget {
            trace_eval_target,
            trace_claim_scales,
            trace_scale,
        },
        row_coefficients,
    ))
}

/// Build the recursive-suffix stage-2 trace table (operation adapter).
///
/// `ring_d` is the level's schedule-derived fold ring dimension; `layout` was
/// derived by the caller from the level geometry.
pub(in crate::protocol::core) fn build_recursive_stage2_trace_table<F, E>(
    ring_d: usize,
    layout: &akita_types::TraceWeightLayout,
    prepared: &PreparedOpeningPoint<F, E>,
    trace_scale: E,
    output_scale: E,
    live_x_cols: usize,
) -> Result<TraceTable<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    dispatch_ring_dim_result!(ring_d, |D| {
        let public_weights = trace_public_weights_recursive::<F, E, D>(prepared, trace_scale)?;
        build_trace_table_scaled(layout, &public_weights, live_x_cols, output_scale)
    })
}

/// Build the root stage-2 trace table (operation adapter).
///
/// `ring_d` / `num_blocks` are extracted level numbers; `layout` was derived by
/// the caller from the level geometry.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn build_root_stage2_trace_table<F, E>(
    ring_d: usize,
    num_blocks: usize,
    layout: &akita_types::TraceWeightLayout,
    opening_batch: &OpeningClaimsLayout,
    prepared_point: &PreparedOpeningPoint<F, E>,
    row_coefficients: &[E],
    trace_claim_scales: Option<&[E]>,
    output_scale: E,
    live_x_cols: usize,
) -> Result<TraceTable<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    dispatch_ring_dim_result!(ring_d, |D| {
        let public_weights = trace_public_weights_root_terms::<F, E, D>(
            num_blocks,
            opening_batch,
            prepared_point,
            row_coefficients,
            trace_claim_scales,
        )?;
        build_trace_table_scaled(layout, &public_weights, live_x_cols, output_scale)
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn build_grouped_root_stage2_trace_table<F, E>(
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
    dispatch_ring_dim_result!(ring_d, |D| {
        let ring_len = D;
        let order = opening_batch.root_group_order()?;
        // Group-major witness: group `g`'s e-hat block sits inside its contiguous
        // `[z_g ‖ e_g ‖ t_g]` stride, at `base_g + z_g`, where `base_g` is the
        // cumulative `z+e+t` width of the groups emitted before it (in
        // `root_group_order()`). Matches `ring_switch_build_w` / `segment_layout`.
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
            let gadget = akita_types::gadget_row_scalars::<F>(
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

pub(in crate::protocol::core) fn scalar_opening_from_folded_ring<F, E, const D: usize>(
    folded_ring: &CyclotomicRing<F, D>,
    prepared_point: &PreparedOpeningPoint<F, E>,
    inner_opening_point: &[E],
    basis: BasisMode,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    if <E as ExtField<F>>::EXT_DEGREE == 1 {
        return (*folded_ring * prepared_point.packed_inner_trusted::<D>()?.sigma_m1())
            .coefficients()
            .first()
            .copied()
            .map(E::lift_base)
            .ok_or_else(|| AkitaError::InvalidInput("empty folded opening ring".to_string()));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }
    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if inner_opening_point.len() > packed_inner_bits
        && inner_opening_point[packed_inner_bits..]
            .iter()
            .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: inner_opening_point.len(),
        });
    }
    let mut point =
        inner_opening_point[..inner_opening_point.len().min(packed_inner_bits)].to_vec();
    point.resize(packed_inner_bits, E::zero());
    let weights = basis_weights(&point, basis)?;
    let packed_inner_point = embed_ring_subfield_vector::<F, E, D>(
        &weights,
        AkitaError::InvalidInput(
            "root opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    recover_ring_subfield_inner_product::<F, E, D>(folded_ring, &packed_inner_point)
}

pub(in crate::protocol::core) fn row_coefficient_rings<F, E, const D: usize>(
    coefficients: &[E],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    coefficients
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, E, D>(
                coefficient,
                AkitaError::InvalidInput(
                    "public-row coefficient does not encode in the ring-subfield basis".to_string(),
                ),
            )
        })
        .collect()
}
