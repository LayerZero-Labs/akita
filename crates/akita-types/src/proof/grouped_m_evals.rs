//! Shared grouped/singleton M-table column evaluation.
//!
//! [`compute_grouped_m_evals_x`] materializes the tau1-weighted M-row column
//! vector `m_evals_x` that the fused stage-2 sumcheck treats as the row
//! polynomial. The prover still materializes this table for stage-2 proving.
//! The verifier replays the same group-major geometry with its structured
//! `RingSwitchDeferredRowEval` path instead of rebuilding the dense vector.

use crate::layout::CommitmentRingDims;
use crate::proof::ring_relation::RingRelationInstance;
use crate::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding, LevelParams, MRowLayout,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_flat_ring_at_pows, scalar_powers};
use akita_challenges::Challenges;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase};

/// Unified M-table column evaluation for singleton and grouped root relations.
///
/// Singleton roots use the scalar/chunked witness layout. Grouped roots use the
/// group-major layout and still reject multi-chunk witness emission.
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// chunking configuration, or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_grouped_m_evals_x")]
pub fn compute_grouped_m_evals_x<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    m_row_layout: MRowLayout,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    let opening_batch = instance.opening_batch();
    lp.witness_chunk.validate()?;
    lp.reject_grouped_multi_chunk("compute_grouped_m_evals_x")?;
    lp.validate_root_opening_batch(opening_batch)?;
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    if alpha_pows.len() != d_a {
        return Err(AkitaError::InvalidSize {
            expected: d_a,
            actual: alpha_pows.len(),
        });
    }
    let alpha_pows_a = alpha_pows;
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let rows = lp.m_row_count_for(opening_batch.num_groups(), m_row_layout)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.n_d_active_for(m_row_layout);
    let levels = r_decomp_levels::<F>(lp.log_basis);
    let order = opening_batch.root_group_order()?;
    let num_chunks = lp.witness_chunk.num_chunks;
    let use_chunked_singleton_layout = opening_batch.num_groups() == 1;
    let mut group_e_offsets = vec![0usize; opening_batch.num_groups()];
    let mut e_total = 0usize;
    let mut z_total = 0usize;
    let mut t_total = 0usize;
    for &group_index in &order {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        group_e_offsets[group_index] = e_total;
        let k_g = group_layout.num_polynomials();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
        let group_z_len = group_lp
            .block_len()
            .checked_mul(group_lp.num_digits_commit())
            .and_then(|n| n.checked_mul(depth_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("grouped z width overflow".to_string()))?;
        let group_z_cols = if use_chunked_singleton_layout {
            group_z_len.checked_mul(num_chunks).ok_or_else(|| {
                AkitaError::InvalidSetup("chunked grouped z width overflow".to_string())
            })?
        } else {
            group_z_len
        };
        z_total = z_total
            .checked_add(group_z_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped z width overflow".to_string()))?;
        let e_len = k_g
            .checked_mul(group_lp.num_blocks())
            .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
            .ok_or_else(|| AkitaError::InvalidSetup("grouped e width overflow".to_string()))?;
        e_total = e_total
            .checked_add(e_len)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped e width overflow".to_string()))?;
        t_total = t_total
            .checked_add(
                k_g.checked_mul(group_lp.num_blocks())
                    .and_then(|n| n.checked_mul(group_lp.a_rows_len()))
                    .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("grouped t width overflow".to_string())
                    })?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("grouped t width overflow".to_string()))?;
    }
    let r_tail_len = rows
        .checked_mul(levels)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped r width overflow".to_string()))?;
    let total_cols = z_total
        .checked_add(e_total)
        .and_then(|n| n.checked_add(t_total))
        .and_then(|n| n.checked_add(r_tail_len))
        .ok_or_else(|| AkitaError::InvalidSetup("grouped M width overflow".to_string()))?;
    let x_len = total_cols.next_power_of_two();
    let mut out = Vec::with_capacity(x_len);

    let d_view = setup
        .shared_matrix
        .ring_view_dyn(lp.d_key.row_len(), e_total, d_d)?;
    let d_rows: Vec<&[F]> = (0..lp.d_key.row_len())
        .map(|r| d_view.row_flat(r))
        .collect::<Result<_, _>>()?;
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let consistency_weight = eq_tau1[0];
    let alpha_pow_d = alpha_pows_d[d_d - 1] * alpha;
    let denom = alpha_pow_d + E::one();

    let mut gamma_offset = 0usize;
    let mut gamma_offsets = vec![0usize; opening_batch.num_groups()];
    for (group_index, offset) in gamma_offsets.iter_mut().enumerate() {
        *offset = gamma_offset;
        gamma_offset = gamma_offset
            .checked_add(opening_batch.group_layout(group_index)?.num_polynomials())
            .ok_or(AkitaError::InvalidProof)?;
    }

    let mut group_segments = Vec::with_capacity(opening_batch.num_groups());
    for (group_index, e_offset) in group_e_offsets.iter().copied().enumerate() {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let opening_point = instance.group_opening_point(group_index)?;
        let ring_multiplier_point = instance.group_ring_multiplier_point(group_index)?;
        let challenges = &instance.group_challenges()[group_index];
        if opening_point.a.len() < group_lp.block_len()
            || opening_point.b.len() != group_lp.num_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "grouped M eval opening-point layout mismatch".to_string(),
            ));
        }
        if ring_multiplier_point.a_len() < group_lp.block_len()
            || ring_multiplier_point.b_len() != group_lp.num_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "grouped M eval multiplier layout mismatch".to_string(),
            ));
        }
        let total_blocks = k_g
            .checked_mul(group_lp.num_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != total_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let c_alphas = match challenges {
            Challenges::Sparse {
                challenges: sparse, ..
            } => sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
                .collect::<Result<Vec<_>, _>>()?,
            Challenges::Tensor { factored: _ } => challenges.evals_at_pows::<F, E>(alpha_pows)?,
        };
        let depth_open = group_lp.num_digits_open();
        let depth_commit = group_lp.num_digits_commit();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
        let log_basis = group_lp.log_basis();
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
        let inner_width = group_lp.a_col_len();
        // Hoist per-group geometry into `Copy` locals so the parallel closures
        // below capture scalars instead of the `!Sync` `&dyn LevelParamsLike`.
        let num_blocks_g = group_lp.num_blocks();
        let block_len_g = group_lp.block_len();
        let t_cols_per_vector = n_a
            .checked_mul(depth_open)
            .and_then(|len| len.checked_mul(num_blocks_g))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("grouped B vector width overflow".to_string())
            })?;
        let b_width = k_g
            .checked_mul(t_cols_per_vector)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped B width overflow".to_string()))?;
        let a_view = setup.shared_matrix.ring_view_dyn(n_a, inner_width, d_a)?;
        let b_view = setup.shared_matrix.ring_view_dyn(n_b, b_width, d_b)?;
        let a_rows: Vec<&[F]> = (0..n_a)
            .map(|r| a_view.row_flat(r))
            .collect::<Result<_, _>>()?;
        let b_rows: Vec<&[F]> = (0..n_b)
            .map(|r| b_view.row_flat(r))
            .collect::<Result<_, _>>()?;
        let a_range = lp.root_a_row_range(opening_batch, group_index, m_row_layout)?;
        let b_range = lp.root_commitment_row_range(opening_batch, group_index, m_row_layout)?;
        let a_weights = &eq_tau1[a_range];
        let b_weights = &eq_tau1[b_range];
        let g_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let g_commit: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis)
            .into_iter()
            .map(E::lift_base)
            .collect();

        let e_segment = cfg_into_iter!(0..(total_blocks * depth_open))
            .map(|x| {
                let dig = x / total_blocks;
                let blk = x % total_blocks;
                let d_phys_col = e_offset + blk * depth_open + dig;
                let mut acc = consistency_weight * c_alphas[blk] * g_open[dig];
                for (di, eq_i) in eq_tau1[d_start..(d_start + n_d_active)].iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_flat_ring_at_pows(
                                &d_rows[di][d_phys_col * d_d..(d_phys_col + 1) * d_d],
                                &alpha_pows_d,
                            );
                    }
                }
                acc
            })
            .collect::<Vec<_>>();

        let mut challenge_sums_by_t_block = vec![E::zero(); total_blocks];
        for claim_idx in 0..k_g {
            for block_idx in 0..num_blocks_g {
                let idx = claim_idx * num_blocks_g + block_idx;
                challenge_sums_by_t_block[idx] += c_alphas[idx];
            }
        }
        let t_compound_per_block = n_a * depth_open;
        let t_segment = cfg_into_iter!(0..(total_blocks * t_compound_per_block))
            .map(|x| {
                let compound_dig = x / total_blocks;
                let blk = x % total_blocks;
                let a_idx = compound_dig / depth_open;
                let digit_idx = compound_dig % depth_open;
                let t_vector_idx = blk / num_blocks_g;
                let block_idx = blk % num_blocks_g;
                let phys_claim_offset =
                    block_idx * t_compound_per_block + a_idx * depth_open + digit_idx;
                let local_col = t_vector_idx * t_cols_per_vector + phys_claim_offset;
                let mut acc = a_weights[a_idx] * challenge_sums_by_t_block[blk] * g_open[digit_idx];
                for (row_idx, eq_i) in b_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_flat_ring_at_pows(
                                &b_rows[row_idx][local_col * d_b..(local_col + 1) * d_b],
                                &alpha_pows_b,
                            );
                    }
                }
                acc
            })
            .collect::<Vec<_>>();

        let z_base = cfg_into_iter!(0..inner_width)
            .map(|k| {
                let block_idx = k / depth_commit;
                let digit_idx = k % depth_commit;
                let a_eval = ring_multiplier_point.eval_a_at_dyn::<E>(block_idx, alpha_pows_a)?;
                let mut acc = consistency_weight * a_eval * g_commit[digit_idx];
                for (a_idx, eq_i) in a_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_flat_ring_at_pows(
                                &a_rows[a_idx][k * d_a..(k + 1) * d_a],
                                alpha_pows_a,
                            );
                    }
                }
                Ok(acc)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let z_segment = cfg_into_iter!(0..(block_len_g * depth_commit * depth_fold))
            .map(|x| {
                let compound_dig = x / block_len_g;
                let global_blk = x % block_len_g;
                let dc = compound_dig / depth_fold;
                let df = compound_dig % depth_fold;
                let phys_k = global_blk * depth_commit + dc;
                -(z_base[phys_k] * fold_gadget[df])
            })
            .collect::<Vec<_>>();
        group_segments.push((z_segment, e_segment, t_segment));
    }

    if use_chunked_singleton_layout {
        let (z_seg, e_seg, t_seg) = group_segments.first().ok_or(AkitaError::InvalidProof)?;
        let group_lp = lp.root_group_params(opening_batch, 0)?;
        let num_blocks = group_lp.num_blocks();
        if num_blocks == 0 {
            return Err(AkitaError::InvalidSetup(
                "chunked grouped M evals require a non-zero block count".to_string(),
            ));
        }
        let blocks_per_chunk = num_blocks.checked_div(num_chunks).ok_or_else(|| {
            AkitaError::InvalidSetup("chunked grouped M eval chunk count is zero".to_string())
        })?;
        if blocks_per_chunk == 0 {
            return Err(AkitaError::InvalidSetup(
                "chunked grouped M eval block window is empty".to_string(),
            ));
        }
        // Singleton chunked layout `[z|e_i|t_i]…[r]`: `z` is replicated per
        // window and `e`/`t` are partitioned by global block.
        for chunk in 0..num_chunks {
            out.extend_from_slice(z_seg);
            let block_lo = chunk * blocks_per_chunk;
            let block_hi = block_lo + blocks_per_chunk;
            for outer in e_seg.chunks_exact(num_blocks) {
                let window = outer
                    .get(block_lo..block_hi)
                    .ok_or(AkitaError::InvalidProof)?;
                out.extend_from_slice(window);
            }
            for outer in t_seg.chunks_exact(num_blocks) {
                let window = outer
                    .get(block_lo..block_hi)
                    .ok_or(AkitaError::InvalidProof)?;
                out.extend_from_slice(window);
            }
        }
    } else {
        // Group-major M-eval columns: each group's `[z_g ‖ e_g ‖ t_g]`
        // contiguously in `root_group_order()`, matching `ring_switch_build_w`.
        for &group_index in &order {
            let (z_seg, e_seg, t_seg) = &group_segments[group_index];
            out.extend_from_slice(z_seg);
            out.extend_from_slice(e_seg);
            out.extend_from_slice(t_seg);
        }
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for eq_weight in eq_tau1.iter().take(rows) {
        for gadget in &r_gadget {
            out.push(-(*eq_weight * denom * *gadget));
        }
    }
    out.resize(x_len, E::zero());
    Ok(out)
}
