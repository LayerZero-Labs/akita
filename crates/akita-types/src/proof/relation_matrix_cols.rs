//! Shared singleton and multi-group relation matrix column evaluation.
//!
//! [`compute_relation_matrix_col_evals`] materializes the tau1-weighted relation-matrix column
//! vector `relation_matrix_col_evals` that the fused stage-2 sumcheck treats as the row
//! polynomial. The prover still materializes this table for stage-2 proving.
//! The verifier replays the same group-major geometry with its structured
//! `RelationMatrixEvaluator` path instead of rebuilding the dense vector.

use crate::layout::CommitmentRingDims;
use crate::proof::ring_relation::RingRelationInstance;
use crate::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    OpeningBlockLayout, RelationMatrixRowLayout,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_challenges::Challenges;
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
};

/// Unified relation matrix column evaluation for singleton and multi-group root relations.
///
/// Singleton roots use the scalar/chunked witness layout. Multi-group roots use the
/// group-major layout and still reject multi-chunk witness emission.
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// chunking configuration, or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_relation_matrix_col_evals")]
pub fn compute_relation_matrix_col_evals<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    opening_layout: OpeningBlockLayout,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    let opening_batch = instance.opening_batch();
    lp.witness_chunk.validate()?;
    lp.reject_multi_group_multi_chunk("compute_relation_matrix_col_evals")?;
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
    let rows =
        lp.relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.n_d_active_for(relation_matrix_row_layout);
    let levels = r_decomp_levels::<F>(lp.log_basis);
    let witness_layout = instance.segment_layout(lp, None)?;
    if witness_layout.relation_rows != rows || witness_layout.quotient_depth != levels {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    let e_total = witness_layout
        .groups
        .iter()
        .try_fold(0usize, |total, group| {
            group
                .num_claims
                .checked_mul(group.num_blocks)
                .and_then(|n| n.checked_mul(group.depth_open))
                .and_then(|width| total.checked_add(width))
                .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
        })?;
    let total_cols = witness_layout.total_len();
    if total_cols > opening_layout.physical_len() {
        return Err(AkitaError::InvalidSize {
            expected: opening_layout.physical_len(),
            actual: total_cols,
        });
    }
    let mut out = vec![E::zero(); opening_layout.opening_len()];

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

    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_id = crate::SemanticGroupId(group_index);
        let witness_group = witness_layout.group(group_id)?;
        let units = witness_layout.units_for_group(group_id)?;
        let k_g = group_layout.num_polynomials();
        let opening_point = instance.group_opening_point(group_index)?;
        let ring_multiplier_point = instance.group_ring_multiplier_point(group_index)?;
        let challenges = &instance.group_challenges()[group_index];
        let group_opening_layout =
            OpeningBlockLayout::new(group_lp.num_blocks(), group_lp.block_len())?;
        if opening_point.a.len() != group_opening_layout.position_stride()
            || opening_point.b.len() != group_lp.num_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval opening-point layout mismatch".to_string(),
            ));
        }
        if ring_multiplier_point.a_len() != group_opening_layout.position_stride()
            || ring_multiplier_point.b_len() != group_lp.num_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval multiplier layout mismatch".to_string(),
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
        let depth_open = witness_group.depth_open;
        let depth_commit = witness_group.depth_commit;
        let depth_fold = witness_group.depth_fold;
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
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let b_width = k_g
            .checked_mul(t_cols_per_vector)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
        let setup_a_view = setup.shared_matrix.ring_view_dyn(n_a, inner_width, d_a)?;
        let b_view = setup.shared_matrix.ring_view_dyn(n_b, b_width, d_b)?;
        let setup_a_rows: Vec<&[F]> = (0..n_a)
            .map(|r| setup_a_view.row_flat(r))
            .collect::<Result<_, _>>()?;
        let b_rows: Vec<&[F]> = (0..n_b)
            .map(|r| b_view.row_flat(r))
            .collect::<Result<_, _>>()?;
        let a_range =
            lp.root_a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        let b_range =
            lp.root_commitment_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        let a_row_weights = &eq_tau1[a_range];
        let b_weights = &eq_tau1[b_range];
        let g_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let commit_gadget: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis)
            .into_iter()
            .map(E::lift_base)
            .collect();

        for claim in 0..k_g {
            for global_block in 0..num_blocks_g {
                let unit = witness_layout.unit_for_block(group_id, global_block)?;
                let challenge_index = claim * num_blocks_g + global_block;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    let d_phys_col =
                        witness_layout.e_setup_col_index(group_id, claim, global_block, digit)?;
                    let mut acc = consistency_weight * c_alphas[challenge_index] * opening_gadget;
                    for (di, eq_i) in eq_tau1[d_start..(d_start + n_d_active)].iter().enumerate() {
                        if !eq_i.is_zero() {
                            acc += *eq_i
                                * eval_flat_ring_at_pows_fast(
                                    &d_rows[di][d_phys_col * d_d..(d_phys_col + 1) * d_d],
                                    &alpha_pows_d,
                                );
                        }
                    }
                    let witness_col = witness_layout.e_index(unit, claim, global_block, digit)?;
                    let opening_col = opening_layout.opening_index_for_physical(witness_col)?;
                    *out.get_mut(opening_col).ok_or(AkitaError::InvalidProof)? = acc;
                }
                for (a_idx, &a_row_weight) in a_row_weights.iter().enumerate() {
                    for (digit, &opening_gadget) in g_open.iter().enumerate() {
                        let local_col = witness_layout.t_setup_col_index(
                            group_id,
                            claim,
                            global_block,
                            a_idx,
                            digit,
                        )?;
                        let mut acc = a_row_weight * c_alphas[challenge_index] * opening_gadget;
                        for (row_idx, eq_i) in b_weights.iter().enumerate() {
                            if !eq_i.is_zero() {
                                acc += *eq_i
                                    * eval_flat_ring_at_pows_fast(
                                        &b_rows[row_idx][local_col * d_b..(local_col + 1) * d_b],
                                        &alpha_pows_b,
                                    );
                            }
                        }
                        let witness_col =
                            witness_layout.t_index(unit, claim, global_block, a_idx, digit)?;
                        let opening_col = opening_layout.opening_index_for_physical(witness_col)?;
                        *out.get_mut(opening_col).ok_or(AkitaError::InvalidProof)? = acc;
                    }
                }
            }
        }

        // For z_hat[blk, dc, df], the column value is:
        //
        // -G_fold[df] * (
        //     tau_consistency * a_alpha[blk] * G_commit[dc]
        //     + sum_r tau_A[r] * A_alpha[r, blk, dc]
        //   ).
        //
        // The first term is the opening row. The second term is the A-row setup
        // contribution. A is already digit-domain, so the A-row setup term does
        // not multiply by G_commit.
        let z_base = cfg_into_iter!(0..inner_width)
            .map(|k| {
                let block_idx = k / depth_commit;
                let digit_idx = k % depth_commit;
                let opening_a_eval =
                    ring_multiplier_point.eval_a_at_dyn::<E>(block_idx, alpha_pows_a)?;
                let mut acc = consistency_weight * opening_a_eval * commit_gadget[digit_idx];
                for (a_idx, eq_i) in a_row_weights.iter().enumerate() {
                    if !eq_i.is_zero() {
                        acc += *eq_i
                            * eval_flat_ring_at_pows_fast(
                                &setup_a_rows[a_idx][k * d_a..(k + 1) * d_a],
                                alpha_pows_a,
                            );
                    }
                }
                Ok(acc)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for unit in units {
            for position in 0..block_len_g {
                for commit_digit in 0..depth_commit {
                    for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                        let phys_k = position * depth_commit + commit_digit;
                        let witness_col =
                            witness_layout.z_index(unit, position, commit_digit, fold_digit)?;
                        let opening_col = opening_layout.opening_index_for_physical(witness_col)?;
                        *out.get_mut(opening_col).ok_or(AkitaError::InvalidProof)? =
                            -(z_base[phys_k] * fold);
                    }
                }
            }
        }
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for (row, eq_weight) in eq_tau1.iter().take(rows).enumerate() {
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let witness_col = witness_layout.r_index(row, digit)?;
            let opening_col = opening_layout.opening_index_for_physical(witness_col)?;
            *out.get_mut(opening_col).ok_or(AkitaError::InvalidProof)? =
                -(*eq_weight * denom * *gadget);
        }
    }
    Ok(out)
}
