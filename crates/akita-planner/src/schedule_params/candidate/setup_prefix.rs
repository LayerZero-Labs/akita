use super::*;

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "test-support"), allow(unreachable_pub))]
/// Derive one secure setup-prefix commitment candidate.
///
/// Returns `None` when the requested geometry has no audited feasible candidate.
pub fn derive_setup_prefix_group(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    log_basis_outer: u32,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    outer_ring_dimension: usize,
) -> Result<Option<PrecommittedLevelParams>, AkitaError> {
    validate_policy(policy)?;
    if outer_ring_dimension == 0
        || !outer_ring_dimension.is_power_of_two()
        || !policy.ring_dimension.is_multiple_of(outer_ring_dimension)
    {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix B dimension must be a power-of-two divisor of its A dimension"
                .to_string(),
        ));
    }
    if n_prefix == 0 || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a nonzero power of two".to_string(),
        ));
    }
    if !n_prefix.is_multiple_of(policy.ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a multiple of the ring dimension".to_string(),
        ));
    }
    if log_basis_outer != log_basis_open {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix checkpoint requires one consuming inner/outer/open basis".to_string(),
        ));
    }
    let ring_slots = n_prefix / policy.ring_dimension;
    let reduced_vars = checked_power_of_two_vars(ring_slots, "setup prefix ring slots")?;
    let prefix_num_vars = checked_power_of_two_vars(n_prefix, "setup prefix field length")?;
    let family = policy.sis_modulus_profile;
    let d = policy.ring_dimension;
    let outer_decomp = DecompositionParams {
        log_basis: log_basis_outer,
        ..policy.decomposition
    };
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(outer_decomp);
    let num_digits_open_val = num_digits_open(open_decomp);
    let mut best: Option<(LayoutCandidateScore, PrecommittedLevelParams)> = None;

    // The current protocol has one Stage-1 range polynomial. Until role-specific
    // range proofs exist, setup-prefix source, commitment, and opening digits use
    // the consuming fold's certified basis and only block geometry is searched.
    let log_basis_inner = log_basis_open;
    let inner_decomp = DecompositionParams {
        log_basis: log_basis_inner,
        ..policy.decomposition
    };
    let num_digits_inner = num_digits_setup_prefix_commit(inner_decomp);
    for block_index_bits in (0..=reduced_vars).rev() {
        let Some(num_live_blocks) = 1usize.checked_shl(block_index_bits as u32) else {
            continue;
        };
        let position_index_bits = reduced_vars - block_index_bits;
        let Some(num_positions_per_block) = 1usize.checked_shl(position_index_bits as u32) else {
            continue;
        };
        let fold_shape = optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
        if num_live_blocks < num_chunks {
            continue;
        }
        let Some(width_s) =
            decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
        else {
            continue;
        };
        let Some(norm_s) = rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            family,
            d,
            inner_decomp,
            log_basis_open,
            ring_challenge_cfg,
            fold_shape,
            false,
            0,
            policy.ring_subfield_norm_bound,
            num_live_blocks,
            1,
            width_s as u64,
        ) else {
            continue;
        };
        let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
            sis_key_at_dimension(
                policy,
                akita_types::SisMatrixRole::Inner,
                policy.ring_dimension,
                norm_s,
            ),
            width_s,
        ) else {
            continue;
        };
        let Some(norm_t) = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            family,
            akita_types::SisMatrixRole::Outer,
            outer_ring_dimension,
            log_basis_open,
        ) else {
            continue;
        };
        let Some(width_t) = decomposed_t_ring_count(
            inner_commit_matrix.output_rank(),
            num_digits_outer,
            num_live_blocks,
            1,
        )
        .and_then(|width| width.checked_mul(d / outer_ring_dimension)) else {
            continue;
        };
        let Ok(outer_commit_matrix) = OuterCommitMatrixParams::try_new_with_min_rank(
            sis_key_at_dimension(
                policy,
                akita_types::SisMatrixRole::Outer,
                outer_ring_dimension,
                norm_t,
            ),
            width_t,
        ) else {
            continue;
        };
        let fold_linf_cap_config =
            FoldWitnessLinfCapConfig::for_fold_level(ring_challenge_cfg, fold_shape, d, width_s)?;
        let challenge = FoldChallengeNorms {
            infinity_norm: fold_shape.effective_infinity_norm(ring_challenge_cfg) as u128,
            l1_norm: fold_shape.effective_l1_mass(ring_challenge_cfg) as u128,
        };
        let (num_digits_fold_one, _) = fold_witness_digit_plan(
            num_live_blocks,
            1,
            policy.decomposition.field_bits(),
            log_basis_open,
            challenge,
            FoldWitnessNorms::new(log_basis_inner, d, 1, false),
            &fold_linf_cap_config,
        )?;
        let layout = PrecommittedGroupDescriptor {
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            num_live_ring_elements_per_claim: ring_slots,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner,
            log_basis_outer,
            inner_ring_dimension: inner_commit_matrix.ring_dimension(),
            outer_ring_dimension: outer_commit_matrix.ring_dimension(),
            n_a: inner_commit_matrix.output_rank(),
            a_coeff_linf_bound: inner_commit_matrix.coeff_linf_bound(),
            n_b: outer_commit_matrix.output_rank(),
            b_coeff_linf_bound: outer_commit_matrix.coeff_linf_bound(),
        };
        let params = PrecommittedLevelParams {
            layout,
            inner_commit_matrix,
            outer_commit_matrix,
            log_basis_open,
            fold_challenge_config: *ring_challenge_cfg,
            num_digits_inner,
            num_digits_outer,
            num_digits_open: num_digits_open_val,
            num_digits_fold_one,
        };
        let physical_width = grouped_segment_rings(
            1,
            num_live_blocks,
            num_chunks,
            num_positions_per_block,
            params.inner_commit_matrix.output_rank(),
            num_digits_inner,
            num_digits_outer,
            num_digits_open_val,
            num_digits_fold_one,
        )?;
        let score =
            layout_candidate_score(physical_width, num_live_blocks, num_chunks, fold_shape)?;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, params));
        }
    }

    Ok(best.map(|(_, params)| params))
}
