use super::*;

/// Build one scalar root-fold candidate for an explicit basis and split.
///
/// `Ok(None)` is the canonical candidate-infeasibility signal used by both
/// schedule optimization and setup-capacity certification.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_root_fold_level_params_candidate(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    num_vars: usize,
    num_claims: usize,
    log_basis: u32,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
    honest_fold_policy: HonestFoldPolicySpec,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    dimensions.validate_role_projection()?;
    let alpha = (dimensions.d_a() as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    if reduced_vars == 0 || block_index_bits >= reduced_vars {
        return Ok(None);
    }
    let num_live_blocks = 1usize.checked_shl(block_index_bits as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root candidate num_live_blocks overflow".to_string())
    })?;
    let root_num_chunks = policy.chunks_at_level(0);
    if num_live_blocks < root_num_chunks {
        return Ok(None);
    }
    let fold_challenge_shape =
        optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
    let position_index_bits = reduced_vars - block_index_bits;
    let num_positions_per_block =
        1usize
            .checked_shl(position_index_bits as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root candidate position count overflow".to_string())
            })?;
    let num_live_ring_elements_per_claim = num_live_blocks
        .checked_mul(num_positions_per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("root candidate source length overflow".into()))?;
    let level_decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let witness_decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let num_digits_inner = num_digits_inner(witness_decomp, true);
    let num_digits_open = num_digits_open(level_decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
    else {
        return Ok(None);
    };
    let Some(num_fold_coeffs) = width_s.checked_mul(dimensions.d_a()) else {
        return Ok(None);
    };
    let Ok(num_digits_fold) = honest_fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: dimensions.d_a(),
        num_claims,
        num_live_blocks,
        num_fold_coeffs,
        log_basis,
        challenge_config: ring_challenge_cfg,
        challenge_shape: fold_challenge_shape,
    }) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        dimensions.d_a(),
        log_basis,
        ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_fold,
        policy.ring_subfield_norm_bound,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(
            policy,
            akita_types::SisMatrixRole::Inner,
            dimensions.d_a(),
            norm_s,
        ),
        width_s,
    ) else {
        return Ok(None);
    };
    let Some(native_width_t) = decomposed_t_ring_count(
        inner_commit_matrix.output_rank(),
        num_digits_open,
        num_live_blocks,
        num_claims,
    ) else {
        return Ok(None);
    };
    let Some((outer_key, width_t)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Outer,
        dimensions.d_a(),
        dimensions.d_b(),
        native_width_t,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Ok(outer_commit_matrix) =
        OuterCommitMatrixParams::try_new_with_min_rank(outer_key, width_t)
    else {
        return Ok(None);
    };
    let Some(native_width_w) =
        decomposed_w_ring_count(num_digits_open, num_live_blocks, num_claims)
    else {
        return Ok(None);
    };
    let Some((open_key, width_w)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Open,
        dimensions.d_a(),
        dimensions.d_d(),
        native_width_w,
        log_basis,
    ) else {
        return Ok(None);
    };
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, width_w)
    else {
        return Ok(None);
    };
    let params = CommittedGroupParams {
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        log_basis_inner: witness_decomp.log_basis,
        log_basis_outer: log_basis,
        log_basis_open: log_basis,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_inner,
        num_digits_outer: num_digits_open,
        num_digits_open,
        num_digits_fold,
        witness_chunk: policy.witness_chunk_for_level(0),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
    };
    Ok(Some(params))
}
