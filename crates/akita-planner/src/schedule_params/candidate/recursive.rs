use super::*;

/// Build one recursive-fold candidate for an explicit ring-element bucket and
/// split. Setup certification uses the maximum current length in each
/// `ceil(log2(ring_elems))` bucket, which dominates every shorter member for
/// the same split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_fold_level_params_candidate(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    num_ring_elems: usize,
    reduced_vars: usize,
    source: InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    if reduced_vars <= 2
        || reduced_vars >= 53
        || block_index_bits == 0
        || block_index_bits >= reduced_vars
    {
        return Ok(None);
    }
    let num_chunks = policy.chunks_at_level(fold_level);
    let num_positions_per_block = 1usize
        .checked_shl((reduced_vars - block_index_bits) as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("recursive candidate position count overflow".to_string())
        })?;
    let num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
    if num_live_blocks < num_chunks {
        return Ok(None);
    }
    let fold_challenge_shape =
        optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let delta_commit = source.num_digits_inner(policy.decomposition, log_basis_inner)?;
    let delta_open = num_digits_open(open_decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, delta_commit) else {
        return Ok(None);
    };
    let Some(num_fold_coeffs) = width_s
        .checked_mul(dimensions.d_a())
        .and_then(|count| count.checked_mul(num_chunks))
    else {
        return Ok(None);
    };
    let fold_policy = BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
        policy.decomposition.field_bits(),
        FoldWitnessNorms::bounded(log_basis_inner, dimensions.d_a()),
    );
    let Ok(num_digits_fold) = fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: dimensions.d_a(),
        num_claims: 1,
        num_live_blocks,
        num_chunks,
        num_fold_coeffs,
        witness_norms: FoldWitnessNorms::bounded(log_basis_inner, dimensions.d_a()),
        log_basis_response: log_basis_open,
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
        log_basis_open,
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
        delta_open,
        num_live_blocks,
        1,
    ) else {
        return Ok(None);
    };
    let Some((outer_key, width_t)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Outer,
        dimensions.d_a(),
        dimensions.d_b(),
        native_width_t,
        log_basis_open,
    ) else {
        return Ok(None);
    };
    let Ok(outer_commit_matrix) =
        OuterCommitMatrixParams::try_new_with_min_rank(outer_key, width_t)
    else {
        return Ok(None);
    };
    let Some(native_width_w) = decomposed_w_ring_count(delta_open, num_live_blocks, 1) else {
        return Ok(None);
    };
    let Some((open_key, width_w)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Open,
        dimensions.d_a(),
        dimensions.d_d(),
        native_width_w,
        log_basis_open,
    ) else {
        return Ok(None);
    };
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, width_w)
    else {
        return Ok(None);
    };
    let params = CommittedGroupParams {
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        log_basis_inner,
        log_basis_outer: log_basis_open,
        log_basis_open,
        inner_commit_matrix,
        outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim: num_ring_elems,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: *ring_challenge_cfg,
        fold_challenge_shape,
        num_digits_inner: delta_commit,
        num_digits_outer: delta_open,
        num_digits_open: delta_open,
        num_digits_fold,
        witness_chunk: policy.witness_chunk_for_level(fold_level),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
    };
    Ok(Some(params))
}

pub(super) fn recursive_candidate_order_key(
    score: LayoutCandidateScore,
    block_index_bits: usize,
) -> (LayoutCandidateScore, std::cmp::Reverse<usize>) {
    (score, std::cmp::Reverse(block_index_bits))
}

struct RecursiveLevelSearch {
    num_chunks: usize,
    num_ring_elems: usize,
    reduced_vars: usize,
    setup_prefixes: Vec<Option<akita_types::SetupPrefixSlotId>>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_recursive_level_search(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let num_chunks = policy.chunks_at_level(fold_level);
    dimensions.validate_role_projection()?;
    if current_witness_len == 0 {
        return Ok(None);
    }
    // The previous fold owns a compact field-coefficient buffer. It need not
    // end on the next A-ring boundary; commitment alignment pads only the
    // transient ring view. Plan from the live coefficient count, rounding up
    // solely to determine the next fold's block geometry.
    let num_ring_elems = current_witness_len.div_ceil(dimensions.d_a());
    let reduced_vars = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness capacity overflow".to_string()))?
        .max(1)
        .trailing_zeros() as usize;

    if reduced_vars <= 2 || reduced_vars >= 53 {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive fold candidate reduced_vars={reduced_vars} is outside \
             the optimizable range [3, 52]"
        )));
    }

    let setup_prefixes = match incoming_setup_prefix {
        Some(natural_len) => {
            let cache = setup_prefix_cache.ok_or_else(|| {
                AkitaError::InvalidSetup("setup-prefix planning requires a search cache".into())
            })?;
            let n_prefix = padded_setup_prefix_len(natural_len);
            let groups = derive_setup_prefix_groups(
                cache,
                policy,
                ring_challenge_cfg,
                requested_fold_shape,
                log_basis_open,
                n_prefix,
                num_chunks,
                dimensions.d_b(),
            )?;
            if groups.is_empty() {
                return Ok(None);
            }
            groups
                .into_iter()
                .map(|group| Some(akita_types::setup_prefix_slot_id(natural_len, group)))
                .collect()
        }
        None => vec![None],
    };
    Ok(Some(RecursiveLevelSearch {
        num_chunks,
        num_ring_elems,
        reduced_vars,
        setup_prefixes,
    }))
}

#[allow(clippy::too_many_arguments)]
fn recursive_level_candidate_for_split(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    search: &RecursiveLevelSearch,
    setup_prefix: Option<&akita_types::SetupPrefixSlotId>,
    source: InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<(LayoutCandidateScore, CommittedGroupParams, usize)>, AkitaError> {
    let Some(mut candidate_params) = recursive_fold_level_params_candidate(
        policy,
        ring_challenge_cfg,
        dimensions,
        search.num_ring_elems,
        search.reduced_vars,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        block_index_bits,
        requested_fold_shape,
    )?
    else {
        return Ok(None);
    };
    candidate_params.payload_mode = payload_mode;
    candidate_params.setup_prefix = setup_prefix.cloned();
    if let Some(prefix) = &candidate_params.setup_prefix {
        let prefix_d_width = prefix
            .commitment_params
            .d_segment_width(candidate_params.role_dims().d_d())?;
        let total_d_width = candidate_params
            .open_commit_matrix
            .input_width()
            .checked_add(prefix_d_width)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup-prefix shared D width overflow".to_string())
            })?;
        candidate_params.open_commit_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
            candidate_params.open_commit_matrix.sis_table_key(),
            total_d_width,
        )?;
    }
    let Some(next_witness_len) = scalar_next_witness_len_if_supported(
        policy.decomposition.field_bits(),
        &candidate_params,
        1,
    )?
    else {
        return Ok(None);
    };
    let score = layout_candidate_score(
        next_witness_len,
        candidate_params.num_live_blocks,
        search.num_chunks,
        candidate_params.fold_challenge_shape,
    )?;
    Ok(Some((score, candidate_params, next_witness_len)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<(CommittedGroupParams, usize)>, AkitaError> {
    let candidates = derive_candidate_level_params_all_splits(
        setup_prefix_cache,
        policy,
        payload_mode,
        ring_challenge_cfg,
        dimensions,
        current_witness_len,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
        requested_fold_shape,
    )?;
    candidates
        .into_iter()
        .map(|(params, next_witness_len)| {
            let score = layout_candidate_score(
                next_witness_len,
                params.num_live_blocks,
                params.witness_chunk.num_chunks,
                params.fold_challenge_shape,
            )?;
            let block_index_bits = params.num_live_blocks.trailing_zeros() as usize;
            Ok((
                recursive_candidate_order_key(score, block_index_bits),
                params,
                next_witness_len,
            ))
        })
        .collect::<Result<Vec<_>, AkitaError>>()
        .map(|scored| {
            scored
                .into_iter()
                .min_by_key(|(order, _, _)| *order)
                .map(|(_, params, next_witness_len)| (params, next_witness_len))
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_all_splits(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(
        setup_prefix_cache,
        policy,
        ring_challenge_cfg,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
        requested_fold_shape,
    )?
    else {
        return Ok(Vec::new());
    };
    let mut candidates = Vec::new();
    for block_index_bits in (1..search.reduced_vars).rev() {
        for setup_prefix in &search.setup_prefixes {
            let Some((_, params, next_witness_len)) = recursive_level_candidate_for_split(
                policy,
                payload_mode,
                ring_challenge_cfg,
                dimensions,
                &search,
                setup_prefix.as_ref(),
                source,
                log_basis_inner,
                log_basis_open,
                fold_level,
                block_index_bits,
                requested_fold_shape,
            )?
            else {
                continue;
            };
            if next_witness_len < current_witness_len {
                candidates.push((params, next_witness_len));
            }
        }
    }
    Ok(candidates)
}
