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
    log_basis: u32,
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
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let delta_commit = num_digits_inner(decomp, false);
    let delta_open = num_digits_open(decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, delta_commit) else {
        return Ok(None);
    };
    let Some(num_fold_coeffs) = width_s.checked_mul(dimensions.d_a()) else {
        return Ok(None);
    };
    let fold_policy = BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
        policy.decomposition.field_bits(),
        FoldWitnessNorms::bounded(decomp.log_basis, dimensions.d_a()),
    );
    let Ok(num_digits_fold) = fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: dimensions.d_a(),
        num_claims: 1,
        num_live_blocks,
        num_fold_coeffs,
        log_basis: decomp.log_basis,
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
        decomp.log_basis,
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
        log_basis,
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
        log_basis,
    ) else {
        return Ok(None);
    };
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, width_w)
    else {
        return Ok(None);
    };
    let params = CommittedGroupParams {
        log_basis_inner: log_basis,
        log_basis_outer: log_basis,
        log_basis_open: log_basis,
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

/// Compute parameters that generate the smallest witness for the next
/// fold level. Note that this is not the optimum case: in the optimum
/// case (similar to `find_schedule`), we should check that current proof
/// size + suffix cost is the smallest. However, as time blows up, we
/// don't do that here.
fn push_recursive_split_candidate(candidates: &mut Vec<usize>, reduced_vars: usize, p: isize) {
    if p <= 0 || p >= reduced_vars as isize {
        return;
    }
    let r = reduced_vars - p as usize;
    if !candidates.contains(&r) {
        candidates.push(r);
    }
}

pub(super) fn seed_recursive_split_candidates(
    num_ring_elems: usize,
    reduced_vars: usize,
    delta_commit: usize,
    delta_open: usize,
    num_chunks: usize,
) -> Vec<usize> {
    if reduced_vars <= 12 {
        return (1..reduced_vars).rev().collect();
    }

    let mut candidates = Vec::new();
    push_recursive_split_candidate(&mut candidates, reduced_vars, 1);
    push_recursive_split_candidate(&mut candidates, reduced_vars, reduced_vars as isize - 1);

    let target_num = 2u128
        .saturating_mul(delta_open as u128)
        .saturating_mul(num_ring_elems as u128);
    let target_den = (delta_commit as u128).saturating_mul(num_chunks.max(1) as u128);
    if target_num > 0 && target_den > 0 {
        let mut center = 1usize;
        let mut best_distance: Option<u128> = None;
        for p in 1..reduced_vars {
            let Some(power) = 1u128.checked_shl((2 * p) as u32) else {
                break;
            };
            let scaled = target_den.saturating_mul(power);
            let distance = scaled.abs_diff(target_num);
            if best_distance.is_none_or(|best| distance < best) {
                center = p;
                best_distance = Some(distance);
            }
        }
        for offset in -5..=5 {
            push_recursive_split_candidate(&mut candidates, reduced_vars, center as isize + offset);
        }
    }

    candidates.sort_by(|left, right| right.cmp(left));
    candidates
}

/// Lower bound on the final layout score for one recursive split.
///
/// The true score is `next_witness_len + challenge_work + chunk_work +
/// imbalance`. For any feasible scalar recursive candidate,
/// `next_witness_len` includes at least `D * (e_hat + t_hat + z_hat)`, with
/// `n_A >= 1`, `num_digits_fold >= 1`, and any setup-prefix / relation-tail
/// terms only increasing the witness. A split whose lower bound already exceeds
/// the current best score therefore cannot become optimal.
#[derive(Clone, Copy)]
pub(super) struct RecursiveSplitLowerBoundInput {
    pub(super) num_ring_elems: usize,
    pub(super) ring_dimension: usize,
    pub(super) reduced_vars: usize,
    pub(super) r: usize,
    pub(super) delta_commit: usize,
    pub(super) delta_open: usize,
    pub(super) num_chunks: usize,
    pub(super) requested_fold_shape: TensorChallengeShape,
}

pub(super) fn recursive_split_lower_bound(input: RecursiveSplitLowerBoundInput) -> Option<usize> {
    if input.r == 0 || input.r >= input.reduced_vars {
        return None;
    }
    let p = input.reduced_vars.checked_sub(input.r)?;
    let num_positions_per_block = 1usize.checked_shl(p as u32)?;
    let num_live_blocks = input.num_ring_elems.div_ceil(num_positions_per_block);

    let e_hat = num_live_blocks.checked_mul(input.delta_open)?;
    let t_hat_floor = e_hat;
    let z_hat_floor = num_positions_per_block
        .checked_mul(input.delta_commit)?
        .checked_mul(input.num_chunks.max(1))?;
    let physical_width_floor = e_hat
        .checked_add(t_hat_floor)?
        .checked_add(z_hat_floor)?
        .checked_mul(input.ring_dimension)?;
    let fold_shape =
        optimize_fold_challenge_shape(input.requested_fold_shape, num_live_blocks).ok()?;
    let challenge_work = match fold_shape {
        TensorChallengeShape::Flat => num_live_blocks,
        TensorChallengeShape::Tensor { fold_low_len } => {
            fold_low_len.checked_add(num_live_blocks.div_ceil(fold_low_len))?
        }
    };
    physical_width_floor
        .checked_add(challenge_work)?
        .checked_add(num_live_blocks)
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
    setup_prefix: Option<akita_types::SetupPrefixSlotId>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_recursive_level_search(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    log_basis: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let num_chunks = policy.chunks_at_level(fold_level);
    dimensions.validate_a_carrier()?;
    if !current_witness_len.is_multiple_of(dimensions.d_a()) {
        return Ok(None);
    }
    let num_ring_elems = current_witness_len / dimensions.d_a();
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

    let setup_prefix = match incoming_setup_prefix {
        Some(natural_len) => {
            if dimensions != CommitmentRingDims::uniform(akita_types::SETUP_OFFLOAD_D_SETUP) {
                return Err(AkitaError::InvalidSetup(
                    "recursive setup planning requires uniform D64".to_string(),
                ));
            }
            let n_prefix = padded_setup_prefix_len(natural_len);
            let Some(group) = derive_setup_prefix_group(
                policy,
                ring_challenge_cfg,
                requested_fold_shape,
                log_basis,
                log_basis,
                n_prefix,
                num_chunks,
                dimensions.d_b(),
            )?
            else {
                return Ok(None);
            };
            Some(akita_types::setup_prefix_slot_id(
                dimensions.d_a(),
                natural_len,
                group,
            ))
        }
        None => None,
    };
    Ok(Some(RecursiveLevelSearch {
        num_chunks,
        num_ring_elems,
        reduced_vars,
        setup_prefix,
    }))
}

#[allow(clippy::too_many_arguments)]
fn recursive_level_candidate_for_split(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    search: &RecursiveLevelSearch,
    log_basis: u32,
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
        log_basis,
        fold_level,
        block_index_bits,
        requested_fold_shape,
    )?
    else {
        return Ok(None);
    };
    candidate_params.setup_prefix = search.setup_prefix.clone();
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
    let next_witness_len = planned_next_witness_len(
        policy.decomposition.field_bits(),
        &candidate_params,
        1,
        search.num_chunks,
    )?;
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
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    log_basis: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Option<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(
        policy,
        ring_challenge_cfg,
        dimensions,
        current_witness_len,
        log_basis,
        fold_level,
        incoming_setup_prefix,
        requested_fold_shape,
    )?
    else {
        return Ok(None);
    };

    // The exhaustive scan visited larger `r` first and retained the first
    // equal-scoring candidate. Keep that tie-break explicit because the
    // seed-first search intentionally evaluates splits in a different order.
    let mut best: Option<(LayoutCandidateScore, usize, CommittedGroupParams, usize)> = None;
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let delta_commit = num_digits_inner(decomp, false);
    let delta_open = num_digits_open(decomp);
    let mut evaluated = Vec::new();
    let mut candidates = seed_recursive_split_candidates(
        search.num_ring_elems,
        search.reduced_vars,
        delta_commit,
        delta_open,
        search.num_chunks,
    );
    candidates.extend((1..search.reduced_vars).rev());

    // Evaluate a square-root-model seed window first, then finish the exact
    // search with a cheap lower-bound filter. The filter may evaluate extra
    // splits when the bound is loose, but it never skips a split that can beat
    // the current best layout score.
    for r in candidates {
        if evaluated.contains(&r) {
            continue;
        }
        evaluated.push(r);
        if let Some((best_score, _, _, _)) = &best {
            if let Some(lower_bound) = recursive_split_lower_bound(RecursiveSplitLowerBoundInput {
                num_ring_elems: search.num_ring_elems,
                ring_dimension: dimensions.d_a(),
                reduced_vars: search.reduced_vars,
                r,
                delta_commit,
                delta_open,
                num_chunks: search.num_chunks,
                requested_fold_shape,
            }) {
                if lower_bound > best_score.0 {
                    continue;
                }
            }
        }
        let Some((score, candidate_params, next_witness_len)) =
            recursive_level_candidate_for_split(
                policy,
                ring_challenge_cfg,
                dimensions,
                &search,
                log_basis,
                fold_level,
                r,
                requested_fold_shape,
            )?
        else {
            continue;
        };
        if best.as_ref().is_none_or(|(best_score, best_r, _, _)| {
            recursive_candidate_order_key(score, r)
                < recursive_candidate_order_key(*best_score, *best_r)
        }) {
            best = Some((score, r, candidate_params, next_witness_len));
        }
    }

    let Some((_, _, candidate_params, next_witness_len)) = best else {
        return Ok(None);
    };

    if next_witness_len >= current_witness_len {
        return Ok(None);
    }

    Ok(Some((candidate_params, next_witness_len)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_all_splits(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    log_basis: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(
        policy,
        ring_challenge_cfg,
        dimensions,
        current_witness_len,
        log_basis,
        fold_level,
        incoming_setup_prefix,
        requested_fold_shape,
    )?
    else {
        return Ok(Vec::new());
    };
    let mut candidates = Vec::new();
    for block_index_bits in (1..search.reduced_vars).rev() {
        let Some((_, params, next_witness_len)) = recursive_level_candidate_for_split(
            policy,
            ring_challenge_cfg,
            dimensions,
            &search,
            log_basis,
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
    Ok(candidates)
}
