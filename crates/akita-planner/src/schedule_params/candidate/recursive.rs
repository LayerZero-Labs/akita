use super::*;

/// Build one recursive-fold candidate for an explicit ring-element bucket and
/// split. Setup certification uses the maximum current length in each
/// `ceil(log2(ring_elems))` bucket, which dominates every shorter member for
/// the same split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_fold_level_params_candidates(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    num_ring_elems: usize,
    reduced_vars: usize,
    log_basis: u32,
    fold_level: usize,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    if reduced_vars <= 2
        || reduced_vars >= 53
        || block_index_bits == 0
        || block_index_bits >= reduced_vars
    {
        return Ok(Vec::new());
    }
    let num_chunks = policy.chunks_at_level(fold_level);
    let num_positions_per_block = 1usize
        .checked_shl((reduced_vars - block_index_bits) as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("recursive candidate position count overflow".to_string())
        })?;
    let num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
    if num_live_blocks < num_chunks {
        return Ok(Vec::new());
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
        return Ok(Vec::new());
    };
    let Some(num_fold_coeffs) = width_s.checked_mul(dimensions.d_a()) else {
        return Ok(Vec::new());
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
        return Ok(Vec::new());
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
        return Ok(Vec::new());
    };
    let inner_key = sis_key_at_dimension(
        policy,
        akita_types::SisMatrixRole::Inner,
        dimensions.d_a(),
        norm_s,
    );
    let Ok(linf_inner_commit_matrix) =
        InnerCommitMatrixParams::try_new_with_min_rank(inner_key, width_s)
    else {
        return Ok(Vec::new());
    };
    // UNSOUND DIAGNOSTIC: late, single-chunk folds may price A against a fixed
    // whole-witness L2 cap. The proof carries the observed norm but the
    // verifier intentionally does not check it yet. Enumerate distinct secure
    // L2 ranks by conservatively widening the collision bucket, then retain
    // the ordinary L∞ candidate as an independent fallback.
    let mut inner_commit_matrices = Vec::new();
    if fold_level >= 3 && num_chunks == 1 {
        // For every current fixed `psi` embedding, this policy value is also
        // its exact squared L2 operator norm: one on the base-field path and
        // two on the degree-2/4 paired-lane paths.
        let collision_l2_sq = role_a_collision_l2_sq_for_response_bound(
            ring_challenge_cfg.challenge_l2_sq_max(),
            policy.ring_subfield_norm_bound,
            UNCHECKED_L2_DIAGNOSTIC_NORM_SQ_CAP,
        );
        if let Some(mut bucket) = collision_l2_sq.and_then(ceil_supported_l2_collision_sq) {
            let mut previous_rank = None;
            while bucket <= (1u128 << 84) {
                let Ok(matrix) =
                    InnerCommitMatrixParams::try_new_with_unchecked_l2_diagnostic_min_rank(
                        inner_key, width_s, bucket,
                    )
                else {
                    break;
                };
                let rank = matrix.output_rank();
                if rank >= linf_inner_commit_matrix.output_rank() {
                    break;
                }
                if previous_rank != Some(rank) {
                    inner_commit_matrices.push(matrix);
                    previous_rank = Some(rank);
                }
                let Some(next_bucket) = bucket.checked_mul(2) else {
                    break;
                };
                bucket = next_bucket;
            }
        }
    }
    inner_commit_matrices.push(linf_inner_commit_matrix);

    let Some(native_width_w) = decomposed_w_ring_count(delta_open, num_live_blocks, 1) else {
        return Ok(Vec::new());
    };
    let Some((open_key, width_w)) = projected_collision_role_price(
        policy,
        akita_types::SisMatrixRole::Open,
        dimensions.d_a(),
        dimensions.d_d(),
        native_width_w,
        log_basis,
    ) else {
        return Ok(Vec::new());
    };
    let Ok(open_commit_matrix) = OpenCommitMatrixParams::try_new_with_min_rank(open_key, width_w)
    else {
        return Ok(Vec::new());
    };

    let mut candidates = Vec::with_capacity(inner_commit_matrices.len());
    for inner_commit_matrix in inner_commit_matrices {
        let Some(native_width_t) = decomposed_t_ring_count(
            inner_commit_matrix.output_rank(),
            delta_open,
            num_live_blocks,
            1,
        ) else {
            continue;
        };
        let Some((outer_key, width_t)) = projected_collision_role_price(
            policy,
            akita_types::SisMatrixRole::Outer,
            dimensions.d_a(),
            dimensions.d_b(),
            native_width_t,
            log_basis,
        ) else {
            continue;
        };
        let Ok(outer_commit_matrix) =
            OuterCommitMatrixParams::try_new_with_min_rank(outer_key, width_t)
        else {
            continue;
        };
        candidates.push(CommittedGroupParams {
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
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
        });
    }
    Ok(candidates)
}

/// Compute parameters that generate the smallest witness for the next
/// fold level. Note that this is not the optimum case: in the optimum
/// case (similar to `find_schedule`), we should check that current proof
/// size + suffix cost is the smallest. However, as time blows up, we
/// don't do that here.
#[cfg(test)]
fn push_recursive_split_candidate(candidates: &mut Vec<usize>, reduced_vars: usize, p: isize) {
    if p <= 0 || p >= reduced_vars as isize {
        return;
    }
    let r = reduced_vars - p as usize;
    if !candidates.contains(&r) {
        candidates.push(r);
    }
}

#[cfg(test)]
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
#[cfg(test)]
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

#[cfg(test)]
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

    let setup_prefix = match incoming_setup_prefix {
        Some(natural_len) => {
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
            Some(akita_types::setup_prefix_slot_id(natural_len, group))
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
fn recursive_level_candidates_for_split(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    search: &RecursiveLevelSearch,
    log_basis: u32,
    fold_level: usize,
    block_index_bits: usize,
    requested_fold_shape: TensorChallengeShape,
) -> Result<Vec<(LayoutCandidateScore, CommittedGroupParams, usize)>, AkitaError> {
    let candidate_params = recursive_fold_level_params_candidates(
        policy,
        ring_challenge_cfg,
        dimensions,
        search.num_ring_elems,
        search.reduced_vars,
        log_basis,
        fold_level,
        block_index_bits,
        requested_fold_shape,
    )?;
    let mut candidates = Vec::with_capacity(candidate_params.len());
    for mut params in candidate_params {
        params.payload_mode = payload_mode;
        params.setup_prefix = search.setup_prefix.clone();
        if let Some(prefix) = &params.setup_prefix {
            let prefix_d_width = prefix
                .commitment_params
                .d_segment_width(params.role_dims().d_d())?;
            let total_d_width = params
                .open_commit_matrix
                .input_width()
                .checked_add(prefix_d_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup-prefix shared D width overflow".to_string())
                })?;
            params.open_commit_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
                params.open_commit_matrix.sis_table_key(),
                total_d_width,
            )?;
        }
        let Some(next_witness_len) = planned_next_witness_len(
            policy.decomposition.field_bits(),
            &params,
            1,
            search.num_chunks,
        )?
        else {
            continue;
        };
        let score = layout_candidate_score(
            next_witness_len,
            params.num_live_blocks,
            search.num_chunks,
            params.fold_challenge_shape,
        )?;
        candidates.push((score, params, next_witness_len));
    }
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_frontier(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
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

    // Keep one best split per A rank. A tighter L2 floor must add a rank path
    // without deleting the old higher-rank path, because their next witness
    // lengths can land in different power-of-two DP states. The diagnostic L2
    // bucket is only an encoding of the selected rank; keeping multiple
    // buckets for the same rank would duplicate equivalent suffix states.
    let mut best = std::collections::BTreeMap::<
        usize,
        (LayoutCandidateScore, usize, CommittedGroupParams, usize),
    >::new();
    for r in (1..search.reduced_vars).rev() {
        for (score, candidate_params, next_witness_len) in recursive_level_candidates_for_split(
            policy,
            payload_mode,
            ring_challenge_cfg,
            dimensions,
            &search,
            log_basis,
            fold_level,
            r,
            requested_fold_shape,
        )? {
            let key = candidate_params.inner_commit_matrix.output_rank();
            if best.get(&key).is_none_or(|(best_score, best_r, _, _)| {
                recursive_candidate_order_key(score, r)
                    < recursive_candidate_order_key(*best_score, *best_r)
            }) {
                best.insert(key, (score, r, candidate_params, next_witness_len));
            }
        }
    }

    Ok(best
        .into_values()
        .filter_map(|(_, _, params, next_witness_len)| {
            (next_witness_len < current_witness_len).then_some((params, next_witness_len))
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_all_splits(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
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
        for (_, params, next_witness_len) in recursive_level_candidates_for_split(
            policy,
            payload_mode,
            ring_challenge_cfg,
            dimensions,
            &search,
            log_basis,
            fold_level,
            block_index_bits,
            requested_fold_shape,
        )? {
            if next_witness_len < current_witness_len {
                candidates.push((params, next_witness_len));
            }
        }
    }
    Ok(candidates)
}
