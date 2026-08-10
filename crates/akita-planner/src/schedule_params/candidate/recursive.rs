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
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    block_index_bits: usize,
    current_witness_len: usize,
    outer_slice_count: akita_types::CommitmentSliceCount,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    if reduced_vars <= 2
        || reduced_vars >= 53
        || block_index_bits == 0
        || block_index_bits >= reduced_vars
    {
        return Ok(None);
    }
    let num_chunks = crate::policy::chunks_at_level(policy, fold_level);
    let num_positions_per_block = 1usize
        .checked_shl((reduced_vars - block_index_bits) as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("recursive candidate position count overflow".to_string())
        })?;
    let num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let delta_commit = source.num_digits_inner(policy.decomposition, log_basis_inner)?;
    let delta_open = num_digits_open(open_decomp);
    let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, delta_commit) else {
        return Ok(None);
    };
    let d_a = dimensions.d_a();
    let Some(num_fold_coeffs) = width_s
        .checked_mul(d_a)
        .and_then(|count| count.checked_mul(num_chunks))
    else {
        return Ok(None);
    };
    let fold_policy = BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
        policy.decomposition.field_bits(),
        FoldWitnessNorms::bounded(log_basis_inner, d_a),
    );
    let Ok(num_digits_fold) = fold_policy.num_digits_fold(HonestFoldSizingQuery {
        ring_dimension: d_a,
        num_claims: 1,
        num_live_blocks,
        num_chunks,
        num_fold_coeffs,
        witness_norms: FoldWitnessNorms::bounded(log_basis_inner, d_a),
        log_basis_response: log_basis_open,
        challenge_config: ring_challenge_cfg,
    }) else {
        return Ok(None);
    };
    let Some(norm_s) = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        d_a,
        log_basis_open,
        ring_challenge_cfg,
        num_digits_fold,
        policy.ring_subfield_norm_bound,
    ) else {
        return Ok(None);
    };
    let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
        sis_key_at_dimension(policy, akita_types::SisMatrixRole::Inner, d_a, norm_s),
        width_s,
    ) else {
        return Ok(None);
    };
    let physical_witness_len = akita_schedules::planner_support::grouped_segment_rings(
        1,
        num_live_blocks,
        num_chunks,
        num_positions_per_block,
        inner_commit_matrix.output_rank(),
        delta_commit,
        delta_open,
        delta_open,
        num_digits_fold,
    )?
    .checked_mul(d_a)
    .ok_or_else(|| AkitaError::InvalidSetup("recursive witness body overflow".into()))?;
    if physical_witness_len >= current_witness_len {
        return Ok(None);
    }
    let Ok(slice_geometry) = akita_types::CommitmentSliceGeometry::try_new(
        outer_slice_count,
        num_live_blocks,
        1,
        inner_commit_matrix.output_rank(),
        delta_open,
        d_a,
        dimensions.d_b(),
    ) else {
        return Ok(None);
    };
    let Some(norm_t) = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        akita_types::SisMatrixRole::Outer,
        dimensions.d_b(),
        log_basis_open,
    ) else {
        return Ok(None);
    };
    let outer_key = sis_key_at_dimension(
        policy,
        akita_types::SisMatrixRole::Outer,
        dimensions.d_b(),
        norm_t,
    );
    let width_t = slice_geometry.physical_input_width();
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
        d_a,
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
        outer_slice_count,
        fold_challenge_config: *ring_challenge_cfg,
        num_digits_inner: delta_commit,
        num_digits_outer: delta_open,
        num_digits_open: delta_open,
        num_digits_fold,
        witness_chunk: crate::policy::witness_chunk_at_level(policy, fold_level),
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

const EXHAUSTIVE_SPLIT_VARIABLE_LIMIT: usize = 12;
const LARGE_SPLIT_BALANCE_RADIUS: isize = 2;

fn bounded_recursive_split_candidates(
    num_ring_elems: usize,
    reduced_vars: usize,
    delta_commit: usize,
    delta_open: usize,
    num_chunks: usize,
) -> Vec<usize> {
    if reduced_vars <= EXHAUSTIVE_SPLIT_VARIABLE_LIMIT {
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
        for offset in -LARGE_SPLIT_BALANCE_RADIUS..=LARGE_SPLIT_BALANCE_RADIUS {
            push_recursive_split_candidate(&mut candidates, reduced_vars, center as isize + offset);
        }
    }

    candidates.sort_by(|left, right| right.cmp(left));
    candidates
}

/// Return the exact split domain selected by the catalog-bound search policy.
pub(crate) fn recursive_split_search_domain(
    search_policy: crate::RecursiveSplitSearchPolicy,
    num_ring_elems: usize,
    reduced_vars: usize,
    delta_commit: usize,
    delta_open: usize,
    num_chunks: usize,
) -> Vec<usize> {
    match search_policy {
        crate::RecursiveSplitSearchPolicy::Exhaustive => (1..reduced_vars).rev().collect(),
        crate::RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1 => {
            bounded_recursive_split_candidates(
                num_ring_elems,
                reduced_vars,
                delta_commit,
                delta_open,
                num_chunks,
            )
        }
    }
}

/// Inputs shared by conservative recursive split bounds.
#[derive(Clone, Copy)]
pub(super) struct RecursiveSplitLowerBoundInput {
    pub(super) num_ring_elems: usize,
    pub(super) ring_dimension: usize,
    pub(super) reduced_vars: usize,
    pub(super) r: usize,
    pub(super) delta_commit: usize,
    pub(super) delta_open: usize,
    pub(super) num_chunks: usize,
}

fn recursive_witness_body_lower_bound(input: RecursiveSplitLowerBoundInput) -> Option<usize> {
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
    Some(physical_width_floor)
}

/// Lower bound on the final layout score for one recursive split.
///
/// The true score adds challenge and chunk work to the next witness. The next
/// witness itself includes at least the physical Z/E/T body returned above;
/// setup-prefix and relation-tail terms can only increase it.
pub(super) fn recursive_split_lower_bound(input: RecursiveSplitLowerBoundInput) -> Option<usize> {
    let physical_width_floor = recursive_witness_body_lower_bound(input)?;
    let p = input.reduced_vars.checked_sub(input.r)?;
    let num_positions_per_block = 1usize.checked_shl(p as u32)?;
    let num_live_blocks = input.num_ring_elems.div_ceil(num_positions_per_block);
    physical_width_floor
        .checked_add(num_live_blocks)?
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
    current_witness_len: usize,
    opening_layout: OpeningClaimsLayout,
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
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let num_chunks = crate::policy::chunks_at_level(policy, fold_level);
    dimensions.validate_role_projection()?;
    let d_a = dimensions.d_a();
    if current_witness_len == 0 {
        return Ok(None);
    }
    // The previous fold owns a compact field-coefficient buffer. It need not
    // end on the next A-ring boundary; commitment alignment pads only the
    // transient ring view. Plan from the live coefficient count, rounding up
    // solely to determine the next fold's block geometry.
    let num_ring_elems = current_witness_len.div_ceil(d_a);
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

    let opening_layout = suffix_opening_layout(current_witness_len, incoming_setup_prefix)?;
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
                log_basis_open,
                n_prefix,
                num_chunks,
                d_a,
                dimensions.d_b(),
                fold_level,
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
        current_witness_len,
        opening_layout,
        setup_prefixes,
    }))
}

#[allow(clippy::too_many_arguments)]
fn attach_recursive_setup_prefix(
    setup_prefix: Option<&akita_types::SetupPrefixSlotId>,
    mut candidate_params: CommittedGroupParams,
) -> Result<CommittedGroupParams, AkitaError> {
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
    Ok(candidate_params)
}

#[allow(clippy::too_many_arguments)]
fn finalize_recursive_level_candidate(
    policy: &PlannerPolicy,
    search: &RecursiveLevelSearch,
    candidate_params: CommittedGroupParams,
) -> Result<Option<(LayoutCandidateScore, CommittedGroupParams, usize)>, AkitaError> {
    let Some(next_witness_len) = planned_next_witness_len(
        policy.decomposition.field_bits(),
        &candidate_params,
        1,
        search.num_chunks,
    )?
    else {
        return Ok(None);
    };
    let score = layout_candidate_score(
        next_witness_len,
        candidate_params.num_live_blocks,
        search.num_chunks,
    )?;
    Ok(Some((score, candidate_params, next_witness_len)))
}

#[allow(clippy::too_many_arguments)]
fn recursive_level_base_candidate_for_split(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    search: &RecursiveLevelSearch,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    block_index_bits: usize,
    outer_slice_count: akita_types::CommitmentSliceCount,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
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
        search.current_witness_len,
        outer_slice_count,
    )?
    else {
        return Ok(None);
    };
    candidate_params.payload_mode = payload_mode;
    Ok(Some(candidate_params))
}

#[derive(Clone, Copy)]
struct RecursiveSplitBounds {
    score: Option<usize>,
    witness_body: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn walk_recursive_splits(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    search: &RecursiveLevelSearch,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    mut admit_split: impl FnMut(usize, RecursiveSplitBounds) -> bool,
    mut visit: impl FnMut(LayoutCandidateScore, usize, CommittedGroupParams, usize),
) -> Result<(), AkitaError> {
    let delta_commit = source.num_digits_inner(policy.decomposition, log_basis_inner)?;
    let delta_open = num_digits_open(DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    });
    let splits = recursive_split_search_domain(
        policy.recursive_split_search_policy,
        search.num_ring_elems,
        search.reduced_vars,
        delta_commit,
        delta_open,
        search.num_chunks,
    );
    for r in splits {
        let lower_bound_input = RecursiveSplitLowerBoundInput {
            num_ring_elems: search.num_ring_elems,
            ring_dimension: dimensions.d_a(),
            reduced_vars: search.reduced_vars,
            r,
            delta_commit,
            delta_open,
            num_chunks: search.num_chunks,
        };
        if !admit_split(
            r,
            RecursiveSplitBounds {
                score: recursive_split_lower_bound(lower_bound_input),
                witness_body: recursive_witness_body_lower_bound(lower_bound_input),
            },
        ) {
            continue;
        }
        let num_positions_per_block = 1usize
            .checked_shl((search.reduced_vars - r) as u32)
            .ok_or_else(|| AkitaError::InvalidSetup("recursive position count overflow".into()))?;
        let num_live_blocks = search.num_ring_elems.div_ceil(num_positions_per_block);
        for setup_prefix in &search.setup_prefixes {
            let mut slice_candidates = Vec::new();
            for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
                if outer_slice_count
                    .validate_for_commitment(fold_level, payload_mode, num_live_blocks)
                    .is_err()
                {
                    continue;
                }
                let Some(base_candidate) = recursive_level_base_candidate_for_split(
                    policy,
                    payload_mode,
                    ring_challenge_cfg,
                    dimensions,
                    search,
                    source,
                    log_basis_inner,
                    log_basis_open,
                    fold_level,
                    r,
                    outer_slice_count,
                )?
                else {
                    continue;
                };
                let candidate_params =
                    attach_recursive_setup_prefix(setup_prefix.as_ref(), base_candidate)?;
                if !candidate_params.compression_sources_supported()? {
                    continue;
                }
                slice_candidates.push(candidate_params);
            }
            for candidate_params in crate::schedule_params::prune_locally_unprofitable_slices(
                policy,
                &search.opening_layout,
                slice_candidates,
            )? {
                let Some((score, params, next_witness_len)) =
                    finalize_recursive_level_candidate(policy, search, candidate_params)?
                else {
                    continue;
                };
                visit(score, r, params, next_witness_len);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
) -> Result<Option<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(
        setup_prefix_cache,
        policy,
        ring_challenge_cfg,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
    )?
    else {
        return Ok(None);
    };

    // Larger `r` wins exact score ties inside the policy-selected domain.
    let mut best: Option<(LayoutCandidateScore, usize, CommittedGroupParams, usize)> = None;
    let best_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
    walk_recursive_splits(
        policy,
        payload_mode,
        ring_challenge_cfg,
        dimensions,
        &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        |_, bounds| {
            best_score
                .get()
                .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0))
        },
        |score, r, candidate_params, next_witness_len| {
            if best.as_ref().is_none_or(|(best_score, best_r, _, _)| {
                recursive_candidate_order_key(score, r)
                    < recursive_candidate_order_key(*best_score, *best_r)
            }) {
                best_score.set(Some(score));
                best = Some((score, r, candidate_params, next_witness_len));
            }
        },
    )?;

    let Some((_, _, candidate_params, next_witness_len)) = best else {
        return Ok(None);
    };

    if next_witness_len >= current_witness_len {
        return Ok(None);
    }

    Ok(Some((candidate_params, next_witness_len)))
}

/// Derive the candidate frontier over recursive splits.
///
/// This is exhaustive through [`EXHAUSTIVE_SPLIT_VARIABLE_LIMIT`] reduced
/// variables and intentionally bounded above that threshold. Large states use
/// [`recursive_split_search_domain`] to keep catalog generation
/// tractable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_split_frontier(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    ring_challenge_cfg: &akita_challenges::SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
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
    )?
    else {
        return Ok(Vec::new());
    };
    let mut candidates = Vec::new();
    walk_recursive_splits(
        policy,
        payload_mode,
        ring_challenge_cfg,
        dimensions,
        &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        |_, bounds| {
            bounds
                .witness_body
                .is_none_or(|bound| bound < current_witness_len)
        },
        |_, _, params, next_witness_len| {
            if next_witness_len < current_witness_len {
                candidates.push((params, next_witness_len));
            }
        },
    )?;
    Ok(candidates)
}
