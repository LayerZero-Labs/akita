use super::*;

#[derive(Clone, Copy)]
struct RecursiveCandidateContext<'a> {
    policy: &'a PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    search: &'a RecursiveLevelSearch,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    require_output_contraction: bool,
}

#[derive(Clone)]
struct RecursiveCandidateCore {
    num_ring_elems: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    num_digits_inner: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
    inner_commit_matrix: InnerCommitMatrixParams,
    open_commit_matrix: OpenCommitMatrixParams,
}

impl RecursiveCandidateContext<'_> {
    fn output_body_contracts(&self, core: &RecursiveCandidateCore) -> Result<bool, AkitaError> {
        if self.opening.is_coefficient_packing() {
            return Ok(true);
        }
        let physical_witness_len = akita_schedules::planner_support::grouped_segment_rings(
            1,
            core.num_live_blocks,
            self.search.num_chunks,
            core.num_positions_per_block,
            core.inner_commit_matrix.output_rank(),
            core.num_digits_inner,
            core.num_digits_open,
            core.num_digits_open,
            core.num_digits_fold,
        )?
        .checked_mul(self.dimensions.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness body overflow".into()))?;
        Ok(physical_witness_len < self.search.current_witness_len)
    }

    /// Build one recursive-fold candidate for an explicit ring-element bucket
    /// and split. Setup certification uses the maximum current length in each
    /// `ceil(log2(ring_elems))` bucket, which dominates every shorter member
    /// for the same split.
    fn candidate_core(
        &self,
        block_index_bits: usize,
    ) -> Result<Option<RecursiveCandidateCore>, AkitaError> {
        let policy = self.policy;
        let ring_challenge_cfg = self.opening.challenge_config();
        let dimensions = self.dimensions;
        let search = self.search;
        let source = self.source;
        let log_basis_inner = self.log_basis_inner;
        let log_basis_open = self.log_basis_open;
        let fold_level = self.fold_level;
        let num_ring_elems = search.num_ring_elems;
        let reduced_vars = search.reduced_vars;
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
        let open_decomp = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let delta_commit = source.num_digits_inner(policy.decomposition, log_basis_inner)?;
        let delta_open = num_digits_open(open_decomp);
        let Some(width_s) = decomposed_s_block_ring_count(num_positions_per_block, delta_commit)
        else {
            return Ok(None);
        };
        let d_a = dimensions.d_a();
        let fold_policy = BalancedSignedDigitFoldPolicy::universal(
            policy.decomposition.field_bits(),
            FoldWitnessNorms::bounded(log_basis_inner, d_a),
        );
        let num_fold_coeffs = width_s
            .checked_mul(d_a)
            .and_then(|count| count.checked_mul(num_chunks))
            .ok_or_else(|| AkitaError::InvalidSetup("fold response width overflow".into()))?;
        let modeled_linf_cap = self.source_moment.and_then(|moment| {
            moment.response_linf_cap(
                ring_challenge_cfg.challenge_l2_sq_max(),
                num_live_blocks,
                num_chunks,
                num_fold_coeffs,
                d_a,
            )
        });
        let Some(inner_candidate) =
            derive_inner_commitment_candidate(InnerCommitmentCandidateRequest {
                policy,
                fold_policy: &fold_policy,
                ring_challenge_cfg: &ring_challenge_cfg,
                challenge_dimension: self.opening.challenge_dimension(d_a),
                dimensions,
                num_claims: 1,
                num_live_ring_elements_per_claim: num_ring_elems,
                num_live_blocks,
                num_positions_per_block,
                num_chunks,
                witness_norms: FoldWitnessNorms::bounded(log_basis_inner, d_a),
                log_basis_open,
                width_s,
                modeled_linf_cap,
            })?
        else {
            return Ok(None);
        };
        let Ok(width_w) = akita_types::opening_d_segment_width(
            self.opening.method(),
            policy.claim_ext_degree,
            d_a,
            dimensions.d_d(),
            delta_open,
            num_live_blocks,
            1,
        ) else {
            return Ok(None);
        };
        let Some((open_key, width_w)) = projected_collision_role_price(
            policy,
            akita_types::SisMatrixRole::Open,
            dimensions.d_d(),
            dimensions.d_d(),
            width_w,
            log_basis_open,
        ) else {
            return Ok(None);
        };
        let Ok(open_commit_matrix) =
            OpenCommitMatrixParams::try_new_with_min_rank(open_key, width_w)
        else {
            return Ok(None);
        };
        Ok(Some(RecursiveCandidateCore {
            num_ring_elems,
            num_positions_per_block,
            num_live_blocks,
            num_digits_inner: delta_commit,
            num_digits_open: delta_open,
            num_digits_fold: inner_candidate.num_digits_fold,
            inner_commit_matrix: inner_candidate.inner_commit_matrix,
            open_commit_matrix,
        }))
    }

    fn candidates_from_core(
        &self,
        core: &RecursiveCandidateCore,
    ) -> Result<Vec<CommittedGroupParams>, AkitaError> {
        let d_a = self.dimensions.d_a();
        if self.require_output_contraction && !self.output_body_contracts(core)? {
            return Ok(Vec::new());
        }
        let source_encoding = akita_types::CommittedSourceEncoding::for_producer(
            self.opening.method(),
            self.policy.claim_ext_degree,
            d_a,
            self.search.current_witness_len.trailing_zeros() as usize,
            false,
        );
        if source_encoding.validate(d_a).is_err() {
            return Ok(Vec::new());
        }
        let mut candidates = Vec::new();
        for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
            if outer_slice_count
                .validate_for_commitment(self.fold_level, self.payload_mode, core.num_live_blocks)
                .is_err()
            {
                continue;
            }
            let Some(outer_commit_matrix) =
                derive_outer_commitment_candidate(OuterCommitmentCandidateRequest {
                    policy: self.policy,
                    dimensions: self.dimensions,
                    payload_mode: self.payload_mode,
                    num_claims: 1,
                    num_live_blocks: core.num_live_blocks,
                    outer_slice_count,
                    log_basis_open: self.log_basis_open,
                    num_digits_outer: core.num_digits_open,
                    inner_output_rank: core.inner_commit_matrix.output_rank(),
                })?
            else {
                continue;
            };
            candidates.push(CommittedGroupParams {
                payload_mode: self.payload_mode,
                source_encoding,
                opening_method: self.opening.method(),
                log_basis_inner: self.log_basis_inner,
                log_basis_outer: self.log_basis_open,
                log_basis_open: self.log_basis_open,
                inner_commit_matrix: core.inner_commit_matrix,
                outer_commit_matrix,
                open_commit_matrix: core.open_commit_matrix,
                num_live_ring_elements_per_claim: core.num_ring_elems,
                num_positions_per_block: core.num_positions_per_block,
                num_live_blocks: core.num_live_blocks,
                outer_slice_count,
                fold_challenge_config: self.opening.challenge_config(),
                num_digits_inner: core.num_digits_inner,
                num_digits_outer: core.num_digits_open,
                num_digits_open: core.num_digits_open,
                num_digits_fold: core.num_digits_fold,
                witness_chunk: crate::policy::witness_chunk_at_level(self.policy, self.fold_level),
                precommitted_groups: Vec::new(),
                setup_prefix: None,
            });
        }
        Ok(candidates)
    }
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
    pub(super) opening_width: usize,
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
        .checked_mul(input.opening_width)?
        .checked_add(t_hat_floor.checked_mul(input.ring_dimension)?)?
        .checked_add(z_hat_floor.checked_mul(input.ring_dimension)?)?;
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
    setup_prefixes: Vec<Option<akita_types::ScheduledSetupPrefix>>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_recursive_level_search(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let num_chunks = policy.chunks_at_level(fold_level);
    dimensions.validate_role_projection()?;
    opening.validate_for(fold_level, policy.claim_ext_degree, dimensions)?;
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
                SetupPrefixSearchRequest {
                    policy,
                    opening,
                    log_basis_open,
                    n_prefix,
                    num_chunks,
                    inner_ring_dimension: d_a,
                    outer_ring_dimension: dimensions.d_b(),
                },
            )?;
            if groups.is_empty() {
                return Ok(None);
            }
            groups
                .into_iter()
                .map(|group| Some(akita_types::scheduled_setup_prefix(natural_len, group)))
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

fn attach_recursive_setup_prefix(
    setup_prefix: Option<&akita_types::ScheduledSetupPrefix>,
    extension_degree: usize,
    mut candidate_params: CommittedGroupParams,
) -> Result<CommittedGroupParams, AkitaError> {
    candidate_params.setup_prefix = setup_prefix.cloned();
    if let Some(prefix) = &candidate_params.setup_prefix {
        let prefix_d_width = prefix
            .commitment_params
            .d_segment_width(extension_degree, candidate_params.role_dims().d_d())?;
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

fn finalize_recursive_level_candidate(
    policy: &PlannerPolicy,
    search: &RecursiveLevelSearch,
    candidate_params: CommittedGroupParams,
) -> Result<Option<(LayoutCandidateScore, CommittedGroupParams, usize)>, AkitaError> {
    let Some(next_witness_len) = planned_next_witness_len(
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
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

#[derive(Clone, Copy)]
struct RecursiveSplitBounds {
    score: Option<usize>,
    witness_body: Option<usize>,
}

impl RecursiveCandidateContext<'_> {
    fn walk_splits(
        &self,
        mut admit_split: impl FnMut(usize, RecursiveSplitBounds) -> bool,
        mut visit: impl FnMut(LayoutCandidateScore, usize, CommittedGroupParams, usize, bool),
    ) -> Result<(), AkitaError> {
        let policy = self.policy;
        let search = self.search;
        let delta_commit = self
            .source
            .num_digits_inner(policy.decomposition, self.log_basis_inner)?;
        let delta_open = num_digits_open(DecompositionParams {
            log_basis: self.log_basis_open,
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
                ring_dimension: self.dimensions.d_a(),
                opening_width: self
                    .opening
                    .physical_coefficient_width(self.dimensions.d_a()),
                reduced_vars: search.reduced_vars,
                r,
                delta_commit,
                delta_open,
                num_chunks: search.num_chunks,
            };
            let bounds = RecursiveSplitBounds {
                score: recursive_split_lower_bound(lower_bound_input),
                witness_body: recursive_witness_body_lower_bound(lower_bound_input),
            };
            if !admit_split(r, bounds) {
                continue;
            }
            let Some(core) = self.candidate_core(r)? else {
                continue;
            };
            let output_body_contracts = if self.require_output_contraction {
                true
            } else {
                self.output_body_contracts(&core)?
            };
            let base_slice_candidates = self.candidates_from_core(&core)?;
            for setup_prefix in &search.setup_prefixes {
                let mut slice_candidates = Vec::new();
                for base_candidate in &base_slice_candidates {
                    let candidate_params = attach_recursive_setup_prefix(
                        setup_prefix.as_ref(),
                        policy.claim_ext_degree,
                        base_candidate.clone(),
                    )?;
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
                    if bounds.score.is_some_and(|bound| bound > score.0)
                        || bounds
                            .witness_body
                            .is_some_and(|bound| bound > next_witness_len)
                    {
                        return Err(AkitaError::InvalidSetup(
                            "recursive split lower bound exceeds a materialized candidate".into(),
                        ));
                    }
                    visit(score, r, params, next_witness_len, output_body_contracts);
                }
            }
        }
        Ok(())
    }
}

fn best_linf_candidate(
    context: &RecursiveCandidateContext<'_>,
) -> Result<Option<(usize, CommittedGroupParams, usize)>, AkitaError> {
    // Larger `r` wins exact score ties inside the policy-selected domain.
    let mut best: Option<(LayoutCandidateScore, usize, CommittedGroupParams, usize)> = None;
    let best_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
    context.walk_splits(
        |_, bounds| {
            best_score
                .get()
                .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0))
        },
        |score, r, candidate_params, next_witness_len, _| {
            if best.as_ref().is_none_or(|(best_score, best_r, _, _)| {
                recursive_candidate_order_key(score, r)
                    < recursive_candidate_order_key(*best_score, *best_r)
            }) {
                best_score.set(Some(score));
                best = Some((score, r, candidate_params, next_witness_len));
            }
        },
    )?;

    Ok(best.and_then(|(_, r, params, next)| {
        (!context.require_output_contraction || next < context.search.current_witness_len)
            .then_some((r, params, next))
    }))
}

#[allow(clippy::too_many_arguments)]
fn append_selective_l2_candidates(
    candidates: &mut Vec<(CommittedGroupParams, usize)>,
    best_modeled: Option<&(usize, CommittedGroupParams, usize)>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    dimensions: CommitmentRingDims,
    search: &RecursiveLevelSearch,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    require_output_contraction: bool,
) -> Result<(), AkitaError> {
    if !policy.selective_l2_response_model_enabled() {
        return Ok(());
    }
    let (Some((block_index_bits, _, _)), Some(source_moment)) = (best_modeled, source_moment)
    else {
        return Ok(());
    };
    let Some(l2_challenge) = akita_challenges::selective_l2_challenge_config(dimensions.d_a())
    else {
        return Ok(());
    };
    let fold_basis = 1usize
        .checked_shl(log_basis_open)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 fold basis overflow".into()))?;
    let response_l2_sq_cap = source_moment.response_l2_sq_cap(l2_challenge.challenge_l2_sq_max());
    let l2_context = RecursiveCandidateContext {
        policy,
        payload_mode,
        opening: PlannerOpeningCandidate::evaluation_trace(l2_challenge),
        dimensions,
        search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment: Some(source_moment),
        require_output_contraction,
    };
    let Some(mut l2_core) = l2_context.candidate_core(*block_index_bits)? else {
        return Ok(());
    };
    let linf_slices = l2_context.candidates_from_core(&l2_core)?;
    if linf_slices.is_empty() {
        return Ok(());
    }
    let linf_rank = l2_core.inner_commit_matrix.output_rank();
    let Some(inner_commit_matrix) = selective_l2_inner_matrix(
        policy,
        SelectiveL2CandidateGeometry {
            fold_level,
            num_claims: 1,
            num_chunks: search.num_chunks,
            inner_width: l2_core.inner_commit_matrix.input_width(),
            ring_dimension: dimensions.d_a(),
            fold_basis,
            fold_digit_count: l2_core.num_digits_fold,
            fold_challenge_config: &l2_challenge,
            response_l2_sq_cap,
            norm_proof_shape: None,
        },
    )?
    else {
        return Ok(());
    };
    if inner_commit_matrix.output_rank() >= linf_rank {
        return Ok(());
    }
    l2_core.inner_commit_matrix = inner_commit_matrix;
    let mut base_slices = l2_context.candidates_from_core(&l2_core)?;
    base_slices.retain(|candidate| {
        linf_slices
            .iter()
            .any(|linf| linf.outer_slice_count == candidate.outer_slice_count)
    });
    for setup_prefix in &search.setup_prefixes {
        let mut sliced = Vec::new();
        for base_params in &base_slices {
            let params = attach_recursive_setup_prefix(
                setup_prefix.as_ref(),
                policy.claim_ext_degree,
                base_params.clone(),
            )?;
            if params.compression_sources_supported()? {
                sliced.push(params);
            }
        }
        for params in crate::schedule_params::prune_locally_unprofitable_slices(
            policy,
            &search.opening_layout,
            sliced,
        )? {
            let Some((_, params, next_witness_len)) =
                finalize_recursive_level_candidate(policy, search, params)?
            else {
                continue;
            };
            if !require_output_contraction || next_witness_len < search.current_witness_len {
                candidates.push((params, next_witness_len));
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
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(
        setup_prefix_cache,
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
    )?
    else {
        return Ok(Vec::new());
    };
    let modeled_context = RecursiveCandidateContext {
        policy,
        payload_mode,
        opening,
        dimensions,
        search: &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        require_output_contraction: true,
    };
    let best_modeled = best_linf_candidate(&modeled_context)?;
    let mut candidates: Vec<_> = best_modeled
        .as_ref()
        .map(|(_, params, next)| (params.clone(), *next))
        .into_iter()
        .collect();
    if source_moment.is_some() {
        let universal_context = RecursiveCandidateContext {
            source_moment: None,
            ..modeled_context
        };
        if let Some((_, params, next)) = best_linf_candidate(&universal_context)? {
            let universal = (params, next);
            if !candidates.contains(&universal) {
                candidates.push(universal);
            }
        }
    }
    if !opening.is_coefficient_packing() {
        append_selective_l2_candidates(
            &mut candidates,
            best_modeled.as_ref(),
            policy,
            payload_mode,
            dimensions,
            &search,
            source,
            log_basis_inner,
            log_basis_open,
            fold_level,
            source_moment,
            true,
        )?;
    }
    Ok(candidates)
}

/// Derive EvaluationTrace parameters used only to certify a direct terminal
/// response. Unlike an emitted recursive fold, this boundary does not require
/// the unused successor witness layout to contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_terminal_candidate_params(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    if opening.is_coefficient_packing() {
        return Err(AkitaError::InvalidSetup(
            "terminal candidates require EvaluationTrace opening parameters".into(),
        ));
    }
    let Some(search) = prepare_recursive_level_search(
        None,
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        None,
    )?
    else {
        return Ok(Vec::new());
    };
    let modeled_context = RecursiveCandidateContext {
        policy,
        payload_mode,
        opening,
        dimensions,
        search: &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        require_output_contraction: false,
    };
    let best_modeled = best_linf_candidate(&modeled_context)?;
    let mut candidates = best_modeled
        .as_ref()
        .map(|(_, params, next)| (params.clone(), *next))
        .into_iter()
        .collect::<Vec<_>>();
    if source_moment.is_some() {
        let universal_context = RecursiveCandidateContext {
            source_moment: None,
            ..modeled_context
        };
        if let Some((_, params, next)) = best_linf_candidate(&universal_context)? {
            let universal = (params, next);
            if !candidates.contains(&universal) {
                candidates.push(universal);
            }
        }
    }
    append_selective_l2_candidates(
        &mut candidates,
        best_modeled.as_ref(),
        policy,
        payload_mode,
        dimensions,
        &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        false,
    )?;
    Ok(candidates.into_iter().map(|(params, _)| params).collect())
}

pub(crate) struct RecursiveCandidateViews {
    pub(crate) terminal: Vec<CommittedGroupParams>,
    pub(crate) folds: Vec<(CommittedGroupParams, usize)>,
}

/// Derive the terminal and fold views of one EvaluationTrace search together.
///
/// Both views use the same split candidates and materialized matrices, but
/// retain their distinct admission rules: terminal construction may use a
/// non-contracting successor, while an emitted fold must contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_recursive_candidate_views(
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    retain_split_frontier: bool,
) -> Result<RecursiveCandidateViews, AkitaError> {
    if opening.is_coefficient_packing() {
        return Err(AkitaError::InvalidSetup(
            "combined terminal/fold search requires EvaluationTrace".into(),
        ));
    }
    let Some(search) = prepare_recursive_level_search(
        None,
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        None,
    )?
    else {
        return Ok(RecursiveCandidateViews {
            terminal: Vec::new(),
            folds: Vec::new(),
        });
    };
    let base_context = RecursiveCandidateContext {
        policy,
        payload_mode,
        opening,
        dimensions,
        search: &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        require_output_contraction: false,
    };
    let mut terminal_pairs = Vec::new();
    let mut folds = Vec::new();
    let mut terminal_best_modeled = None;
    let mut fold_best_modeled = None;

    for (source_index, candidate_source_moment) in [source_moment, None].into_iter().enumerate() {
        if source_index != 0 && source_moment.is_none() {
            break;
        }
        let context = RecursiveCandidateContext {
            source_moment: candidate_source_moment,
            ..base_context
        };
        let terminal_best_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
        let fold_best_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
        let mut terminal_best = None;
        let mut fold_best = None;
        context.walk_splits(
            |_, bounds| {
                let terminal_admits = terminal_best_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0));
                let fold_admits = if retain_split_frontier {
                    let frontier_admits = bounds
                        .witness_body
                        .is_none_or(|bound| bound < current_witness_len);
                    frontier_admits
                        || (source_index == 0
                            && fold_best_score.get().is_none_or(|score| {
                                bounds.score.is_none_or(|bound| bound <= score.0)
                            }))
                } else {
                    fold_best_score
                        .get()
                        .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0))
                };
                terminal_admits || fold_admits
            },
            |score, r, params, next_witness_len, output_body_contracts| {
                if terminal_best
                    .as_ref()
                    .is_none_or(|(best_score, best_r, _, _)| {
                        recursive_candidate_order_key(score, r)
                            < recursive_candidate_order_key(*best_score, *best_r)
                    })
                {
                    terminal_best_score.set(Some(score));
                    terminal_best = Some((score, r, params.clone(), next_witness_len));
                }
                if !output_body_contracts {
                    return;
                }
                if fold_best.as_ref().is_none_or(|(best_score, best_r, _, _)| {
                    recursive_candidate_order_key(score, r)
                        < recursive_candidate_order_key(*best_score, *best_r)
                }) {
                    fold_best_score.set(Some(score));
                    fold_best = Some((score, r, params.clone(), next_witness_len));
                }
                if retain_split_frontier
                    && next_witness_len < current_witness_len
                    && !folds.contains(&(params.clone(), next_witness_len))
                {
                    folds.push((params, next_witness_len));
                }
            },
        )?;

        if let Some((_, r, params, next)) = terminal_best {
            let candidate = (params, next);
            if !terminal_pairs.contains(&candidate) {
                terminal_pairs.push(candidate.clone());
            }
            if source_index == 0 {
                terminal_best_modeled = Some((r, candidate.0, candidate.1));
            }
        }
        if let Some((_, r, params, next)) = fold_best {
            if source_index == 0 && next < current_witness_len {
                fold_best_modeled = Some((r, params.clone(), next));
            }
            if !retain_split_frontier
                && next < current_witness_len
                && !folds.contains(&(params.clone(), next))
            {
                folds.push((params, next));
            }
        }
    }

    append_selective_l2_candidates(
        &mut terminal_pairs,
        terminal_best_modeled.as_ref(),
        policy,
        payload_mode,
        dimensions,
        &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        false,
    )?;
    append_selective_l2_candidates(
        &mut folds,
        fold_best_modeled.as_ref(),
        policy,
        payload_mode,
        dimensions,
        &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        true,
    )?;
    Ok(RecursiveCandidateViews {
        terminal: terminal_pairs
            .into_iter()
            .map(|(params, _)| params)
            .collect(),
        folds,
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_linf_candidate_level_params(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
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
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
    )?
    else {
        return Ok(None);
    };
    let context = RecursiveCandidateContext {
        policy,
        payload_mode,
        opening,
        dimensions,
        search: &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment: None,
        require_output_contraction: true,
    };
    Ok(best_linf_candidate(&context)?.map(|(_, params, next)| (params, next)))
}

/// Derive the candidate frontier over recursive splits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_split_frontier(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    derive_candidate_level_params_split_frontier_with_bounds(
        setup_prefix_cache,
        policy,
        payload_mode,
        opening,
        dimensions,
        current_witness_len,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
        source_moment,
        true,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_candidate_level_params_split_frontier_without_bounds(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    derive_candidate_level_params_split_frontier_with_bounds(
        setup_prefix_cache,
        policy,
        payload_mode,
        opening,
        dimensions,
        current_witness_len,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
        source_moment,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_candidate_level_params_split_frontier_with_bounds(
    setup_prefix_cache: Option<&mut SetupPrefixSearchCache>,
    policy: &PlannerPolicy,
    payload_mode: akita_types::CommitmentPayloadMode,
    opening: PlannerOpeningCandidate,
    dimensions: CommitmentRingDims,
    current_witness_len: usize,
    source: crate::InnerBasisSource,
    log_basis_inner: u32,
    log_basis_open: u32,
    fold_level: usize,
    incoming_setup_prefix: Option<usize>,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    apply_split_bounds: bool,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(
        setup_prefix_cache,
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        incoming_setup_prefix,
    )?
    else {
        return Ok(Vec::new());
    };
    let modeled_context = RecursiveCandidateContext {
        policy,
        payload_mode,
        opening,
        dimensions,
        search: &search,
        source,
        log_basis_inner,
        log_basis_open,
        fold_level,
        source_moment,
        require_output_contraction: true,
    };
    let mut candidates = Vec::new();
    let mut best_modeled_with_score: Option<(
        LayoutCandidateScore,
        usize,
        CommittedGroupParams,
        usize,
    )> = None;
    let best_modeled_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
    for (source_index, candidate_source_moment) in [source_moment, None].into_iter().enumerate() {
        if candidate_source_moment.is_none() && source_moment.is_none() && !candidates.is_empty() {
            break;
        }
        let context = RecursiveCandidateContext {
            source_moment: candidate_source_moment,
            ..modeled_context
        };
        context.walk_splits(
            |_, bounds| {
                if !apply_split_bounds {
                    return true;
                }
                let frontier_admits = bounds
                    .witness_body
                    .is_none_or(|bound| bound < current_witness_len);
                if source_index != 0 {
                    return frontier_admits;
                }
                let best_search_admits = best_modeled_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0));
                frontier_admits || best_search_admits
            },
            |score, r, params, next_witness_len, _| {
                if source_index == 0
                    && best_modeled_with_score
                        .as_ref()
                        .is_none_or(|(best_score, best_r, _, _)| {
                            recursive_candidate_order_key(score, r)
                                < recursive_candidate_order_key(*best_score, *best_r)
                        })
                {
                    best_modeled_score.set(Some(score));
                    best_modeled_with_score = Some((score, r, params.clone(), next_witness_len));
                }
                if next_witness_len < current_witness_len
                    && !candidates.contains(&(params.clone(), next_witness_len))
                {
                    candidates.push((params, next_witness_len));
                }
            },
        )?;
        if source_moment.is_none() {
            break;
        }
    }
    let best_modeled = best_modeled_with_score
        .and_then(|(_, r, params, next)| (next < current_witness_len).then_some((r, params, next)));
    if !opening.is_coefficient_packing() {
        append_selective_l2_candidates(
            &mut candidates,
            best_modeled.as_ref(),
            policy,
            payload_mode,
            dimensions,
            &search,
            source,
            log_basis_inner,
            log_basis_open,
            fold_level,
            source_moment,
            true,
        )?;
    }
    Ok(candidates)
}

#[cfg(all(test, feature = "catalog-gen"))]
mod tests {
    use super::*;

    #[test]
    fn combined_terminal_and_fold_views_match_independent_searches() {
        use akita_config::{
            policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
        };
        use akita_types::InnerCommitSecurityRoute;

        type Recursive = RecursiveCommitmentConfig<OneHot>;
        let policy = policy_of::<Recursive>();
        let opening = PlannerOpeningCandidate::evaluation_trace(
            Recursive::ring_challenge_config(64).expect("challenge config"),
        );
        let dimensions = CommitmentRingDims::uniform(64);
        let source = crate::InnerBasisSource::BalancedDigits { log_basis: 4 };
        let source_moment = crate::response_model::SourceMomentEstimate::new(1_000_000);
        for retain_split_frontier in [false, true] {
            let expected_terminal = derive_terminal_candidate_params(
                &policy,
                akita_types::CommitmentPayloadMode::Compressed,
                opening,
                dimensions,
                948_672,
                source,
                4,
                4,
                3,
                source_moment,
            )
            .expect("terminal search");
            let expected_folds = if retain_split_frontier {
                derive_candidate_level_params_split_frontier(
                    Some(&mut SetupPrefixSearchCache::default()),
                    &policy,
                    akita_types::CommitmentPayloadMode::Compressed,
                    opening,
                    dimensions,
                    948_672,
                    source,
                    4,
                    4,
                    3,
                    None,
                    source_moment,
                )
            } else {
                derive_candidate_level_params(
                    Some(&mut SetupPrefixSearchCache::default()),
                    &policy,
                    akita_types::CommitmentPayloadMode::Compressed,
                    opening,
                    dimensions,
                    948_672,
                    source,
                    4,
                    4,
                    3,
                    None,
                    source_moment,
                )
            }
            .expect("fold search");
            let actual = derive_recursive_candidate_views(
                &policy,
                akita_types::CommitmentPayloadMode::Compressed,
                opening,
                dimensions,
                948_672,
                source,
                4,
                4,
                3,
                source_moment,
                retain_split_frontier,
            )
            .expect("combined search");

            assert_eq!(actual.terminal, expected_terminal);
            assert_eq!(actual.folds, expected_folds);
            assert!(actual.folds.iter().any(|(params, _)| matches!(
                params.inner_commit_matrix.security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )));
        }
    }

    #[test]
    fn combined_views_keep_a_noncontracting_terminal_candidate() {
        use akita_config::{
            policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
        };

        type Recursive = RecursiveCommitmentConfig<OneHot>;
        let policy = policy_of::<Recursive>();
        let opening = PlannerOpeningCandidate::evaluation_trace(
            Recursive::ring_challenge_config(64).expect("challenge config"),
        );
        let source = crate::InnerBasisSource::BalancedDigits { log_basis: 4 };
        let mut witnessed_boundary = false;
        for current_witness_len in [1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16] {
            let views = derive_recursive_candidate_views(
                &policy,
                akita_types::CommitmentPayloadMode::Raw,
                opening,
                CommitmentRingDims::uniform(64),
                current_witness_len,
                source,
                4,
                4,
                2,
                None,
                false,
            )
            .expect("combined search");
            if !views.terminal.is_empty() && views.folds.is_empty() {
                witnessed_boundary = true;
                break;
            }
        }
        assert!(
            witnessed_boundary,
            "the fixture must exercise a terminal winner rejected by fold contraction"
        );
    }

    #[test]
    fn late_consumer_keeps_setup_prefix_slices_eligible() {
        use akita_config::{
            policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
        };

        type Recursive = RecursiveCommitmentConfig<OneHot>;
        let policy = policy_of::<Recursive>();
        let challenge = Recursive::ring_challenge_config(64).expect("challenge config");
        let mut cache = SetupPrefixSearchCache::default();
        let search = prepare_recursive_level_search(
            Some(&mut cache),
            &policy,
            PlannerOpeningCandidate::evaluation_trace(challenge),
            CommitmentRingDims::uniform(64),
            1 << 16,
            4,
            2,
            Some(1 << 12),
        )
        .expect("late consumer search")
        .expect("eligible recursive level");

        assert!(search.setup_prefixes.iter().flatten().any(|slot| {
            slot.commitment_params.layout.outer_slice_count > akita_types::CommitmentSliceCount::ONE
        }));
    }
}
