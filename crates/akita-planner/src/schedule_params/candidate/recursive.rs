use super::*;

mod split;

pub(crate) use split::recursive_split_search_domain;
use split::recursive_witness_body_lower_bound;
pub(super) use split::{
    recursive_candidate_order_key, recursive_split_lower_bound, RecursiveSplitLowerBoundInput,
};

#[derive(Clone, Copy)]
pub(crate) struct RecursiveCandidateRequest<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) payload_mode: akita_types::CommitmentPayloadMode,
    pub(crate) opening: PlannerOpeningCandidate,
    pub(crate) dimensions: CommitmentRingDims,
    pub(crate) current_witness_len: usize,
    pub(crate) source: crate::InnerBasisSource,
    pub(crate) log_basis_inner: u32,
    pub(crate) log_basis_open: u32,
    pub(crate) fold_level: usize,
    pub(crate) source_moment: Option<crate::response_model::SourceMomentEstimate>,
}

pub(crate) enum RecursiveSetupPrefix<'a> {
    None,
    Search {
        cache: &'a mut SetupPrefixSearchCache,
        natural_len: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SplitBoundPolicy {
    Enabled,
    #[cfg(test)]
    DisabledForOracle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FoldCandidatePolicy {
    Best,
    Frontier(SplitBoundPolicy),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SuccessorPolicy {
    AllowNonContracting,
    RequireContraction,
}

impl SuccessorPolicy {
    fn admits(self, current_witness_len: usize, next_witness_len: usize) -> bool {
        self == Self::AllowNonContracting || next_witness_len < current_witness_len
    }
}

#[derive(Clone, Copy)]
struct RecursiveCandidateContext<'request, 'policy> {
    request: &'request RecursiveCandidateRequest<'policy>,
    search: &'request RecursiveLevelSearch,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    successor_policy: SuccessorPolicy,
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

impl RecursiveCandidateContext<'_, '_> {
    /// Build one recursive-fold candidate for an explicit ring-element bucket
    /// and split. Setup certification uses the maximum current length in each
    /// `ceil(log2(ring_elems))` bucket, which dominates every shorter member
    /// for the same split.
    fn candidate_core(
        &self,
        block_index_bits: usize,
    ) -> Result<Option<RecursiveCandidateCore>, AkitaError> {
        let request = self.request;
        let policy = request.policy;
        let ring_challenge_cfg = request.opening.challenge_config();
        let dimensions = request.dimensions;
        let search = self.search;
        let source = request.source;
        let log_basis_inner = request.log_basis_inner;
        let log_basis_open = request.log_basis_open;
        let fold_level = request.fold_level;
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
        let fold_policy =
            BalancedSignedDigitFoldPolicy::universal(policy.decomposition.field_bits());
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
                challenge_dimension: request.opening.challenge_dimension(d_a),
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
            request.opening.method(),
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
        let request = self.request;
        let d_a = request.dimensions.d_a();
        let source_encoding = akita_types::CommittedSourceEncoding::for_producer(
            request.opening.method(),
            request.policy.claim_ext_degree,
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
                .validate_for_commitment(
                    request.fold_level,
                    request.payload_mode,
                    core.num_live_blocks,
                )
                .is_err()
            {
                continue;
            }
            let Some(outer_commit_matrix) =
                derive_outer_commitment_candidate(OuterCommitmentCandidateRequest {
                    policy: request.policy,
                    dimensions: request.dimensions,
                    payload_mode: request.payload_mode,
                    num_claims: 1,
                    num_live_blocks: core.num_live_blocks,
                    outer_slice_count,
                    log_basis_open: request.log_basis_open,
                    num_digits_outer: core.num_digits_open,
                    inner_output_rank: core.inner_commit_matrix.output_rank(),
                })?
            else {
                continue;
            };
            candidates.push(CommittedGroupParams {
                payload_mode: request.payload_mode,
                source_encoding,
                opening_method: request.opening.method(),
                log_basis_inner: request.log_basis_inner,
                log_basis_outer: request.log_basis_open,
                log_basis_open: request.log_basis_open,
                inner_commit_matrix: core.inner_commit_matrix,
                outer_commit_matrix,
                open_commit_matrix: core.open_commit_matrix,
                num_live_ring_elements_per_claim: core.num_ring_elems,
                num_positions_per_block: core.num_positions_per_block,
                num_live_blocks: core.num_live_blocks,
                outer_slice_count,
                fold_challenge_config: request.opening.challenge_config(),
                num_digits_inner: core.num_digits_inner,
                num_digits_outer: core.num_digits_open,
                num_digits_open: core.num_digits_open,
                num_digits_fold: core.num_digits_fold,
                witness_chunk: crate::policy::witness_chunk_at_level(
                    request.policy,
                    request.fold_level,
                ),
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
struct RecursiveLevelSearch {
    num_chunks: usize,
    num_ring_elems: usize,
    reduced_vars: usize,
    current_witness_len: usize,
    opening_layout: OpeningClaimsLayout,
    setup_prefixes: Vec<Option<akita_types::GroupOpenPhaseParams>>,
}

fn prepare_recursive_level_search(
    request: &RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let RecursiveCandidateRequest {
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        ..
    } = *request;
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

    let incoming_setup_prefix = match &setup_prefix {
        RecursiveSetupPrefix::None => None,
        RecursiveSetupPrefix::Search { natural_len, .. } => Some(*natural_len),
    };
    let opening_layout = suffix_opening_layout(current_witness_len, incoming_setup_prefix)?;
    let setup_prefixes = match setup_prefix {
        RecursiveSetupPrefix::Search { cache, natural_len } => {
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
        RecursiveSetupPrefix::None => vec![None],
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
    setup_prefix: Option<&akita_types::GroupOpenPhaseParams>,
    extension_degree: usize,
    mut candidate_params: CommittedGroupParams,
) -> Result<CommittedGroupParams, AkitaError> {
    candidate_params.setup_prefix = setup_prefix.cloned();
    if let Some(prefix) = &candidate_params.setup_prefix {
        let prefix_d_width =
            prefix.d_segment_width(extension_degree, candidate_params.role_dims().d_d())?;
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

impl RecursiveCandidateContext<'_, '_> {
    fn walk_splits(
        &self,
        mut admit_split: impl FnMut(usize, RecursiveSplitBounds) -> bool,
        mut visit: impl FnMut(LayoutCandidateScore, usize, CommittedGroupParams, usize),
    ) -> Result<(), AkitaError> {
        let request = self.request;
        let policy = request.policy;
        let search = self.search;
        let delta_commit = request
            .source
            .num_digits_inner(policy.decomposition, request.log_basis_inner)?;
        let delta_open = num_digits_open(DecompositionParams {
            log_basis: request.log_basis_open,
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
                ring_dimension: request.dimensions.d_a(),
                opening_width: request.opening.method().physical_coefficient_width(
                    policy.claim_ext_degree,
                    request.dimensions.d_a(),
                )?,
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
                    visit(score, r, params, next_witness_len);
                }
            }
        }
        Ok(())
    }
}

fn best_linf_candidate(
    context: &RecursiveCandidateContext<'_, '_>,
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
        |score, r, candidate_params, next_witness_len| {
            if !context
                .successor_policy
                .admits(context.search.current_witness_len, next_witness_len)
            {
                return;
            }
            if best.as_ref().is_none_or(|(best_score, best_r, _, _)| {
                recursive_candidate_order_key(score, r)
                    < recursive_candidate_order_key(*best_score, *best_r)
            }) {
                best_score.set(Some(score));
                best = Some((score, r, candidate_params, next_witness_len));
            }
        },
    )?;

    Ok(best.map(|(_, r, params, next)| (r, params, next)))
}

fn append_selective_l2_candidates(
    candidates: &mut Vec<(CommittedGroupParams, usize)>,
    best_modeled: Option<&(usize, CommittedGroupParams, usize)>,
    request: &RecursiveCandidateRequest<'_>,
    search: &RecursiveLevelSearch,
    successor_policy: SuccessorPolicy,
) -> Result<(), AkitaError> {
    let RecursiveCandidateRequest {
        policy,
        dimensions,
        log_basis_open,
        fold_level,
        source_moment,
        ..
    } = *request;
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
    let l2_request = RecursiveCandidateRequest {
        opening: PlannerOpeningCandidate::evaluation_trace(l2_challenge),
        source_moment: Some(source_moment),
        ..*request
    };
    let l2_context = RecursiveCandidateContext {
        request: &l2_request,
        search,
        source_moment: Some(source_moment),
        successor_policy,
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
            if successor_policy.admits(search.current_witness_len, next_witness_len) {
                candidates.push((params, next_witness_len));
            }
        }
    }
    Ok(())
}

fn derive_best_fold_candidates(
    request: RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(&request, setup_prefix)? else {
        return Ok(Vec::new());
    };
    let modeled_context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: SuccessorPolicy::RequireContraction,
    };
    let best_modeled = best_linf_candidate(&modeled_context)?;
    let mut candidates: Vec<_> = best_modeled
        .as_ref()
        .map(|(_, params, next)| (params.clone(), *next))
        .into_iter()
        .collect();
    if request.source_moment.is_some() {
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
    if !request.opening.is_coefficient_packing() {
        append_selective_l2_candidates(
            &mut candidates,
            best_modeled.as_ref(),
            &request,
            &search,
            SuccessorPolicy::RequireContraction,
        )?;
    }
    Ok(candidates)
}

/// Derive EvaluationTrace parameters used only to certify a direct terminal
/// response. Unlike an emitted recursive fold, this boundary does not require
/// the unused successor witness layout to contract.
pub(crate) fn derive_terminal_candidates(
    request: RecursiveCandidateRequest<'_>,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    if request.opening.is_coefficient_packing() {
        return Err(AkitaError::InvalidSetup(
            "terminal candidates require EvaluationTrace opening parameters".into(),
        ));
    }
    let Some(search) = prepare_recursive_level_search(&request, RecursiveSetupPrefix::None)? else {
        return Ok(Vec::new());
    };
    let modeled_context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: SuccessorPolicy::AllowNonContracting,
    };
    let best_modeled = best_linf_candidate(&modeled_context)?;
    let mut candidates = best_modeled
        .as_ref()
        .map(|(_, params, next)| (params.clone(), *next))
        .into_iter()
        .collect::<Vec<_>>();
    if request.source_moment.is_some() {
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
        &request,
        &search,
        SuccessorPolicy::AllowNonContracting,
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
pub(crate) fn derive_recursive_candidate_views(
    request: RecursiveCandidateRequest<'_>,
    fold_policy: FoldCandidatePolicy,
) -> Result<RecursiveCandidateViews, AkitaError> {
    if request.opening.is_coefficient_packing() {
        return Err(AkitaError::InvalidSetup(
            "combined terminal/fold search requires EvaluationTrace".into(),
        ));
    }
    let (retain_split_frontier, split_bounds) = match fold_policy {
        FoldCandidatePolicy::Best => (false, SplitBoundPolicy::Enabled),
        FoldCandidatePolicy::Frontier(bounds) => (true, bounds),
    };
    let Some(search) = prepare_recursive_level_search(&request, RecursiveSetupPrefix::None)? else {
        return Ok(RecursiveCandidateViews {
            terminal: Vec::new(),
            folds: Vec::new(),
        });
    };
    let base_context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: SuccessorPolicy::AllowNonContracting,
    };
    let mut terminal_pairs = Vec::new();
    let mut folds = Vec::new();
    let mut terminal_best_modeled = None;
    let mut fold_best_modeled = None;

    for (source_index, candidate_source_moment) in
        [request.source_moment, None].into_iter().enumerate()
    {
        if source_index != 0 && request.source_moment.is_none() {
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
                if !split_bounds.is_enabled() {
                    return true;
                }
                let terminal_admits = terminal_best_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0));
                let fold_admits = if retain_split_frontier {
                    let frontier_admits = bounds
                        .witness_body
                        .is_none_or(|bound| bound < request.current_witness_len);
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
            |score, r, params, next_witness_len| {
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
                if next_witness_len >= request.current_witness_len {
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
                    && next_witness_len < request.current_witness_len
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
            if source_index == 0 && next < request.current_witness_len {
                fold_best_modeled = Some((r, params.clone(), next));
            }
            if !retain_split_frontier
                && next < request.current_witness_len
                && !folds.contains(&(params.clone(), next))
            {
                folds.push((params, next));
            }
        }
    }

    append_selective_l2_candidates(
        &mut terminal_pairs,
        terminal_best_modeled.as_ref(),
        &request,
        &search,
        SuccessorPolicy::AllowNonContracting,
    )?;
    append_selective_l2_candidates(
        &mut folds,
        fold_best_modeled.as_ref(),
        &request,
        &search,
        SuccessorPolicy::RequireContraction,
    )?;
    Ok(RecursiveCandidateViews {
        terminal: terminal_pairs
            .into_iter()
            .map(|(params, _)| params)
            .collect(),
        folds,
    })
}

/// Derive recursive fold candidates under the requested retention policy.
pub(crate) fn derive_fold_candidates(
    request: RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
    policy: FoldCandidatePolicy,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    match policy {
        FoldCandidatePolicy::Best => derive_best_fold_candidates(request, setup_prefix),
        FoldCandidatePolicy::Frontier(bounds) => {
            derive_fold_candidate_frontier(request, setup_prefix, bounds)
        }
    }
}

fn derive_fold_candidate_frontier(
    request: RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
    split_bounds: SplitBoundPolicy,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(&request, setup_prefix)? else {
        return Ok(Vec::new());
    };
    let modeled_context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: SuccessorPolicy::RequireContraction,
    };
    let mut candidates = Vec::new();
    let mut best_modeled_with_score: Option<(
        LayoutCandidateScore,
        usize,
        CommittedGroupParams,
        usize,
    )> = None;
    let best_modeled_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
    for (source_index, candidate_source_moment) in
        [request.source_moment, None].into_iter().enumerate()
    {
        if candidate_source_moment.is_none()
            && request.source_moment.is_none()
            && !candidates.is_empty()
        {
            break;
        }
        let context = RecursiveCandidateContext {
            source_moment: candidate_source_moment,
            ..modeled_context
        };
        context.walk_splits(
            |_, bounds| {
                if !split_bounds.is_enabled() {
                    return true;
                }
                let frontier_admits = bounds
                    .witness_body
                    .is_none_or(|bound| bound < request.current_witness_len);
                if source_index != 0 {
                    return frontier_admits;
                }
                let best_search_admits = best_modeled_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0));
                frontier_admits || best_search_admits
            },
            |score, r, params, next_witness_len| {
                if source_index == 0
                    && next_witness_len < request.current_witness_len
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
                if next_witness_len < request.current_witness_len
                    && !candidates.contains(&(params.clone(), next_witness_len))
                {
                    candidates.push((params, next_witness_len));
                }
            },
        )?;
        if request.source_moment.is_none() {
            break;
        }
    }
    let best_modeled = best_modeled_with_score.map(|(_, r, params, next)| (r, params, next));
    if !request.opening.is_coefficient_packing() {
        append_selective_l2_candidates(
            &mut candidates,
            best_modeled.as_ref(),
            &request,
            &search,
            SuccessorPolicy::RequireContraction,
        )?;
    }
    Ok(candidates)
}

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "recursive/tests.rs"]
mod tests;
