use super::*;

mod frontier;
mod level_search;
mod split;

use frontier::derive_fold_candidate_frontier;
use level_search::{
    attach_recursive_setup_prefix, finalize_recursive_level_candidate,
    prepare_recursive_level_search, RecursiveLevelSearch,
};
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
        relation_transitions: &[RelationTransition],
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
            for transition in relation_transitions {
                candidates.push(CommittedGroupParams::try_new(
                    // A recursive candidate consumes no frozen groups, so its own
                    // new group is the whole list.
                    vec![akita_types::GroupOpenPhaseParams {
                        profile: akita_types::GroupCommitPhaseParams {
                            version: akita_types::GroupCommitPhaseParams::VERSION,
                            // It commits one polynomial over the witness arriving at
                            // its level.
                            group: akita_types::PolynomialGroupLayout::singleton(
                                akita_types::padded_boolean_opening_vars(
                                    request.current_witness_len,
                                )?,
                            ),
                            blocks: akita_types::BlockGeometry::new(
                                core.num_ring_elems,
                                core.num_positions_per_block,
                                core.num_live_blocks,
                            ),
                            outer_slice_count,
                            inner: akita_types::RoleParams::new(
                                akita_types::GadgetDigits::new(
                                    request.log_basis_inner,
                                    core.num_digits_inner,
                                ),
                                core.inner_commit_matrix,
                            ),
                            outer: akita_types::RoleParams::new(
                                akita_types::GadgetDigits::new(
                                    request.log_basis_open,
                                    core.num_digits_open,
                                ),
                                outer_commit_matrix,
                            ),
                        },
                        opening: akita_types::GroupOpeningPlan {
                            opening_method: request.opening.method(),
                            fold_challenge_config: request.opening.challenge_config(),
                            log_basis_open: request.log_basis_open,
                            num_digits_open: core.num_digits_open,
                            num_digits_fold: core.num_digits_fold,
                        },
                        setup_natural_len: None,
                    }],
                    core.open_commit_matrix,
                    request.payload_mode,
                    transition.mode(),
                    source_encoding,
                    crate::policy::witness_chunk_at_level(request.policy, request.fold_level),
                )?);
            }
        }
        Ok(candidates)
    }
}

#[derive(Clone, Copy)]
struct RecursiveSplitBounds {
    score: Option<usize>,
    witness_body: Option<usize>,
}

impl RecursiveCandidateContext<'_, '_> {
    fn walk_splits(
        &self,
        relation_transitions: &[RelationTransition],
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
            let base_slice_candidates = self.candidates_from_core(&core, relation_transitions)?;
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
                for transition in relation_transitions {
                    let mode_slices = slice_candidates
                        .iter()
                        .filter(|params| params.ring_relation_mode == transition.mode())
                        .cloned()
                        .collect();
                    for candidate_params in
                        crate::schedule_params::prune_locally_unprofitable_slices(
                            policy,
                            &search.opening_layout,
                            mode_slices,
                        )?
                    {
                        let relation_mode = candidate_params.ring_relation_mode;
                        let Some((score, params, next_witness_len)) =
                            finalize_recursive_level_candidate(policy, search, candidate_params)?
                        else {
                            continue;
                        };
                        if relation_mode == akita_types::RingRelationMode::QuotientLift
                            && (bounds.score.is_some_and(|bound| bound > score.0)
                                || bounds
                                    .witness_body
                                    .is_some_and(|bound| bound > next_witness_len))
                        {
                            return Err(AkitaError::InvalidSetup(
                                "recursive split lower bound exceeds a materialized candidate"
                                    .into(),
                            ));
                        }
                        visit(score, r, params, next_witness_len);
                    }
                }
            }
        }
        Ok(())
    }
}

type BestLinfCandidate = (usize, CommittedGroupParams, usize);

fn best_linf_candidates(
    context: &RecursiveCandidateContext<'_, '_>,
) -> Result<Vec<BestLinfCandidate>, AkitaError> {
    best_linf_candidates_for(context, RelationTransition::quotient_only())
}

fn best_linf_candidates_for(
    context: &RecursiveCandidateContext<'_, '_>,
    relation_transitions: &[RelationTransition],
) -> Result<Vec<BestLinfCandidate>, AkitaError> {
    // Larger `r` wins exact score ties independently for each relation mode.
    let mut best = std::collections::BTreeMap::<
        akita_types::RingRelationMode,
        (LayoutCandidateScore, usize, CommittedGroupParams, usize),
    >::new();
    let best_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
    context.walk_splits(
        relation_transitions,
        |_, bounds| {
            relation_transitions.len() != 1
                || best_score
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
            let mode = candidate_params.ring_relation_mode;
            if best.get(&mode).is_none_or(|(best_score, best_r, _, _)| {
                recursive_candidate_order_key(score, r)
                    < recursive_candidate_order_key(*best_score, *best_r)
            }) {
                if relation_transitions.len() == 1 {
                    best_score.set(Some(score));
                }
                best.insert(mode, (score, r, candidate_params, next_witness_len));
            }
        },
    )?;

    Ok(best
        .into_values()
        .map(|(_, r, params, next)| (r, params, next))
        .collect())
}

fn append_selective_l2_candidates(
    candidates: &mut Vec<(CommittedGroupParams, usize)>,
    best_modeled: Option<&(usize, CommittedGroupParams, usize)>,
    request: &RecursiveCandidateRequest<'_>,
    search: &RecursiveLevelSearch,
    successor_policy: SuccessorPolicy,
    relation_transition: RelationTransition,
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
    let relation_transitions = std::slice::from_ref(&relation_transition);
    let linf_slices = l2_context.candidates_from_core(&l2_core, relation_transitions)?;
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
    let mut base_slices = l2_context.candidates_from_core(&l2_core, relation_transitions)?;
    base_slices.retain(|candidate| {
        linf_slices
            .iter()
            .any(|linf| linf.outer_slice_count() == candidate.outer_slice_count())
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
    relation_transitions: &[RelationTransition],
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
    let best_modeled = best_linf_candidates_for(&modeled_context, relation_transitions)?;
    let mut candidates: Vec<_> = best_modeled
        .iter()
        .map(|(_, params, next)| (params.clone(), *next))
        .collect();
    if request.source_moment.is_some() {
        let universal_context = RecursiveCandidateContext {
            source_moment: None,
            ..modeled_context
        };
        for (_, params, next) in best_linf_candidates_for(&universal_context, relation_transitions)?
        {
            let universal = (params, next);
            if !candidates.contains(&universal) {
                candidates.push(universal);
            }
        }
    }
    if !request.opening.is_coefficient_packing() {
        for best in &best_modeled {
            let transition = relation_transitions
                .iter()
                .copied()
                .find(|transition| transition.mode() == best.1.ring_relation_mode)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "mode-specific Linf candidate has no relation transition".into(),
                    )
                })?;
            append_selective_l2_candidates(
                &mut candidates,
                Some(best),
                &request,
                &search,
                SuccessorPolicy::RequireContraction,
                transition,
            )?;
        }
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
    let best_modeled = best_linf_candidates(&modeled_context)?;
    let mut candidates = best_modeled
        .iter()
        .map(|(_, params, next)| (params.clone(), *next))
        .collect::<Vec<_>>();
    if request.source_moment.is_some() {
        let universal_context = RecursiveCandidateContext {
            source_moment: None,
            ..modeled_context
        };
        for (_, params, next) in best_linf_candidates(&universal_context)? {
            let universal = (params, next);
            if !candidates.contains(&universal) {
                candidates.push(universal);
            }
        }
    }
    for best in &best_modeled {
        append_selective_l2_candidates(
            &mut candidates,
            Some(best),
            &request,
            &search,
            SuccessorPolicy::AllowNonContracting,
            RelationTransition::quotient_only()[0],
        )?;
    }
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
    relation_transitions: &[RelationTransition],
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
    let mut fold_best_modeled = Vec::new();
    let search_transitions = RelationTransition::with_terminal_quotient(relation_transitions);

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
        let mut fold_best = std::collections::BTreeMap::<
            akita_types::RingRelationMode,
            (LayoutCandidateScore, usize, CommittedGroupParams, usize),
        >::new();
        context.walk_splits(
            search_transitions,
            |_, bounds| {
                if !split_bounds.is_enabled() {
                    return true;
                }
                let terminal_admits = terminal_best_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0));
                let fold_admits = if relation_transitions.len() != 1 {
                    true
                } else if retain_split_frontier {
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
                let mode = params.ring_relation_mode;
                if mode == akita_types::RingRelationMode::QuotientLift
                    && terminal_best
                        .as_ref()
                        .is_none_or(|(best_score, best_r, _, _)| {
                            recursive_candidate_order_key(score, r)
                                < recursive_candidate_order_key(*best_score, *best_r)
                        })
                {
                    terminal_best_score.set(Some(score));
                    terminal_best = Some((score, r, params.clone(), next_witness_len));
                }
                if !relation_transitions
                    .iter()
                    .any(|transition| transition.mode() == mode)
                    || next_witness_len >= request.current_witness_len
                {
                    return;
                }
                if fold_best
                    .get(&mode)
                    .is_none_or(|(best_score, best_r, _, _)| {
                        recursive_candidate_order_key(score, r)
                            < recursive_candidate_order_key(*best_score, *best_r)
                    })
                {
                    if relation_transitions.len() == 1 {
                        fold_best_score.set(Some(score));
                    }
                    fold_best.insert(mode, (score, r, params.clone(), next_witness_len));
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
        for (_, r, params, next) in fold_best.into_values() {
            if source_index == 0 && next < request.current_witness_len {
                fold_best_modeled.push((r, params.clone(), next));
            }
            if !retain_split_frontier
                && next < request.current_witness_len
                && !folds.contains(&(params.clone(), next))
            {
                folds.push((params, next));
            }
        }
    }

    if let Some(best) = terminal_best_modeled.as_ref() {
        append_selective_l2_candidates(
            &mut terminal_pairs,
            Some(best),
            &request,
            &search,
            SuccessorPolicy::AllowNonContracting,
            RelationTransition::quotient_only()[0],
        )?;
    }
    for best in &fold_best_modeled {
        let transition = relation_transitions
            .iter()
            .copied()
            .find(|transition| transition.mode() == best.1.ring_relation_mode)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "mode-specific fold winner has no relation transition".into(),
                )
            })?;
        append_selective_l2_candidates(
            &mut folds,
            Some(best),
            &request,
            &search,
            SuccessorPolicy::RequireContraction,
            transition,
        )?;
    }
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
    relation_transitions: &[RelationTransition],
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    match policy {
        FoldCandidatePolicy::Best => {
            derive_best_fold_candidates(request, setup_prefix, relation_transitions)
        }
        FoldCandidatePolicy::Frontier(bounds) => {
            derive_fold_candidate_frontier(request, setup_prefix, bounds, relation_transitions)
        }
    }
}

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "recursive/tests.rs"]
mod tests;
