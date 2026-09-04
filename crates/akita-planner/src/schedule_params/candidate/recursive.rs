use super::*;

mod frontier;
mod level_search;
#[cfg(all(test, feature = "catalog-gen"))]
mod oracle;
mod split;
mod views;

use frontier::derive_fold_candidate_frontier;
use level_search::{
    attach_recursive_setup_prefix, finalize_recursive_level_candidate,
    prepare_recursive_level_search, RecursiveLevelSearch,
};
#[cfg(all(test, feature = "catalog-gen"))]
pub(crate) use oracle::{
    derive_unpruned_fold_candidates_for_oracle, derive_unpruned_terminal_candidates_for_oracle,
};
pub(crate) use split::recursive_split_search_domain;
use split::recursive_witness_body_lower_bound;
pub(super) use split::{
    recursive_candidate_order_key, recursive_split_lower_bound, RecursiveSplitLowerBoundInput,
};
pub(crate) use views::derive_recursive_candidate_views;

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
    pub(crate) relation_traversal_order: RelationTraversalOrder,
}

enum RecursiveSetupPrefix<'a> {
    None,
    Search {
        cache: &'a mut SetupPrefixSearchCache,
        natural_len: usize,
    },
}

pub(crate) enum RecursiveFoldWork<'a> {
    Direct {
        relation_domain: RelationSearchDomain,
    },
    SetupPrefixed {
        cache: &'a mut SetupPrefixSearchCache,
        natural_len: usize,
    },
}

impl<'a> RecursiveFoldWork<'a> {
    pub(crate) const fn direct(relation_domain: RelationSearchDomain) -> Self {
        Self::Direct { relation_domain }
    }

    pub(crate) fn setup_prefixed(
        cache: &'a mut SetupPrefixSearchCache,
        natural_len: usize,
    ) -> Self {
        Self::SetupPrefixed { cache, natural_len }
    }

    fn into_search_parts(self) -> (RecursiveSetupPrefix<'a>, RelationSearchDomain) {
        match self {
            Self::Direct { relation_domain } => (RecursiveSetupPrefix::None, relation_domain),
            Self::SetupPrefixed { cache, natural_len } => (
                RecursiveSetupPrefix::Search { cache, natural_len },
                RelationSearchDomain::QuotientOnly,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SplitBoundPolicy {
    Enabled,
    #[cfg(all(test, feature = "catalog-gen"))]
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
        num_chunks: usize,
        delta_commit: usize,
        delta_open: usize,
    ) -> Result<Option<RecursiveCandidateCore>, AkitaError> {
        let request = self.request;
        let policy = request.policy;
        let ring_challenge_cfg = request.opening.challenge_config();
        let dimensions = request.dimensions;
        let search = self.search;
        let log_basis_inner = request.log_basis_inner;
        let log_basis_open = request.log_basis_open;
        let num_ring_elems = search.num_ring_elems;
        let reduced_vars = search.reduced_vars;
        if reduced_vars <= 2
            || reduced_vars >= 53
            || block_index_bits == 0
            || block_index_bits >= reduced_vars
        {
            return Ok(None);
        }
        let num_positions_per_block = 1usize
            .checked_shl((reduced_vars - block_index_bits) as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive candidate position count overflow".to_string())
            })?;
        let num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
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
        relation_domain: RelationSearchDomain,
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
            for transition in relation_domain.transitions_in(self.request.relation_traversal_order)
            {
                let params = CommittedGroupParams::try_new(
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
                    *transition,
                    source_encoding,
                    crate::policy::witness_chunk_at_level(request.policy, request.fold_level),
                )?;
                candidates.push(params);
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
        relation_domain: RelationSearchDomain,
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
        let opening_width = request
            .opening
            .method()
            .physical_coefficient_width(policy.claim_ext_degree, request.dimensions.d_a())?;
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
                opening_width,
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
            let Some(core) = self.candidate_core(r, search.num_chunks, delta_commit, delta_open)?
            else {
                continue;
            };
            let base_slice_candidates = self.candidates_from_core(&core, relation_domain)?;
            for setup_prefix in &search.setup_prefixes {
                for transition in
                    relation_domain.transitions_in(self.request.relation_traversal_order)
                {
                    let mut mode_slices = Vec::with_capacity(base_slice_candidates.len());
                    for base_candidate in base_slice_candidates
                        .iter()
                        .filter(|candidate| candidate.ring_relation_mode == *transition)
                    {
                        let params = attach_recursive_setup_prefix(
                            setup_prefix.as_ref(),
                            policy.claim_ext_degree,
                            base_candidate.clone(),
                        )?;
                        if params.compression_sources_supported()? {
                            mode_slices.push(params);
                        }
                    }
                    for params in crate::schedule_params::prune_locally_unprofitable_slices(
                        policy,
                        &search.opening_layout,
                        mode_slices,
                    )? {
                        let relation_mode = *transition;
                        let Some((score, params, next_witness_len)) =
                            finalize_recursive_level_candidate(policy, search, params)?
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

fn best_linf_candidates_for(
    context: &RecursiveCandidateContext<'_, '_>,
    relation_domain: RelationSearchDomain,
) -> Result<Vec<BestLinfCandidate>, AkitaError> {
    // Larger `r` wins exact score ties independently for each relation mode.
    let mut best = std::collections::BTreeMap::<
        akita_types::RingRelationMode,
        (LayoutCandidateScore, usize, CommittedGroupParams, usize),
    >::new();
    let best_score = std::cell::Cell::new(None::<LayoutCandidateScore>);
    context.walk_splits(
        relation_domain,
        |_, bounds| {
            relation_domain.has_multiple_modes()
                || best_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0))
        },
        |score, r, candidate, next_witness_len| {
            if !context
                .successor_policy
                .admits(context.search.current_witness_len, next_witness_len)
            {
                return;
            }
            let mode = candidate.ring_relation_mode;
            if best.get(&mode).is_none_or(|(best_score, best_r, _, _)| {
                recursive_candidate_order_key(score, r)
                    < recursive_candidate_order_key(*best_score, *best_r)
            }) {
                if !relation_domain.has_multiple_modes() {
                    best_score.set(Some(score));
                }
                best.insert(mode, (score, r, candidate, next_witness_len));
            }
        },
    )?;

    Ok(best
        .into_values()
        .map(|(_, r, candidate, next)| (r, candidate, next))
        .collect())
}

fn all_linf_candidates_for(
    context: &RecursiveCandidateContext<'_, '_>,
    relation_domain: RelationSearchDomain,
) -> Result<Vec<BestLinfCandidate>, AkitaError> {
    let mut candidates = Vec::new();
    context.walk_splits(
        relation_domain,
        |_, _| true,
        |_, split, candidate, next_witness_len| {
            if context
                .successor_policy
                .admits(context.search.current_witness_len, next_witness_len)
                && !candidates
                    .iter()
                    .any(|(_, existing, next)| existing == &candidate && *next == next_witness_len)
            {
                candidates.push((split, candidate, next_witness_len));
            }
        },
    )?;
    Ok(candidates)
}

fn append_selective_l2_candidates(
    candidates: &mut Vec<(CommittedGroupParams, usize)>,
    best_modeled: Option<&BestLinfCandidate>,
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
    let l2_delta_commit = l2_request
        .source
        .num_digits_inner(policy.decomposition, l2_request.log_basis_inner)?;
    let l2_delta_open = num_digits_open(DecompositionParams {
        log_basis: l2_request.log_basis_open,
        ..policy.decomposition
    });
    let Some(mut l2_core) = l2_context.candidate_core(
        *block_index_bits,
        search.num_chunks,
        l2_delta_commit,
        l2_delta_open,
    )?
    else {
        return Ok(());
    };
    let relation_transition = best_modeled
        .map(|(_, candidate, _)| candidate.ring_relation_mode)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 candidate is missing its relation".into()))?;
    let relation_domain = RelationSearchDomain::for_mode(relation_transition);
    let linf_slices = l2_context.candidates_from_core(&l2_core, relation_domain)?;
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
    let mut base_slices = l2_context.candidates_from_core(&l2_core, relation_domain)?;
    base_slices.retain(|candidate| {
        linf_slices
            .iter()
            .any(|linf| linf.outer_slice_count() == candidate.outer_slice_count())
    });
    for setup_prefix in &search.setup_prefixes {
        let mut sliced = Vec::with_capacity(base_slices.len());
        for base_candidate in &base_slices {
            let params = attach_recursive_setup_prefix(
                setup_prefix.as_ref(),
                policy.claim_ext_degree,
                base_candidate.clone(),
            )?;
            if params.compression_sources_supported()? {
                sliced.push(params);
            }
        }
        let retained_slices = crate::schedule_params::prune_locally_unprofitable_slices(
            policy,
            &search.opening_layout,
            sliced,
        )?;
        for params in retained_slices {
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
    relation_domain: RelationSearchDomain,
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
    let best_modeled = best_linf_candidates_for(&modeled_context, relation_domain)?;
    let mut candidates: Vec<_> = best_modeled
        .iter()
        .map(|(_, candidate, next)| (candidate.clone(), *next))
        .collect();
    if request.source_moment.is_some() {
        let universal_context = RecursiveCandidateContext {
            source_moment: None,
            ..modeled_context
        };
        for (_, candidate, next) in best_linf_candidates_for(&universal_context, relation_domain)? {
            let universal = (candidate, next);
            if !candidates.contains(&universal) {
                candidates.push(universal);
            }
        }
    }
    if !request.opening.is_coefficient_packing() {
        for best in &best_modeled {
            append_selective_l2_candidates(
                &mut candidates,
                Some(best),
                &request,
                &search,
                SuccessorPolicy::RequireContraction,
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
    let retain_setup_frontier = matches!(
        request.policy.selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    );
    let modeled = if retain_setup_frontier {
        all_linf_candidates_for(&modeled_context, RelationSearchDomain::QuotientOnly)?
    } else {
        best_linf_candidates_for(&modeled_context, RelationSearchDomain::QuotientOnly)?
    };
    let mut candidates = modeled
        .iter()
        .map(|(_, candidate, next)| (candidate.clone(), *next))
        .collect::<Vec<_>>();
    if request.source_moment.is_some() {
        let universal_context = RecursiveCandidateContext {
            source_moment: None,
            ..modeled_context
        };
        let universal = if retain_setup_frontier {
            all_linf_candidates_for(&universal_context, RelationSearchDomain::QuotientOnly)?
        } else {
            best_linf_candidates_for(&universal_context, RelationSearchDomain::QuotientOnly)?
        };
        for (_, candidate, next) in universal {
            let universal = (candidate, next);
            if !candidates.contains(&universal) {
                candidates.push(universal);
            }
        }
    }
    for best in &modeled {
        append_selective_l2_candidates(
            &mut candidates,
            Some(best),
            &request,
            &search,
            SuccessorPolicy::AllowNonContracting,
        )?;
    }
    Ok(candidates
        .into_iter()
        .map(|(candidate, _)| candidate)
        .collect())
}

/// Derive recursive fold candidates under the requested retention policy.
pub(crate) fn derive_fold_candidates(
    request: RecursiveCandidateRequest<'_>,
    work: RecursiveFoldWork<'_>,
    policy: FoldCandidatePolicy,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let (setup_prefix, relation_domain) = work.into_search_parts();
    match policy {
        FoldCandidatePolicy::Best => {
            derive_best_fold_candidates(request, setup_prefix, relation_domain)
        }
        FoldCandidatePolicy::Frontier(bounds) => {
            derive_fold_candidate_frontier(request, setup_prefix, bounds, relation_domain)
        }
    }
}

#[cfg(all(test, feature = "catalog-gen"))]
#[path = "recursive/tests.rs"]
mod tests;
