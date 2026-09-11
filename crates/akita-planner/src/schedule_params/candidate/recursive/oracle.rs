use super::*;

#[derive(Clone, Copy)]
enum ReferenceCandidatePurpose {
    Fold(RelationSearchDomain),
    Terminal,
}

impl ReferenceCandidatePurpose {
    const fn successor_policy(self) -> SuccessorPolicy {
        match self {
            Self::Fold(_) => SuccessorPolicy::RequireContraction,
            Self::Terminal => SuccessorPolicy::AllowNonContracting,
        }
    }

    const fn relation_domain(self) -> RelationSearchDomain {
        match self {
            Self::Fold(domain) => domain,
            Self::Terminal => RelationSearchDomain::QuotientOnly,
        }
    }
}

fn retain_finalized_candidates(
    request: &RecursiveCandidateRequest<'_>,
    search: &RecursiveLevelSearch,
    successor_policy: SuccessorPolicy,
    base_candidates: impl IntoIterator<Item = CommittedGroupParams>,
    candidates: &mut Vec<(CommittedGroupParams, usize)>,
) -> Result<(), AkitaError> {
    for candidate in base_candidates {
        if !candidate.compression_sources_supported()? {
            continue;
        }
        let Some((_, params, next_witness_len)) =
            finalize_recursive_level_candidate(request.policy, search, candidate)?
        else {
            continue;
        };
        if !successor_policy.admits(request.current_witness_len, next_witness_len) {
            continue;
        }
        let candidate = (params, next_witness_len);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    Ok(())
}

fn independently_enumerated_l2_candidates(
    context: &RecursiveCandidateContext<'_, '_>,
    split: usize,
    relation_domain: RelationSearchDomain,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    let request = context.request;
    if request.opening.is_coefficient_packing()
        || !request.policy.selective_l2_response_model_enabled()
    {
        return Ok(Vec::new());
    }
    let Some(source_moment) = context.source_moment else {
        return Ok(Vec::new());
    };
    let Some(l2_challenge) =
        akita_challenges::selective_l2_challenge_config(request.dimensions.d_a())
    else {
        return Ok(Vec::new());
    };
    let fold_basis = 1usize
        .checked_shl(request.log_basis_open)
        .ok_or_else(|| AkitaError::InvalidSetup("reference L2 fold basis overflow".into()))?;
    let response_l2_sq_cap = source_moment.response_l2_sq_cap(l2_challenge.challenge_l2_sq_max());
    let l2_request = RecursiveCandidateRequest {
        opening: PlannerOpeningCandidate::evaluation_trace(l2_challenge),
        source_moment: Some(source_moment),
        ..*request
    };
    let l2_context = RecursiveCandidateContext {
        request: &l2_request,
        search: context.search,
        source_moment: Some(source_moment),
        successor_policy: context.successor_policy,
    };
    let delta_commit = l2_request
        .source
        .num_digits_inner(request.policy.decomposition, l2_request.log_basis_inner)?;
    let delta_open = num_digits_open(DecompositionParams {
        log_basis: l2_request.log_basis_open,
        ..request.policy.decomposition
    });
    let Some(mut l2_core) =
        l2_context.candidate_core(split, context.search.num_chunks, delta_commit, delta_open)?
    else {
        return Ok(Vec::new());
    };
    let l2_challenge_linf_slices = l2_context.candidates_from_core(&l2_core, relation_domain)?;
    if l2_challenge_linf_slices.is_empty() {
        return Ok(Vec::new());
    }
    let l2_challenge_linf_rank = l2_core.inner_commit_matrix.output_rank();
    let Some(inner_commit_matrix) = selective_l2_inner_matrix(
        request.policy,
        SelectiveL2CandidateGeometry {
            fold_level: request.fold_level,
            num_claims: 1,
            num_chunks: context.search.num_chunks,
            inner_width: l2_core.inner_commit_matrix.input_width(),
            ring_dimension: request.dimensions.d_a(),
            fold_basis,
            fold_digit_count: l2_core.num_digits_fold,
            fold_challenge_config: &l2_challenge,
            response_l2_sq_cap,
            norm_proof_shape: None,
        },
    )?
    else {
        return Ok(Vec::new());
    };
    if inner_commit_matrix.output_rank() >= l2_challenge_linf_rank {
        return Ok(Vec::new());
    }
    l2_core.inner_commit_matrix = inner_commit_matrix;
    let mut candidates = l2_context.candidates_from_core(&l2_core, relation_domain)?;
    candidates.retain(|candidate| {
        l2_challenge_linf_slices.iter().any(|linf| {
            linf.ring_relation_mode == candidate.ring_relation_mode
                && linf.outer_slice_count() == candidate.outer_slice_count()
        })
    });
    Ok(candidates)
}

/// Enumerate every candidate admitted by the reference purpose without
/// production split bounds, slice pruning, objective selection, or the
/// production split walker.
///
/// This reference path shares only the canonical materializers for an explicit
/// split. Linf and modeled L2 candidates are both enumerated without reusing
/// the production choice of which split should receive the L2 alternative.
fn enumerate_unpruned_candidates(
    request: RecursiveCandidateRequest<'_>,
    purpose: ReferenceCandidatePurpose,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(&request, RecursiveSetupPrefix::None)? else {
        return Ok(Vec::new());
    };
    let relation_domain = purpose.relation_domain();
    let base_context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: purpose.successor_policy(),
    };
    let delta_commit = request
        .source
        .num_digits_inner(request.policy.decomposition, request.log_basis_inner)?;
    let delta_open = num_digits_open(DecompositionParams {
        log_basis: request.log_basis_open,
        ..request.policy.decomposition
    });
    let mut candidates = Vec::new();
    for (source_index, source_moment) in [request.source_moment, None].into_iter().enumerate() {
        if source_index != 0 && request.source_moment.is_none() {
            break;
        }
        let context = RecursiveCandidateContext {
            source_moment,
            ..base_context
        };
        for split in (1..search.reduced_vars).rev() {
            let Some(core) =
                context.candidate_core(split, search.num_chunks, delta_commit, delta_open)?
            else {
                continue;
            };
            let linf_slices = context.candidates_from_core(&core, relation_domain)?;
            let l2_slices = if source_index == 0 {
                independently_enumerated_l2_candidates(&context, split, relation_domain)?
            } else {
                Vec::new()
            };
            retain_finalized_candidates(
                &request,
                &search,
                purpose.successor_policy(),
                linf_slices,
                &mut candidates,
            )?;
            retain_finalized_candidates(
                &request,
                &search,
                purpose.successor_policy(),
                l2_slices,
                &mut candidates,
            )?;
        }
    }
    Ok(candidates)
}

pub(crate) fn derive_unpruned_fold_candidates_for_oracle(
    request: RecursiveCandidateRequest<'_>,
    relation_domain: RelationSearchDomain,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    enumerate_unpruned_candidates(request, ReferenceCandidatePurpose::Fold(relation_domain))
}

pub(crate) fn derive_unpruned_terminal_candidates_for_oracle(
    request: RecursiveCandidateRequest<'_>,
) -> Result<Vec<CommittedGroupParams>, AkitaError> {
    Ok(
        enumerate_unpruned_candidates(request, ReferenceCandidatePurpose::Terminal)?
            .into_iter()
            .map(|(candidate, _)| candidate)
            .collect(),
    )
}
