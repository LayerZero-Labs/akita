use super::*;

pub(super) fn derive_fold_candidate_frontier(
    request: RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
    split_bounds: SplitBoundPolicy,
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
    let mut candidates = Vec::new();
    let mut best_modeled_with_score = std::collections::BTreeMap::<
        akita_types::RingRelationMode,
        (LayoutCandidateScore, usize, CommittedGroupParams, usize),
    >::new();
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
            relation_transitions,
            |_, bounds| {
                if !split_bounds.is_enabled() {
                    return true;
                }
                if relation_transitions.len() != 1 {
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
                let mode = params.ring_relation_mode;
                if source_index == 0
                    && next_witness_len < request.current_witness_len
                    && best_modeled_with_score.get(&mode).is_none_or(
                        |(best_score, best_r, _, _)| {
                            recursive_candidate_order_key(score, r)
                                < recursive_candidate_order_key(*best_score, *best_r)
                        },
                    )
                {
                    if relation_transitions.len() == 1 {
                        best_modeled_score.set(Some(score));
                    }
                    best_modeled_with_score
                        .insert(mode, (score, r, params.clone(), next_witness_len));
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
    let best_modeled = best_modeled_with_score
        .into_values()
        .map(|(_, r, params, next)| (r, params, next))
        .collect::<Vec<_>>();
    if !request.opening.is_coefficient_packing() {
        for best in &best_modeled {
            let transition = relation_transitions
                .iter()
                .copied()
                .find(|transition| transition.mode() == best.1.ring_relation_mode)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "mode-specific frontier winner has no relation transition".into(),
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
