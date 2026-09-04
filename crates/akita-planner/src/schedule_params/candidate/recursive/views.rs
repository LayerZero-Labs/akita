use super::*;

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
    relation_domain: RelationSearchDomain,
) -> Result<RecursiveCandidateViews, AkitaError> {
    if request.opening.is_coefficient_packing() {
        return Err(AkitaError::InvalidSetup(
            "combined terminal/fold search requires EvaluationTrace".into(),
        ));
    }
    if matches!(
        request.policy.selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ) {
        return Ok(RecursiveCandidateViews {
            terminal: derive_terminal_candidates(request)?,
            folds: derive_fold_candidates(
                request,
                RecursiveFoldWork::direct(relation_domain),
                fold_policy,
            )?,
        });
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
    let search_domain = relation_domain.including_terminal_quotient();

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
            search_domain,
            |_, bounds| {
                if !split_bounds.is_enabled() {
                    return true;
                }
                let terminal_admits = terminal_best_score
                    .get()
                    .is_none_or(|score| bounds.score.is_none_or(|bound| bound <= score.0));
                let fold_admits = if relation_domain.has_multiple_modes() {
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
            |score, split, candidate, next_witness_len| {
                let mode = candidate.ring_relation_mode;
                if mode == akita_types::RingRelationMode::QuotientLift
                    && terminal_best
                        .as_ref()
                        .is_none_or(|(best_score, best_split, _, _)| {
                            recursive_candidate_order_key(score, split)
                                < recursive_candidate_order_key(*best_score, *best_split)
                        })
                {
                    terminal_best_score.set(Some(score));
                    terminal_best = Some((score, split, candidate.clone(), next_witness_len));
                }
                if !relation_domain.admits(candidate.ring_relation_mode)
                    || next_witness_len >= request.current_witness_len
                {
                    return;
                }
                if fold_best
                    .get(&mode)
                    .is_none_or(|(best_score, best_split, _, _)| {
                        recursive_candidate_order_key(score, split)
                            < recursive_candidate_order_key(*best_score, *best_split)
                    })
                {
                    if !relation_domain.has_multiple_modes() {
                        fold_best_score.set(Some(score));
                    }
                    fold_best.insert(mode, (score, split, candidate.clone(), next_witness_len));
                }
                if retain_split_frontier && !folds.contains(&(candidate.clone(), next_witness_len))
                {
                    folds.push((candidate, next_witness_len));
                }
            },
        )?;

        if let Some((_, split, relation_candidate, next)) = terminal_best {
            let candidate = (relation_candidate, next);
            if !terminal_pairs.contains(&candidate) {
                terminal_pairs.push(candidate.clone());
            }
            if source_index == 0 {
                terminal_best_modeled = Some((split, candidate.0, candidate.1));
            }
        }
        for (_, split, candidate, next) in fold_best.into_values() {
            if source_index == 0 {
                fold_best_modeled.push((split, candidate.clone(), next));
            }
            if !retain_split_frontier && !folds.contains(&(candidate.clone(), next)) {
                folds.push((candidate, next));
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
        )?;
    }
    for best in &fold_best_modeled {
        append_selective_l2_candidates(
            &mut folds,
            Some(best),
            &request,
            &search,
            SuccessorPolicy::RequireContraction,
        )?;
    }
    Ok(RecursiveCandidateViews {
        terminal: terminal_pairs
            .into_iter()
            .map(|(candidate, _)| candidate)
            .collect(),
        folds,
    })
}
