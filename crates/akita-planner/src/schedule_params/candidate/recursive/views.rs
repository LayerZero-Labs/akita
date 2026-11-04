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
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenExactProofAndWorkV6
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
    let mut terminal_best_modeled = Vec::new();
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
        let terminal_best = std::cell::RefCell::new(SplitFrontier::default());
        let fold_best = std::cell::RefCell::new(std::collections::BTreeMap::<
            akita_params::RingRelationMode,
            SplitFrontier,
        >::new());
        context.walk_splits(
            search_domain,
            |_, bounds| {
                if !split_bounds.is_enabled() {
                    return true;
                }
                let terminal_admits = terminal_best.borrow().admits(bounds);
                let fold_best_admits = || {
                    fold_best
                        .borrow()
                        .values()
                        .all(|frontier| frontier.admits(bounds))
                };
                let fold_admits = if relation_domain.has_multiple_modes() {
                    true
                } else if retain_split_frontier {
                    let frontier_admits = bounds
                        .witness_body
                        .is_none_or(|bound| bound < request.current_witness_len);
                    frontier_admits || (source_index == 0 && fold_best_admits())
                } else {
                    fold_best_admits()
                };
                terminal_admits || fold_admits
            },
            |score, split, candidate, next_witness_len| {
                let mode = candidate.ring_relation_mode;
                let is_terminal = mode == akita_params::RingRelationMode::QuotientLift;
                let is_fold = relation_domain.admits(candidate.ring_relation_mode)
                    && next_witness_len < request.current_witness_len;
                if !is_terminal && !is_fold {
                    return Ok(());
                }
                let nonce_bytes = fold_nonce_bytes(&candidate)?;
                if let (true, Some(nonce_bytes)) = (is_terminal, nonce_bytes) {
                    terminal_best.borrow_mut().offer(
                        score,
                        split,
                        candidate.clone(),
                        next_witness_len,
                        nonce_bytes,
                    )?;
                }
                if !is_fold {
                    return Ok(());
                }
                if retain_split_frontier
                    && !folds
                        .iter()
                        .any(|(existing, next)| existing == &candidate && *next == next_witness_len)
                {
                    folds.push((candidate.clone(), next_witness_len));
                }
                let Some(nonce_bytes) = nonce_bytes else {
                    return Ok(());
                };
                fold_best.borrow_mut().entry(mode).or_default().offer(
                    score,
                    split,
                    candidate,
                    next_witness_len,
                    nonce_bytes,
                )
            },
        )?;

        for (split, relation_candidate, next) in terminal_best.into_inner().into_candidates() {
            let candidate = (relation_candidate, next);
            if !terminal_pairs.contains(&candidate) {
                terminal_pairs.push(candidate.clone());
            }
            if source_index == 0 {
                terminal_best_modeled.push((split, candidate.0, candidate.1));
            }
        }
        for (split, candidate, next) in fold_best
            .into_inner()
            .into_values()
            .flat_map(SplitFrontier::into_candidates)
        {
            if source_index == 0 {
                fold_best_modeled.push((split, candidate.clone(), next));
            }
            if !retain_split_frontier
                && !folds.iter().any(|(existing, existing_next)| {
                    existing == &candidate && *existing_next == next
                })
            {
                folds.push((candidate, next));
            }
        }
    }

    append_selective_l2_candidates(
        &mut terminal_pairs,
        &terminal_best_modeled,
        &request,
        &search,
        SuccessorPolicy::AllowNonContracting,
    )?;
    append_selective_l2_candidates(
        &mut folds,
        &fold_best_modeled,
        &request,
        &search,
        SuccessorPolicy::RequireContraction,
    )?;
    Ok(RecursiveCandidateViews {
        terminal: terminal_pairs
            .into_iter()
            .map(|(candidate, _)| candidate)
            .collect(),
        folds,
    })
}
