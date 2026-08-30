use super::*;

/// Enumerate every contracting recursive candidate without production split
/// bounds, slice pruning, or objective selection.
pub(crate) fn derive_unpruned_fold_candidates_for_oracle(
    request: RecursiveCandidateRequest<'_>,
    relation_domain: RelationSearchDomain,
) -> Result<Vec<(RecursiveRelationCandidate, usize)>, AkitaError> {
    let Some(search) = prepare_recursive_level_search(&request, RecursiveSetupPrefix::None)? else {
        return Ok(Vec::new());
    };
    let base_context = RecursiveCandidateContext {
        request: &request,
        search: &search,
        source_moment: request.source_moment,
        successor_policy: SuccessorPolicy::RequireContraction,
    };
    let mut candidates = Vec::new();
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
        let mut linf = Vec::new();
        context.walk_splits(
            relation_domain,
            SliceRetention::Exhaustive,
            |_, _| true,
            |_, split, params, next_witness_len| {
                if next_witness_len < request.current_witness_len {
                    linf.push((split, params, next_witness_len));
                }
            },
        )?;
        for (_, params, next) in &linf {
            let candidate = (params.clone(), *next);
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
        if source_index == 0 {
            for best in &linf {
                append_selective_l2_candidates(
                    &mut candidates,
                    Some(best),
                    &request,
                    &search,
                    SuccessorPolicy::RequireContraction,
                    SliceRetention::Exhaustive,
                )?;
            }
        }
    }
    Ok(candidates)
}
