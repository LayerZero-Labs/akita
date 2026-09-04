use super::*;
use crate::schedule_params::ReducedTransitionRejection;

struct OpeningSearch<'a> {
    state: SuffixState,
    depth: usize,
    open_log_basis: u32,
    opening_layout: &'a OpeningClaimsLayout,
    require_child_fold: bool,
    guide_scope: Option<GuideScope>,
}

struct ChildPlan<'a> {
    params: &'a CommittedGroupParams,
    next_witness_len: usize,
    next_source_moment: Option<crate::response_model::SourceMomentEstimate>,

    natural_setup_field_len: usize,
    direct_edge_is_admissible: bool,
    prune_direct_edge: bool,
}

struct PlannedChildren {
    direct: Option<Arc<SuffixResult>>,
    offloaded: Option<Arc<SuffixResult>>,
}

fn plan_candidate_children(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    search: &OpeningSearch<'_>,
    plan: ChildPlan<'_>,
) -> Result<PlannedChildren, AkitaError> {
    let state = search.state;
    let direct_child = if !plan.direct_edge_is_admissible || plan.prune_direct_edge {
        None
    } else if search.depth == MAX_RECURSION_DEPTH {
        Some(empty_suffix_result())
    } else {
        Some(derive_selected_suffix_schedule(
            ctx,
            memo,
            SuffixState {
                level: state.level + 1,
                current_witness_len: plan.next_witness_len,
                current_lb: search.open_log_basis,
                source_moment: plan.next_source_moment,
                dimension_ceiling: plan.params.role_dims(),
                topology: state
                    .topology
                    .direct_successor(plan.params.payload_mode, plan.params.ring_relation_mode),
            },
            search.depth + 1,
        )?)
    };
    let offload_search_enabled = ctx.policy.recursive_setup_planning
        && ctx
            .policy
            .recursive_setup_search_policy
            .admits_offloaded_edge_at(state.level)
        // An offloaded edge accepts only a child suffix with at least two
        // folds. That topology cannot fit at the last two depths.
        && search.depth + 2 < MAX_RECURSION_DEPTH;
    if offload_search_enabled
        && plan.params.payload_mode.is_compressed()
        && plan.params.ring_relation_mode.is_reduced_evaluation()
    {
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_reduced_rejection(ReducedTransitionRejection::OutgoingSetupOffload);
        }
    }
    let offloaded_child = SuffixTopology::offloaded_successor(
        plan.params.ring_relation_mode,
        plan.params.payload_mode,
        plan.natural_setup_field_len,
    )
    .filter(|_| offload_search_enabled)
    .map(|topology| {
        derive_selected_suffix_schedule(
            ctx,
            memo,
            SuffixState {
                level: state.level + 1,
                current_witness_len: plan.next_witness_len,
                current_lb: search.open_log_basis,
                source_moment: plan.next_source_moment,
                dimension_ceiling: plan.params.role_dims(),
                topology,
            },
            search.depth + 1,
        )
    })
    .transpose()?;
    Ok(PlannedChildren {
        direct: direct_child,
        offloaded: offloaded_child,
    })
}

fn price_planned_fold_candidate(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    search: &OpeningSearch<'_>,
    guide: Option<(CompleteObjectiveBound, Option<usize>)>,
    candidate: PlannedFoldCandidate,
    frontiers: &mut StateFrontiers,
) -> Result<(), AkitaError> {
    let state = search.state;
    let PlannedFoldCandidate {
        params,
        next_witness_len,
        opening_reduction_bytes: _,
        next_source_moment,
    } = candidate;
    if let Some(natural_prefix_len) = state.topology.incoming_setup_prefix() {
        let padded_prefix_len = akita_types::padded_setup_prefix_len(natural_prefix_len);
        if !offloaded_witness_contracts(
            state.current_witness_len,
            state.current_lb,
            padded_prefix_len,
            ctx.policy.decomposition.field_bits(),
            next_witness_len,
            search.open_log_basis,
            ctx.policy.min_offloaded_witness_contraction,
        )? {
            return Ok(());
        }
    }
    let natural_len = guide.and_then(|(_, natural_len)| natural_len).map_or_else(
        || active_setup_field_len(&params, search.opening_layout),
        Ok,
    )?;
    let direct_edge_is_admissible =
        state
            .topology
            .incoming_setup_prefix()
            .is_none_or(|incoming_len| {
                akita_types::padded_setup_prefix_len(natural_len)
                    < akita_types::padded_setup_prefix_len(incoming_len)
            });
    if matches!(
        ctx.policy.selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ) && matches!(search.guide_scope, Some(GuideScope::CompleteRoot))
        && guide.is_some_and(|(lower_bound, _)| {
            complete_root_setup_bound_is_strictly_worse(lower_bound, &frontiers.projected)
        })
    {
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_guided_direct_edge_prune();
        }
        return Ok(());
    }
    let prune_direct_edge = if direct_edge_is_admissible {
        guide
            .zip(search.guide_scope)
            .map(|((lower_bound, _), guide_scope)| {
                direct_edge_bound_is_strictly_worse(
                    ctx.policy,
                    guide_scope,
                    &params,
                    natural_len,
                    lower_bound,
                    &frontiers.projected,
                )
            })
            .transpose()?
            .unwrap_or(false)
    } else {
        false
    };
    if prune_direct_edge {
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_guided_direct_edge_prune();
        }
        if !ctx.policy.recursive_setup_planning {
            return Ok(());
        }
    }
    let children = plan_candidate_children(
        ctx,
        memo,
        search,
        ChildPlan {
            params: &params,
            next_witness_len,
            next_source_moment,

            natural_setup_field_len: natural_len,
            direct_edge_is_admissible,
            prune_direct_edge,
        },
    )?;
    price_level_candidate_with_children(
        ctx,
        state,
        search.opening_layout,
        LevelCandidateEdge {
            params: &params,
            next_witness_len,
            natural_setup_field_len: natural_len,
            require_child_fold: search.require_child_fold,
        },
        CandidateChildren {
            direct: children.direct.as_deref(),
            offloaded: children.offloaded.as_deref(),
        },
        frontiers,
    )
}

fn finish_state(retains_setup_projection: bool, frontiers: StateFrontiers) -> SuffixResult {
    let mut payload_only = BTreeMap::new();
    let mut setup_and_payload = BTreeMap::new();
    for (key, choices) in frontiers.projected.by_parent_cost {
        if retains_setup_projection {
            setup_and_payload.insert(key, choices.into_objective_choices());
        } else {
            let candidates = choices.into_payload_candidates();
            if !candidates.is_empty() {
                payload_only.insert(key, candidates);
            }
        }
    }
    SuffixResult {
        payload_only,
        setup_and_payload,
    }
}

#[allow(clippy::too_many_arguments)]
fn process_candidate_batch(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    state: SuffixState,
    depth: usize,
    open_log_basis: u32,
    opening_layout: &OpeningClaimsLayout,
    is_root_level: bool,
    require_child_fold: bool,
    generated: candidates::GeneratedCandidates,
    frontiers: &mut StateFrontiers,
) -> Result<(), AkitaError> {
    let generated_candidate_count = generated
        .terminal
        .len()
        .saturating_add(generated.folds.len());
    let terminal_candidate_count = generated.terminal.len();
    for candidate in generated.terminal {
        let natural_len = active_setup_field_len(&candidate.params, opening_layout)?;
        price_terminal_candidate(
            ctx,
            state,
            &candidate.params,
            candidate.opening_reduction_bytes,
            natural_len,
            frontiers,
        )?;
    }
    let candidates_with_source =
        attach_source_moments(ctx, state, is_root_level, opening_layout, generated.folds)?;
    // Complete-root candidates are traversed in exact lower-bound order.
    // Recursive states do not have that global admission rule and retain
    // local Pareto pruning.
    let candidates = if is_root_level {
        candidates_with_source
    } else {
        prune::level_candidates(opening_layout, candidates_with_source)?
    };
    if let Some(diagnostics) = ctx.diagnostics {
        diagnostics.record_candidates(
            generated_candidate_count,
            terminal_candidate_count.saturating_add(candidates.len()),
        );
    }
    if candidates.is_empty() {
        return Ok(());
    }
    let incoming_setup_prefix = state.topology.incoming_setup_prefix();
    let guide_scope = GuideScope::for_state(ctx.policy, is_root_level, incoming_setup_prefix);
    let traversal = candidate_traversal(ctx.policy, guide_scope, opening_layout, candidates)?;
    let search = OpeningSearch {
        state,
        depth,
        open_log_basis,
        opening_layout,
        require_child_fold,
        guide_scope,
    };
    for (guide, candidate) in traversal {
        price_planned_fold_candidate(ctx, memo, &search, guide, candidate, frontiers)?;
    }
    Ok(())
}

/// Derive the suffix frontier for the selected recursive schedule at
/// `(level, current_witness_len, current_lb)`.
pub(crate) fn derive_selected_suffix_schedule(
    ctx: &SuffixCtx<'_>,
    memo: &mut ScheduleMemo,
    state: SuffixState,
    depth: usize,
) -> Result<Arc<SuffixResult>, AkitaError> {
    let policy = ctx.policy;
    let relation_phase = state.topology.relation_phase();
    if let Some(diagnostics) = ctx.diagnostics {
        diagnostics.record_suffix_call(relation_phase);
    }
    let memo_key = state.memo_key(policy);
    if depth <= MAX_RECURSION_DEPTH {
        let cached = memo.get(&memo_key);
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_memo_result(relation_phase, cached.is_some());
        }
        if let Some(cached) = cached {
            return Ok(Arc::clone(cached));
        }
    }
    if depth > MAX_RECURSION_DEPTH {
        return Ok(empty_suffix_result());
    }
    if policy.selective_l2_response_model_enabled()
        && !(ctx.level_zero_is_root && state.level == 0)
        && state.source_moment.is_none()
    {
        return Err(AkitaError::InvalidSetup(
            "recursive suffix is missing its response source moment".into(),
        ));
    }
    let incoming_setup_prefix = state.topology.incoming_setup_prefix();
    let retains_setup_projection =
        incoming_setup_prefix.is_some() || (ctx.level_zero_is_root && state.level == 0);
    let candidate_domain = candidates::CandidateDomain::prepare(ctx, state)?;
    let root_level_key = candidate_domain.root_level_key;
    let opening_layout = &candidate_domain.opening_layout;
    let mut frontiers = StateFrontiers::new();
    for open_log_basis in candidate_domain.opening_basis_range.clone() {
        if root_level_key.is_some() {
            candidate_domain.visit_root_batches(ctx, state, open_log_basis, |generated| {
                process_candidate_batch(
                    ctx,
                    memo,
                    state,
                    depth,
                    open_log_basis,
                    opening_layout,
                    true,
                    candidate_domain.require_child_fold,
                    generated,
                    &mut frontiers,
                )
            })?;
        } else {
            let generated = candidate_domain.generate_for_opening_basis(
                ctx,
                state,
                open_log_basis,
                &mut memo.setup_prefixes,
            )?;
            process_candidate_batch(
                ctx,
                memo,
                state,
                depth,
                open_log_basis,
                opening_layout,
                false,
                candidate_domain.require_child_fold,
                generated,
                &mut frontiers,
            )?;
        }
    }
    if let Some(diagnostics) = ctx.diagnostics {
        diagnostics.record_completed_state(frontiers.candidate_count());
    }
    let result = Arc::new(finish_state(retains_setup_projection, frontiers));
    memo.insert(memo_key, Arc::clone(&result), ctx.diagnostics);
    Ok(result)
}
