use super::*;
use crate::schedule_params::ReducedTransitionRejection;

struct OpeningSearch<'a> {
    state: SuffixState<'a>,
    depth: usize,
    open_log_basis: u32,
    opening_layout: &'a OpeningClaimsLayout,
    require_child_fold: bool,
    guide_scope: Option<GuideScope>,
    query_search: QuerySearch,
}

fn adaptation_guide_allows_state(ctx: &SuffixCtx<'_>, state: SuffixState<'_>) -> bool {
    let Some(guide) = ctx.adaptation_guide else {
        return true;
    };
    if state.level == 0 {
        return state.topology.incoming_setup_prefix().is_none();
    }
    if let Some(step) = guide.recursive_folds.get(state.level - 1) {
        return state.topology.incoming_setup_prefix().is_some()
            == step.params.setup_prefix().is_some();
    }
    state.level == guide.recursive_folds.len() + 1
        && state.topology.incoming_setup_prefix().is_none()
}

/// What a fold candidate's edges into its successors need, decided before
/// any successor is planned.
struct CandidateEdges {
    natural_setup_field_len: usize,
    direct_edge_is_admissible: bool,
    prune_direct_edge: bool,
}

enum EdgeDecision {
    /// The candidate enters no frontier. `pruned` records a guided prune.
    Dropped {
        pruned: bool,
    },
    Planned(CandidateEdges),
}

/// The successor states one planned fold candidate requests: they share the
/// fold's output and differ only in topology.
struct ChildRequests {
    query: QuerySearch,
    input_chunks: Option<Arc<akita_params::WitnessChunkShape>>,
    level: usize,
    next_witness_len: usize,
    open_log_basis: u32,
    next_source_moment: Option<crate::response_model::SourceMomentEstimate>,
    dimension_ceiling: CommitmentRingDims,
    direct: Option<DirectChild>,
    offloaded: Option<SuffixTopology>,
}

enum DirectChild {
    /// The recursion bound ends the schedule with this fold's successor.
    Empty,
    Search(SuffixTopology),
}

impl ChildRequests {
    fn state(&self, topology: SuffixTopology) -> SuffixState<'_> {
        SuffixState {
            input_chunks: self.input_chunks.as_ref(),
            level: self.level,
            current_witness_len: self.next_witness_len,
            current_lb: self.open_log_basis,
            source_moment: self.next_source_moment,
            dimension_ceiling: self.dimension_ceiling,
            topology,
        }
    }
}

/// Decides `candidate`'s edges against the state's current `frontiers`.
fn candidate_edges(
    ctx: &SuffixCtx<'_>,
    search: &OpeningSearch<'_>,
    guide: Option<(CompleteObjectiveBound, usize)>,
    candidate: &PlannedFoldCandidate,
    frontiers: &StateFrontiers,
) -> Result<EdgeDecision, AkitaError> {
    let state = search.state;
    let params = &candidate.params;
    if let Some(natural_prefix_len) = state.topology.incoming_setup_prefix() {
        let padded_prefix_len = akita_params::padded_setup_prefix_len(natural_prefix_len);
        if !offloaded_witness_contracts(
            state.current_witness_len,
            state.current_lb,
            padded_prefix_len,
            ctx.policy.decomposition.field_bits(),
            candidate.next_witness_len,
            search.open_log_basis,
            ctx.policy.min_offloaded_witness_contraction,
        )? {
            return Ok(EdgeDecision::Dropped { pruned: false });
        }
    }
    let natural_len = guide
        .map(|(_, natural_len)| natural_len)
        .map_or_else(|| active_setup_field_len(params, search.opening_layout), Ok)?;
    let direct_edge_is_admissible =
        state
            .topology
            .incoming_setup_prefix()
            .is_none_or(|incoming_len| {
                akita_params::padded_setup_prefix_len(natural_len)
                    < akita_params::padded_setup_prefix_len(incoming_len)
            });
    if matches!(
        ctx.policy.selection_policy,
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenExactProofAndWorkV6
    ) && matches!(search.guide_scope, Some(GuideScope::CompleteRoot))
        && guide.is_some_and(|(lower_bound, _)| {
            complete_root_setup_bound_is_strictly_worse(lower_bound, &frontiers.projected)
        })
    {
        return Ok(EdgeDecision::Dropped { pruned: true });
    }
    let prune_direct_edge = if direct_edge_is_admissible {
        guide
            .zip(search.guide_scope)
            .map(|((lower_bound, _), guide_scope)| {
                direct_edge_bound_is_strictly_worse(
                    ctx.policy,
                    guide_scope,
                    params,
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
    if prune_direct_edge && !ctx.policy.recursive_setup_planning {
        return Ok(EdgeDecision::Dropped { pruned: true });
    }
    Ok(EdgeDecision::Planned(CandidateEdges {
        natural_setup_field_len: natural_len,
        direct_edge_is_admissible,
        prune_direct_edge,
    }))
}

/// The successor states `candidate` requests under `edges`, or `None` when
/// its query budget admits none.
fn child_requests(
    ctx: &SuffixCtx<'_>,
    search: &OpeningSearch<'_>,
    candidate: &PlannedFoldCandidate,
    edges: &CandidateEdges,
    diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
) -> Result<Option<ChildRequests>, AkitaError> {
    let state = search.state;
    let params = &candidate.params;
    let Some(query) = search
        .query_search
        .child(ctx, state, search.opening_layout, params)?
    else {
        return Ok(None);
    };
    let guided_successor_is_offloaded = ctx.adaptation_guide.map(|guide| {
        guide
            .recursive_folds
            .get(state.level)
            .is_some_and(|successor| successor.params.setup_prefix().is_some())
    });
    let offload_forced = ctx.zk.is_some_and(|zk| zk.forces_offload_from(state.level));
    let direct = if guided_successor_is_offloaded == Some(true)
        || offload_forced
        || !edges.direct_edge_is_admissible
        || edges.prune_direct_edge
    {
        None
    } else if search.depth == MAX_RECURSION_DEPTH {
        Some(DirectChild::Empty)
    } else {
        Some(DirectChild::Search(state.topology.direct_successor(
            params.payload_mode,
            params.ring_relation_mode,
        )))
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
        && params.payload_mode.is_compressed()
        && params.ring_relation_mode.is_reduced_evaluation()
    {
        if let Some(diagnostics) = diagnostics {
            diagnostics.record_reduced_rejection(ReducedTransitionRejection::OutgoingSetupOffload);
        }
    }
    let offloaded = SuffixTopology::offloaded_successor(
        params.ring_relation_mode,
        params.payload_mode,
        edges.natural_setup_field_len,
    )
    .filter(|_| guided_successor_is_offloaded != Some(false) && offload_search_enabled);
    Ok(Some(ChildRequests {
        query,
        input_chunks: candidate.chunk_shape.map(Arc::new),
        level: state.level + 1,
        next_witness_len: candidate.next_witness_len,
        open_log_basis: search.open_log_basis,
        next_source_moment: candidate.next_source_moment,
        dimension_ceiling: params.role_dims(),
        direct,
        offloaded,
    }))
}

fn price_planned_fold_candidate(
    ctx: &SuffixCtx<'_>,
    memo: &ScheduleMemo,
    search: &OpeningSearch<'_>,
    guide: Option<(CompleteObjectiveBound, usize)>,
    candidate: PlannedFoldCandidate,
    frontiers: &mut StateFrontiers,
) -> Result<(), AkitaError> {
    let edges = match candidate_edges(ctx, search, guide, &candidate, frontiers)? {
        EdgeDecision::Dropped { pruned } => {
            if pruned {
                if let Some(diagnostics) = ctx.diagnostics {
                    diagnostics.record_guided_direct_edge_prune();
                }
            }
            return Ok(());
        }
        EdgeDecision::Planned(edges) => edges,
    };
    if edges.prune_direct_edge {
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_guided_direct_edge_prune();
        }
    }
    let (direct, offloaded) =
        match child_requests(ctx, search, &candidate, &edges, ctx.diagnostics)? {
            None => (None, None),
            Some(requests) => {
                let direct = requests
                    .direct
                    .as_ref()
                    .map(|child| match child {
                        DirectChild::Empty => Ok(empty_suffix_result()),
                        DirectChild::Search(topology) => derive_selected_suffix_schedule(
                            ctx,
                            memo,
                            requests.state(*topology),
                            search.depth + 1,
                            requests.query.clone(),
                        ),
                    })
                    .transpose()?;
                let offloaded = requests
                    .offloaded
                    .map(|topology| {
                        derive_selected_suffix_schedule(
                            ctx,
                            memo,
                            requests.state(topology),
                            search.depth + 1,
                            requests.query.clone(),
                        )
                    })
                    .transpose()?;
                (direct, offloaded)
            }
        };
    price_level_candidate_with_children(
        ctx,
        search.state,
        search.opening_layout,
        LevelCandidateEdge {
            params: &candidate.params,
            natural_setup_field_len: edges.natural_setup_field_len,
            require_child_fold: search.require_child_fold,
        },
        CandidateChildren {
            direct: direct.as_deref(),
            offloaded: offloaded.as_deref(),
        },
        frontiers,
    )
}

/// How many candidates of a state's traversal have their successors
/// completed together, once `waves_done` waves have been. A guided
/// traversal prunes against the frontier its earlier candidates built, so a
/// candidate whose successors are completed in a later wave may be pruned
/// before they are.
///
/// The root's waves double from one candidate up to the pool's width. Its
/// candidates are in global lower-bound order and its successors are the
/// largest subtrees: the first completions build the frontier that prunes
/// most of the rest, so early pool-width waves completed subtrees the
/// traversal then pruned (1.5x the states of a sequential search for the
/// 2^25 response statement). Doubling bounds that waste by the last wave
/// and still fills the pool when many candidates survive (masked catalogs).
/// Every other state completes all its candidates' successors at once: for
/// setup-prefixed states, pool-width waves completed the same states as one
/// wave on the response statements and only added barriers
/// (`jolt:specs/whiteout-o2b-planner.md`).
fn successor_wave(guide_scope: Option<GuideScope>, waves_done: u32) -> usize {
    #[cfg(feature = "parallel")]
    if matches!(guide_scope, Some(GuideScope::CompleteRoot)) {
        return 1usize
            .checked_shl(waves_done)
            .map_or(usize::MAX, |wave| wave.min(rayon::current_num_threads()));
    }
    #[cfg(not(feature = "parallel"))]
    let _ = (guide_scope, waves_done);
    usize::MAX
}

/// Completes, in parallel, the successor states that `traversal`'s
/// candidates request under the state's current `frontiers`, so the ordered
/// traversal that follows reads them from `memo`. `search` is the state's
/// search; each candidate's opening layout replaces its layout.
///
/// This only fills the memo, whose entries are functions of their keys, so
/// the traversal selects exactly what it selects without it. Prunes are
/// decided against the frontier at the start of the wave; within the wave
/// the traversal's frontiers only grow, so it may prune a candidate whose
/// successors were completed here (wasted work, never a different result).
/// This skips a successor only when it is already completed or claimed by
/// another thread, when deciding its candidate's edges failed, or when the
/// traversal's request differs (a query budget that admits no successor);
/// the traversal completes any skipped successor it needs. Every successor
/// is completed unconstrained by the query budget, which is how a
/// restricted request first looks it up. An error is dropped here and
/// reported by the traversal if it reaches that state.
#[cfg(feature = "parallel")]
fn complete_successors(
    ctx: &SuffixCtx<'_>,
    memo: &ScheduleMemo,
    search: &OpeningSearch<'_>,
    traversal: &[(
        Option<(CompleteObjectiveBound, usize)>,
        PlannedFoldCandidate,
    )],
    frontiers: &StateFrontiers,
) {
    use rayon::prelude::*;

    let mut requests = Vec::new();
    for (guide, candidate) in traversal {
        let Ok(opening_layout) = candidate
            .params
            .opening_layout_for_final_group(candidate.params.group())
        else {
            continue;
        };
        let candidate_search = OpeningSearch {
            opening_layout: &opening_layout,
            query_search: QuerySearch::Unconstrained,
            ..*search
        };
        let Ok(EdgeDecision::Planned(edges)) =
            candidate_edges(ctx, &candidate_search, *guide, candidate, frontiers)
        else {
            continue;
        };
        if let Ok(Some(request)) = child_requests(ctx, &candidate_search, candidate, &edges, None) {
            requests.push(request);
        }
    }
    let mut pending = std::collections::HashSet::new();
    let states: Vec<_> = requests
        .iter()
        .flat_map(|request| {
            let direct = match request.direct {
                Some(DirectChild::Search(topology)) => Some(topology),
                Some(DirectChild::Empty) | None => None,
            };
            direct
                .into_iter()
                .chain(request.offloaded)
                .map(|topology| request.state(topology))
        })
        .filter(|state| {
            let key = state.memo_key(ctx.policy);
            !memo.is_completed_or_claimed(&key) && pending.insert(key)
        })
        .collect();
    let depth = search.depth + 1;
    states.into_par_iter().for_each(|state| {
        // The traversal reports this state's error if it reaches the state.
        let _ =
            derive_selected_suffix_schedule(ctx, memo, state, depth, QuerySearch::Unconstrained);
    });
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
pub(super) fn process_candidate_batch(
    ctx: &SuffixCtx<'_>,
    memo: &ScheduleMemo,
    state: SuffixState<'_>,
    depth: usize,
    open_log_basis: u32,
    opening_layout: &OpeningClaimsLayout,
    is_root_level: bool,
    require_child_fold: bool,
    generated: candidates::GeneratedCandidates,
    frontiers: &mut StateFrontiers,
    query_search: &QuerySearch,
) -> Result<(), AkitaError> {
    let generated_candidate_count = generated
        .terminal
        .len()
        .saturating_add(generated.folds.len());
    let terminal_candidate_count = generated.terminal.len();
    for candidate in generated.terminal {
        let natural_len = active_setup_field_len(&candidate, opening_layout)?;
        price_terminal_candidate(ctx, state, query_search, &candidate, natural_len, frontiers)?;
    }
    let candidates_with_source = attach_source_moments(ctx, state, is_root_level, generated.folds)?;
    // Complete-root candidates are traversed in exact lower-bound order.
    // Recursive states do not have that global admission rule and retain
    // local Pareto pruning.
    let candidates = if is_root_level || matches!(query_search, QuerySearch::Restricted(_)) {
        candidates_with_source
    } else {
        prune::level_candidates(
            opening_layout,
            state.input_chunks.is_some(),
            candidates_with_source,
        )?
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
    let guide_scope = GuideScope::for_state(is_root_level, incoming_setup_prefix);
    let mut traversal =
        candidate_traversal(ctx.policy, guide_scope, opening_layout, candidates)?.peekable();
    let mut waves_done = 0;
    while traversal.peek().is_some() {
        let wave: Vec<_> = traversal
            .by_ref()
            .take(successor_wave(guide_scope, waves_done))
            .collect();
        waves_done = waves_done.saturating_add(1);
        #[cfg(feature = "parallel")]
        complete_successors(
            ctx,
            memo,
            &OpeningSearch {
                state,
                depth,
                open_log_basis,
                opening_layout,
                require_child_fold,
                guide_scope,
                query_search: QuerySearch::Unconstrained,
            },
            &wave,
            frontiers,
        );
        for (guide, candidate) in wave {
            let opening_layout = candidate
                .params
                .opening_layout_for_final_group(candidate.params.group())?;
            let search = OpeningSearch {
                state,
                depth,
                open_log_basis,
                opening_layout: &opening_layout,
                require_child_fold,
                guide_scope,
                query_search: query_search.clone(),
            };
            price_planned_fold_candidate(ctx, memo, &search, guide, candidate, frontiers)?;
        }
    }
    Ok(())
}

/// Derive the suffix frontier for the selected recursive schedule at
/// `(level, current_witness_len, current_lb)`.
pub(crate) fn derive_selected_suffix_schedule(
    ctx: &SuffixCtx<'_>,
    memo: &ScheduleMemo,
    state: SuffixState<'_>,
    depth: usize,
    query_search: QuerySearch,
) -> Result<Arc<SuffixResult>, AkitaError> {
    if !adaptation_guide_allows_state(ctx, state) {
        return Ok(empty_suffix_result());
    }
    let policy = ctx.policy;
    let relation_phase = state.topology.relation_phase();
    if let Some(diagnostics) = ctx.diagnostics {
        diagnostics.record_suffix_call(relation_phase);
    }
    let memo_key = state.memo_key(policy);
    // Memo entries remain budget-independent objective optima. A restricted
    // caller reuses that result when it fits its exact root prefix; otherwise
    // it recomputes this state locally and never replaces the primary entry.
    if let QuerySearch::Restricted(prefix) = &query_search {
        let cached = memo.get(&memo_key).map_or_else(
            || derive_selected_suffix_schedule(ctx, memo, state, depth, QuerySearch::Unconstrained),
            Ok,
        )?;
        if prefix.admits_result(policy, ctx.challenge_order, &cached)? {
            if let Some(diagnostics) = ctx.diagnostics {
                diagnostics.record_memo_result(relation_phase, true);
            }
            return Ok(cached);
        }
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_memo_result(relation_phase, false);
        }
    } else if matches!(query_search, QuerySearch::Unconstrained) && depth <= MAX_RECURSION_DEPTH {
        let cached = memo.get(&memo_key);
        if let Some(diagnostics) = ctx.diagnostics {
            diagnostics.record_memo_result(relation_phase, cached.is_some());
        }
        if let Some(cached) = cached {
            return Ok(cached);
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
    #[cfg(feature = "parallel")]
    let _claim = matches!(query_search, QuerySearch::Unconstrained).then(|| memo.claim(&memo_key));
    let candidate_domain = candidates::CandidateDomain::prepare(ctx, state)?;
    let root_level_key = candidate_domain.root_level_key;
    let opening_layout = &candidate_domain.opening_layout;
    let mut frontiers = StateFrontiers::new();
    for open_log_basis in candidate_domain.opening_basis_range.clone() {
        if root_level_key.is_some() {
            let generated = candidate_domain.root_candidates(ctx, open_log_basis)?;
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
                &query_search,
            )?;
        } else {
            let generated = candidate_domain.generate_recursive_for_opening_basis(
                ctx,
                state,
                open_log_basis,
                &memo.setup_prefixes,
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
                &query_search,
            )?;
        }
    }
    if let Some(diagnostics) = ctx.diagnostics {
        diagnostics.record_completed_state(frontiers.candidate_count());
    }
    let result = Arc::new(finish_state(retains_setup_projection, frontiers));
    if matches!(query_search, QuerySearch::Unconstrained) {
        memo.insert(memo_key, Arc::clone(&result), ctx.diagnostics);
    }
    Ok(result)
}
