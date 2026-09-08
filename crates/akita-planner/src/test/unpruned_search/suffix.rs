use super::*;

type OpeningWork = (crate::schedule_params::PlannerOpeningCandidate, usize);

fn evaluation_trace_work(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    dimensions: CommitmentRingDims,
) -> Result<Option<OpeningWork>, AkitaError> {
    let opening_shape = akita_types::PolynomialGroupLayout::singleton(
        akita_types::padded_boolean_opening_vars(state.input_witness_len)?,
    );
    let Ok(ring_challenge) = (ctx.ring_challenge_config)(dimensions.d_a()) else {
        return Ok(None);
    };
    let Some(bytes) = try_extension_opening_reduction_level_bytes(
        ctx.policy.challenge_field_bits()?,
        ctx.policy.claim_ext_degree,
        opening_shape,
    )?
    else {
        return Ok(None);
    };
    Ok(Some((
        crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(ring_challenge),
        bytes,
    )))
}

fn retain_terminal_candidates(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    log_basis: u32,
    dimensions: CommitmentRingDims,
    trace_work: Option<OpeningWork>,
    work: &mut OracleWork,
    frontier: &mut OracleFrontier,
) -> Result<(), AkitaError> {
    let Some((opening, opening_reduction_bytes)) = trace_work else {
        return Ok(());
    };
    for &payload_mode in state.payload_phase.candidate_modes(state.level, false) {
        let request = RecursiveCandidateRequest {
            policy: ctx.policy,
            payload_mode,
            opening,
            dimensions,
            current_witness_len: state.input_witness_len,
            source: crate::InnerBasisSource::BalancedDigits {
                log_basis: state.current_log_basis,
            },
            log_basis_inner: state.current_log_basis,
            log_basis_open: log_basis,
            fold_level: state.level,
            source_moment: state.source_moment,
            relation_traversal_order: RelationTraversalOrder::Canonical,
            guide: None,
        };
        for params in derive_unpruned_terminal_candidates_for_oracle(request)? {
            work.record_candidate_route(&params)?;
            if let Some(candidate) = terminal(ctx, state, opening_reduction_bytes, &params)? {
                retain_frontier_candidate(frontier, candidate)?;
            }
        }
    }
    Ok(())
}

fn next_source_moment(
    policy: &PlannerPolicy,
    state: UnprunedState,
    params: &CommittedGroupParams,
) -> Result<Option<crate::response_model::SourceMomentEstimate>, AkitaError> {
    if !policy.selective_l2_response_model_enabled() {
        return Ok(None);
    }
    let opening_layout = suffix_opening_layout(state.input_witness_len, None)?;
    Ok(Some(crate::response_model::next_source_moment(
        params,
        &opening_layout,
        &[state.source_moment.ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned response source moment is missing".into())
        })?],
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
    )?))
}

struct FoldOpening {
    log_basis: u32,
    dimensions: CommitmentRingDims,
    opening: crate::schedule_params::PlannerOpeningCandidate,
    reduction_bytes: usize,
}

fn visit_fold_opening(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    fold_opening: FoldOpening,
    memo: &mut OracleMemo,
    work: &mut OracleWork,
    frontier: &mut OracleFrontier,
) -> Result<(), AkitaError> {
    let policy = ctx.policy;
    for &payload_mode in state.payload_phase.candidate_modes(state.level, false) {
        for relation_transition in relation::transitions(state.relation_state, state.level) {
            let request = RecursiveCandidateRequest {
                policy,
                payload_mode,
                opening: fold_opening.opening,
                dimensions: fold_opening.dimensions,
                current_witness_len: state.input_witness_len,
                source: crate::InnerBasisSource::BalancedDigits {
                    log_basis: state.current_log_basis,
                },
                log_basis_inner: state.current_log_basis,
                log_basis_open: fold_opening.log_basis,
                fold_level: state.level,
                source_moment: state.source_moment,
                relation_traversal_order: RelationTraversalOrder::Canonical,
                guide: None,
            };
            let fold_candidates = derive_unpruned_fold_candidates_for_oracle(
                request,
                RelationSearchDomain::for_mode(relation_transition.mode),
            )?;
            if relation_transition.mode.is_reduced_evaluation() {
                work.record_reduced_fold_candidates(fold_candidates.len())?;
            }
            for (candidate, output_witness_len) in fold_candidates {
                if candidate.ring_relation_mode != relation_transition.mode {
                    return Err(AkitaError::InvalidSetup(
                        "oracle candidate relation token disagrees with its requested transition"
                            .into(),
                    ));
                }
                let params = candidate;
                work.record_candidate_route(&params)?;
                let next_state = UnprunedState {
                    level: state.level + 1,
                    input_witness_len: output_witness_len,
                    current_log_basis: fold_opening.log_basis,
                    source_moment: next_source_moment(policy, state, &params)?,
                    dimension_ceiling: params.role_dims(),
                    payload_phase: state.payload_phase.after(params.payload_mode),
                    relation_state: relation_transition.next_state,
                };
                visit_suffixes(ctx, next_state, memo, work, &mut |child| {
                    retain_frontier_candidate(
                        frontier,
                        prepend_fold(
                            policy,
                            state.level,
                            state.input_witness_len,
                            output_witness_len,
                            fold_opening.reduction_bytes,
                            &params,
                            &child,
                        )?,
                    )
                })?;
            }
        }
    }
    Ok(())
}

pub(super) fn visit_suffixes(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    memo: &mut OracleMemo,
    work: &mut OracleWork,
    visitor: &mut dyn FnMut(ScheduleCandidate) -> Result<(), AkitaError>,
) -> Result<(), AkitaError> {
    if let Some(cached) = memo.get(&state) {
        for candidate in Arc::clone(cached).iter().cloned() {
            visitor(candidate)?;
        }
        return Ok(());
    }
    work.visit_suffix_state()?;
    if state.level > MAX_ORACLE_RECURSION_DEPTH {
        return Ok(());
    }
    let mut frontier = OracleFrontier::default();
    let (min_log_basis, max_log_basis) =
        crate::policy::log_basis_search_range_at_level(ctx.policy, state.level);
    for log_basis in min_log_basis.max(state.current_log_basis)..=max_log_basis {
        for dimensions in dimension_candidates(ctx.policy, state.level, state.dimension_ceiling)? {
            let trace_work = evaluation_trace_work(ctx, state, dimensions)?;
            retain_terminal_candidates(
                ctx,
                state,
                log_basis,
                dimensions,
                trace_work,
                work,
                &mut frontier,
            )?;
            let fold_work = if state.level <= 1 {
                crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
                    state.level,
                    ctx.policy.claim_ext_degree,
                    dimensions,
                )?
                .into_iter()
                .map(|opening| (opening, 0))
                .collect::<Vec<_>>()
            } else {
                trace_work.into_iter().collect()
            };
            for (opening, reduction_bytes) in fold_work {
                visit_fold_opening(
                    ctx,
                    state,
                    FoldOpening {
                        log_basis,
                        dimensions,
                        opening,
                        reduction_bytes,
                    },
                    memo,
                    work,
                    &mut frontier,
                )?;
            }
        }
    }
    let frontier = Arc::new(frontier.into_candidates());
    memo.insert(state, Arc::clone(&frontier));
    for candidate in frontier.iter().cloned() {
        visitor(candidate)?;
    }
    Ok(())
}
