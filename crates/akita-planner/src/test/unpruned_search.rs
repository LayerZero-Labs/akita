use super::*;
use akita_schedules::planner_support::MAX_RECURSION_DEPTH;

struct UnprunedCtx<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_config: &'a dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    key: PolynomialGroupLayout,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct UnprunedState {
    level: usize,
    input_witness_len: usize,
    current_log_basis: u32,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    dimension_ceiling: CommitmentRingDims,
    payload_phase: akita_types::CommitmentPayloadPhase,
    cutover_level: Option<usize>,
}

type UnprunedMemo = Vec<(UnprunedState, Arc<Vec<ScheduleCandidate>>)>;

fn packing_opening_domain(
    level: usize,
    extension_degree: usize,
    dimensions: CommitmentRingDims,
) -> Result<Vec<crate::schedule_params::PlannerOpeningCandidate>, AkitaError> {
    crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
        level,
        extension_degree,
        dimensions,
    )
}

fn derive_unpruned_fold_candidates(
    mut request: RecursiveCandidateRequest<'_>,
    relation_mode: akita_types::RingRelationMode,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let exhaustive = request.fold_level < akita_schedules::ADAPTIVE_SEARCH_LEVELS;
    if !exhaustive {
        request.source_moment = None;
    }
    let fold_policy = if exhaustive {
        FoldCandidatePolicy::Frontier(SplitBoundPolicy::DisabledForOracle)
    } else {
        FoldCandidatePolicy::Best
    };
    derive_fold_candidates(
        request,
        RecursiveSetupPrefix::None,
        fold_policy,
        RelationTransition::for_mode(relation_mode),
    )
}

fn terminal_schedule_candidate(
    policy: &PlannerPolicy,
    key: PolynomialGroupLayout,
    level: usize,
    input_witness_len: usize,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
    field_bits: u32,
    opening_reduction_bytes: usize,
    params: &CommittedGroupParams,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    let Some((mut terminal, terminal_bytes)) = suffix_dp::try_terminal_direct_suffix_cost(
        policy,
        input_witness_len,
        params,
        field_bits,
        key,
        level,
        None,
        source_moment,
    )?
    else {
        return Ok(None);
    };
    let payload_bytes = opening_reduction_bytes
        .checked_add(terminal_bytes)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned traversal terminal proof size overflow".into())
        })?;
    terminal.estimated_direct_payload_bytes = opening_reduction_bytes;
    Ok(Some(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_types::active_setup_field_len(
                params,
                &suffix_opening_layout(input_witness_len, None)?,
            )?,
        ),
        cost: PackedProofCost::new(payload_bytes, 0)?,
        setup_field_elements: terminal_setup_field_elements(&terminal.params)?,
        folds: CandidateFoldChain::default(),
        terminal: Arc::new(terminal),
    }))
}

fn prepend_fold_candidate(
    field_bits: u32,
    challenge_field_bits: u32,
    input_witness_len: usize,
    output_witness_len: usize,
    opening_reduction_bytes: usize,
    params: &CommittedGroupParams,
    child: &ScheduleCandidate,
) -> Result<ScheduleCandidate, AkitaError> {
    let binding = if child.folds.is_empty() {
        akita_types::NextWitnessBindingPolicy::TerminalInnerState
    } else {
        akita_types::NextWitnessBindingPolicy::OuterPayload
    };
    let direct_bytes = level_proof_bytes(
        field_bits,
        challenge_field_bits,
        params,
        child.first_fold_params(),
        output_witness_len,
        Some(binding),
    )?
    .checked_add(opening_reduction_bytes)
    .ok_or_else(|| {
        AkitaError::InvalidSetup("unpruned traversal fold proof size overflow".into())
    })?;
    Ok(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_types::active_setup_field_len(
                params,
                &suffix_opening_layout(input_witness_len, None)?,
            )?,
        ),
        cost: child.cost.checked_prepend(direct_bytes, 0)?,
        setup_field_elements: level_setup_field_elements(params)?.max(child.setup_field_elements),
        folds: child.folds.prepend(CandidateFoldStep {
            params: Arc::new(params.clone()),
            input_witness_len,
            output_witness_len,
            estimated_direct_payload_bytes: direct_bytes,
            estimated_stage3_payload_bytes: 0,
        }),
        terminal: Arc::clone(&child.terminal),
    })
}

fn enumerate_suffixes(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    memo: &mut UnprunedMemo,
) -> Result<Arc<Vec<ScheduleCandidate>>, AkitaError> {
    if let Some((_, suffixes)) = memo.iter().find(|(cached, _)| *cached == state) {
        return Ok(Arc::clone(suffixes));
    }
    let UnprunedCtx {
        policy,
        ring_challenge_config,
        key,
    } = *ctx;
    let UnprunedState {
        level,
        input_witness_len,
        current_log_basis,
        source_moment,
        dimension_ceiling,
        payload_phase,
        cutover_level,
    } = state;
    if level > MAX_RECURSION_DEPTH {
        return Ok(Arc::new(Vec::new()));
    }
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = policy.challenge_field_bits()?;
    let (min_log_basis, max_log_basis) =
        crate::policy::log_basis_search_range_at_level(policy, level);
    let terminal_opening_shape = akita_types::PolynomialGroupLayout::singleton(
        akita_types::padded_boolean_opening_vars(input_witness_len)?,
    );
    let mut schedules = Vec::new();
    for log_basis in min_log_basis.max(current_log_basis)..=max_log_basis {
        for candidate_dimensions in dimension_candidates(policy, level, dimension_ceiling)? {
            let trace_work = ring_challenge_config(candidate_dimensions.d_a())
                .ok()
                .and_then(|ring_challenge| {
                    try_extension_opening_reduction_level_bytes(
                        challenge_field_bits,
                        policy.claim_ext_degree,
                        terminal_opening_shape,
                    )
                    .transpose()
                    .map(|result| {
                        result.map(|eor_bytes| {
                            (
                                crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(
                                    ring_challenge,
                                ),
                                eor_bytes,
                            )
                        })
                    })
                })
                .transpose()?;
            if let Some((trace_opening, terminal_eor_bytes)) = trace_work {
                for &payload_mode in payload_phase.candidate_modes(level, false) {
                    for params in derive_terminal_candidates(RecursiveCandidateRequest {
                        policy,
                        payload_mode,
                        opening: trace_opening,
                        dimensions: candidate_dimensions,
                        current_witness_len: input_witness_len,
                        source: crate::InnerBasisSource::BalancedDigits {
                            log_basis: current_log_basis,
                        },
                        log_basis_inner: current_log_basis,
                        log_basis_open: log_basis,
                        fold_level: level,
                        source_moment,
                    })? {
                        if let Some(candidate) = terminal_schedule_candidate(
                            policy,
                            key,
                            level,
                            input_witness_len,
                            source_moment,
                            field_bits,
                            terminal_eor_bytes,
                            &params,
                        )? {
                            schedules.push(candidate);
                        }
                    }
                }
            }

            let fold_work = if level <= 1 {
                packing_opening_domain(level, policy.claim_ext_degree, candidate_dimensions)?
                    .into_iter()
                    .map(|opening| (opening, 0))
                    .collect::<Vec<_>>()
            } else {
                trace_work.into_iter().collect()
            };
            for (opening, opening_reduction_bytes) in fold_work {
                for &payload_mode in payload_phase.candidate_modes(level, false) {
                    let ring_relation_mode =
                        if cutover_level.is_some_and(|cutover| level >= cutover) {
                            akita_types::RingRelationMode::ReducedEvaluation
                        } else {
                            akita_types::RingRelationMode::QuotientLift
                        };
                    {
                        let request = RecursiveCandidateRequest {
                            policy,
                            payload_mode,
                            opening,
                            dimensions: candidate_dimensions,
                            current_witness_len: input_witness_len,
                            source: crate::InnerBasisSource::BalancedDigits {
                                log_basis: current_log_basis,
                            },
                            log_basis_inner: current_log_basis,
                            log_basis_open: log_basis,
                            fold_level: level,
                            source_moment,
                        };
                        for (params, output_witness_len) in
                            derive_unpruned_fold_candidates(request, ring_relation_mode)?
                        {
                            let child_ceiling = params.role_dims();
                            let next_source_moment = if policy.selective_l2_response_model_enabled()
                            {
                                let opening_layout =
                                    suffix_opening_layout(input_witness_len, None)?;
                                Some(crate::response_model::next_source_moment(
                                    &params,
                                    &opening_layout,
                                    &[source_moment.ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "unpruned response source moment is missing".into(),
                                        )
                                    })?],
                                    field_bits,
                                    policy.claim_ext_degree,
                                )?)
                            } else {
                                None
                            };
                            for child in enumerate_suffixes(
                                ctx,
                                UnprunedState {
                                    level: level + 1,
                                    input_witness_len: output_witness_len,
                                    current_log_basis: log_basis,
                                    source_moment: next_source_moment,
                                    dimension_ceiling: child_ceiling,
                                    payload_phase: payload_phase.after(params.payload_mode),
                                    cutover_level,
                                },
                                memo,
                            )?
                            .iter()
                            {
                                schedules.push(prepend_fold_candidate(
                                    field_bits,
                                    challenge_field_bits,
                                    input_witness_len,
                                    output_witness_len,
                                    opening_reduction_bytes,
                                    &params,
                                    child,
                                )?);
                            }
                        }
                    }
                }
            }
        }
    }
    let schedules = Arc::new(schedules);
    memo.push((state, Arc::clone(&schedules)));
    Ok(schedules)
}

pub(super) fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    find_schedule_with_cutover_order(
        key,
        policy,
        honest_fold_policy,
        ring_challenge_config,
        false,
    )
}

pub(super) fn find_schedule_with_cutover_order(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    reverse_cutover_order: bool,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate()?;
    akita_schedules::planner_support::validate_policy(policy)?;

    let field_bits = policy.decomposition.field_bits();
    let input_witness_len = 1usize.checked_shl(key.num_vars() as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("unpruned traversal root witness too large".into())
    })?;
    let (min_log_basis, max_log_basis) = crate::policy::log_basis_search_range_at_level(policy, 0);
    let mut complete = Vec::new();
    let schedule_key = akita_types::AkitaScheduleLookupKey::single(key);
    let ctx = UnprunedCtx {
        policy,
        ring_challenge_config: &ring_challenge_config,
        key,
    };
    // This remains an exhaustive oracle: memoization only reuses the complete
    // candidate set for an identical suffix state. In particular, it applies
    // none of the dominance or lower-bound pruning used by the production DP.
    let mut memo = UnprunedMemo::new();
    let inner_source =
        root_inner_basis_source(honest_fold_policy, policy.decomposition.log_commit_bound);
    let (min_inner_basis, max_inner_basis) = inner_source.search_range(policy)?;
    let mut cutover_levels = std::iter::once(None)
        .chain((2..=MAX_RECURSION_DEPTH).map(Some))
        .collect::<Vec<_>>();
    if reverse_cutover_order {
        cutover_levels.reverse();
    }
    for log_basis in min_log_basis..=max_log_basis {
        for inner_basis in min_inner_basis..=max_inner_basis {
            for root_dimensions in
                dimension_candidates(policy, 0, initial_dimension_ceiling(policy)?)?
            {
                let alpha = root_dimensions.d_a().trailing_zeros() as usize;
                let reduced_vars = key.num_vars().saturating_sub(alpha);
                if reduced_vars == 0 {
                    continue;
                }
                let root_openings =
                    packing_opening_domain(0, policy.claim_ext_degree, root_dimensions)?;
                for root_opening in root_openings {
                    for (root_params, output_witness_len) in
                        crate::planner::root_level_candidates_for_basis(
                            &schedule_key,
                            honest_fold_policy,
                            &[],
                            policy,
                            root_dimensions,
                            root_opening,
                            &[],
                            inner_basis,
                            log_basis,
                        )?
                    {
                        let next_source_moment = if policy.selective_l2_response_model_enabled() {
                            let opening_layout = schedule_key.opening_layout()?;
                            let source_groups = crate::response_model::root_group_source_moments(
                                &root_params,
                                &opening_layout,
                                honest_fold_policy,
                                &[],
                                policy.decomposition,
                            )?;
                            Some(crate::response_model::next_source_moment(
                                &root_params,
                                &opening_layout,
                                &source_groups,
                                field_bits,
                                policy.claim_ext_degree,
                            )?)
                        } else {
                            None
                        };
                        for &cutover_level in &cutover_levels {
                            for suffix in enumerate_suffixes(
                                &ctx,
                                UnprunedState {
                                    level: 1,
                                    input_witness_len: output_witness_len,
                                    current_log_basis: log_basis,
                                    source_moment: next_source_moment,
                                    dimension_ceiling: root_dimensions,
                                    payload_phase:
                                        akita_types::CommitmentPayloadPhase::CompressedPrefix,
                                    cutover_level,
                                },
                                &mut memo,
                            )?
                            .iter()
                            {
                                let opening_layout = schedule_key.opening_layout()?;
                                let first_direct_setup_field_len = NonZeroUsize::new(
                                    active_setup_field_len(&root_params, &opening_layout)?,
                                )
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "unpruned root setup field length must be nonzero".into(),
                                    )
                                })?;
                                let child_is_terminal = suffix.folds.is_empty();
                                let root_bytes = level_proof_bytes(
                                    field_bits,
                                    policy.challenge_field_bits()?,
                                    &root_params,
                                    suffix.first_fold_params(),
                                    output_witness_len,
                                    Some(if child_is_terminal {
                                        akita_types::NextWitnessBindingPolicy::TerminalInnerState
                                    } else {
                                        akita_types::NextWitnessBindingPolicy::OuterPayload
                                    }),
                                )?;
                                let folds = suffix.folds.prepend(CandidateFoldStep {
                                    params: Arc::new(root_params.clone()),
                                    input_witness_len,
                                    output_witness_len,
                                    estimated_direct_payload_bytes: root_bytes,
                                    estimated_stage3_payload_bytes: 0,
                                });
                                let payload_cost = suffix.cost.checked_prepend(root_bytes, 0)?;
                                let mut candidate = ScheduleCandidate {
                                    first_direct_setup_field_len: Some(
                                        first_direct_setup_field_len,
                                    ),
                                    cost: payload_cost,
                                    setup_field_elements: level_setup_field_elements(&root_params)?
                                        .max(suffix.setup_field_elements),
                                    folds,
                                    terminal: Arc::clone(&suffix.terminal),
                                };
                                let nonce_bits =
                                akita_schedules::planner_support::candidate_grinding_nonce_bits(
                                    policy,
                                    &schedule_key.opening_layout()?,
                                    &candidate.folds.to_vec(),
                                    candidate.terminal.as_ref(),
                                )?;
                                candidate.cost = candidate.cost.with_nonce_bits(nonce_bits)?;
                                complete.push(candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    let supported = complete
        .iter()
        .filter(|candidate| policy.admits_setup_field_elements(candidate.setup_field_elements));
    let Some(selected) = select_complete_candidate(policy, supported, None)?.cloned() else {
        return Err(AkitaError::UnsupportedSchedule(
            "unpruned traversal found no complete schedule".into(),
        ));
    };
    let cached_first_direct_setup_field_len =
        if policy.selection_policy == crate::SelectionPolicyId::MinFirstDirectSetupThenPayload {
            selected.first_direct_setup_field_len.map(NonZeroUsize::get)
        } else {
            None
        };
    materialize_candidate_schedule(
        selected.cost.proof_bytes(),
        selected.setup_field_elements,
        cached_first_direct_setup_field_len,
        policy,
        &schedule_key.opening_layout()?,
        selected.folds.to_vec(),
        selected.terminal.as_ref().clone(),
    )
}
