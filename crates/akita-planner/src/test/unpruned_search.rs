use super::*;
use akita_schedules::planner_support::MAX_RECURSION_DEPTH;

struct UnprunedCtx<'a> {
    policy: &'a PlannerPolicy,
    dimensions: &'a RingDimensionSearchDomain,
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
}

type UnprunedMemo = Vec<(UnprunedState, Arc<Vec<ScheduleCandidate>>)>;

fn packing_opening_domain(
    level: usize,
    extension_degree: usize,
    dimensions: CommitmentRingDims,
) -> Vec<crate::schedule_params::PlannerOpeningCandidate> {
    akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS
        .iter()
        .copied()
        .filter_map(|challenge_subring_dimension| {
            crate::schedule_params::PlannerOpeningCandidate::coefficient_packing(
                level,
                extension_degree,
                dimensions,
                challenge_subring_dimension,
            )
            .ok()
        })
        .collect()
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
        dimensions,
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
    } = state;
    if level > MAX_RECURSION_DEPTH {
        return Ok(Arc::new(Vec::new()));
    }
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = policy.challenge_field_bits()?;
    let (min_log_basis, max_log_basis) =
        crate::policy::log_basis_search_range_at_level(policy, level);
    let mut schedules = Vec::new();
    let early_packing_available = level <= 1
        && dimensions.candidates().iter().any(|candidate_dimensions| {
            let suffix_dimensions = CommitmentRingDims::uniform(ADAPTIVE_SUFFIX_RING_DIMENSION);
            let dimension_is_admissible = if level >= akita_schedules::ADAPTIVE_SEARCH_LEVELS {
                *candidate_dimensions == suffix_dimensions
            } else {
                componentwise_dimensions_at_most(*candidate_dimensions, dimension_ceiling)
            };
            dimension_is_admissible
                && !packing_opening_domain(level, policy.claim_ext_degree, *candidate_dimensions)
                    .is_empty()
        });

    for log_basis in min_log_basis.max(current_log_basis)..=max_log_basis {
        for candidate_dimensions in dimensions.candidates() {
            let suffix_dimensions = CommitmentRingDims::uniform(ADAPTIVE_SUFFIX_RING_DIMENSION);
            if level >= akita_schedules::ADAPTIVE_SEARCH_LEVELS {
                if *candidate_dimensions != suffix_dimensions {
                    continue;
                }
            } else if !componentwise_dimensions_at_most(*candidate_dimensions, dimension_ceiling) {
                continue;
            }
            let trace_work = ring_challenge_config(candidate_dimensions.d_a())
                .ok()
                .and_then(|ring_challenge| {
                    try_extension_opening_reduction_level_bytes(
                        challenge_field_bits,
                        policy.claim_ext_degree,
                        level,
                        key,
                        input_witness_len,
                        candidate_dimensions.d_a(),
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
            let derive_candidates =
                |opening, payload_mode| -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
                    if level < akita_schedules::ADAPTIVE_SEARCH_LEVELS {
                        derive_candidate_level_params_split_frontier_without_bounds(
                            None,
                            policy,
                            payload_mode,
                            opening,
                            *candidate_dimensions,
                            input_witness_len,
                            crate::InnerBasisSource::BalancedDigits {
                                log_basis: current_log_basis,
                            },
                            current_log_basis,
                            log_basis,
                            level,
                            None,
                            source_moment,
                        )
                    } else {
                        Ok(derive_linf_candidate_level_params(
                            None,
                            policy,
                            payload_mode,
                            opening,
                            *candidate_dimensions,
                            input_witness_len,
                            crate::InnerBasisSource::BalancedDigits {
                                log_basis: current_log_basis,
                            },
                            current_log_basis,
                            log_basis,
                            level,
                            None,
                        )?
                        .into_iter()
                        .collect())
                    }
                };

            if dimensions.candidates().len() == 1
                || level >= akita_schedules::ADAPTIVE_SEARCH_LEVELS
            {
                if let Some((trace_opening, terminal_eor_bytes)) = trace_work {
                    for &payload_mode in payload_phase.candidate_modes(level, false) {
                        for params in derive_terminal_candidate_params(
                            policy,
                            payload_mode,
                            trace_opening,
                            *candidate_dimensions,
                            input_witness_len,
                            crate::InnerBasisSource::BalancedDigits {
                                log_basis: current_log_basis,
                            },
                            current_log_basis,
                            log_basis,
                            level,
                            source_moment,
                        )? {
                            if let Some((mut terminal, terminal_bytes)) =
                                suffix_dp::try_terminal_direct_suffix_cost(
                                    policy,
                                    input_witness_len,
                                    &params,
                                    field_bits,
                                    key,
                                    level,
                                    None,
                                    source_moment,
                                )?
                            {
                                let direct_bytes = akita_types::proof_size::FOLD_GRIND_NONCE_BYTES
                                    .checked_add(terminal_eor_bytes)
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "unpruned traversal terminal proof size overflow"
                                                .into(),
                                        )
                                    })?;
                                terminal.estimated_direct_payload_bytes = direct_bytes;
                                schedules.push(ScheduleCandidate {
                                    first_direct_setup_field_len: std::num::NonZeroUsize::new(
                                        akita_types::active_setup_field_len(
                                            &params,
                                            &suffix_opening_layout(input_witness_len, None)?,
                                        )?,
                                    ),
                                    total_bytes: direct_bytes
                                        .checked_add(terminal_bytes)
                                        .ok_or_else(|| {
                                            AkitaError::InvalidSetup(
                                                "unpruned traversal terminal proof size overflow"
                                                    .into(),
                                            )
                                        })?,
                                    setup_field_elements: terminal_setup_field_elements(
                                        &terminal.params,
                                    )?,
                                    folds: CandidateFoldChain::default(),
                                    terminal: Arc::new(terminal),
                                });
                            }
                        }
                    }
                }
            }

            let fold_work = if level <= 1 {
                let packing =
                    packing_opening_domain(level, policy.claim_ext_degree, *candidate_dimensions);
                if !early_packing_available {
                    trace_work.into_iter().collect()
                } else {
                    packing
                        .into_iter()
                        .map(|opening| (opening, 0))
                        .collect::<Vec<_>>()
                }
            } else {
                trace_work.into_iter().collect()
            };
            for (opening, opening_reduction_bytes) in fold_work {
                for &payload_mode in payload_phase.candidate_modes(level, false) {
                    for (params, output_witness_len) in derive_candidates(opening, payload_mode)? {
                        let child_ceiling = if level + 1 >= akita_schedules::ADAPTIVE_SEARCH_LEVELS
                        {
                            CommitmentRingDims::uniform(ADAPTIVE_SUFFIX_RING_DIMENSION)
                        } else {
                            params.role_dims()
                        };
                        let next_source_moment = if policy.selective_l2_response_model_enabled() {
                            let opening_layout = suffix_opening_layout(input_witness_len, None)?;
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
                            },
                            memo,
                        )?
                        .iter()
                        {
                            let child_is_terminal = child.folds.is_empty();
                            let direct_bytes = level_proof_bytes(
                                field_bits,
                                challenge_field_bits,
                                &params,
                                child.first_fold_params(),
                                output_witness_len,
                                Some(if child_is_terminal {
                                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                                } else {
                                    akita_types::NextWitnessBindingPolicy::OuterPayload
                                }),
                            )?
                            .checked_add(opening_reduction_bytes)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "unpruned traversal fold proof size overflow".into(),
                                )
                            })?;
                            let folds = child.folds.prepend(CandidateFoldStep {
                                params: Arc::new(params.clone()),
                                input_witness_len,
                                output_witness_len,
                                estimated_direct_payload_bytes: direct_bytes,
                                estimated_stage3_payload_bytes: 0,
                            });
                            schedules.push(ScheduleCandidate {
                                first_direct_setup_field_len: std::num::NonZeroUsize::new(
                                    akita_types::active_setup_field_len(
                                        &params,
                                        &suffix_opening_layout(input_witness_len, None)?,
                                    )?,
                                ),
                                total_bytes: direct_bytes
                                    .checked_add(child.total_bytes)
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "unpruned traversal proof size overflow".into(),
                                        )
                                    })?,
                                setup_field_elements: level_setup_field_elements(&params)?
                                    .max(child.setup_field_elements),
                                folds,
                                terminal: Arc::clone(&child.terminal),
                            });
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
    dimensions: &RingDimensionSearchDomain,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
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
        dimensions,
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
    let root_packing_available = dimensions.candidates().iter().any(|dimensions| {
        !packing_opening_domain(0, policy.claim_ext_degree, *dimensions).is_empty()
    });

    for log_basis in min_log_basis..=max_log_basis {
        for inner_basis in min_inner_basis..=max_inner_basis {
            for root_dimensions in dimensions.candidates() {
                let alpha = root_dimensions.d_a().trailing_zeros() as usize;
                let reduced_vars = key.num_vars().saturating_sub(alpha);
                if reduced_vars == 0 {
                    continue;
                }
                let packing = packing_opening_domain(0, policy.claim_ext_degree, *root_dimensions);
                let root_openings = if !root_packing_available {
                    ring_challenge_config(root_dimensions.d_a())
                        .ok()
                        .map(crate::schedule_params::PlannerOpeningCandidate::evaluation_trace)
                        .into_iter()
                        .collect()
                } else {
                    packing
                };
                for root_opening in root_openings {
                    for (root_params, output_witness_len) in
                        crate::planner::root_level_candidates_for_basis(
                            &schedule_key,
                            honest_fold_policy,
                            &[],
                            policy,
                            *root_dimensions,
                            root_opening,
                            &[],
                            input_witness_len,
                            inner_basis,
                            log_basis,
                            true,
                        )?
                    {
                        let next_source_moment = if policy.selective_l2_response_model_enabled() {
                            let opening_layout = schedule_key.opening_layout()?;
                            let source_groups = crate::response_model::root_group_source_moments(
                                &root_params,
                                &opening_layout,
                                honest_fold_policy,
                                &[],
                                field_bits,
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
                        for suffix in enumerate_suffixes(
                            &ctx,
                            UnprunedState {
                                level: 1,
                                input_witness_len: output_witness_len,
                                current_log_basis: log_basis,
                                source_moment: next_source_moment,
                                dimension_ceiling: *root_dimensions,
                                payload_phase:
                                    akita_types::CommitmentPayloadPhase::CompressedPrefix,
                            },
                            &mut memo,
                        )?
                        .iter()
                        {
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
                            )?
                            .checked_add(if root_opening.is_coefficient_packing() {
                                0
                            } else {
                                try_extension_opening_reduction_level_bytes(
                                    policy.challenge_field_bits()?,
                                    policy.claim_ext_degree,
                                    0,
                                    key,
                                    input_witness_len,
                                    root_dimensions.d_a(),
                                )?
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "unpruned root ET fallback has no EOR geometry".into(),
                                    )
                                })?
                            })
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "unpruned traversal root proof size overflow".into(),
                                )
                            })?;
                            let folds = suffix.folds.prepend(CandidateFoldStep {
                                params: Arc::new(root_params.clone()),
                                input_witness_len,
                                output_witness_len,
                                estimated_direct_payload_bytes: root_bytes,
                                estimated_stage3_payload_bytes: 0,
                            });
                            complete.push(ScheduleCandidate {
                                first_direct_setup_field_len: None,
                                total_bytes: root_bytes
                                    .checked_add(suffix.total_bytes)
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "unpruned traversal proof size overflow".into(),
                                        )
                                    })?,
                                setup_field_elements: level_setup_field_elements(&root_params)?
                                    .max(suffix.setup_field_elements),
                                folds,
                                terminal: Arc::clone(&suffix.terminal),
                            });
                        }
                    }
                }
            }
        }
    }

    let supported = complete
        .iter()
        .filter(|candidate| policy.admits_setup_field_elements(candidate.setup_field_elements));
    let Some(selected) = select_complete_candidate(policy, supported)?.cloned() else {
        return Err(AkitaError::UnsupportedSchedule(
            "unpruned traversal found no complete schedule".into(),
        ));
    };
    materialize_candidate_schedule(
        selected.total_bytes,
        selected.setup_field_elements,
        None,
        selected.folds.to_vec(),
        selected.terminal.as_ref().clone(),
    )
}
