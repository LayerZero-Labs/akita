use super::*;
use std::sync::Arc;

struct UnprunedCtx<'a> {
    policy: &'a PlannerPolicy,
    dimensions: &'a RingDimensionSearchDomain,
    ring_challenge_config: &'a dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    key: PolynomialGroupLayout,
}

#[derive(Clone, Copy)]
struct UnprunedState {
    level: usize,
    input_witness_len: usize,
    current_log_basis: u32,
    dimension_ceiling: CommitmentRingDims,
    payload_phase: akita_types::CommitmentPayloadPhase,
}

fn enumerate_suffixes(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
) -> Result<Vec<ScheduleCandidate>, AkitaError> {
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
        dimension_ceiling,
        payload_phase,
    } = state;
    if level > MAX_RECURSION_DEPTH {
        return Ok(Vec::new());
    }
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = policy.challenge_field_bits()?;
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(level);
    let (min_inner_basis, max_inner_basis) =
        policy.inner_basis_search_range(crate::InnerBasisSource::BalancedDigits {
            log_basis: current_log_basis,
        })?;
    let mut schedules = Vec::new();

    for log_basis in min_log_basis.max(current_log_basis)..=max_log_basis {
        for inner_basis in min_inner_basis..=max_inner_basis {
            for candidate_dimensions in dimensions.candidates() {
                let suffix_dimensions =
                    CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION);
                if level >= MIXED_SEARCH_FOLD_LEVELS {
                    if *candidate_dimensions != suffix_dimensions {
                        continue;
                    }
                } else if !componentwise_dimensions_at_most(
                    *candidate_dimensions,
                    dimension_ceiling,
                ) {
                    continue;
                }
                let Some(eor_bytes) = try_extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    policy.claim_ext_degree,
                    level,
                    key,
                    input_witness_len,
                    candidate_dimensions.d_a(),
                )?
                else {
                    continue;
                };
                let Ok(ring_challenge) = ring_challenge_config(candidate_dimensions.d_a()) else {
                    continue;
                };
                for &payload_mode in payload_phase.candidate_modes(level, false) {
                    let candidates = if level < MIXED_SEARCH_FOLD_LEVELS {
                        derive_candidate_level_params_all_splits(
                            None,
                            policy,
                            payload_mode,
                            &ring_challenge,
                            *candidate_dimensions,
                            input_witness_len,
                            crate::InnerBasisSource::BalancedDigits {
                                log_basis: current_log_basis,
                            },
                            inner_basis,
                            log_basis,
                            level,
                            None,
                        )?
                    } else {
                        derive_candidate_level_params(
                            None,
                            policy,
                            payload_mode,
                            &ring_challenge,
                            *candidate_dimensions,
                            input_witness_len,
                            crate::InnerBasisSource::BalancedDigits {
                                log_basis: current_log_basis,
                            },
                            inner_basis,
                            log_basis,
                            level,
                            None,
                        )?
                        .into_iter()
                        .collect()
                    };

                    for (params, output_witness_len) in candidates {
                        let params = Arc::new(params);
                        let terminal_candidate = if dimensions.candidates().len() == 1
                            || level >= MIXED_SEARCH_FOLD_LEVELS
                        {
                            suffix_dp::try_terminal_direct_suffix_cost(
                                input_witness_len,
                                &params,
                                field_bits,
                                key,
                                level,
                                None,
                            )?
                        } else {
                            None
                        };
                        if let Some((mut terminal, terminal_bytes)) = terminal_candidate {
                            let direct_bytes = akita_types::proof_size::FOLD_GRIND_NONCE_BYTES
                                .checked_add(eor_bytes)
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "unpruned traversal terminal proof size overflow".into(),
                                    )
                                })?;
                            terminal.estimated_direct_payload_bytes = direct_bytes;
                            schedules.push(ScheduleCandidate {
                                first_direct_setup_field_len: Some(
                                    akita_types::active_setup_field_len(
                                        &params,
                                        &suffix_opening_layout(input_witness_len, None)?,
                                    )?,
                                ),
                                total_bytes: direct_bytes.checked_add(terminal_bytes).ok_or_else(
                                    || {
                                        AkitaError::InvalidSetup(
                                            "unpruned traversal terminal proof size overflow"
                                                .into(),
                                        )
                                    },
                                )?,
                                setup_field_elements: terminal_setup_field_elements(
                                    &terminal.params,
                                )?,
                                folds: CandidateFoldChain::default(),
                                terminal: Arc::new(terminal),
                            });
                        }

                        let child_ceiling = if level + 1 >= MIXED_SEARCH_FOLD_LEVELS {
                            CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION)
                        } else {
                            params.role_dims()
                        };
                        for child in enumerate_suffixes(
                            ctx,
                            UnprunedState {
                                level: level + 1,
                                input_witness_len: output_witness_len,
                                current_log_basis: log_basis,
                                dimension_ceiling: child_ceiling,
                                payload_phase: payload_phase.after(params.payload_mode),
                            },
                        )? {
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
                            .checked_add(eor_bytes)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "unpruned traversal fold proof size overflow".into(),
                                )
                            })?;
                            let folds = child.folds.prepend(CandidateFoldStep {
                                params: Arc::clone(&params),
                                input_witness_len,
                                output_witness_len,
                                estimated_direct_payload_bytes: direct_bytes,
                                estimated_stage3_payload_bytes: 0,
                            });
                            schedules.push(ScheduleCandidate {
                                first_direct_setup_field_len: Some(
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
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    let inner_source =
        root_inner_basis_source(honest_fold_policy, policy.decomposition.log_commit_bound);
    let (min_inner_basis, max_inner_basis) = policy.inner_basis_search_range(inner_source)?;
    let mut complete = Vec::new();
    let schedule_key = akita_types::AkitaScheduleLookupKey::single(key);
    let ctx = UnprunedCtx {
        policy,
        dimensions,
        ring_challenge_config: &ring_challenge_config,
        key,
    };

    for log_basis in min_log_basis..=max_log_basis {
        for inner_basis in min_inner_basis..=max_inner_basis {
            for root_dimensions in dimensions.candidates() {
                let alpha = root_dimensions.d_a().trailing_zeros() as usize;
                let reduced_vars = key.num_vars().saturating_sub(alpha);
                if reduced_vars == 0 {
                    continue;
                }
                let Ok(ring_challenge) = ring_challenge_config(root_dimensions.d_a()) else {
                    continue;
                };
                for (root_params, output_witness_len) in
                    crate::planner::root_level_candidates_for_basis(
                        &schedule_key,
                        honest_fold_policy,
                        &[],
                        policy,
                        *root_dimensions,
                        &ring_challenge,
                        &ring_challenge_config,
                        input_witness_len,
                        inner_basis,
                        log_basis,
                        true,
                    )?
                {
                    let root_params = Arc::new(root_params);
                    let Some(eor_bytes) = try_extension_opening_reduction_level_bytes(
                        policy.challenge_field_bits()?,
                        policy.claim_ext_degree,
                        0,
                        key,
                        input_witness_len,
                        root_dimensions.d_a(),
                    )?
                    else {
                        continue;
                    };
                    for suffix in enumerate_suffixes(
                        &ctx,
                        UnprunedState {
                            level: 1,
                            input_witness_len: output_witness_len,
                            current_log_basis: log_basis,
                            dimension_ceiling: *root_dimensions,
                            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
                        },
                    )? {
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
                        .checked_add(eor_bytes)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "unpruned traversal root proof size overflow".into(),
                            )
                        })?;
                        let folds = suffix.folds.prepend(CandidateFoldStep {
                            params: Arc::clone(&root_params),
                            input_witness_len,
                            output_witness_len,
                            estimated_direct_payload_bytes: root_bytes,
                            estimated_stage3_payload_bytes: 0,
                        });
                        complete.push(ScheduleCandidate {
                            first_direct_setup_field_len: None,
                            total_bytes: root_bytes.checked_add(suffix.total_bytes).ok_or_else(
                                || {
                                    AkitaError::InvalidSetup(
                                        "unpruned traversal proof size overflow".into(),
                                    )
                                },
                            )?,
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

    let supported = complete
        .iter()
        .filter(|candidate| policy.admits_setup_field_elements(candidate.setup_field_elements));
    let Some(selected) = select_complete_candidate(policy, supported)?.cloned() else {
        return Err(AkitaError::UnsupportedSchedule(
            "unpruned traversal found no complete schedule".into(),
        ));
    };
    selected.materialize()
}
