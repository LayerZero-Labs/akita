use std::collections::HashMap;

use super::*;

type MixedMemo = HashMap<(usize, usize, u32, usize, usize, usize), Vec<ScheduleCandidate>>;

fn score(candidate: &ScheduleCandidate) -> MixedScore {
    MixedScore {
        setup_field_elements: candidate.setup_field_elements,
        proof_bytes: candidate.total_bytes,
    }
}

fn dominates_score(left: MixedScore, right: MixedScore) -> bool {
    left.setup_field_elements <= right.setup_field_elements
        // A setup-only improvement is not safe to prune: a parent can mask
        // both setup footprints with the same envelope, after which the
        // descriptor tie-break must still see both candidates.
        && left.proof_bytes < right.proof_bytes
}

fn dominates(left: &ScheduleCandidate, right: &ScheduleCandidate) -> bool {
    dominates_score(score(left), score(right))
}

fn insert_frontier(frontier: &mut Vec<ScheduleCandidate>, candidate: ScheduleCandidate) {
    let same_first_step =
        |other: &ScheduleCandidate| other.first_fold_params() == candidate.first_fold_params();
    if frontier
        .iter()
        .any(|other| same_first_step(other) && dominates(other, &candidate))
    {
        return;
    }
    frontier.retain(|other| !same_first_step(other) || !dominates(&candidate, other));
    frontier.push(candidate);
}

fn insert_supported(
    policy: &PlannerPolicy,
    frontier: &mut Vec<ScheduleCandidate>,
    candidate: ScheduleCandidate,
) {
    if candidate.setup_field_elements <= policy.max_setup_envelope_field_elements {
        insert_frontier(frontier, candidate);
    }
}

#[allow(clippy::too_many_arguments)]
fn suffix_frontier(
    policy: &PlannerPolicy,
    dimensions: &RingDimensionSearchDomain,
    ring_challenge_config: &dyn Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: &dyn Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    key: PolynomialGroupLayout,
    level: usize,
    input_witness_len: usize,
    current_log_basis: u32,
    dimension_ceiling: CommitmentRingDims,
    memo: &mut MixedMemo,
) -> Result<Vec<ScheduleCandidate>, AkitaError> {
    if level > MAX_RECURSION_DEPTH {
        return Ok(Vec::new());
    }
    let memo_key = (
        level,
        input_witness_len,
        current_log_basis,
        dimension_ceiling.d_a(),
        dimension_ceiling.d_b(),
        dimension_ceiling.d_d(),
    );
    if let Some(cached) = memo.get(&memo_key) {
        return Ok(cached.clone());
    }

    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = policy.challenge_field_bits()?;
    let requested_fold_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level,
        input_witness_len,
    });
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(level);
    let mut frontier = Vec::new();

    for log_basis in min_log_basis.max(current_log_basis)..=max_log_basis {
        for candidate_dimensions in dimensions.candidates() {
            let suffix_dimensions = CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION);
            if level >= MIXED_SEARCH_FOLD_LEVELS {
                if *candidate_dimensions != suffix_dimensions {
                    continue;
                }
            } else if !componentwise_dimensions_at_most(*candidate_dimensions, dimension_ceiling) {
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
            let candidates = if level < MIXED_SEARCH_FOLD_LEVELS {
                derive_candidate_level_params_all_splits(
                    policy,
                    &ring_challenge,
                    *candidate_dimensions,
                    input_witness_len,
                    log_basis,
                    level,
                    None,
                    requested_fold_shape,
                )?
            } else {
                derive_candidate_level_params(
                    policy,
                    &ring_challenge,
                    *candidate_dimensions,
                    input_witness_len,
                    log_basis,
                    level,
                    None,
                    requested_fold_shape,
                )?
                .into_iter()
                .collect()
            };

            for (params, output_witness_len) in candidates {
                if level >= MIXED_SEARCH_FOLD_LEVELS {
                    if let Some((mut terminal, terminal_bytes)) =
                        suffix_dp::try_terminal_direct_suffix_cost(
                            input_witness_len,
                            &params,
                            field_bits,
                            key,
                            level,
                            None,
                        )?
                    {
                        let direct_bytes = akita_types::proof_size::FOLD_GRIND_NONCE_BYTES
                            .checked_add(eor_bytes)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "mixed terminal proof size overflow".into(),
                                )
                            })?;
                        terminal.estimated_direct_payload_bytes = direct_bytes;
                        insert_supported(
                            policy,
                            &mut frontier,
                            ScheduleCandidate {
                                first_direct_setup_field_len: Some(
                                    akita_types::active_setup_field_len(
                                        &params,
                                        &suffix_opening_layout(input_witness_len, None)?,
                                    )?,
                                ),
                                total_bytes: direct_bytes.checked_add(terminal_bytes).ok_or_else(
                                    || {
                                        AkitaError::InvalidSetup(
                                            "mixed terminal proof size overflow".into(),
                                        )
                                    },
                                )?,
                                setup_field_elements: terminal_setup_field_elements(
                                    &terminal.params,
                                )?,
                                folds: Vec::new(),
                                terminal,
                            },
                        );
                    }
                }

                let child_ceiling = if level + 1 >= MIXED_SEARCH_FOLD_LEVELS {
                    CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION)
                } else {
                    params.role_dims()
                };
                for child in suffix_frontier(
                    policy,
                    dimensions,
                    ring_challenge_config,
                    fold_shape,
                    key,
                    level + 1,
                    output_witness_len,
                    log_basis,
                    child_ceiling,
                    memo,
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
                            akita_types::NextWitnessBindingPolicy::OuterCommitment
                        }),
                    )?
                    .checked_add(eor_bytes)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("mixed fold proof size overflow".into())
                    })?;
                    let mut folds = Vec::with_capacity(child.folds.len() + 1);
                    folds.push(CandidateFoldStep {
                        params: params.clone(),
                        input_witness_len,
                        output_witness_len,
                        estimated_direct_payload_bytes: direct_bytes,
                        estimated_stage3_payload_bytes: 0,
                    });
                    folds.extend(child.folds.iter().cloned());
                    insert_supported(
                        policy,
                        &mut frontier,
                        ScheduleCandidate {
                            first_direct_setup_field_len: Some(
                                akita_types::active_setup_field_len(
                                    &params,
                                    &suffix_opening_layout(input_witness_len, None)?,
                                )?,
                            ),
                            total_bytes: direct_bytes.checked_add(child.total_bytes).ok_or_else(
                                || AkitaError::InvalidSetup("mixed proof size overflow".into()),
                            )?,
                            setup_field_elements: level_setup_field_elements(&params)?
                                .max(child.setup_field_elements),
                            folds,
                            terminal: child.terminal,
                        },
                    );
                }
            }
        }
    }
    memo.insert(memo_key, frontier.clone());
    Ok(frontier)
}

pub(super) fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    dimensions: &RingDimensionSearchDomain,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let input_witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed root witness too large".into()))?;
    let root_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len,
    });
    let root_num_chunks = policy.chunks_at_level(0);
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    let mut memo = MixedMemo::new();
    let mut complete = Vec::new();

    for log_basis in min_log_basis..=max_log_basis {
        for root_dimensions in dimensions.candidates() {
            let alpha = root_dimensions.d_a().trailing_zeros() as usize;
            let reduced_vars = key.num_vars().saturating_sub(alpha);
            if reduced_vars == 0 {
                continue;
            }
            let min_block_bits = if reduced_vars >= 3 { 1 } else { 0 };
            let max_block_bits = (reduced_vars - 1).min(usize::BITS as usize - 1);
            let Ok(ring_challenge) = ring_challenge_config(root_dimensions.d_a()) else {
                continue;
            };
            for block_bits in (min_block_bits..=max_block_bits).rev() {
                let Some(root_params) = scalar_root_fold_level_params_candidate(
                    policy,
                    &ring_challenge,
                    *root_dimensions,
                    key.num_vars(),
                    key.num_polynomials(),
                    log_basis,
                    block_bits,
                    root_shape,
                )?
                else {
                    continue;
                };
                let output_witness_len = intermediate_w_ring_element_count_for_chunks(
                    field_bits,
                    &root_params,
                    key.num_polynomials(),
                    root_num_chunks,
                )?
                .checked_mul(root_dimensions.d_a())
                .ok_or_else(|| AkitaError::InvalidSetup("mixed root witness overflow".into()))?;
                if output_witness_len
                    .checked_mul(log_basis as usize)
                    .ok_or_else(|| AkitaError::InvalidSetup("mixed bit length overflow".into()))?
                    >= input_witness_len
                        .checked_mul(field_bits as usize)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("mixed input bit length overflow".into())
                        })?
                {
                    continue;
                }
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
                for suffix in suffix_frontier(
                    policy,
                    dimensions,
                    &ring_challenge_config,
                    &fold_shape,
                    key,
                    1,
                    output_witness_len,
                    log_basis,
                    *root_dimensions,
                    &mut memo,
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
                            akita_types::NextWitnessBindingPolicy::OuterCommitment
                        }),
                    )?
                    .checked_add(eor_bytes)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("mixed root proof size overflow".into())
                    })?;
                    let mut folds = Vec::with_capacity(suffix.folds.len() + 1);
                    folds.push(CandidateFoldStep {
                        params: root_params.clone(),
                        input_witness_len,
                        output_witness_len,
                        estimated_direct_payload_bytes: root_bytes,
                        estimated_stage3_payload_bytes: 0,
                    });
                    folds.extend(suffix.folds.iter().cloned());
                    let candidate = ScheduleCandidate {
                        first_direct_setup_field_len: None,
                        total_bytes: root_bytes.checked_add(suffix.total_bytes).ok_or_else(
                            || AkitaError::InvalidSetup("mixed proof size overflow".into()),
                        )?,
                        setup_field_elements: level_setup_field_elements(&root_params)?
                            .max(suffix.setup_field_elements),
                        folds,
                        terminal: suffix.terminal,
                    };
                    if candidate.setup_field_elements <= policy.max_setup_envelope_field_elements {
                        complete.push(candidate);
                    }
                }
            }
        }
    }

    let mut scored = complete
        .into_iter()
        .map(|candidate| {
            let descriptor =
                candidate_schedule_descriptor_bytes(&candidate, policy.ring_dimension)?;
            Ok((
                MixedScore {
                    setup_field_elements: candidate.setup_field_elements,
                    proof_bytes: candidate.total_bytes,
                },
                descriptor,
                candidate,
            ))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    scored.sort_by(|left, right| (&left.0, &left.1).cmp(&(&right.0, &right.1)));
    let Some((_, _, selected)) = scored.into_iter().next() else {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no mixed-D schedule with at least two folds for num_vars={}, num_polynomials={}",
            key.num_vars(),
            key.num_polynomials()
        )));
    };
    materialize_candidate_schedule(
        selected.total_bytes,
        selected.setup_field_elements,
        policy.ring_dimension,
        None,
        selected.folds,
        selected.terminal,
    )
}

#[cfg(test)]
mod tests {
    use super::{dominates_score, MixedScore};

    #[test]
    fn frontier_keeps_lower_payload_child_until_parent_masks_setup() {
        let lower_setup = MixedScore {
            setup_field_elements: 10,
            proof_bytes: 20,
        };
        let lower_payload = MixedScore {
            setup_field_elements: 15,
            proof_bytes: 10,
        };
        assert!(!dominates_score(lower_setup, lower_payload));
        assert!(!dominates_score(lower_payload, lower_setup));

        let parent_setup = 20;
        let lower_setup_complete = MixedScore {
            setup_field_elements: parent_setup.max(lower_setup.setup_field_elements),
            proof_bytes: lower_setup.proof_bytes,
        };
        let lower_payload_complete = MixedScore {
            setup_field_elements: parent_setup.max(lower_payload.setup_field_elements),
            proof_bytes: lower_payload.proof_bytes,
        };
        assert!(lower_payload_complete < lower_setup_complete);
    }

    #[test]
    fn frontier_keeps_equal_payload_alternatives_for_descriptor_ties() {
        let lower_setup = MixedScore {
            setup_field_elements: 10,
            proof_bytes: 20,
        };
        let higher_setup = MixedScore {
            setup_field_elements: 15,
            proof_bytes: 20,
        };

        assert!(!dominates_score(lower_setup, higher_setup));
        assert!(!dominates_score(higher_setup, lower_setup));

        let parent_setup = 20;
        assert_eq!(
            parent_setup.max(lower_setup.setup_field_elements),
            parent_setup.max(higher_setup.setup_field_elements)
        );
    }
}
