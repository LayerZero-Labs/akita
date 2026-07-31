use std::collections::HashMap;

use super::*;

type MixedMemo = HashMap<(usize, usize, u32, usize, usize, usize), Vec<ScheduleCandidate>>;

fn score(candidate: &ScheduleCandidate) -> Result<MixedScore, AkitaError> {
    let matrix_field_elements = candidate.folds.iter().try_fold(
        terminal_matrix_field_elements(&candidate.terminal.params)?,
        |total, fold| {
            total
                .checked_add(level_matrix_field_elements(&fold.params)?)
                .ok_or_else(|| AkitaError::InvalidSetup("schedule matrix work overflow".into()))
        },
    )?;
    Ok(MixedScore {
        setup_field_elements: candidate.setup_field_elements,
        matrix_field_elements,
        proof_bytes: candidate.total_bytes,
    })
}

fn dominates_score(left: MixedScore, right: MixedScore) -> bool {
    left.setup_field_elements <= right.setup_field_elements
        // A setup-only improvement is not safe to prune: a parent can mask
        // both setup footprints with the same envelope, after which the
        // descriptor tie-break must still see both candidates.
        && left.proof_bytes < right.proof_bytes
}

fn role_dimension(params: &CommittedGroupParams, role: akita_types::SisMatrixRole) -> usize {
    match role {
        akita_types::SisMatrixRole::Outer => params.role_dims().d_b(),
        akita_types::SisMatrixRole::Open => params.role_dims().d_d(),
        akita_types::SisMatrixRole::Inner => params.role_dims().d_a(),
    }
}

fn role_width(params: &CommittedGroupParams, role: akita_types::SisMatrixRole) -> usize {
    match role {
        akita_types::SisMatrixRole::Outer => params.outer_width(),
        akita_types::SisMatrixRole::Open => params.d_matrix_width(),
        akita_types::SisMatrixRole::Inner => params.inner_width(),
    }
}

fn smaller_admitted_dimension_has_rank_one(
    policy: &PlannerPolicy,
    dimensions: &RingDimensionSearchDomain,
    params: &CommittedGroupParams,
    role: akita_types::SisMatrixRole,
) -> bool {
    let carrier = params.role_dims().d_a();
    let selected = role_dimension(params, role);
    let Some(native_width) = role_width(params, role)
        .checked_mul(selected)
        .and_then(|width| width.checked_div(carrier))
    else {
        return false;
    };
    dimensions.candidates().iter().any(|candidate| {
        if candidate.d_a() != carrier {
            return false;
        }
        let candidate_dimension = match role {
            akita_types::SisMatrixRole::Outer => candidate.d_b(),
            akita_types::SisMatrixRole::Open => {
                if candidate.d_b() != params.role_dims().d_b() {
                    return false;
                }
                candidate.d_d()
            }
            akita_types::SisMatrixRole::Inner => candidate.d_a(),
        };
        candidate_dimension < selected
            && candidate::projected_collision_role_price(
                policy,
                role,
                carrier,
                candidate_dimension,
                native_width,
                params.log_basis_open,
            )
            .and_then(|(key, width)| {
                u64::try_from(width)
                    .ok()
                    .and_then(|width| akita_types::sis::min_secure_rank(key, width))
            }) == Some(1)
    })
}

fn mixed_candidate_is_admitted(
    policy: &PlannerPolicy,
    dimensions: &RingDimensionSearchDomain,
    params: &CommittedGroupParams,
) -> bool {
    let mixed_policy = dimensions.mixed_policy();
    let inner_total_dimension = params
        .inner_commit_matrix
        .output_rank()
        .checked_mul(params.role_dims().d_a());
    if mixed_policy
        .max_inner_total_dimension
        .is_some_and(|cap| inner_total_dimension.is_none_or(|dimension| dimension > cap))
    {
        return false;
    }
    if mixed_policy.stop_outer_at_rank_one
        && smaller_admitted_dimension_has_rank_one(
            policy,
            dimensions,
            params,
            akita_types::SisMatrixRole::Outer,
        )
    {
        return false;
    }
    if mixed_policy.stop_opening_at_rank_one
        && smaller_admitted_dimension_has_rank_one(
            policy,
            dimensions,
            params,
            akita_types::SisMatrixRole::Open,
        )
    {
        return false;
    }
    true
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn lcm(left: u128, right: u128) -> Option<u128> {
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

fn balanced_scale(scores: &[MixedScore]) -> Result<(MixedScore, u128), AkitaError> {
    let minima = MixedScore {
        setup_field_elements: scores
            .iter()
            .map(|score| score.setup_field_elements)
            .min()
            .unwrap_or(1),
        matrix_field_elements: scores
            .iter()
            .map(|score| score.matrix_field_elements)
            .min()
            .unwrap_or(1),
        proof_bytes: scores
            .iter()
            .map(|score| score.proof_bytes)
            .min()
            .unwrap_or(1),
    };
    let scale = [
        minima.setup_field_elements,
        minima.matrix_field_elements,
        minima.proof_bytes,
    ]
    .into_iter()
    .try_fold(1u128, |scale, minimum| lcm(scale, minimum as u128))
    .ok_or_else(|| AkitaError::InvalidSetup("balanced score scale overflow".into()))?;
    Ok((minima, scale))
}

fn balanced_key(
    score: MixedScore,
    minima: MixedScore,
    scale: u128,
) -> Result<(u128, u128), AkitaError> {
    let setup = (score.setup_field_elements as u128)
        .checked_mul(scale / minima.setup_field_elements as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("balanced setup score overflow".into()))?;
    let matrix = (score.matrix_field_elements as u128)
        .checked_mul(scale / minima.matrix_field_elements as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("balanced matrix score overflow".into()))?;
    let proof = (score.proof_bytes as u128)
        .checked_mul(scale / minima.proof_bytes as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("balanced proof score overflow".into()))?;
    let aggregate = setup
        .checked_add(matrix)
        .and_then(|value| value.checked_add(proof))
        .ok_or_else(|| AkitaError::InvalidSetup("balanced aggregate score overflow".into()))?;
    Ok((setup.max(matrix).max(proof), aggregate))
}

fn dominates(
    left: &ScheduleCandidate,
    right: &ScheduleCandidate,
    objective: MixedScheduleObjective,
) -> Result<bool, AkitaError> {
    let left = score(left)?;
    let right = score(right)?;
    Ok(match objective {
        MixedScheduleObjective::MinimumSetupThenProof => dominates_score(left, right),
        MixedScheduleObjective::Balanced => {
            left.setup_field_elements <= right.setup_field_elements
                && left.matrix_field_elements <= right.matrix_field_elements
                && left.proof_bytes <= right.proof_bytes
                && (left.matrix_field_elements < right.matrix_field_elements
                    || left.proof_bytes < right.proof_bytes)
        }
    })
}

fn insert_frontier(
    frontier: &mut Vec<ScheduleCandidate>,
    candidate: ScheduleCandidate,
    objective: MixedScheduleObjective,
) -> Result<(), AkitaError> {
    let same_first_step =
        |other: &ScheduleCandidate| other.first_fold_params() == candidate.first_fold_params();
    for other in frontier.iter() {
        if same_first_step(other) && dominates(other, &candidate, objective)? {
            return Ok(());
        }
    }
    let mut retained = Vec::with_capacity(frontier.len() + 1);
    for other in frontier.drain(..) {
        if !same_first_step(&other) || !dominates(&candidate, &other, objective)? {
            retained.push(other);
        }
    }
    *frontier = retained;
    frontier.push(candidate);
    Ok(())
}

fn insert_supported(
    policy: &PlannerPolicy,
    objective: MixedScheduleObjective,
    frontier: &mut Vec<ScheduleCandidate>,
    candidate: ScheduleCandidate,
) -> Result<(), AkitaError> {
    if candidate.setup_field_elements <= policy.max_setup_envelope_field_elements {
        insert_frontier(frontier, candidate, objective)?;
    }
    Ok(())
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
    let requested_fold_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level,
        input_witness_len,
    });
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(level);
    let eor_bytes = extension_opening_reduction_level_bytes(
        field_bits * policy.chal_ext_degree as u32,
        policy.claim_ext_degree,
        level,
        key,
        input_witness_len,
        dimension_ceiling.d_a(),
    )?;
    let mut frontier = Vec::new();

    for log_basis in min_log_basis.max(current_log_basis)..=max_log_basis {
        for candidate_dimensions in dimensions.candidates() {
            let mixed_policy = dimensions.mixed_policy();
            let suffix_dimensions = mixed_policy.suffix_dimensions;
            if level >= mixed_policy.mixed_fold_levels {
                if *candidate_dimensions != suffix_dimensions {
                    continue;
                }
            } else if !componentwise_dimensions_at_most(*candidate_dimensions, dimension_ceiling) {
                continue;
            }
            let Ok(ring_challenge) = ring_challenge_config(candidate_dimensions.d_a()) else {
                continue;
            };
            let candidates = if level < mixed_policy.mixed_fold_levels {
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
                if !mixed_candidate_is_admitted(policy, dimensions, &params) {
                    continue;
                }
                if level >= mixed_policy.mixed_fold_levels {
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
                            mixed_policy.objective,
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
                        )?;
                    }
                }

                let child_ceiling = if level + 1 >= mixed_policy.mixed_fold_levels {
                    mixed_policy.suffix_dimensions
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
                        field_bits * policy.chal_ext_degree as u32,
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
                        mixed_policy.objective,
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
                    )?;
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
                if !mixed_candidate_is_admitted(policy, dimensions, &root_params) {
                    continue;
                }
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
                let eor_bytes = extension_opening_reduction_level_bytes(
                    field_bits * policy.chal_ext_degree as u32,
                    policy.claim_ext_degree,
                    0,
                    key,
                    input_witness_len,
                    root_dimensions.d_a(),
                )?;
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
                        field_bits * policy.chal_ext_degree as u32,
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
            Ok((score(&candidate)?, descriptor, candidate))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    match dimensions.mixed_policy().objective {
        MixedScheduleObjective::MinimumSetupThenProof => scored.sort_by(|left, right| {
            (left.0.setup_field_elements, left.0.proof_bytes, &left.1).cmp(&(
                right.0.setup_field_elements,
                right.0.proof_bytes,
                &right.1,
            ))
        }),
        MixedScheduleObjective::Balanced => {
            let scores = scored.iter().map(|entry| entry.0).collect::<Vec<_>>();
            let (minima, scale) = balanced_scale(&scores)?;
            let mut balanced = scored
                .into_iter()
                .map(|(score, descriptor, candidate)| {
                    Ok((
                        balanced_key(score, minima, scale)?,
                        score,
                        descriptor,
                        candidate,
                    ))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            balanced.sort_by(|left, right| {
                (
                    left.0,
                    left.1.setup_field_elements,
                    left.1.proof_bytes,
                    left.1.matrix_field_elements,
                    &left.2,
                )
                    .cmp(&(
                        right.0,
                        right.1.setup_field_elements,
                        right.1.proof_bytes,
                        right.1.matrix_field_elements,
                        &right.2,
                    ))
            });
            scored = balanced
                .into_iter()
                .map(|(_, score, descriptor, candidate)| (score, descriptor, candidate))
                .collect();
        }
    }
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
            matrix_field_elements: 10,
            proof_bytes: 20,
        };
        let lower_payload = MixedScore {
            setup_field_elements: 15,
            matrix_field_elements: 10,
            proof_bytes: 10,
        };
        assert!(!dominates_score(lower_setup, lower_payload));
        assert!(!dominates_score(lower_payload, lower_setup));

        let parent_setup = 20;
        let lower_setup_complete = MixedScore {
            setup_field_elements: parent_setup.max(lower_setup.setup_field_elements),
            matrix_field_elements: lower_setup.matrix_field_elements,
            proof_bytes: lower_setup.proof_bytes,
        };
        let lower_payload_complete = MixedScore {
            setup_field_elements: parent_setup.max(lower_payload.setup_field_elements),
            matrix_field_elements: lower_payload.matrix_field_elements,
            proof_bytes: lower_payload.proof_bytes,
        };
        assert!(lower_payload_complete < lower_setup_complete);
    }

    #[test]
    fn frontier_keeps_equal_payload_alternatives_for_descriptor_ties() {
        let lower_setup = MixedScore {
            setup_field_elements: 10,
            matrix_field_elements: 10,
            proof_bytes: 20,
        };
        let higher_setup = MixedScore {
            setup_field_elements: 15,
            matrix_field_elements: 10,
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
