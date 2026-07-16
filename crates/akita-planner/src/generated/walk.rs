//! Canonical walker for compact generated schedule rows.
//!
//! [`walk_generated_schedule_entry`] is the single implementation shared by
//! runtime materialization ([`crate::schedule_from_entry`]) and admissibility
//! checks ([`super::validate::validate_generated_schedule_entry`]). Both paths
//! expand every fold step once, audit SIS ranks via
//! [`GeneratedFoldStep::expand_to_level_params`], and recompute witness
//! transitions and proof-byte totals.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, Prime128OffsetA7F7};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_level_bytes, level_proof_bytes,
    segment_typed_witness_shape_from_groups, AkitaScheduleInputs, AkitaScheduleLookupKey,
    CleartextWitnessShape, DirectStep, FoldStep, LevelParams, PolynomialGroupLayout,
    PrecommittedLevelParams, RelationMatrixRowLayout, Schedule, SetupContributionMode, Step,
};

use crate::generated::{
    validate_entry_key, GeneratedFoldStep, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::group_batch::multi_group_root_precommitted_groups;
use crate::schedule_params::planned_next_witness_len;
use crate::PlannerPolicy;

pub(crate) struct GeneratedEntryWalkOutput {
    pub total_bytes: usize,
    pub schedule: Schedule,
}

pub(crate) fn walk_generated_schedule_entry(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    key.validate()?;
    validate_entry_key(entry, key)?;
    entry.validate()?;
    reject_scalar_recursive_catalog_row(entry, key)?;

    if key.precommitteds.is_empty() {
        return walk_scalar_generated_schedule_entry(
            entry,
            key.final_group,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }

    walk_multi_group_generated_schedule_entry(
        entry,
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

fn reject_scalar_recursive_catalog_row(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
) -> Result<(), AkitaError> {
    if !key.precommitteds.is_empty() {
        return Ok(());
    }
    for step in entry.steps {
        if let GeneratedStep::FoldWithSetupMetadata(meta) = step {
            if meta.setup_contribution_mode == SetupContributionMode::Recursive {
                return Err(AkitaError::InvalidSetup(
                    "scalar lookup keys (empty precommitteds) do not support recursive setup \
                     contribution; grouped-batch scheduling requires genuine precommits"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn walk_scalar_generated_schedule_entry(
    entry: &GeneratedScheduleTableEntry,
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits
        .checked_mul(policy.chal_ext_degree as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "generated schedule challenge field bit width overflow".to_string(),
            )
        })?;
    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;

    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len: Option<usize> = None;
    let mut last_fold_lp: Option<LevelParams> = None;
    let mut total_bytes = 0usize;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(_) | GeneratedStep::FoldWithSetupMetadata(_) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                let num_claims = if fold_level == 0 {
                    key.num_polynomials()
                } else {
                    1
                };
                let mut lp = expand_validated_fold_level(
                    step,
                    key,
                    policy,
                    ring_challenge_config,
                    fold_challenge_shape_at_level,
                    fold_level,
                    current_w_len,
                    num_claims,
                )?;
                // Stamp the per-level chunk layout (the expander defaults it to
                // single-chunk); the pricing below uses the same count.
                lp.witness_chunk = policy.witness_chunk_for_level(fold_level);
                let num_polynomials = if fold_level == 0 {
                    key.num_polynomials()
                } else {
                    1
                };
                if is_terminal && lp.has_precommitted_groups() {
                    return Err(AkitaError::InvalidSetup(
                        "grouped terminal fold must be followed by another fold".to_string(),
                    ));
                }
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let len = planned_next_witness_len(
                        field_bits,
                        &lp,
                        num_polynomials,
                        RelationMatrixRowLayout::WithoutDBlock,
                        lp.witness_chunk.num_chunks,
                    )?;
                    terminal_witness_field_len = Some(len);
                    (len, None, RelationMatrixRowLayout::WithoutDBlock)
                } else {
                    let len = planned_next_witness_len(
                        field_bits,
                        &lp,
                        num_polynomials,
                        RelationMatrixRowLayout::WithDBlock,
                        lp.witness_chunk.num_chunks,
                    )?;
                    if next.fold_step().is_none() {
                        return Err(AkitaError::InvalidSetup(
                            "generated non-terminal successor must be a fold step".to_string(),
                        ));
                    }
                    let mut next_lp = expand_validated_fold_level(
                        next,
                        key,
                        policy,
                        ring_challenge_config,
                        fold_challenge_shape_at_level,
                        fold_level + 1,
                        len,
                        1,
                    )?;
                    next_lp.witness_chunk = policy.witness_chunk_for_level(fold_level + 1);
                    (len, Some(next_lp), RelationMatrixRowLayout::WithDBlock)
                };

                let level_bytes = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &lp,
                    next_lp.as_ref(),
                    next_w_len,
                    1,
                    layout,
                )
                .checked_add(extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    policy.claim_ext_degree,
                    fold_level,
                    key,
                    current_w_len,
                )?)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("generated level byte count overflow".to_string())
                })?;
                total_bytes = total_bytes.checked_add(level_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
                })?;

                steps.push(Step::Fold(FoldStep {
                    params: lp.clone(),
                    current_w_len,
                    next_w_len,
                    level_bytes,
                }));
                last_fold_lp = Some(lp);
                fold_level += 1;
                current_w_len = next_w_len;
            }
            GeneratedStep::Direct(direct) => {
                let (witness_shape, direct_current_w_len, params) = if fold_level == 0 {
                    let params = direct
                        .commit
                        .as_ref()
                        .map(|commit| {
                            validate_block_geometry(commit, key, policy, 0, expected_root_w_len)?;
                            validate_log_basis(commit.log_basis, policy)?;
                            let fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
                                num_vars: key.num_vars(),
                                level: 0,
                                current_w_len: expected_root_w_len,
                            });
                            let lp = commit.expand_to_root_direct_commit_params(
                                policy,
                                ring_challenge_config,
                                expected_root_w_len,
                                fold_shape,
                                key.num_polynomials(),
                            )?;
                            validate_expanded_level_params(
                                &lp,
                                commit,
                                policy,
                                0,
                                key.num_polynomials(),
                            )
                        })
                        .transpose()?;
                    (
                        CleartextWitnessShape::FieldElements(expected_root_w_len),
                        expected_root_w_len,
                        params,
                    )
                } else {
                    let len = terminal_witness_field_len.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing precomputed witness length".to_string(),
                        )
                    })?;
                    let terminal_lp = last_fold_lp.as_ref().ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing predecessor fold params".to_string(),
                        )
                    })?;
                    let num_polynomials = if fold_level == 1 {
                        key.num_polynomials()
                    } else {
                        1
                    };
                    // The terminal-direct (cleartext) witness is single-chunk by
                    // construction: the prover emits the global folded response
                    // and one shared `r̂` tail (`num_segments = 1`). Chunking the
                    // cleartext tail is unsupported, so the last fold level must be
                    // single-chunk; reject loudly here instead of letting the
                    // prover hit a cryptic layout mismatch at prove time.
                    if terminal_lp.witness_chunk.num_chunks > 1 {
                        return Err(AkitaError::InvalidSetup(
                            "terminal-direct witness does not support a multi-chunk last fold level"
                                .to_string(),
                        ));
                    }
                    let witness_shape = segment_typed_witness_shape_from_groups(
                        terminal_lp,
                        field_bits,
                        [(
                            terminal_lp as &dyn akita_types::LevelParamsLike,
                            num_polynomials,
                            num_polynomials,
                            1,
                        )],
                        1,
                    )?;
                    (witness_shape, len, None)
                };
                if direct_current_w_len == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "generated direct step has zero witness length".to_string(),
                    ));
                }
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                total_bytes = total_bytes.checked_add(direct_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("generated proof byte total overflow".to_string())
                })?;
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct_current_w_len,
                    witness_shape,
                    direct_bytes,
                    params,
                }));
            }
        }
    }

    if total_bytes == 0 {
        return Err(AkitaError::InvalidSetup(
            "generated schedule validates to zero proof bytes".to_string(),
        ));
    }

    let schedule = Schedule { steps, total_bytes };

    Ok(GeneratedEntryWalkOutput {
        total_bytes,
        schedule,
    })
}

/// Walk one multi-group-root generated catalog row into a runtime [`Schedule`].
fn walk_multi_group_generated_schedule_entry(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    let expected_root_w_len = 1usize
        .checked_shl(key.final_group.num_vars() as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group root witness length overflow".into())
        })?;
    let root_fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.final_group.num_vars(),
        level: 0,
        current_w_len: expected_root_w_len,
    });
    let extension_opening_width = policy.claim_ext_degree;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits
        .checked_mul(policy.chal_ext_degree as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "generated multi-group schedule challenge field bit width overflow".to_string(),
            )
        })?;
    let root_eor_key =
        PolynomialGroupLayout::new(key.final_group.num_vars(), key.num_polynomials()?);

    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut total_bytes = 0usize;
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len: Option<usize> = None;
    let mut last_fold_lp: Option<LevelParams> = None;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(_) | GeneratedStep::FoldWithSetupMetadata(_) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated multi-group schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                let inputs = AkitaScheduleInputs {
                    num_vars: key.final_group.num_vars(),
                    level: fold_level,
                    current_w_len,
                };
                let fold_shape = if fold_level == 0 {
                    root_fold_shape
                } else {
                    fold_challenge_shape_at_level(inputs)
                };
                let mut lp = if fold_level == 0 {
                    let (precommitted_groups, precommitted_d_width) =
                        multi_group_root_precommitted_groups(key, policy, ring_challenge_config)?;
                    validate_expanded_precommitted_groups(key, &precommitted_groups)?;
                    expand_multi_group_root_fold_step(
                        step,
                        policy,
                        ring_challenge_config,
                        fold_shape,
                        key.final_group.num_polynomials(),
                        precommitted_groups,
                        precommitted_d_width,
                    )?
                } else {
                    expand_fold_step(
                        step,
                        policy,
                        ring_challenge_config,
                        fold_level,
                        current_w_len,
                        fold_shape,
                        1,
                    )?
                };

                lp.witness_chunk = policy.witness_chunk_for_level(fold_level);
                if is_terminal && lp.has_precommitted_groups() {
                    return Err(AkitaError::InvalidSetup(
                        "grouped terminal fold must be followed by another fold".to_string(),
                    ));
                }

                let (next_w_len, next_lp, layout) = if is_terminal {
                    let len = planned_next_witness_len(
                        field_bits,
                        &lp,
                        1,
                        RelationMatrixRowLayout::WithoutDBlock,
                        lp.witness_chunk.num_chunks,
                    )?;
                    terminal_witness_field_len = Some(len);
                    (len, None, RelationMatrixRowLayout::WithoutDBlock)
                } else {
                    let len = if fold_level == 0 {
                        let opening_batch = key.opening_layout()?;
                        lp.next_w_len::<Prime128OffsetA7F7>(
                            &opening_batch,
                            RelationMatrixRowLayout::WithDBlock,
                        )?
                    } else {
                        planned_next_witness_len(
                            field_bits,
                            &lp,
                            1,
                            RelationMatrixRowLayout::WithDBlock,
                            lp.witness_chunk.num_chunks,
                        )?
                    };
                    if next.fold_step().is_none() {
                        return Err(AkitaError::InvalidSetup(
                            "generated multi-group non-terminal successor must be a fold step"
                                .to_string(),
                        ));
                    }
                    let next_inputs = AkitaScheduleInputs {
                        num_vars: key.final_group.num_vars(),
                        level: fold_level + 1,
                        current_w_len: len,
                    };
                    let mut next_lp = expand_fold_step(
                        next,
                        policy,
                        ring_challenge_config,
                        fold_level + 1,
                        len,
                        fold_challenge_shape_at_level(next_inputs),
                        1,
                    )?;
                    next_lp.witness_chunk = policy.witness_chunk_for_level(fold_level + 1);
                    (len, Some(next_lp), RelationMatrixRowLayout::WithDBlock)
                };

                let level_bytes = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &lp,
                    next_lp.as_ref(),
                    next_w_len,
                    1,
                    layout,
                )
                .checked_add(extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    extension_opening_width,
                    fold_level,
                    root_eor_key,
                    current_w_len,
                )?)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated multi-group level byte count overflow".to_string(),
                    )
                })?;
                total_bytes = total_bytes.checked_add(level_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated multi-group proof byte total overflow".to_string(),
                    )
                })?;
                last_fold_lp = Some(lp.clone());
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len,
                    next_w_len,
                    level_bytes,
                }));
                fold_level += 1;
                current_w_len = next_w_len;
            }
            GeneratedStep::Direct(direct) => {
                let (witness_shape, direct_current_w_len, params) = if fold_level == 0 {
                    let direct_current_w_len = key.opening_layout()?.root_direct_witness_len()?;
                    let fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
                        num_vars: key.final_group.num_vars(),
                        level: 0,
                        current_w_len: direct_current_w_len,
                    });
                    let params = match direct.commit {
                        Some(commit) => {
                            let (precommitted_groups, precommitted_d_width) =
                                multi_group_root_precommitted_groups(
                                    key,
                                    policy,
                                    ring_challenge_config,
                                )?;
                            validate_expanded_precommitted_groups(key, &precommitted_groups)?;
                            Some(commit.expand_to_multi_group_root_level_params(
                                policy,
                                ring_challenge_config,
                                fold_shape,
                                key.final_group.num_polynomials(),
                                precommitted_groups,
                                precommitted_d_width,
                            )?)
                        }
                        None => None,
                    };
                    (
                        CleartextWitnessShape::FieldElements(direct_current_w_len),
                        direct_current_w_len,
                        params,
                    )
                } else {
                    let len = terminal_witness_field_len.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing precomputed witness length".to_string(),
                        )
                    })?;
                    let terminal_lp = last_fold_lp.as_ref().ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing predecessor fold params".to_string(),
                        )
                    })?;
                    let witness_shape = segment_typed_witness_shape_from_groups(
                        terminal_lp,
                        field_bits,
                        [(terminal_lp as &dyn akita_types::LevelParamsLike, 1, 1, 1)],
                        1,
                    )?;
                    (witness_shape, len, None)
                };
                if direct_current_w_len == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "generated multi-group direct step has zero witness length".to_string(),
                    ));
                }
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                total_bytes = total_bytes.checked_add(direct_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated multi-group proof byte total overflow".to_string(),
                    )
                })?;
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct_current_w_len,
                    witness_shape,
                    direct_bytes,
                    params,
                }));
            }
        }
    }

    if total_bytes == 0 {
        return Err(AkitaError::InvalidSetup(
            "generated multi-group schedule validates to zero proof bytes".to_string(),
        ));
    }

    let schedule = Schedule { steps, total_bytes };

    Ok(GeneratedEntryWalkOutput {
        total_bytes,
        schedule,
    })
}

fn validate_expanded_precommitted_groups(
    key: &AkitaScheduleLookupKey,
    groups: &[PrecommittedLevelParams],
) -> Result<(), AkitaError> {
    if groups.len() != key.precommitteds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "multi-group root precommitted group count mismatch: expected {}, got {}",
            key.precommitteds.len(),
            groups.len()
        )));
    }
    for (expected, actual) in key.precommitteds.iter().zip(groups) {
        if &actual.layout != expected {
            return Err(AkitaError::InvalidSetup(
                "multi-group root expanded precommitted layout does not match frozen key"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn expand_validated_fold_level(
    step: &GeneratedStep,
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    fold_level: usize,
    current_w_len: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    let fold = step
        .fold_step()
        .ok_or_else(|| AkitaError::InvalidSetup("generated expected a fold step".to_string()))?;
    validate_block_geometry(fold, key, policy, fold_level, current_w_len)?;
    validate_log_basis(fold.log_basis, policy)?;
    let inputs = AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: fold_level,
        current_w_len,
    };
    let lp = expand_fold_step(
        step,
        policy,
        ring_challenge_config,
        fold_level,
        current_w_len,
        fold_challenge_shape_at_level(inputs),
        num_claims,
    )?;
    validate_expanded_level_params(&lp, fold, policy, fold_level, num_claims)
}

#[allow(clippy::too_many_arguments)]
fn expand_fold_step(
    step: &GeneratedStep,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_level: usize,
    current_w_len: usize,
    fold_shape: TensorChallengeShape,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    match step {
        GeneratedStep::Fold(fold) => fold.expand_to_level_params(
            policy,
            ring_challenge_config,
            fold_level,
            current_w_len,
            fold_shape,
            num_claims,
        ),
        GeneratedStep::FoldWithSetupMetadata(fold) => fold.expand_to_level_params(
            policy,
            ring_challenge_config,
            fold_level,
            current_w_len,
            fold_shape,
            num_claims,
        ),
        GeneratedStep::Direct(_) => Err(AkitaError::InvalidSetup(
            "generated expected a fold step".to_string(),
        )),
    }
}

fn expand_multi_group_root_fold_step(
    step: &GeneratedStep,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: TensorChallengeShape,
    main_num_polys: usize,
    precommitted_groups: Vec<PrecommittedLevelParams>,
    precommitted_d_width: usize,
) -> Result<LevelParams, AkitaError> {
    match step {
        GeneratedStep::Fold(fold) => fold.expand_to_multi_group_root_level_params(
            policy,
            ring_challenge_config,
            fold_shape,
            main_num_polys,
            precommitted_groups,
            precommitted_d_width,
        ),
        GeneratedStep::FoldWithSetupMetadata(fold) => fold.expand_to_multi_group_root_level_params(
            policy,
            ring_challenge_config,
            fold_shape,
            main_num_polys,
            precommitted_groups,
            precommitted_d_width,
        ),
        GeneratedStep::Direct(_) => Err(AkitaError::InvalidSetup(
            "generated expected a fold step".to_string(),
        )),
    }
}

fn validate_log_basis(log_basis: u32, policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let (min, max) = policy.basis_range;
    if log_basis < min || log_basis > max {
        return Err(AkitaError::InvalidSetup(format!(
            "generated fold step log_basis={log_basis} outside policy range [{min}, {max}]"
        )));
    }
    Ok(())
}

fn validate_block_geometry(
    step: &GeneratedFoldStep,
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    fold_level: usize,
    current_w_len: usize,
) -> Result<(), AkitaError> {
    if step.ring_d as usize != policy.ring_dimension || step.ring_d == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "generated fold step ring dimension {} does not match policy D={}",
            step.ring_d, policy.ring_dimension
        )));
    }
    if policy.ring_dimension == 0 || !policy.ring_dimension.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "generated schedule policy ring dimension must be a nonzero power of two".to_string(),
        ));
    }
    let block_index_bits = step.block_index_bits as usize;
    let position_index_bits = step.position_index_bits as usize;
    let block_index_domain_size = 1usize.checked_shl(step.block_index_bits).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "generated schedule 2^block_index_bits overflows usize".to_string(),
        )
    })?;
    let num_live_blocks = step.num_live_blocks as usize;
    if num_live_blocks == 0
        || num_live_blocks > block_index_domain_size
        || num_live_blocks
            .checked_next_power_of_two()
            .is_none_or(|domain| domain != block_index_domain_size)
    {
        return Err(AkitaError::InvalidSetup(
            "generated schedule exact live block count disagrees with block-index domain"
                .to_string(),
        ));
    }
    let num_positions_per_block =
        1usize
            .checked_shl(step.position_index_bits)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated schedule 2^position_index_bits overflows usize".to_string(),
                )
            })?;
    if fold_level == 0 {
        // A small root-direct polynomial may occupy only a prefix of its first
        // ring. Count that padded ring as live source storage; recursive
        // witnesses remain exactly ring-aligned below.
        let num_live_ring_elements_per_claim = current_w_len.div_ceil(policy.ring_dimension);
        let derived_num_live_blocks =
            num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
        if num_live_blocks != derived_num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "generated root exact live block mismatch: stored={num_live_blocks}, derived={derived_num_live_blocks}"
            )));
        }
        let alpha = policy.ring_dimension.trailing_zeros() as usize;
        if position_index_bits
            .checked_add(block_index_bits)
            .and_then(|n| n.checked_add(alpha))
            != Some(key.num_vars().max(alpha))
        {
            return Err(AkitaError::InvalidSetup(
                "generated root geometry variable split disagrees with padded key domain"
                    .to_string(),
            ));
        }
        return Ok(());
    }

    if current_w_len == 0 || !current_w_len.is_multiple_of(policy.ring_dimension) {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive fold level {fold_level} has invalid current_w_len={current_w_len}"
        )));
    }
    let num_ring_elems = current_w_len / policy.ring_dimension;
    let reduced_vars = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| {
            AkitaError::InvalidSetup("generated recursive witness length overflow".to_string())
        })?
        .max(1)
        .trailing_zeros() as usize;
    if position_index_bits.checked_add(block_index_bits) != Some(reduced_vars) {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive geometry mismatch at level {fold_level}: position_index_bits={position_index_bits}, block_index_bits={block_index_bits}, reduced_vars={reduced_vars}"
        )));
    }
    let derived_num_live_blocks = num_ring_elems.div_ceil(num_positions_per_block);
    if num_live_blocks != derived_num_live_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive exact live block mismatch at level {fold_level}: stored={num_live_blocks}, derived={derived_num_live_blocks}"
        )));
    }
    Ok(())
}

fn validate_expanded_level_params(
    lp: &LevelParams,
    step: &GeneratedFoldStep,
    policy: &PlannerPolicy,
    fold_level: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    if lp.position_index_bits() != step.position_index_bits as usize
        || lp.block_index_bits() != step.block_index_bits as usize
        || lp.num_live_blocks != step.num_live_blocks as usize
    {
        return Err(AkitaError::InvalidSetup(
            "expanded generated level has mismatched block geometry".to_string(),
        ));
    }
    if lp.log_basis != step.log_basis {
        return Err(AkitaError::InvalidSetup(
            "expanded generated level has mismatched log_basis".to_string(),
        ));
    }
    if fold_level > 0 && lp.onehot_chunk_size != 0 {
        return Err(AkitaError::InvalidSetup(
            "generated recursive level must not carry one-hot root metadata".to_string(),
        ));
    }
    if fold_level == 0
        && policy.decomposition.log_commit_bound == 1
        && policy.onehot_chunk_size == 0
    {
        return Err(AkitaError::InvalidSetup(
            "one-hot root requires onehot_chunk_size > 0".to_string(),
        ));
    }
    if fold_level == 0
        && policy.decomposition.log_commit_bound == 1
        && lp.onehot_chunk_size != policy.onehot_chunk_size
    {
        return Err(AkitaError::InvalidSetup(
            "generated one-hot root has mismatched chunk size".to_string(),
        ));
    }
    lp.num_digits_fold(num_claims, policy.decomposition.field_bits())?;
    Ok(lp.clone())
}
