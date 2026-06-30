//! Canonical walker for compact generated schedule rows.
//!
//! [`walk_generated_schedule_entry`] is the single implementation shared by
//! runtime materialization ([`crate::schedule_from_entry`]) and admissibility
//! checks ([`super::validate::validate_generated_schedule_entry`]). Both paths
//! expand every fold step once, audit SIS ranks via
//! [`GeneratedFoldStep::expand_to_level_params`], and recompute witness
//! transitions and proof-byte totals.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_level_bytes, level_proof_bytes,
    segment_typed_witness_shape, w_ring_element_count_with_counts_for_layout_bits,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep,
    LevelParams, MRowLayout, Schedule, Step,
};

use crate::generated::{
    GeneratedFoldStep, GeneratedScheduleKey, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::PlannerPolicy;

pub(crate) struct GeneratedEntryWalkOutput {
    pub total_bytes: usize,
    pub schedule: Schedule,
}

pub(crate) fn walk_generated_schedule_entry(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedEntryWalkOutput, AkitaError> {
    key.validate()?;
    validate_entry_key(entry, key)?;
    entry.validate()?;

    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits
        .checked_mul(policy.chal_ext_degree as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "generated schedule challenge field bit width overflow".to_string(),
            )
        })?;
    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;

    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len: Option<usize> = None;
    let mut last_fold_lp: Option<LevelParams> = None;
    let mut total_bytes = 0usize;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(level) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                let num_claims = if fold_level == 0 {
                    key.num_polynomials
                } else {
                    1
                };
                let lp = expand_validated_fold_level(
                    level,
                    key,
                    policy,
                    ring_challenge_config,
                    fold_challenge_shape_at_level,
                    fold_level,
                    current_w_len,
                    num_claims,
                )?;
                let (num_polynomials, num_public_rows) = if fold_level == 0 {
                    (key.num_polynomials, 1)
                } else {
                    (1, 1)
                };
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let ring_len = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        num_polynomials,
                        num_public_rows,
                        MRowLayout::WithoutDBlock,
                    )?;
                    let len = checked_ring_field_len(ring_len, lp.ring_dimension)?;
                    terminal_witness_field_len = Some(len);
                    (len, None, MRowLayout::WithoutDBlock)
                } else {
                    let ring_len = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        num_polynomials,
                        num_public_rows,
                        MRowLayout::WithDBlock,
                    )?;
                    let len = checked_ring_field_len(ring_len, lp.ring_dimension)?;
                    let GeneratedStep::Fold(next_level) = next else {
                        return Err(AkitaError::InvalidSetup(
                            "generated non-terminal successor must be a fold step".to_string(),
                        ));
                    };
                    let next_lp = expand_validated_fold_level(
                        next_level,
                        key,
                        policy,
                        ring_challenge_config,
                        fold_challenge_shape_at_level,
                        fold_level + 1,
                        len,
                        1,
                    )?;
                    (len, Some(next_lp), MRowLayout::WithDBlock)
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
                        .map(|commit| {
                            expand_validated_fold_level(
                                &commit,
                                key,
                                policy,
                                ring_challenge_config,
                                fold_challenge_shape_at_level,
                                0,
                                expected_root_w_len,
                                key.num_polynomials,
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
                        key.num_polynomials
                    } else {
                        1
                    };
                    let witness_shape = segment_typed_witness_shape(
                        terminal_lp,
                        field_bits,
                        num_polynomials,
                        num_polynomials,
                        1,
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

#[allow(clippy::too_many_arguments)]
fn expand_validated_fold_level(
    step: &GeneratedFoldStep,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    fold_level: usize,
    current_w_len: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    validate_fold_geometry(step, key, policy, fold_level, current_w_len)?;
    validate_log_basis(step.log_basis, policy)?;
    let inputs = AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: fold_level,
        current_w_len,
    };
    let lp = step.expand_to_level_params(
        policy,
        ring_challenge_config,
        fold_level,
        current_w_len,
        fold_challenge_shape_at_level(inputs),
        num_claims,
    )?;
    validate_expanded_level_params(&lp, step, policy, fold_level, num_claims)
}

fn validate_entry_key(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
) -> Result<(), AkitaError> {
    let expected = GeneratedScheduleKey {
        num_vars: key.num_vars,
        num_polynomials: key.num_polynomials,
    };
    if entry.key != expected {
        return Err(AkitaError::InvalidSetup(format!(
            "generated schedule key mismatch: entry key {:?}, requested key {:?}",
            entry.key, expected
        )));
    }
    Ok(())
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

fn validate_fold_geometry(
    step: &GeneratedFoldStep,
    key: AkitaScheduleLookupKey,
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
    let r_vars = step.r_vars as usize;
    let m_vars = step.m_vars as usize;
    let num_blocks = 1usize.checked_shl(step.r_vars).ok_or_else(|| {
        AkitaError::InvalidSetup("generated schedule 2^r_vars overflows usize".to_string())
    })?;
    let block_len = 1usize.checked_shl(step.m_vars).ok_or_else(|| {
        AkitaError::InvalidSetup("generated schedule 2^m_vars overflows usize".to_string())
    })?;
    let ring_capacity = num_blocks
        .checked_mul(block_len)
        .and_then(|n| n.checked_mul(policy.ring_dimension))
        .ok_or_else(|| AkitaError::InvalidSetup("generated root capacity overflow".to_string()))?;

    if fold_level == 0 {
        if ring_capacity < current_w_len {
            return Err(AkitaError::InvalidSetup(format!(
                "generated root geometry under-covers witness: capacity={ring_capacity}, witness={current_w_len}"
            )));
        }
        let alpha = policy.ring_dimension.trailing_zeros() as usize;
        if m_vars
            .checked_add(r_vars)
            .and_then(|n| n.checked_add(alpha))
            .is_none_or(|bits| bits < key.num_vars)
        {
            return Err(AkitaError::InvalidSetup(
                "generated root geometry has too few variables for key".to_string(),
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
    if m_vars.checked_add(r_vars) != Some(reduced_vars) {
        return Err(AkitaError::InvalidSetup(format!(
            "generated recursive geometry mismatch at level {fold_level}: m_vars={m_vars}, r_vars={r_vars}, reduced_vars={reduced_vars}"
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
    let expected_num_blocks = 1usize.checked_shl(step.r_vars).ok_or_else(|| {
        AkitaError::InvalidSetup("generated schedule 2^r_vars overflows usize".to_string())
    })?;
    if lp.num_blocks != expected_num_blocks {
        return Err(AkitaError::InvalidSetup(
            "expanded generated level has mismatched num_blocks".to_string(),
        ));
    }
    if lp.m_vars != step.m_vars as usize || lp.r_vars != step.r_vars as usize {
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
    if step.tier_split.is_some() != step.n_f.is_some() {
        return Err(AkitaError::InvalidSetup(
            "generated tiered step must set both tier_split and n_f, or neither".to_string(),
        ));
    }
    if !policy.tiered && (step.tier_split.is_some() || step.n_f.is_some()) {
        return Err(AkitaError::InvalidSetup(
            "generated tiered step is not allowed by the planner policy".to_string(),
        ));
    }
    lp.num_digits_fold(num_claims, policy.decomposition.field_bits())?;
    Ok(lp.clone())
}

fn checked_ring_field_len(ring_len: usize, ring_dimension: usize) -> Result<usize, AkitaError> {
    ring_len.checked_mul(ring_dimension).ok_or_else(|| {
        AkitaError::InvalidSetup("generated next witness length overflow".to_string())
    })
}
