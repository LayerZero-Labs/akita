//! Schedule resolution: catalog validation, cache-then-generate.
//!
//! [`resolve_schedule`] is the single entry point the runtime (config,
//! prover, verifier) uses to obtain a [`Schedule`] for a lookup key. When a
//! preset supplies a catalog, identity is validated and the compact entry
//! is expanded via [`schedule_from_entry`]; on a miss (or no catalog) the
//! schedule is regenerated with the offline DP search [`crate::find_schedule`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    segment_typed_witness_shape, w_ring_element_count_with_counts_for_layout_bits,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep,
    GroupBatchAkitaScheduleLookupKey, LevelParams, MRowLayout, Schedule, Step,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::{
    group_batch_table_entry, table_entry, GeneratedGroupBatchScheduleTableEntry,
    GeneratedScheduleKey, GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::group_batch::{
    grouped_root_direct_witness_len, grouped_root_next_w_len, grouped_root_precommitted_groups,
};
use crate::PlannerPolicy;
use crate::{find_group_batch_schedule, find_schedule};

///
/// Convert the public runtime lookup key into a generated-table lookup key.
pub const fn generated_schedule_lookup_key(key: AkitaScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        num_vars: key.num_vars,
        num_polynomials: key.num_polynomials,
    }
}

/// Resolve the runtime [`Schedule`] using an explicit optional catalog.
pub fn resolve_schedule(
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    let Some(table) = catalog else {
        return find_schedule(
            key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    };
    validate_catalog_identity(
        &table,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?;
    if let Some(entry) = table_entry(table, generated_schedule_lookup_key(key)) {
        return schedule_from_entry(
            entry,
            key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }
    find_schedule(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

/// Resolve a grouped-root schedule without falling back to a scalar table key.
pub fn resolve_group_batch_schedule(
    key: &GroupBatchAkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    if let Some(table) = catalog {
        validate_catalog_identity(
            &table,
            policy,
            &ring_challenge_config,
            &fold_challenge_shape_at_level,
        )?;
        if let Some(entry) = group_batch_table_entry(table, key) {
            return schedule_from_group_batch_entry(
                entry,
                key,
                policy,
                ring_challenge_config,
                fold_challenge_shape_at_level,
            );
        }
    }
    find_group_batch_schedule(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

/// Build the runtime [`Schedule`] for a compact generated grouped-root entry.
pub(crate) fn schedule_from_group_batch_entry(
    entry: &GeneratedGroupBatchScheduleTableEntry,
    key: &GroupBatchAkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    entry.validate()?;
    let extension_opening_width = policy.claim_ext_degree;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits * policy.chal_ext_degree as u32;
    let root_eor_key = AkitaScheduleLookupKey::new(key.main.num_vars, key.num_polynomials()?);

    let expected_root_w_len = 1usize
        .checked_shl(key.main.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped root witness length overflow".into()))?;
    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut total = 0usize;
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len: Option<usize> = None;
    let mut last_fold_lp: Option<LevelParams> = None;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(level) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated grouped schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                if fold_level == 0 && is_terminal {
                    return Err(AkitaError::InvalidSetup(
                        "grouped terminal root folds are not supported yet".to_string(),
                    ));
                }
                let inputs = AkitaScheduleInputs {
                    num_vars: key.main.num_vars,
                    level: fold_level,
                    current_w_len,
                };
                let fold_shape = fold_challenge_shape_at_level(inputs);
                let lp = if fold_level == 0 {
                    let (precommitted_groups, precommitted_d_width) =
                        grouped_root_precommitted_groups(
                            key,
                            policy,
                            &ring_challenge_config,
                            fold_shape,
                        )?;
                    level.expand_to_grouped_root_level_params(
                        policy,
                        &ring_challenge_config,
                        fold_shape,
                        key.main.num_polynomials,
                        precommitted_groups,
                        precommitted_d_width,
                    )?
                } else {
                    level.expand_to_level_params(
                        policy,
                        &ring_challenge_config,
                        fold_level,
                        current_w_len,
                        fold_shape,
                        1,
                    )?
                };

                let mul_d = |ring: usize, lp: &LevelParams| -> Result<usize, AkitaError> {
                    ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated grouped next witness length overflow".to_string(),
                        )
                    })
                };
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let ring = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        1,
                        1,
                        MRowLayout::WithoutDBlock,
                    )?;
                    let len = mul_d(ring, &lp)?;
                    terminal_witness_field_len = Some(len);
                    (len, None, MRowLayout::WithoutDBlock)
                } else {
                    let len = if fold_level == 0 {
                        grouped_root_next_w_len(
                            field_bits,
                            &lp,
                            key.main.num_polynomials,
                            MRowLayout::WithDBlock,
                        )?
                    } else {
                        let ring = w_ring_element_count_with_counts_for_layout_bits(
                            field_bits,
                            &lp,
                            1,
                            1,
                            MRowLayout::WithDBlock,
                        )?;
                        mul_d(ring, &lp)?
                    };
                    let GeneratedStep::Fold(next_level) = next else {
                        return Err(AkitaError::InvalidSetup(
                            "generated grouped non-terminal successor must be a fold step"
                                .to_string(),
                        ));
                    };
                    let next_inputs = AkitaScheduleInputs {
                        num_vars: key.main.num_vars,
                        level: fold_level + 1,
                        current_w_len: len,
                    };
                    let next_lp = next_level.expand_to_level_params(
                        policy,
                        &ring_challenge_config,
                        fold_level + 1,
                        len,
                        fold_challenge_shape_at_level(next_inputs),
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
                ) + extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    extension_opening_width,
                    fold_level,
                    root_eor_key,
                    current_w_len,
                )?;
                total = total.checked_add(level_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
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
                    let direct_current_w_len = grouped_root_direct_witness_len(key)?;
                    let fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
                        num_vars: key.main.num_vars,
                        level: 0,
                        current_w_len: direct_current_w_len,
                    });
                    let params = match direct.commit {
                        Some(commit) => {
                            let (precommitted_groups, precommitted_d_width) =
                                grouped_root_precommitted_groups(
                                    key,
                                    policy,
                                    &ring_challenge_config,
                                    fold_shape,
                                )?;
                            Some(commit.expand_to_grouped_root_level_params(
                                policy,
                                &ring_challenge_config,
                                fold_shape,
                                key.main.num_polynomials,
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
                    let witness_shape =
                        segment_typed_witness_shape(terminal_lp, field_bits, 1, 1, 1, 1)?;
                    (witness_shape, len, None)
                };
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                total = total.checked_add(direct_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
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

    Ok(Schedule {
        steps,
        total_bytes: total,
    })
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: AkitaScheduleLookupKey,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    if extension_opening_width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (
            extension_opening_width.saturating_mul(key.num_polynomials),
            key.num_vars,
        )
    } else {
        (extension_opening_width, padded_boolean_vars(current_w_len)?)
    };
    extension_opening_reduction_proof_bytes(
        challenge_field_bits,
        partials,
        opening_vars,
        extension_opening_width,
    )
}

/// Build the runtime [`Schedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    entry.validate()?;
    let extension_opening_width = policy.claim_ext_degree;
    let field_bits = policy.decomposition.field_bits();
    let challenge_field_bits = field_bits * policy.chal_ext_degree as u32;

    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut total = 0usize;
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut terminal_witness_field_len: Option<usize> = None;
    let mut last_fold_lp: Option<LevelParams> = None;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(level) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                let inputs = AkitaScheduleInputs {
                    num_vars: key.num_vars,
                    level: fold_level,
                    current_w_len,
                };
                let level_num_claims = if fold_level == 0 {
                    key.num_polynomials
                } else {
                    1
                };
                let lp = level.expand_to_level_params(
                    policy,
                    &ring_challenge_config,
                    fold_level,
                    current_w_len,
                    fold_challenge_shape_at_level(inputs),
                    level_num_claims,
                )?;
                let (num_polynomials, num_public_rows) = if fold_level == 0 {
                    (key.num_polynomials, 1)
                } else {
                    (1, 1)
                };
                let mul_d = |ring: usize| -> Result<usize, AkitaError> {
                    ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated next witness length overflow".to_string(),
                        )
                    })
                };
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let ring = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        num_polynomials,
                        num_public_rows,
                        MRowLayout::WithoutDBlock,
                    )?;
                    let len = mul_d(ring)?;
                    terminal_witness_field_len = Some(len);
                    (len, None, MRowLayout::WithoutDBlock)
                } else {
                    let ring = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        num_polynomials,
                        num_public_rows,
                        MRowLayout::WithDBlock,
                    )?;
                    let len = mul_d(ring)?;
                    let GeneratedStep::Fold(next_level) = next else {
                        return Err(AkitaError::InvalidSetup(
                            "generated non-terminal successor must be a fold step".to_string(),
                        ));
                    };
                    let next_inputs = AkitaScheduleInputs {
                        num_vars: key.num_vars,
                        level: fold_level + 1,
                        current_w_len: len,
                    };
                    let next_lp = next_level.expand_to_level_params(
                        policy,
                        &ring_challenge_config,
                        fold_level + 1,
                        len,
                        fold_challenge_shape_at_level(next_inputs),
                        1,
                    )?;
                    (len, Some(next_lp), MRowLayout::WithDBlock)
                };
                // Single commitment group at one point: one public row per level.
                let num_claims_here = 1;
                let level_bytes = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &lp,
                    next_lp.as_ref(),
                    next_w_len,
                    num_claims_here,
                    layout,
                ) + extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    extension_opening_width,
                    fold_level,
                    key,
                    current_w_len,
                )?;
                total = total.checked_add(level_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
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
                    let params = match direct.commit {
                        Some(commit) => commit
                            .expand_to_level_params(
                                policy,
                                &ring_challenge_config,
                                0,
                                expected_root_w_len,
                                fold_challenge_shape_at_level(AkitaScheduleInputs {
                                    num_vars: key.num_vars,
                                    level: 0,
                                    current_w_len: expected_root_w_len,
                                }),
                                key.num_polynomials,
                            )
                            .ok(),
                        None => None,
                    };
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
                    let terminal_fold_level = fold_level.saturating_sub(1);
                    let terminal_lp = last_fold_lp.as_ref().ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing predecessor fold params".to_string(),
                        )
                    })?;
                    let num_polynomials = if terminal_fold_level == 0 {
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
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                total = total.checked_add(direct_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
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

    Ok(Schedule {
        steps,
        total_bytes: total,
    })
}

pub fn estimate_proof_bytes(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<usize, AkitaError> {
    Ok(schedule_from_entry(
        entry,
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?
    .total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{DecompositionParams, SisModulusFamily};

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_family: SisModulusFamily::Q128,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 4,
            chal_ext_degree: 4,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
            tiered: false,
        }
    }

    fn ring_challenge_config(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn fold_shape(_: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    #[test]
    fn resolve_schedule_none_matches_find_schedule() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let via_resolve = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect("resolve");
        let via_find =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find");
        assert_eq!(via_resolve.total_bytes, via_find.total_bytes);
    }

    #[test]
    fn resolve_schedule_rejects_zero_dimension_key() {
        let key = AkitaScheduleLookupKey::new(0, 1);
        let policy = flat_policy();

        let err = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect_err("zero-arity key must be rejected");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
