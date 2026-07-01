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
    direct_witness_bytes, extension_opening_reduction_level_bytes, level_proof_bytes,
    segment_typed_witness_shape, w_ring_element_count_with_counts_for_layout_bits,
    AkitaScheduleInputs, AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep,
    GroupBatchAkitaScheduleLookupKey, LevelParams, MRowLayout, Schedule, Step,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::walk::walk_generated_schedule_entry;
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

/// Build the runtime [`Schedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    Ok(walk_generated_schedule_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?
    .schedule)
}

pub fn estimate_proof_bytes(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<usize, AkitaError> {
    Ok(walk_generated_schedule_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?
    .total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog_identity::expected_catalog_identity;
    use crate::generated::{
        validate_generated_schedule_entry, GeneratedDirectStep, GeneratedFoldStep,
        GeneratedScheduleTable, GeneratedStep,
    };
    use akita_types::{DecompositionParams, LevelParams, SisModulusFamily, Step};

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

    fn generated_fold_step(lp: &LevelParams) -> GeneratedFoldStep {
        GeneratedFoldStep {
            ring_d: lp.ring_dimension as u32,
            log_basis: lp.log_basis,
            m_vars: lp.m_vars as u32,
            r_vars: lp.r_vars as u32,
            n_a: lp.a_key.row_len() as u32,
            n_b: lp.b_key.row_len() as u32,
            n_d: lp.d_key.row_len() as u32,
            tier_split: if lp.tier_split > 1 {
                Some(lp.tier_split as u32)
            } else {
                None
            },
            n_f: lp.f_key.as_ref().map(|f| f.row_len() as u32),
        }
    }

    fn generated_steps_from_schedule(schedule: &Schedule) -> Vec<GeneratedStep> {
        schedule
            .steps
            .iter()
            .map(|step| match step {
                Step::Fold(fold) => GeneratedStep::Fold(generated_fold_step(&fold.params)),
                Step::Direct(direct) => GeneratedStep::Direct(GeneratedDirectStep {
                    commit: direct.params.as_ref().map(generated_fold_step),
                }),
            })
            .collect()
    }

    fn generated_entry_from_steps(
        key: AkitaScheduleLookupKey,
        steps: Vec<GeneratedStep>,
    ) -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            key: generated_schedule_lookup_key(key),
            steps: Box::leak(steps.into_boxed_slice()),
        }
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

    #[test]
    fn validate_generated_entry_accepts_materialized_dp_schedule() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find schedule");
        let entry = generated_entry_from_steps(key, generated_steps_from_schedule(&schedule));

        validate_generated_schedule_entry(
            &entry,
            key,
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect("generated entry should validate");
    }

    #[test]
    fn validate_generated_entry_rejects_overstated_b_rank() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        match steps
            .iter_mut()
            .find(|step| matches!(step, GeneratedStep::Fold(_)))
            .expect("schedule should contain a fold")
        {
            GeneratedStep::Fold(fold) => fold.n_b += 1,
            GeneratedStep::Direct(_) => unreachable!("find guaranteed a fold"),
        }
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            key,
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect_err("overstated B rank must be rejected");

        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("b-rank mismatch")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_generated_entry_rejects_overstated_a_rank() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        match steps
            .iter_mut()
            .find(|step| matches!(step, GeneratedStep::Fold(_)))
            .expect("schedule should contain a fold")
        {
            GeneratedStep::Fold(fold) => fold.n_a += 1,
            GeneratedStep::Direct(_) => unreachable!("find guaranteed a fold"),
        }
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            key,
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect_err("overstated A rank must be rejected");

        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("a-rank mismatch")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_generated_entry_rejects_understated_a_rank() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        match steps
            .iter_mut()
            .find(|step| matches!(step, GeneratedStep::Fold(_)))
            .expect("schedule should contain a fold")
        {
            GeneratedStep::Fold(fold) => {
                assert!(fold.n_a > 1, "test needs n_a > 1 to understate rank");
                fold.n_a -= 1;
            }
            GeneratedStep::Direct(_) => unreachable!("find guaranteed a fold"),
        }
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            key,
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect_err("understated A rank must be rejected");

        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("a-rank mismatch")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_schedule_rejects_corrupt_table_hit() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        match steps
            .iter_mut()
            .find(|step| matches!(step, GeneratedStep::Fold(_)))
            .expect("schedule should contain a fold")
        {
            GeneratedStep::Fold(fold) => fold.n_d += 1,
            GeneratedStep::Direct(_) => unreachable!("find guaranteed a fold"),
        }
        let entry = generated_entry_from_steps(key, steps);
        let entries: &'static [GeneratedScheduleTableEntry] =
            Box::leak(vec![entry].into_boxed_slice());
        let identity = expected_catalog_identity(
            "test",
            &policy,
            entries,
            &[],
            ring_challenge_config,
            fold_shape,
        )
        .expect("identity");
        let table = GeneratedScheduleTable {
            entries,
            group_batch_entries: &[],
            identity,
        };

        let err = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, Some(table))
            .expect_err("corrupt table hit must be rejected");

        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("d-rank mismatch")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn walk_validate_matches_materialize_total_bytes() {
        let key = AkitaScheduleLookupKey::new(20, 1);
        let policy = flat_policy();
        let schedule =
            find_schedule(key, &policy, ring_challenge_config, fold_shape).expect("find schedule");
        let entry = generated_entry_from_steps(key, generated_steps_from_schedule(&schedule));

        let validated =
            estimate_proof_bytes(&entry, key, &policy, ring_challenge_config, fold_shape)
                .expect("validate bytes");
        let materialized =
            schedule_from_entry(&entry, key, &policy, ring_challenge_config, fold_shape)
                .expect("materialize");

        assert_eq!(validated, materialized.total_bytes);
    }
}
