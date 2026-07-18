//! Schedule resolution: catalog validation, cache-then-generate.
//!
//! [`resolve_schedule`] is the single entry point the runtime (config,
//! prover, verifier) uses to obtain a [`Schedule`] for a lookup key. When a
//! preset supplies a catalog, identity is validated and the compact entry
//! is expanded via [`schedule_from_entry`]; on a miss (or no catalog) the
//! schedule is regenerated with the offline key-shaped DP search
//! [`crate::find_group_batch_schedule`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, PolynomialGroupLayout, Schedule};

use crate::catalog_identity::validate_catalog_identity;
use crate::find_group_batch_schedule;
use crate::find_schedule;
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{table_entry, GeneratedScheduleTable, GeneratedScheduleTableEntry};
use crate::schedule_params::validate_policy_witness_chunk;
use crate::PlannerPolicy;

/// Resolve the runtime [`Schedule`] using an explicit optional catalog.
pub fn resolve_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    resolve_group_batch_schedule(
        &AkitaScheduleLookupKey::single(key),
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
        catalog,
    )
}

/// Resolve a multi-group-root schedule without falling back to a scalar table key.
pub fn resolve_group_batch_schedule(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<Schedule, AkitaError> {
    key.validate()?;
    validate_policy_witness_chunk(policy)?;
    let scalar_recursive_key = key.precommitteds.is_empty() && policy.recursive_setup_planning;
    if !scalar_recursive_key {
        if let Some(table) = catalog {
            validate_catalog_identity(
                &table,
                policy,
                &ring_challenge_config,
                &fold_challenge_shape_at_level,
            )?;
            if let Some(entry) = table_entry(table, key) {
                match schedule_from_entry(
                    entry,
                    key,
                    policy,
                    &ring_challenge_config,
                    &fold_challenge_shape_at_level,
                ) {
                    Ok(schedule) => {
                        schedule.validate_structure()?;
                        return Ok(schedule);
                    }
                    Err(err) if unsupported_grouped_table_hit_error(&err) => {}
                    Err(err) => return Err(err),
                }
            }
        }
    }
    if scalar_recursive_key {
        let mut scalar_policy = *policy;
        scalar_policy.recursive_setup_planning = false;
        let schedule = find_schedule(
            key.final_group,
            &scalar_policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )?;
        schedule.validate_structure()?;
        return Ok(schedule);
    }
    let schedule = find_group_batch_schedule(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;
    schedule.validate_structure()?;
    Ok(schedule)
}

fn unsupported_grouped_table_hit_error(err: &AkitaError) -> bool {
    let msg = err.to_string();
    msg.contains("grouped terminal fold must be followed by another fold")
        || msg.contains("terminal fold must be scalar")
}

/// Build the runtime [`Schedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Schedule, AkitaError> {
    let schedule = walk_generated_schedule_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?
    .schedule;
    schedule.validate_structure()?;
    Ok(schedule)
}

pub fn estimate_proof_bytes(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
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
    use crate::find_group_batch_schedule;
    use crate::find_schedule;
    use crate::generated::{
        validate_generated_schedule_entry, GeneratedFoldStep, GeneratedScheduleTable, GeneratedStep,
    };
    use akita_types::{
        AkitaScheduleLookupKey, ChunkedWitnessCfg, DecompositionParams, LevelParams,
        MultiChunkProfileId, PolynomialGroupLayout, PrecommittedGroupParams, SisModulusProfileId,
        SisTableDigest, Step, DEFAULT_SIS_SECURITY_POLICY,
    };

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: DEFAULT_SIS_SECURITY_POLICY,
            sis_table_digest: SisTableDigest::CURRENT,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        }
    }

    fn ring_challenge_config(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::pm1_only(1))
    }

    fn fold_shape(_: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    fn find_single_schedule(
        key: PolynomialGroupLayout,
        policy: &PlannerPolicy,
    ) -> Result<Schedule, AkitaError> {
        find_schedule(key, policy, ring_challenge_config, fold_shape)
    }

    fn generated_fold_step(lp: &LevelParams) -> GeneratedFoldStep {
        GeneratedFoldStep {
            ring_d: lp.ring_dimension as u32,
            log_basis: lp.log_basis,
            position_index_bits: lp.position_index_bits() as u32,
            block_index_bits: lp.block_index_bits() as u32,
            num_live_blocks: lp.num_live_blocks as u32,
            n_a: lp.a_key.row_len() as u32,
            n_b: lp.b_key.row_len() as u32,
            n_d: lp.d_key.row_len() as u32,
        }
    }

    fn generated_steps_from_schedule(schedule: &Schedule) -> Vec<GeneratedStep> {
        schedule
            .steps
            .iter()
            .map(|step| match step {
                Step::Fold(fold) => GeneratedStep::Fold(generated_fold_step(&fold.params)),
                Step::Direct(_) => GeneratedStep::Direct,
            })
            .collect()
    }

    fn generated_entry_from_steps(
        key: PolynomialGroupLayout,
        steps: Vec<GeneratedStep>,
    ) -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            final_group: key,
            precommitteds: &[],
            steps: Box::leak(steps.into_boxed_slice()),
        }
    }

    fn mutate_first_generated_fold_step(
        steps: &mut [GeneratedStep],
        mutate: impl FnOnce(&mut GeneratedFoldStep),
    ) {
        for step in steps.iter_mut() {
            match step {
                GeneratedStep::Fold(fold) => {
                    mutate(fold);
                    return;
                }
                GeneratedStep::FoldWithSetupMetadata(fold) => {
                    mutate(&mut fold.fold);
                    return;
                }
                GeneratedStep::Direct => {}
            }
        }
        panic!("schedule should contain a fold");
    }

    #[test]
    fn resolve_schedule_none_matches_key_planner() {
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let via_resolve = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect("resolve");
        let via_find = find_single_schedule(key, &policy).expect("find");
        assert_eq!(via_resolve.total_bytes, via_find.total_bytes);
    }

    #[test]
    fn multi_chunk_schedule_ends_with_single_chunk_terminal_fold() {
        // Chunked leading levels are allowed when the planner can route through a
        // single-chunk fold before the terminal-direct tail.
        let mut policy = flat_policy();
        policy.decomposition.log_open_bound = Some(128);
        policy.witness_chunk = ChunkedWitnessCfg {
            num_chunks: 8,
            num_activated_levels: 2,
        };
        let key = PolynomialGroupLayout::new(30, 1);
        let schedule = find_single_schedule(key, &policy).expect("schedule");
        let last_fold = schedule
            .steps
            .iter()
            .rev()
            .find_map(|step| match step {
                Step::Fold(fold) => Some(fold),
                _ => None,
            })
            .expect("fold-then-direct schedule");
        assert_eq!(last_fold.params.witness_chunk.num_chunks, 1);
    }

    #[test]
    fn multi_chunk_does_not_perturb_single_chunk_schedule() {
        // A policy with default (single-chunk) witness_chunk must reproduce the
        // exact schedule of today's planner for the same key.
        let key = PolynomialGroupLayout::new(22, 1);
        let base = flat_policy();
        let mut explicit_default = flat_policy();
        explicit_default.witness_chunk = ChunkedWitnessCfg::default();
        let a = find_single_schedule(key, &base);
        let b = find_single_schedule(key, &explicit_default);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.total_bytes, b.total_bytes);
                assert_eq!(a.steps.len(), b.steps.len());
            }
            (Err(AkitaError::UnsupportedSchedule(_)), Err(AkitaError::UnsupportedSchedule(_))) => {}
            (a, b) => panic!("default chunk policy diverged: implicit={a:?}, explicit={b:?}"),
        }
    }

    #[test]
    fn all_multi_chunk_profiles_use_single_chunk_terminal() {
        let key = PolynomialGroupLayout::new(30, 1);
        for profile in MultiChunkProfileId::ALL {
            let mut policy = flat_policy();
            policy.decomposition.log_open_bound = Some(128);
            policy.witness_chunk = profile.cfg();
            let schedule = find_single_schedule(key, &policy)
                .unwrap_or_else(|err| panic!("profile {profile:?} must plan at nv=30: {err:?}"));
            let last_fold = schedule
                .steps
                .iter()
                .rev()
                .find_map(|step| match step {
                    Step::Fold(fold) => Some(fold),
                    _ => None,
                })
                .expect("fold-then-direct schedule");
            assert_eq!(
                last_fold.params.witness_chunk.num_chunks, 1,
                "profile {profile:?} must end with a single-chunk terminal fold"
            );
        }
    }

    #[test]
    fn key_planner_rejects_non_power_of_two_chunks() {
        let mut policy = flat_policy();
        policy.witness_chunk = ChunkedWitnessCfg {
            num_chunks: 6,
            num_activated_levels: 2,
        };
        let key = PolynomialGroupLayout::new(30, 1);
        let err = find_single_schedule(key, &policy)
            .expect_err("non-power-of-two chunk count must be rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn resolve_group_batch_schedule_delegates_single_group_to_scalar() {
        let final_group = PolynomialGroupLayout::new(20, 1);
        let key = AkitaScheduleLookupKey::single(final_group);
        let policy = flat_policy();

        let via_multi_group =
            resolve_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape, None);
        let via_scalar = resolve_schedule(
            final_group,
            &policy,
            ring_challenge_config,
            fold_shape,
            None,
        );

        match (via_multi_group, via_scalar) {
            (Ok(via_multi_group), Ok(via_scalar)) => {
                assert_eq!(via_multi_group.total_bytes, via_scalar.total_bytes);
                assert_eq!(via_multi_group.steps.len(), via_scalar.steps.len());
            }
            (Err(AkitaError::UnsupportedSchedule(_)), Err(AkitaError::UnsupportedSchedule(_))) => {}
            (via_multi_group, via_scalar) => panic!(
                "single-group resolve diverged: grouped={via_multi_group:?}, scalar={via_scalar:?}"
            ),
        }
    }

    #[test]
    fn resolve_schedule_rejects_zero_dimension_key() {
        let key = PolynomialGroupLayout::new(0, 1);
        let policy = flat_policy();

        let err = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, None)
            .expect_err("zero-arity key must be rejected");

        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn validate_generated_entry_accepts_materialized_dp_schedule() {
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let entry = generated_entry_from_steps(key, generated_steps_from_schedule(&schedule));

        validate_generated_schedule_entry(
            &entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect("generated entry should validate");
    }

    #[test]
    fn validate_generated_entry_rejects_overstated_b_rank() {
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        mutate_first_generated_fold_step(&mut steps, |fold| fold.n_b += 1);
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            &AkitaScheduleLookupKey::single(key),
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
    fn validate_generated_entry_rejects_inexact_num_live_blocks() {
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        mutate_first_generated_fold_step(&mut steps, |fold| fold.num_live_blocks -= 1);
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect_err("inexact live block count must be rejected");

        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("live block")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_generated_entry_rejects_overstated_a_rank() {
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        mutate_first_generated_fold_step(&mut steps, |fold| fold.n_a += 1);
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            &AkitaScheduleLookupKey::single(key),
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
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        mutate_first_generated_fold_step(&mut steps, |fold| {
            assert!(fold.n_a > 1, "test needs n_a > 1 to understate rank");
            fold.n_a -= 1;
        });
        let entry = generated_entry_from_steps(key, steps);

        let err = validate_generated_schedule_entry(
            &entry,
            &AkitaScheduleLookupKey::single(key),
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
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let mut steps = generated_steps_from_schedule(&schedule);
        mutate_first_generated_fold_step(&mut steps, |fold| fold.n_d += 1);
        let entry = generated_entry_from_steps(key, steps);
        let entries: &'static [GeneratedScheduleTableEntry] =
            Box::leak(vec![entry].into_boxed_slice());
        let identity =
            expected_catalog_identity("test", &policy, entries, ring_challenge_config, fold_shape)
                .expect("identity");
        let table = GeneratedScheduleTable { entries, identity };

        let err = resolve_schedule(key, &policy, ring_challenge_config, fold_shape, Some(table))
            .expect_err("corrupt table hit must be rejected");

        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("d-rank mismatch")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn walk_validate_matches_materialize_total_bytes() {
        let key = PolynomialGroupLayout::new(30, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
        let entry = generated_entry_from_steps(key, generated_steps_from_schedule(&schedule));

        let validated = estimate_proof_bytes(
            &entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            ring_challenge_config,
            fold_shape,
        )
        .expect("validate bytes");
        let materialized = schedule_from_entry(
            &entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            ring_challenge_config,
            fold_shape,
        )
        .expect("materialize");

        assert_eq!(validated, materialized.total_bytes);
    }

    fn multi_group_sample_key() -> AkitaScheduleLookupKey {
        let pre_key = PolynomialGroupLayout::new(16, 1);
        let policy = flat_policy();
        let pre = PrecommittedGroupParams::from_params(
            pre_key,
            find_single_schedule(pre_key, &policy)
                .expect("precommit schedule")
                .steps
                .first()
                .and_then(|step| match step {
                    Step::Direct(_) => None,
                    Step::Fold(fold) => Some(&fold.params),
                })
                .expect("commit params"),
        );
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![pre],
        }
    }

    fn generated_group_entry_from_steps(
        key: &AkitaScheduleLookupKey,
        steps: Vec<GeneratedStep>,
    ) -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            final_group: key.final_group,
            precommitteds: Box::leak(key.precommitteds.clone().into_boxed_slice()),
            steps: Box::leak(steps.into_boxed_slice()),
        }
    }

    #[test]
    fn validate_generated_multi_group_entry_accepts_materialized_dp_schedule() {
        let key = multi_group_sample_key();
        let policy = flat_policy();
        let schedule = find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
            .expect("multi-group schedule");
        let entry =
            generated_group_entry_from_steps(&key, generated_steps_from_schedule(&schedule));

        validate_generated_schedule_entry(
            &entry,
            &key,
            &policy,
            &ring_challenge_config,
            &fold_shape,
        )
        .expect("multi-group generated entry should validate");
    }

    #[test]
    fn resolve_group_batch_schedule_rejects_stale_one_fold_table_hit() {
        let key = multi_group_sample_key();
        let mut policy = flat_policy();
        policy.decomposition.log_open_bound = Some(128);
        let regenerated =
            find_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape)
                .expect("multi-group schedule");
        let root = regenerated
            .steps
            .first()
            .and_then(|step| match step {
                Step::Fold(fold) => Some(generated_fold_step(&fold.params)),
                Step::Direct(_) => None,
            })
            .expect("multi-group root fold");
        let entry = generated_group_entry_from_steps(
            &key,
            vec![GeneratedStep::Fold(root), GeneratedStep::Direct],
        );
        let entries: &'static [GeneratedScheduleTableEntry] =
            Box::leak(vec![entry].into_boxed_slice());
        let identity =
            expected_catalog_identity("test", &policy, entries, ring_challenge_config, fold_shape)
                .expect("identity");
        let table = GeneratedScheduleTable { entries, identity };

        let error = resolve_group_batch_schedule(
            &key,
            &policy,
            ring_challenge_config,
            fold_shape,
            Some(table),
        )
        .expect_err("stale one-fold table hit must be rejected");

        assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
    }
}
