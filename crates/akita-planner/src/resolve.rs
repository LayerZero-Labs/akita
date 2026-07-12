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
    if let Some(table) = catalog {
        validate_catalog_identity(
            &table,
            policy,
            &ring_challenge_config,
            &fold_challenge_shape_at_level,
        )?;
        if let Some(entry) = table_entry(table, key) {
            return schedule_from_entry(
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

/// Build the runtime [`Schedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
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
    use crate::generated::{
        validate_generated_schedule_entry, GeneratedDirectStep, GeneratedFoldStep,
        GeneratedScheduleTable, GeneratedStep,
    };
    use crate::schedule_params::find_single_group_schedule;
    use akita_types::{
        AkitaScheduleLookupKey, ChunkedWitnessCfg, DecompositionParams, LevelParams,
        MultiChunkProfileId, PolynomialGroupLayout, PrecommittedGroupParams, SisModulusFamily,
        Step, DEFAULT_SIS_SECURITY_BITS,
    };

    fn flat_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_family: SisModulusFamily::Q128,
            min_sis_security_bits: DEFAULT_SIS_SECURITY_BITS,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 4,
            chal_ext_degree: 4,
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
        find_single_group_schedule(key, policy, ring_challenge_config, fold_shape)
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
            setup_prefix_group: None,
            setup_contribution_mode: lp.setup_contribution_mode,
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
        key: PolynomialGroupLayout,
        steps: Vec<GeneratedStep>,
    ) -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            final_group: key,
            precommitteds: &[],
            steps: Box::leak(steps.into_boxed_slice()),
        }
    }

    #[test]
    fn resolve_schedule_none_matches_key_planner() {
        let key = PolynomialGroupLayout::new(20, 1);
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
        policy.witness_chunk = ChunkedWitnessCfg {
            num_chunks: 8,
            num_activated_levels: 2,
        };
        let key = PolynomialGroupLayout::new(24, 1);
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
        let a = find_single_schedule(key, &base).expect("a");
        let b = find_single_schedule(key, &explicit_default).expect("b");
        assert_eq!(a.total_bytes, b.total_bytes);
        assert_eq!(a.steps.len(), b.steps.len());
    }

    #[test]
    fn all_multi_chunk_profiles_use_single_chunk_terminal() {
        let key = PolynomialGroupLayout::new(24, 1);
        for profile in MultiChunkProfileId::ALL {
            let mut policy = flat_policy();
            policy.witness_chunk = profile.cfg();
            let schedule = find_single_schedule(key, &policy)
                .unwrap_or_else(|err| panic!("profile {profile:?} must plan at nv=24: {err:?}"));
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
        let key = PolynomialGroupLayout::new(20, 1);
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
            resolve_group_batch_schedule(&key, &policy, ring_challenge_config, fold_shape, None)
                .expect("single-group resolve should delegate to scalar path");
        let via_scalar = resolve_schedule(
            final_group,
            &policy,
            ring_challenge_config,
            fold_shape,
            None,
        )
        .expect("scalar resolve");

        assert_eq!(via_multi_group.total_bytes, via_scalar.total_bytes);
        assert_eq!(via_multi_group.steps.len(), via_scalar.steps.len());
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
        let key = PolynomialGroupLayout::new(20, 1);
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
        let key = PolynomialGroupLayout::new(20, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
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
    fn validate_generated_entry_rejects_overstated_a_rank() {
        let key = PolynomialGroupLayout::new(20, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
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
        let key = PolynomialGroupLayout::new(20, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
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
        let key = PolynomialGroupLayout::new(20, 1);
        let policy = flat_policy();
        let schedule = find_single_schedule(key, &policy).expect("find schedule");
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
        let key = PolynomialGroupLayout::new(20, 1);
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
        let pre_key = PolynomialGroupLayout::new(10, 1);
        let policy = flat_policy();
        let pre = PrecommittedGroupParams::from_params(
            pre_key,
            find_single_schedule(pre_key, &policy)
                .expect("precommit schedule")
                .steps
                .first()
                .and_then(|step| match step {
                    Step::Direct(direct) => direct.params.as_ref(),
                    Step::Fold(fold) => Some(&fold.params),
                })
                .expect("commit params"),
        );
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 2),
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
}
