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
    AkitaScheduleInputs, AkitaScheduleLookupKey, GroupBatchAkitaScheduleLookupKey, Schedule,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{
    table_entry, GeneratedScheduleKey, GeneratedScheduleTable, GeneratedScheduleTableEntry,
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
///
/// Phase 1 has no generated grouped entries yet, so catalog handling only
/// validates identity before delegating to the grouped DP fallback.
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
