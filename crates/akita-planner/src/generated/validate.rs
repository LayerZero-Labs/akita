//! Public validation for generated schedule rows.
//!
//! Delegates to the shared generated-schedule walkers; see
//! [`validate_generated_schedule_entry`] and
//! [`validate_generated_group_batch_schedule_entry`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, CommitmentGroupScheduleKey};

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::walk::{
    walk_generated_group_batch_schedule_entry, walk_generated_schedule_entry,
};
use crate::generated::{
    runtime_key_from_generated, GeneratedGroupBatchScheduleTableEntry, GeneratedScheduleTable,
    GeneratedScheduleTableEntry,
};
use crate::PlannerPolicy;

/// Validate every generated row in a catalog against a public policy.
pub fn validate_generated_schedule_table(
    catalog: &GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    validate_catalog_identity(
        catalog,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;
    for entry in catalog.entries {
        let key = CommitmentGroupScheduleKey::new(entry.key.num_vars, entry.key.num_polynomials);
        validate_generated_schedule_entry(
            entry,
            key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )?;
    }
    for entry in catalog.group_batch_entries {
        let key = runtime_key_from_generated(&entry.key);
        validate_generated_group_batch_schedule_entry(
            entry,
            &key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )?;
    }
    Ok(())
}

/// Validate one generated schedule row without running planner search.
pub fn validate_generated_schedule_entry(
    entry: &GeneratedScheduleTableEntry,
    key: CommitmentGroupScheduleKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    walk_generated_schedule_entry(
        entry,
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
    .map(|_| ())
}

/// Validate one generated grouped-root schedule row without running planner search.
pub fn validate_generated_group_batch_schedule_entry(
    entry: &GeneratedGroupBatchScheduleTableEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    walk_generated_group_batch_schedule_entry(
        entry,
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
    .map(|_| ())
}
