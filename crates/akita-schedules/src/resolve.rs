//! Strict runtime schedule resolution.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, FoldSchedule, PolynomialGroupLayout,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{table_entry, GeneratedFoldScheduleEntry, GeneratedScheduleTable};
use crate::runtime::validate_policy;
use crate::PlannerPolicy;

/// Resolve a runtime schedule using only the enabled generated catalog.
///
/// A missing catalog or missing row is unsupported input. This function never
/// invokes planner search.
pub fn resolve_group_batch_schedule(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<FoldSchedule, AkitaError> {
    key.validate()?;
    validate_policy(policy)?;
    let table = catalog.ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "schedule catalog is not enabled for request {:?}",
            key
        ))
    })?;
    validate_catalog_identity(
        &table,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?;
    let entry = table_entry(table, key).ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!("no generated schedule row for request {:?}", key))
    })?;
    schedule_from_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )
}

/// Resolve a scalar-root runtime schedule using only the enabled generated catalog.
pub fn resolve_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<FoldSchedule, AkitaError> {
    resolve_group_batch_schedule(
        &AkitaScheduleLookupKey::single(key),
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
        catalog,
    )
}

/// Build the runtime [`FoldSchedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedFoldScheduleEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<FoldSchedule, AkitaError> {
    let schedule = walk_generated_schedule_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?
    .planned_schedule
    .schedule;
    schedule.validate_structure()?;
    Ok(schedule)
}

/// Estimate proof bytes for a generated row without planner search.
pub fn estimate_proof_bytes(
    entry: &GeneratedFoldScheduleEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<usize, AkitaError> {
    walk_generated_schedule_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?
    .planned_schedule
    .estimate
    .estimated_proof_payload_bytes()
}
