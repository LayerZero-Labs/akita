//! FoldSchedule resolution: catalog validation, cache-then-generate.
//!
//! [`resolve_schedule`] is the single entry point the runtime (config,
//! prover, verifier) uses to obtain a [`FoldSchedule`] for a lookup key. When a
//! preset supplies a catalog, identity is validated and the compact entry
//! is expanded via [`schedule_from_entry`]; on a miss (or no catalog) the
//! schedule is regenerated with the offline key-shaped DP search
//! [`crate::find_group_batch_schedule`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, FoldSchedule, PolynomialGroupLayout,
};

use crate::catalog_identity::validate_catalog_identity;
use crate::find_group_batch_schedule;
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{table_entry, GeneratedFoldScheduleEntry, GeneratedScheduleTable};
use crate::schedule_params::{
    find_schedule_prioritizing_first_direct_setup, validate_policy_witness_chunk,
};
use crate::PlannerPolicy;

/// Resolve the runtime [`FoldSchedule`] using an explicit optional catalog.
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

/// Resolve a multi-group-root schedule without falling back to a scalar table key.
pub fn resolve_group_batch_schedule(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<FoldSchedule, AkitaError> {
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
        let scalar_policy = policy.direct_only();
        let planned = find_schedule_prioritizing_first_direct_setup(
            key.final_group,
            &scalar_policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )?;
        planned.schedule.validate_structure()?;
        return Ok(planned.schedule);
    }
    let planned = find_group_batch_schedule(
        key,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}

fn unsupported_grouped_table_hit_error(err: &AkitaError) -> bool {
    let msg = err.to_string();
    msg.contains("grouped terminal fold must be followed by another fold")
        || msg.contains("terminal fold must be scalar")
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
