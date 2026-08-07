//! Public validation for generated schedule rows.
//!
//! Delegates to the shared generated-schedule walkers; see
//! [`validate_generated_schedule_entry`].

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::AkitaScheduleLookupKey;

use crate::catalog_identity::validate_catalog_identity;
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{GeneratedFoldScheduleEntry, GeneratedScheduleTable};
use crate::PlannerPolicy;

/// Validate every generated row in a catalog against a public policy.
pub fn validate_generated_schedule_table(
    catalog: &GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    validate_catalog_identity(catalog, policy, ring_challenge_config)?;
    for entry in catalog.entries {
        let key = entry.to_runtime_lookup_key();
        validate_generated_schedule_entry(entry, &key, policy, ring_challenge_config)?;
    }
    for row in catalog.precommitted_profiles {
        row.expand_to_committed_profile(policy)?;
    }
    Ok(())
}

/// Validate one generated schedule row without running planner search.
pub fn validate_generated_schedule_entry(
    entry: &GeneratedFoldScheduleEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    walk_generated_schedule_entry(entry, key, policy, ring_challenge_config).map(|_| ())
}
