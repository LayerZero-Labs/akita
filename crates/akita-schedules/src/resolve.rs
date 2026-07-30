//! Strict runtime schedule resolution.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    schedule_row_digest, AkitaScheduleInputs, AkitaScheduleLookupKey, CommittedGroupBatchProfile,
    CommittedGroupProfile, FoldSchedule, OpeningScheduleSelection, PolynomialGroupLayout,
};

use crate::catalog_identity::{selection_catalog_identity, validate_catalog_identity};
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{table_entry, GeneratedFoldScheduleEntry, GeneratedScheduleTable};
use crate::runtime::validate_policy;
use crate::PlannerPolicy;

const MAX_RESOLVED_CATALOG_ROWS: usize = 1 << 14;

/// One generated row resolved to the exact verifier schedule and public identity.
#[derive(Clone, Debug)]
pub struct ResolvedScheduleRow {
    selection: OpeningScheduleSelection,
    profiles: CommittedGroupBatchProfile,
    schedule: FoldSchedule,
}

impl ResolvedScheduleRow {
    /// Construct a row already authorized by a configuration-owned catalog.
    ///
    /// This validates the exact committed profiles, expanded schedule, and row
    /// digest. The caller remains responsible for authorizing the catalog
    /// identity; generated configurations do that through their catalog
    /// identity check before calling this constructor.
    pub fn try_new(
        selection: OpeningScheduleSelection,
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        field_bits: u32,
    ) -> Result<Self, AkitaError> {
        profiles.validate(field_bits)?;
        schedule.validate_structure()?;
        if schedule_row_digest(&profiles, &schedule)? != selection.row_digest {
            return Err(AkitaError::InvalidSetup(
                "schedule row digest does not match the supplied profiles and schedule".to_string(),
            ));
        }
        Ok(Self {
            selection,
            profiles,
            schedule,
        })
    }

    /// Batch-level public schedule selection.
    pub const fn selection(&self) -> OpeningScheduleSelection {
        self.selection
    }

    /// Exact ordered committed profiles accepted by this row.
    pub fn profiles(&self) -> &CommittedGroupBatchProfile {
        &self.profiles
    }

    /// Exact expanded schedule consumed by proving and verification.
    pub fn schedule(&self) -> &FoldSchedule {
        &self.schedule
    }

    /// Consume the resolved row into its expanded schedule.
    pub fn into_schedule(self) -> FoldSchedule {
        self.schedule
    }
}

fn profiles_for_entry(
    entry: &GeneratedFoldScheduleEntry,
    schedule: &FoldSchedule,
) -> CommittedGroupBatchProfile {
    CommittedGroupBatchProfile {
        final_group: CommittedGroupProfile::from_params(
            entry.root.final_group.layout,
            &schedule.root.params.final_group.commitment,
        ),
        precommitteds: entry
            .root
            .precommitted_groups
            .iter()
            .map(|group| group.descriptor)
            .collect(),
    }
}

fn materialize_catalog_rows(
    table: GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<Vec<ResolvedScheduleRow>, AkitaError> {
    if table.entries.is_empty() || table.entries.len() > MAX_RESOLVED_CATALOG_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule catalog row count {} is outside 1..={MAX_RESOLVED_CATALOG_ROWS}",
            table.entries.len()
        )));
    }
    validate_catalog_identity(
        &table,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;

    let mut rows = Vec::with_capacity(table.entries.len());
    let mut digests = Vec::with_capacity(table.entries.len());
    for entry in table.entries {
        let key = entry.to_runtime_lookup_key();
        let schedule = schedule_from_entry(
            entry,
            &key,
            policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )?;
        let profiles = profiles_for_entry(entry, &schedule);
        let row_digest = schedule_row_digest(&profiles, &schedule)?;
        digests.push(row_digest);
        rows.push((row_digest, profiles, schedule));
    }
    let catalog_identity = selection_catalog_identity(&table.identity, &digests)?;
    let mut resolved = rows
        .into_iter()
        .map(|(row_digest, profiles, schedule)| {
            ResolvedScheduleRow::try_new(
                OpeningScheduleSelection {
                    catalog_identity,
                    row_digest,
                },
                profiles,
                schedule,
                policy.decomposition.field_bits(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    resolved.sort_by_key(|row| row.selection.row_digest);
    Ok(resolved)
}

/// Resolve an explicit public schedule selection without planner/key search.
///
/// Catalog and row identities are recomputed from exact expanded rows. The
/// final lookup is a bounded binary search over fixed-width digests.
///
/// This is a transitional resolver: generated artifacts do not yet embed a
/// digest-sorted row index, so each call materializes at most
/// [`MAX_RESOLVED_CATALOG_ROWS`] entries before lookup. The intended generated
/// form emits the exact row digests and resolves directly without this
/// verifier-side catalog walk.
pub fn resolve_generated_schedule_selection(
    selection: OpeningScheduleSelection,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<ResolvedScheduleRow, AkitaError> {
    validate_policy(policy)?;
    let table = catalog.ok_or_else(|| {
        AkitaError::UnsupportedSchedule("schedule catalog is not enabled".to_string())
    })?;
    let rows = materialize_catalog_rows(
        table,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?;
    let actual_catalog_identity = rows
        .first()
        .map(|row| row.selection.catalog_identity)
        .ok_or_else(|| AkitaError::InvalidSetup("empty validated schedule catalog".to_string()))?;
    if selection.catalog_identity != actual_catalog_identity {
        return Err(AkitaError::UnsupportedSchedule(
            "schedule catalog identity is not enabled by this configuration".to_string(),
        ));
    }
    let index = rows
        .binary_search_by_key(&selection.row_digest, |row| row.selection.row_digest)
        .map_err(|_| {
            AkitaError::UnsupportedSchedule(
                "selected schedule row is not present in the configured catalog".to_string(),
            )
        })?;
    rows.get(index).cloned().ok_or_else(|| {
        AkitaError::InvalidSetup("resolved schedule row index is out of bounds".to_string())
    })
}

/// Select the generated row matching a prover/planner lookup request.
///
/// This is the pre-commit counterpart of
/// [`resolve_generated_schedule_selection`]: it returns the same resolved
/// handle so the caller can retain its public selection for proving.
pub fn select_generated_schedule_row(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<ResolvedScheduleRow, AkitaError> {
    key.validate(policy.decomposition.field_bits())?;
    validate_policy(policy)?;
    let table = catalog.ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "schedule catalog is not enabled for request {:?}",
            key
        ))
    })?;
    let entry = table_entry(table, key).ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!("no generated schedule row for request {:?}", key))
    })?;
    let selected_schedule = schedule_from_entry(
        entry,
        key,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?;
    let selected_profiles = profiles_for_entry(entry, &selected_schedule);
    let selected_digest = schedule_row_digest(&selected_profiles, &selected_schedule)?;
    let rows = materialize_catalog_rows(
        table,
        policy,
        &ring_challenge_config,
        &fold_challenge_shape_at_level,
    )?;
    let index = rows
        .binary_search_by_key(&selected_digest, |row| row.selection.row_digest)
        .map_err(|_| {
            AkitaError::InvalidSetup(
                "selected generated row is missing from its resolved catalog".to_string(),
            )
        })?;
    rows.get(index).cloned().ok_or_else(|| {
        AkitaError::InvalidSetup("selected schedule row index is out of bounds".to_string())
    })
}

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
    key.validate(policy.decomposition.field_bits())?;
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
        &AkitaScheduleLookupKey::single(
            key,
            akita_types::GroupSource::from_config(policy.decomposition, policy.onehot_chunk_size),
        ),
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
