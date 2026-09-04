//! Strict runtime schedule resolution.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{
    root_input_witness_len, schedule_row_digest, validate_schedule_ring_dims,
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, FoldSchedule, GroupCommitPhaseParams,
    OpeningScheduleSelection,
};

use crate::audit::audit_resolved_schedule;
use crate::catalog_identity::{identity_digest, policy_digest, validate_catalog_identity};
use crate::generated::walk::walk_generated_schedule_entry;
use crate::generated::{table_entry_range, GeneratedFoldScheduleEntry, GeneratedScheduleTable};
use crate::runtime::planned_next_witness_len;
use crate::runtime::validate_policy;
use crate::PlannerPolicy;

const MAX_RESOLVED_CATALOG_ROWS: usize = 1 << 14;

static MATERIALIZED_CATALOGS: LazyLock<
    Mutex<HashMap<MaterializedCatalogCacheKey, Arc<MaterializedCatalog>>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MaterializedCatalogCacheKey {
    entries_ptr: usize,
    entries_len: usize,
    identity_digest: [u8; 32],
    policy_digest: [u8; 32],
}

struct MaterializedCatalog {
    rows_by_digest: Vec<ResolvedScheduleRow>,
    entry_row_digests: Vec<akita_types::ScheduleRowDigest>,
}

fn lock_materialized_catalogs() -> Result<
    std::sync::MutexGuard<'static, HashMap<MaterializedCatalogCacheKey, Arc<MaterializedCatalog>>>,
    AkitaError,
> {
    MATERIALIZED_CATALOGS.lock().map_err(|_| {
        AkitaError::InvalidSetup("materialized schedule catalog cache poisoned".to_string())
    })
}

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
    /// digest. The caller remains responsible for admitting the row from its
    /// configured catalog.
    pub fn try_new(
        selection: OpeningScheduleSelection,
        profiles: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
        policy: &PlannerPolicy,
    ) -> Result<Self, AkitaError> {
        audit_resolved_schedule(&profiles, &schedule, policy)?;
        validate_schedule_ring_dims(&schedule)?;
        validate_canonical_transition_lengths(&profiles, &schedule, policy)?;
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

    /// Check that opening claims have the exact layout authorized by this row.
    pub fn validate_opening_layout(
        &self,
        opening_batch: &akita_types::OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        if self.profiles.opening_layout()? != *opening_batch {
            return Err(AkitaError::InvalidInput(
                "committed-group descriptors do not match the opening layout".to_string(),
            ));
        }
        Ok(())
    }

    /// Consume the resolved row into its expanded schedule.
    pub fn into_schedule(self) -> FoldSchedule {
        self.schedule
    }
}

fn validate_canonical_transition_lengths(
    profiles: &CommittedGroupBatchProfile,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<(), AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    let root_params = &schedule.root.params;
    let expected_root_input = root_input_witness_len(root_params);
    if schedule.root.input_witness_len != expected_root_input {
        return Err(AkitaError::InvalidSetup(format!(
            "root input witness length {} is not canonical; expected {expected_root_input}",
            schedule.root.input_witness_len
        )));
    }
    let expected_root_output = if root_params.has_preceding_groups() {
        root_params.output_witness_len_for_field_bits(
            field_bits,
            policy.claim_ext_degree,
            &profiles.opening_layout()?,
        )?
    } else {
        planned_next_witness_len(
            field_bits,
            policy.claim_ext_degree,
            root_params,
            profiles.final_group.group.num_polynomials(),
            root_params.witness_chunk.num_chunks,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "root schedule uses unsupported compression source geometry".to_string(),
            )
        })?
    };
    if schedule.root.output_witness_len != expected_root_output {
        return Err(AkitaError::InvalidSetup(format!(
            "root output witness length {} is not canonical; expected {expected_root_output}",
            schedule.root.output_witness_len
        )));
    }

    let mut expected_input = expected_root_output;
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        if step.input_witness_len != expected_input {
            return Err(AkitaError::InvalidSetup(format!(
                "recursive fold {index} input witness length {} is not canonical; expected {expected_input}",
                step.input_witness_len
            )));
        }
        let expected_output = planned_next_witness_len(
            field_bits,
            policy.claim_ext_degree,
            &step.params,
            1,
            step.params.witness_chunk.num_chunks,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "recursive fold {index} uses unsupported compression source geometry"
            ))
        })?;
        if step.output_witness_len != expected_output {
            return Err(AkitaError::InvalidSetup(format!(
                "recursive fold {index} output witness length {} is not canonical; expected {expected_output}",
                step.output_witness_len
            )));
        }
        expected_input = expected_output;
    }
    if schedule.terminal.input_witness_len != expected_input {
        return Err(AkitaError::InvalidSetup(format!(
            "terminal input witness length {} is not canonical; expected {expected_input}",
            schedule.terminal.input_witness_len
        )));
    }
    Ok(())
}

fn profiles_for_entry(
    entry: &GeneratedFoldScheduleEntry,
    schedule: &FoldSchedule,
) -> Result<CommittedGroupBatchProfile, AkitaError> {
    Ok(CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            entry.final_group,
            &schedule.root.params,
        )?,
        precommitteds: entry
            .root
            .precommitted_groups
            .iter()
            .map(|group| group.group.profile)
            .collect(),
    })
}

fn materialize_catalog_rows_uncached(
    table: GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<MaterializedCatalog, AkitaError> {
    let mut rows = Vec::with_capacity(table.entries.len());
    let mut digests = Vec::with_capacity(table.entries.len());
    for entry in table.entries {
        let key = entry.to_runtime_lookup_key();
        let schedule = schedule_from_entry(entry, &key, policy, ring_challenge_config)?;
        let profiles = profiles_for_entry(entry, &schedule)?;
        let row_digest = schedule_row_digest(&profiles, &schedule)?;
        digests.push(row_digest);
        rows.push((row_digest, profiles, schedule));
    }
    let mut resolved = rows
        .into_iter()
        .map(|(row_digest, profiles, schedule)| {
            ResolvedScheduleRow::try_new(
                OpeningScheduleSelection { row_digest },
                profiles,
                schedule,
                policy,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    resolved.sort_by_key(|row| row.selection.row_digest);
    if resolved
        .windows(2)
        .any(|pair| pair[0].selection.row_digest == pair[1].selection.row_digest)
    {
        return Err(AkitaError::InvalidSetup(
            "schedule catalog contains duplicate full-row identities".to_string(),
        ));
    }
    Ok(MaterializedCatalog {
        rows_by_digest: resolved,
        entry_row_digests: digests,
    })
}

fn materialized_catalog(
    table: GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<Arc<MaterializedCatalog>, AkitaError> {
    if table.entries.is_empty() || table.entries.len() > MAX_RESOLVED_CATALOG_ROWS {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule catalog row count {} is outside 1..={MAX_RESOLVED_CATALOG_ROWS}",
            table.entries.len()
        )));
    }

    // This call is intentionally outside the materialization cache. The
    // catalog validator cheaply rechecks runtime hooks on its own cache hits,
    // so a caller cannot reuse rows under different hook behavior.
    validate_catalog_identity(&table, policy, ring_challenge_config)?;
    let cache_key = MaterializedCatalogCacheKey {
        entries_ptr: table.entries.as_ptr() as usize,
        entries_len: table.entries.len(),
        identity_digest: identity_digest(&table.identity),
        policy_digest: policy_digest(policy),
    };
    if let Some(cached) = lock_materialized_catalogs()?.get(&cache_key).cloned() {
        return Ok(cached);
    }

    let materialized = Arc::new(materialize_catalog_rows_uncached(
        table,
        policy,
        ring_challenge_config,
    )?);
    let mut cache = lock_materialized_catalogs()?;
    Ok(cache
        .entry(cache_key)
        .or_insert_with(|| Arc::clone(&materialized))
        .clone())
}

/// Resolve an explicit public schedule selection without planner/key search.
///
/// Row identities are recomputed from exact expanded rows. The final lookup is
/// a bounded binary search over fixed-width digests in the configured catalog.
///
pub fn resolve_generated_schedule_selection(
    selection: OpeningScheduleSelection,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<ResolvedScheduleRow, AkitaError> {
    validate_policy(policy)?;
    let table = catalog.ok_or_else(|| {
        AkitaError::UnsupportedSchedule("schedule catalog is not enabled".to_string())
    })?;
    let catalog = materialized_catalog(table, policy, &ring_challenge_config)?;
    let index = catalog
        .rows_by_digest
        .binary_search_by_key(&selection.row_digest, |row| row.selection.row_digest)
        .map_err(|_| {
            AkitaError::UnsupportedSchedule(
                "selected schedule row is not present in the configured catalog".to_string(),
            )
        })?;
    catalog.rows_by_digest.get(index).cloned().ok_or_else(|| {
        AkitaError::InvalidSetup("resolved schedule row index is out of bounds".to_string())
    })
}

fn resolve_generated_catalog_row_matching(
    key: &AkitaScheduleLookupKey,
    exact_profiles: Option<&CommittedGroupBatchProfile>,
    policy: &PlannerPolicy,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    table: GeneratedScheduleTable,
) -> Result<ResolvedScheduleRow, AkitaError> {
    let catalog = materialized_catalog(table, policy, ring_challenge_config)?;
    let candidate_range = table_entry_range(table, key);
    if candidate_range.is_empty() {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no generated schedule row for request {:?}",
            key
        )));
    }
    let selected_digest = candidate_range
        .map(|index| {
            catalog
                .entry_row_digests
                .get(index)
                .copied()
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "generated schedule entry index is out of bounds".to_string(),
                    )
                })
        })
        .filter_map(|digest| match digest {
            Ok(digest) => {
                let row = catalog
                    .rows_by_digest
                    .binary_search_by_key(&digest, |row| row.selection.row_digest)
                    .ok()
                    .and_then(|index| catalog.rows_by_digest.get(index));
                match row {
                    Some(row)
                        if exact_profiles.is_none_or(|profiles| row.profiles() == profiles) =>
                    {
                        Some(Ok(digest))
                    }
                    Some(_) => None,
                    None => Some(Err(AkitaError::InvalidSetup(
                        "generated row is missing from its materialized catalog".to_string(),
                    ))),
                }
            }
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(if exact_profiles.is_some() {
                "no generated schedule row matches the exact committed profiles".to_string()
            } else {
                format!("no generated schedule row for request {:?}", key)
            })
        })?;
    let index = catalog
        .rows_by_digest
        .binary_search_by_key(&selected_digest, |row| row.selection.row_digest)
        .map_err(|_| {
            AkitaError::InvalidSetup(
                "selected generated row is missing from its resolved catalog".to_string(),
            )
        })?;
    catalog.rows_by_digest.get(index).cloned().ok_or_else(|| {
        AkitaError::InvalidSetup("selected schedule row index is out of bounds".to_string())
    })
}

/// Resolve the generated row matching a runtime catalog lookup request.
///
/// This is the pre-commit counterpart of
/// [`resolve_generated_schedule_selection`]: it returns the same resolved
/// handle so the caller can retain its public selection for proving.
pub fn resolve_generated_catalog_row_for_key(
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
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
    resolve_generated_catalog_row_matching(key, None, policy, &ring_challenge_config, table)
}

/// Resolve the canonical generated row matching exact committed profiles.
pub fn resolve_generated_catalog_row_for_profiles(
    key: &AkitaScheduleLookupKey,
    profiles: &CommittedGroupBatchProfile,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    catalog: Option<GeneratedScheduleTable>,
) -> Result<ResolvedScheduleRow, AkitaError> {
    key.validate(policy.decomposition.field_bits())?;
    profiles.validate(policy.decomposition.field_bits())?;
    validate_policy(policy)?;
    let table = catalog.ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "schedule catalog is not enabled for request {:?}",
            key
        ))
    })?;
    resolve_generated_catalog_row_matching(
        key,
        Some(profiles),
        policy,
        &ring_challenge_config,
        table,
    )
}

/// Build the runtime [`FoldSchedule`] for a compact generated entry.
pub fn schedule_from_entry(
    entry: &GeneratedFoldScheduleEntry,
    key: &AkitaScheduleLookupKey,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<FoldSchedule, AkitaError> {
    let schedule = walk_generated_schedule_entry(entry, key, policy, &ring_challenge_config)?
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
) -> Result<usize, AkitaError> {
    walk_generated_schedule_entry(entry, key, policy, &ring_challenge_config)?
        .planned_schedule
        .estimate
        .estimated_proof_payload_bytes()
}
