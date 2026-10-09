//! Extension seam: canonical offline catalog reports for LayerZero-Labs/akita#45.
//!
//! Artifact generators share the snapshot format, revision comparison and
//! materialized-row reporting through this `catalog-gen` API. Calculations and
//! formatting remain owned here; command-line and output handling stay with the
//! calling generator.

use crate::EmitSpec;
use akita_params::{
    schedule_row_digest, CommittedGroupBatchProfile, FoldSchedule, GroupCommitPhaseParams,
    ScheduleLookupKey,
};

mod policy;
mod snapshot;

use policy::catalog_policy_signature;
pub use snapshot::{
    compare_snapshots, parse_snapshot, write_snapshot, CatalogRevisionComparison,
    CatalogSnapshotRow,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogRowMetrics {
    setup_fields: usize,
    first_direct_setup_capacity: Option<usize>,
    proof_bytes: usize,
    fold_levels: usize,
    row_digest: String,
    policy_signature: String,
}

fn row_digest_hex(key: &ScheduleLookupKey, schedule: &FoldSchedule) -> Result<String, String> {
    let final_group =
        GroupCommitPhaseParams::try_from_params(key.final_group, &schedule.root.params)
            .map_err(|error| format!("derive final committed profile: {error}"))?;
    let profiles = CommittedGroupBatchProfile {
        final_group,
        precommitteds: key.precommitteds.clone(),
    };
    let digest = schedule_row_digest(&profiles, schedule)
        .map_err(|error| format!("derive schedule row digest: {error}"))?;
    Ok(digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn catalog_row_metrics(
    spec: &EmitSpec,
    key: &ScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<CatalogRowMetrics, String> {
    let proof_bytes =
        akita_schedules::expanded_schedule_proof_estimate_bytes(key, schedule, &spec.policy)
            .map_err(|error| format!("estimate proof payload: {error}"))?;
    let setup_fields = akita_params::setup_matrix_capacity_for_schedule(schedule)
        .map_err(|error| format!("estimate setup capacity: {error}"))?
        .num_field_elements;
    let first_direct_setup_capacity = Some(
        akita_schedules::planner_support::first_direct_setup_capacity_for_schedule(
            schedule,
            &key.opening_layout()
                .map_err(|error| format!("derive opening layout: {error}"))?,
        )
        .map_err(|error| format!("estimate first direct setup capacity: {error}"))?,
    );
    Ok(CatalogRowMetrics {
        setup_fields,
        first_direct_setup_capacity,
        proof_bytes,
        fold_levels: schedule.num_fold_levels(),
        row_digest: row_digest_hex(key, schedule)?,
        policy_signature: catalog_policy_signature(spec, schedule)?,
    })
}

fn catalog_logical_key(key: &ScheduleLookupKey) -> String {
    use std::fmt::Write as _;

    let mut logical = format!(
        "final={}:{};precommitted=",
        key.final_group.num_vars(),
        key.final_group.num_polynomials(),
    );
    for (index, precommitted) in key.precommitteds.iter().enumerate() {
        if index != 0 {
            logical.push(',');
        }
        write!(
            logical,
            "{}:{}",
            precommitted.group.num_vars(),
            precommitted.group.num_polynomials(),
        )
        .expect("writing to String cannot fail");
    }
    logical
}

fn catalog_lookup_key_digest(key: &ScheduleLookupKey) -> String {
    akita_params::digest_descriptor_bytes(&key.canonical_descriptor_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn catalog_snapshot_row(
    spec: &EmitSpec,
    key: &ScheduleLookupKey,
    schedule: &FoldSchedule,
    logical_key: String,
) -> Result<snapshot::CatalogSnapshotRow, String> {
    let metrics = catalog_row_metrics(spec, key, schedule)?;
    Ok(snapshot::CatalogSnapshotRow {
        schema: snapshot::SnapshotSchema::Current,
        family: spec.family_name.to_string(),
        logical_key,
        lookup_key_digest: catalog_lookup_key_digest(key),
        setup_fields: metrics.setup_fields,
        first_direct_setup_capacity: metrics.first_direct_setup_capacity,
        proof_bytes: metrics.proof_bytes,
        fold_levels: metrics.fold_levels,
        row_digest: metrics.row_digest,
        policy: metrics.policy_signature,
    })
}

/// Derive canonical snapshot evidence from materialized generation requests.
///
/// The lookup digest retains each exact producer descriptor; the displayed key
/// adds producer contracts when equal displayed geometries need disambiguation.
pub fn materialized_snapshot_rows(
    spec: &EmitSpec,
    entries: &[crate::emit::MaterializedEntry],
) -> Result<Vec<snapshot::CatalogSnapshotRow>, String> {
    let mut logical_key_counts = std::collections::BTreeMap::new();
    for entry in entries {
        *logical_key_counts
            .entry(catalog_logical_key(&entry.key()))
            .or_insert(0usize) += 1;
    }
    entries
        .iter()
        .map(|entry| {
            let key = entry.key();
            let mut logical_key = catalog_logical_key(&key);
            let is_ambiguous = logical_key_counts
                .get(&logical_key)
                .copied()
                .unwrap_or_default()
                > 1;
            let producers_match_family = entry
                .precommitted_producers()
                .iter()
                .all(|producer| producer.source_contract() == spec.source_contract);
            if is_ambiguous && !producers_match_family {
                use std::fmt::Write as _;
                logical_key.push_str(";producer_contracts=");
                for (index, producer) in entry.precommitted_producers().iter().enumerate() {
                    if index != 0 {
                        logical_key.push(',');
                    }
                    let contract = producer.source_contract();
                    match contract.class() {
                        akita_params::sis::CommittedSourceClass::UnitOneHot {
                            source_chunk_size,
                        } => write!(
                            logical_key,
                            "onehot(chunk={source_chunk_size},bound={})",
                            contract.decomposition().log_commit_bound,
                        ),
                        akita_params::sis::CommittedSourceClass::BalancedSignedDigit => write!(
                            logical_key,
                            "balanced(bound={})",
                            contract.decomposition().log_commit_bound,
                        ),
                    }
                    .map_err(|error| format!("write producer contract key: {error}"))?;
                }
            }
            catalog_snapshot_row(spec, &key, entry.schedule(), logical_key)
        })
        .collect()
}
