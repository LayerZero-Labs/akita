//! Per-set planning: catalog, limits, profiles, and sizes, without
//! committing to anything.

use crate::description::SetDescription;
use akita_config::preprocessing::set_record_code_fields;
use akita_config::{
    required_setup_prefix_slot_ids_for_schedule, CommitmentConfig, SetupRequirements,
    TrustedScheduleCatalog,
};
use akita_error::AkitaError;
use akita_params::sis::SisModulusProfileId;
use akita_params::{
    verifier_setup_matrix_capacity_for_schedule, ChunkedWitnessCfg, FoldSchedule,
    ScheduleRowDigest, SetupPrefixSlotId,
};
use akita_serialization::{AkitaDeserialize, Valid};
use akita_types::preprocessing::{parse_hex, profile_key, ProfileKey, SetRecord, StreamIdentity};
use akita_types::AkitaSetupSeed;
use std::collections::BTreeMap;
use std::path::Path;

/// Everything generation needs to know about one set.
pub(crate) struct SetPlan {
    pub(crate) record: SetRecord,
    pub(crate) stream: StreamIdentity,
    pub(crate) field: SisModulusProfileId,
    /// A representative slot id for every profile the set needs.
    pub(crate) profiles: BTreeMap<ProfileKey, SetupPrefixSlotId>,
    /// Field elements a prover of the set derives.
    pub(crate) prover_capacity: usize,
    pub(crate) catalog_bytes: Vec<u8>,
    /// Rows within the limits, for guest files.
    pub(crate) rows: Vec<RowPlan>,
}

/// One admitted row, for guest files.
pub(crate) struct RowPlan {
    pub(crate) digest: ScheduleRowDigest,
    pub(crate) view: Vec<u8>,
    pub(crate) schedule: FoldSchedule,
    pub(crate) verifier_capacity: usize,
}

/// Plan one set under the compiled config `Cfg`.
pub(crate) fn plan_set<Cfg: CommitmentConfig>(
    repo_root: &Path,
    set: &SetDescription,
    setup_seed: &AkitaSetupSeed,
) -> Result<SetPlan, AkitaError>
where
    Cfg::Field: Valid + AkitaDeserialize<Context = ()>,
{
    let path = repo_root.join(&set.catalog);
    let catalog_bytes = std::fs::read(&path).map_err(|error| {
        AkitaError::InvalidInput(format!("cannot read catalog {}: {error}", path.display()))
    })?;
    let chunking = ChunkedWitnessCfg {
        num_chunks: set.num_chunks,
        num_activated_levels: set.num_activated_levels,
    };
    let catalog = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&catalog_bytes, chunking)?;
    let rows = catalog.rows_within_setup_capacity(set.max_num_vars, set.max_num_batched_polys)?;
    if rows.is_empty() {
        return Err(AkitaError::InvalidInput(format!(
            "set {:?} admits no row of its catalog",
            set.name
        )));
    }
    let mut profiles = BTreeMap::new();
    for row in &rows {
        for id in required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            &row.profiles().opening_layout()?,
        )? {
            insert_profile(&mut profiles, id);
        }
    }
    let catalog_profiles = profiles.keys().copied().collect::<Vec<_>>();
    let universe = match &set.program_rows {
        None => Vec::new(),
        Some(program_rows) => {
            for file in &program_rows.universe_slots {
                for id in read_slot_ids(&repo_root.join(file))? {
                    insert_profile(&mut profiles, id);
                }
            }
            profiles.keys().copied().collect()
        }
    };
    let prover_capacity =
        SetupRequirements::from_catalog(&catalog, set.max_num_vars, set.max_num_batched_polys)?
            .matrix_capacity()
            .num_field_elements;
    let row_plans = rows
        .iter()
        .map(|row| {
            Ok(RowPlan {
                digest: row.selection().row_digest,
                view: catalog.to_verifier_view(&[row.selection()])?,
                schedule: row.schedule().clone(),
                verifier_capacity: verifier_setup_matrix_capacity_for_schedule(
                    row.schedule(),
                    &row.profiles().opening_layout()?,
                )?
                .num_field_elements,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let code = set_record_code_fields::<Cfg>();
    let stream = StreamIdentity::for_field::<Cfg::Field>(setup_seed)?;
    let record = SetRecord {
        name: set.name.clone(),
        config_family: code.config_family,
        decomposition: code.decomposition,
        sis_modulus_profile_tag: code.sis_modulus_profile_tag,
        sis_table_digest: code.sis_table_digest,
        sis_l2_table_digest: code.sis_l2_table_digest,
        algorithm_identity: code.algorithm_identity,
        stream_id: stream.stream_id(),
        num_chunks: set.num_chunks as u64,
        num_activated_levels: set.num_activated_levels as u64,
        max_num_vars: set.max_num_vars as u64,
        max_num_batched_polys: set.max_num_batched_polys as u64,
        catalog_digest: catalog.catalog_digest(),
        catalog_profiles,
        admits_program_rows: set.program_rows.is_some(),
        universe,
    };
    Ok(SetPlan {
        record,
        stream,
        field: Cfg::sis_modulus_profile(),
        profiles,
        prover_capacity,
        catalog_bytes,
        rows: row_plans,
    })
}

/// Keep one slot id per profile: the smallest, so the choice is canonical.
fn insert_profile(profiles: &mut BTreeMap<ProfileKey, SetupPrefixSlotId>, id: SetupPrefixSlotId) {
    profiles
        .entry(profile_key(&id.commitment_profile))
        .and_modify(|kept| {
            if id < *kept {
                *kept = id.clone();
            }
        })
        .or_insert(id);
}

/// Read a universe file: one hex-encoded uncompressed slot id per line.
pub(crate) fn read_slot_ids(path: &Path) -> Result<Vec<SetupPrefixSlotId>, AkitaError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        AkitaError::InvalidInput(format!("cannot read {}: {error}", path.display()))
    })?;
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let id = SetupPrefixSlotId::deserialize_uncompressed_exact(&parse_hex(line)?, &())
                .map_err(|error| {
                    AkitaError::InvalidInput(format!(
                        "invalid slot id in {}: {error}",
                        path.display()
                    ))
                })?;
            id.check()
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            Ok(id)
        })
        .collect()
}
