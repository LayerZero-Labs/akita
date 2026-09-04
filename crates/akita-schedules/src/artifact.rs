//! Versioned trusted JSON schedule catalog artifacts.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::instance_descriptor::{
    digest_descriptor_bytes, AKITA_INSTANCE_DESCRIPTOR_VERSION,
};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, FoldSchedule, OpeningScheduleSelection,
};
use serde::{Deserialize, Serialize};

use crate::policy_digest::policy_digest;
use crate::resolve::ResolvedScheduleRow;
use crate::traversal::{visit_schedule_groups, ScheduleGroup, ScheduleGroupPosition};
use crate::PlannerPolicy;

const ARTIFACT_MAGIC: [u8; 8] = *b"AKSCHD01";
const ARTIFACT_VERSION: u32 = 1;
const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAX_FAMILY_NAME_BYTES: usize = 128;
pub(crate) const MAX_TRUSTED_CATALOG_ROWS: usize = 1 << 14;

#[derive(Debug, Serialize, Deserialize)]
struct ScheduleCatalogArtifactV1 {
    magic: [u8; 8],
    version: u32,
    protocol_epoch: u32,
    policy_digest: [u8; 32],
    family_name: String,
    rows: Vec<ScheduleCatalogArtifactRowV1>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScheduleCatalogArtifactRowV1 {
    profiles: CommittedGroupBatchProfile,
    schedule: FoldSchedule,
}

/// An owned, validated schedule catalog supplied through a trusted parameter path.
///
/// Proofs carry only an [`OpeningScheduleSelection`]. Both the honest prover
/// lookup and verifier digest lookup resolve through this same object.
#[derive(Clone, Debug)]
pub struct TrustedScheduleCatalog {
    family_name: String,
    policy_digest: [u8; 32],
    catalog_digest: [u8; 32],
    rows_by_digest: Vec<ResolvedScheduleRow>,
    rows_by_key: Vec<(AkitaScheduleLookupKey, usize)>,
}

impl TrustedScheduleCatalog {
    /// Build a catalog from expanded rows and validate every verifier consumed field.
    pub fn try_new(
        family_name: impl Into<String>,
        rows: impl IntoIterator<Item = (CommittedGroupBatchProfile, FoldSchedule)>,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        let family_name = family_name.into();
        validate_family_name(&family_name)?;
        let rows = rows.into_iter().collect::<Vec<_>>();
        if rows.is_empty() || rows.len() > MAX_TRUSTED_CATALOG_ROWS {
            return Err(AkitaError::InvalidSetup(format!(
                "trusted schedule catalog row count {} is outside 1..={MAX_TRUSTED_CATALOG_ROWS}",
                rows.len()
            )));
        }

        let mut resolved = Vec::with_capacity(rows.len());
        for (profiles, schedule) in rows {
            let row = ResolvedScheduleRow::try_new(profiles, schedule, policy)?;
            validate_schedule_challenge_hooks(row.schedule(), &ring_challenge_config)?;
            resolved.push(row);
        }
        resolved.sort_by_key(|row| row.selection().row_digest);
        let mut rows_by_key = resolved
            .iter()
            .enumerate()
            .map(|(index, row)| (key_for_profiles(row.profiles()), index))
            .collect::<Vec<_>>();
        rows_by_key.sort_by(|(left_key, left_index), (right_key, right_index)| {
            left_key.canonical_cmp(right_key).then_with(|| {
                resolved[*left_index]
                    .selection()
                    .row_digest
                    .cmp(&resolved[*right_index].selection().row_digest)
            })
        });
        let has_duplicate_lookup_key = rows_by_key
            .windows(2)
            .any(|pair| pair[0].0.canonical_cmp(&pair[1].0).is_eq());
        if has_duplicate_lookup_key {
            return Err(AkitaError::InvalidSetup(
                "trusted schedule catalog contains a duplicate prover lookup key".to_string(),
            ));
        }
        if resolved
            .windows(2)
            .any(|pair| pair[0].selection() == pair[1].selection())
        {
            return Err(AkitaError::InvalidSetup(
                "trusted schedule catalog contains duplicate row identities".to_string(),
            ));
        }

        let policy_digest = policy_digest(policy);
        let catalog_digest = catalog_digest(&family_name, policy_digest, &resolved);
        Ok(Self {
            family_name,
            policy_digest,
            catalog_digest,
            rows_by_digest: resolved,
            rows_by_key,
        })
    }

    /// Decode and validate one complete trusted catalog artifact.
    pub fn from_artifact_bytes(
        bytes: &[u8],
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        if bytes.is_empty() || bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact byte length {} is outside 1..={MAX_ARTIFACT_BYTES}",
                bytes.len()
            )));
        }
        let artifact: ScheduleCatalogArtifactV1 =
            serde_json::from_slice(bytes).map_err(|error| {
                AkitaError::InvalidSetup(format!("invalid schedule artifact encoding: {error}"))
            })?;
        let canonical = encode_artifact(&artifact)?;
        if canonical != bytes {
            return Err(AkitaError::InvalidSetup(
                "schedule artifact is not in canonical JSON form".to_string(),
            ));
        }
        if artifact.magic != ARTIFACT_MAGIC || artifact.version != ARTIFACT_VERSION {
            return Err(AkitaError::InvalidSetup(
                "unsupported schedule artifact format".to_string(),
            ));
        }
        if artifact.protocol_epoch != AKITA_INSTANCE_DESCRIPTOR_VERSION {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact protocol epoch {} does not match runtime epoch {}",
                artifact.protocol_epoch, AKITA_INSTANCE_DESCRIPTOR_VERSION
            )));
        }
        if artifact.family_name != expected_family_name {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact family {:?} does not match trusted family {:?}",
                artifact.family_name, expected_family_name
            )));
        }
        if artifact.policy_digest != policy_digest(policy) {
            return Err(AkitaError::InvalidSetup(
                "schedule artifact policy does not match the runtime config".to_string(),
            ));
        }
        let catalog = Self::try_new(
            artifact.family_name,
            artifact
                .rows
                .into_iter()
                .map(|row| (row.profiles, row.schedule)),
            policy,
            ring_challenge_config,
        )?;
        if catalog.to_artifact_bytes()? != bytes {
            return Err(AkitaError::InvalidSetup(
                "schedule artifact rows are not in canonical digest order".to_string(),
            ));
        }
        Ok(catalog)
    }

    /// Encode this validated catalog as the canonical versioned artifact.
    pub fn to_artifact_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let artifact = ScheduleCatalogArtifactV1 {
            magic: ARTIFACT_MAGIC,
            version: ARTIFACT_VERSION,
            protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            policy_digest: self.policy_digest,
            family_name: self.family_name.clone(),
            rows: self
                .rows_by_digest
                .iter()
                .map(|row| ScheduleCatalogArtifactRowV1 {
                    profiles: row.profiles().clone(),
                    schedule: row.schedule().clone(),
                })
                .collect(),
        };
        let bytes = encode_artifact(&artifact)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(format!(
                "encoded schedule artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
            )));
        }
        Ok(bytes)
    }

    /// Stable family label carried by the trusted artifact.
    pub fn family_name(&self) -> &str {
        &self.family_name
    }

    /// Digest of the validated policy and ordered semantic row identities.
    pub const fn catalog_digest(&self) -> [u8; 32] {
        self.catalog_digest
    }

    /// Check that this catalog belongs to the expected family and runtime policy.
    pub fn validate_binding(
        &self,
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<(), AkitaError> {
        if self.family_name != expected_family_name {
            return Err(AkitaError::InvalidSetup(format!(
                "trusted schedule family {:?} does not match expected family {:?}",
                self.family_name, expected_family_name
            )));
        }
        if self.policy_digest != policy_digest(policy) {
            return Err(AkitaError::InvalidSetup(
                "trusted schedule policy does not match the runtime config".to_string(),
            ));
        }
        for row in &self.rows_by_digest {
            validate_schedule_challenge_hooks(row.schedule(), &ring_challenge_config)?;
        }
        Ok(())
    }

    /// Validated rows in canonical row-digest order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &ResolvedScheduleRow> {
        self.rows_by_digest.iter()
    }

    /// Number of admitted rows.
    pub fn len(&self) -> usize {
        self.rows_by_digest.len()
    }

    /// Whether the catalog contains no rows. Valid catalogs are never empty.
    pub fn is_empty(&self) -> bool {
        self.rows_by_digest.is_empty()
    }

    /// Resolve the proof supplied row digest. No key search or planner search runs here.
    pub fn resolve_selection(
        &self,
        selection: OpeningScheduleSelection,
    ) -> Result<ResolvedScheduleRow, AkitaError> {
        let index = self
            .rows_by_digest
            .binary_search_by_key(&selection.row_digest, |row| row.selection().row_digest)
            .map_err(|_| {
                AkitaError::UnsupportedSchedule(
                    "selected schedule row is not present in the trusted catalog".to_string(),
                )
            })?;
        self.rows_by_digest.get(index).cloned().ok_or_else(|| {
            AkitaError::InvalidSetup("trusted schedule row index is out of bounds".to_string())
        })
    }

    /// Resolve the canonical honest prover row for a runtime key.
    pub fn resolve_key(
        &self,
        key: &AkitaScheduleLookupKey,
    ) -> Result<ResolvedScheduleRow, AkitaError> {
        self.resolve_key_matching(key, None)
    }

    /// Resolve the canonical honest prover row for exact committed profiles.
    pub fn resolve_profiles(
        &self,
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<ResolvedScheduleRow, AkitaError> {
        self.resolve_key_matching(&key_for_profiles(profiles), Some(profiles))
    }

    fn resolve_key_matching(
        &self,
        key: &AkitaScheduleLookupKey,
        exact_profiles: Option<&CommittedGroupBatchProfile>,
    ) -> Result<ResolvedScheduleRow, AkitaError> {
        let row_for_index = |row_index: usize| {
            self.rows_by_digest.get(row_index).ok_or_else(|| {
                AkitaError::InvalidSetup("trusted schedule key index is out of bounds".to_string())
            })
        };
        let start = self
            .rows_by_key
            .partition_point(|(row_key, _)| row_key.canonical_cmp(key).is_lt());
        let (row_key, row_index) = self
            .rows_by_key
            .get(start)
            .ok_or_else(|| unsupported_schedule_lookup(key, exact_profiles.is_some()))?;
        let row = row_for_index(*row_index)?;
        if !row_key.canonical_cmp(key).is_eq()
            || exact_profiles.is_some_and(|profiles| row.profiles() != profiles)
        {
            return Err(unsupported_schedule_lookup(key, exact_profiles.is_some()));
        }
        Ok(row.clone())
    }
}

fn encode_artifact(artifact: &ScheduleCatalogArtifactV1) -> Result<Vec<u8>, AkitaError> {
    serde_json::to_vec_pretty(artifact).map_err(|error| {
        AkitaError::InvalidSetup(format!("failed to encode schedule artifact: {error}"))
    })
}

fn unsupported_schedule_lookup(key: &AkitaScheduleLookupKey, exact_profiles: bool) -> AkitaError {
    AkitaError::UnsupportedSchedule(if exact_profiles {
        "no trusted schedule row matches the exact committed profiles".to_string()
    } else {
        format!("no trusted schedule row for request {key:?}")
    })
}

fn validate_family_name(family_name: &str) -> Result<(), AkitaError> {
    if family_name.is_empty() || family_name.len() > MAX_FAMILY_NAME_BYTES {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule family name length {} is outside 1..={MAX_FAMILY_NAME_BYTES}",
            family_name.len()
        )));
    }
    Ok(())
}

fn key_for_profiles(profiles: &CommittedGroupBatchProfile) -> AkitaScheduleLookupKey {
    AkitaScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
}

fn catalog_digest(
    family_name: &str,
    policy_digest: [u8; 32],
    rows: &[ResolvedScheduleRow],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(32 + family_name.len() + rows.len() * 32 + 32);
    bytes.extend_from_slice(b"AKITA-TRUSTED-SCHEDULE-CATALOG-V1");
    bytes.extend_from_slice(&(family_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(family_name.as_bytes());
    bytes.extend_from_slice(&policy_digest);
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        bytes.extend_from_slice(row.selection().row_digest.as_bytes());
    }
    digest_descriptor_bytes(&bytes)
}

fn validate_schedule_challenge_hooks(
    schedule: &FoldSchedule,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    let validate = |actual: SparseChallengeConfig,
                    method: akita_types::OpeningMethod,
                    ring_dimension: usize,
                    uses_l2: bool,
                    position: ScheduleGroupPosition| {
        let expected = match method {
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "{position} uses unsupported challenge subring D={challenge_subring_dimension}"
                    ))
                })?,
            akita_types::OpeningMethod::EvaluationTrace if uses_l2 => {
                akita_challenges::selective_l2_challenge_config(ring_dimension).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "{position} has no selective L2 challenge config for D={ring_dimension}"
                    ))
                })?
            }
            akita_types::OpeningMethod::EvaluationTrace => {
                ring_challenge_config(ring_dimension)?
            }
        };
        if actual != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "{position} challenge config does not match the trusted runtime hook for D={ring_dimension}"
            )));
        }
        Ok(())
    };

    visit_schedule_groups(schedule, |group| match group {
        ScheduleGroup::Frozen {
            position, params, ..
        } => validate(
            params.fold_challenge_config(),
            params.opening_method(),
            params.inner_commit_matrix_params().ring_dimension(),
            matches!(
                params.inner_commit_matrix_params().security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            position,
        ),
        ScheduleGroup::Final {
            position, params, ..
        } => validate(
            params.fold_challenge_config(),
            params.opening_method(),
            params.d_a(),
            matches!(
                params.inner().matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            position,
        ),
        ScheduleGroup::Terminal {
            position, params, ..
        } => validate(
            params.fold_challenge_config,
            akita_types::OpeningMethod::EvaluationTrace,
            params.d_a(),
            matches!(
                params.inner.matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            position,
        ),
    })
}
