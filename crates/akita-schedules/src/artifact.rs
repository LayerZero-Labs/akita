//! Versioned trusted JSON schedule catalog artifacts.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::instance_descriptor::{
    digest_descriptor_bytes, AKITA_INSTANCE_DESCRIPTOR_VERSION,
};
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, FoldSchedule,
    OpeningScheduleSelection,
};
use serde::{Deserialize, Serialize};

use crate::catalog_identity::policy_digest;
use crate::resolve::ResolvedScheduleRow;
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
    rows_by_key: Vec<usize>,
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
            validate_schedule_challenge_hooks(&schedule, &ring_challenge_config)?;
            let row_digest = schedule_row_digest(&profiles, &schedule)?;
            resolved.push(ResolvedScheduleRow::try_new(
                OpeningScheduleSelection { row_digest },
                profiles,
                schedule,
                policy,
            )?);
        }
        resolved.sort_by_key(|row| row.selection().row_digest);
        let mut rows_by_key = (0..resolved.len()).collect::<Vec<_>>();
        rows_by_key.sort_by(|left_index, right_index| {
            profiles_key_cmp(
                resolved[*left_index].profiles(),
                resolved[*right_index].profiles(),
            )
            .then_with(|| {
                resolved[*left_index]
                    .selection()
                    .row_digest
                    .cmp(&resolved[*right_index].selection().row_digest)
            })
        });
        let has_duplicate_lookup_key = rows_by_key.windows(2).any(|pair| {
            let Some(left) = pair.first().and_then(|index| resolved.get(*index)) else {
                return false;
            };
            let Some(right) = pair.get(1).and_then(|index| resolved.get(*index)) else {
                return false;
            };
            profiles_key_cmp(left.profiles(), right.profiles()).is_eq()
        });
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
        let canonical = serde_json::to_vec(&artifact).map_err(|error| {
            AkitaError::InvalidSetup(format!("failed to canonicalize schedule artifact: {error}"))
        })?;
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
        Self::try_new(
            artifact.family_name,
            artifact
                .rows
                .into_iter()
                .map(|row| (row.profiles, row.schedule)),
            policy,
            ring_challenge_config,
        )
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
        let bytes = serde_json::to_vec(&artifact).map_err(|error| {
            AkitaError::InvalidSetup(format!("failed to encode schedule artifact: {error}"))
        })?;
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
        let mut selected: Option<&ResolvedScheduleRow> = None;
        let row_for_index = |row_index: usize| {
            self.rows_by_digest.get(row_index).ok_or_else(|| {
                AkitaError::InvalidSetup("trusted schedule key index is out of bounds".to_string())
            })
        };
        let start = self.rows_by_key.partition_point(|&row_index| {
            row_for_index(row_index)
                .is_ok_and(|row| profiles_key_cmp_runtime(row.profiles(), key).is_lt())
        });
        let end = self.rows_by_key.partition_point(|&row_index| {
            row_for_index(row_index)
                .is_ok_and(|row| !profiles_key_cmp_runtime(row.profiles(), key).is_gt())
        });
        for &row_index in self.rows_by_key.get(start..end).ok_or_else(|| {
            AkitaError::InvalidSetup("trusted schedule key range is out of bounds".to_string())
        })? {
            let row = self.rows_by_digest.get(row_index).ok_or_else(|| {
                AkitaError::InvalidSetup("trusted schedule key index is out of bounds".to_string())
            })?;
            if exact_profiles.is_none_or(|profiles| row.profiles() == profiles)
                && selected.is_none_or(|current| {
                    row.selection().row_digest < current.selection().row_digest
                })
            {
                selected = Some(row);
            }
        }
        selected.cloned().ok_or_else(|| {
            AkitaError::UnsupportedSchedule(if exact_profiles.is_some() {
                "no trusted schedule row matches the exact committed profiles".to_string()
            } else {
                format!("no trusted schedule row for request {key:?}")
            })
        })
    }
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

fn profiles_key_cmp(
    left: &CommittedGroupBatchProfile,
    right: &CommittedGroupBatchProfile,
) -> std::cmp::Ordering {
    let left_main = (
        left.final_group.group.num_vars(),
        left.final_group.group.num_polynomials(),
    );
    let right_main = (
        right.final_group.group.num_vars(),
        right.final_group.group.num_polynomials(),
    );
    left_main
        .cmp(&right_main)
        .then_with(|| left.precommitteds.len().cmp(&right.precommitteds.len()))
        .then_with(|| {
            left.precommitteds
                .iter()
                .map(akita_types::GroupCommitPhaseParams::canonical_descriptor_bytes)
                .cmp(
                    right
                        .precommitteds
                        .iter()
                        .map(akita_types::GroupCommitPhaseParams::canonical_descriptor_bytes),
                )
        })
}

fn profiles_key_cmp_runtime(
    profiles: &CommittedGroupBatchProfile,
    key: &AkitaScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (
        profiles.final_group.group.num_vars(),
        profiles.final_group.group.num_polynomials(),
    );
    let right_main = (
        key.final_group.num_vars(),
        key.final_group.num_polynomials(),
    );
    left_main
        .cmp(&right_main)
        .then_with(|| profiles.precommitteds.len().cmp(&key.precommitteds.len()))
        .then_with(|| {
            profiles
                .precommitteds
                .iter()
                .map(akita_types::GroupCommitPhaseParams::canonical_descriptor_bytes)
                .cmp(
                    key.precommitteds
                        .iter()
                        .map(akita_types::GroupCommitPhaseParams::canonical_descriptor_bytes),
                )
        })
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
                    context: &str| {
        let expected = match method {
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "{context} uses unsupported challenge subring D={challenge_subring_dimension}"
                    ))
                })?,
            akita_types::OpeningMethod::EvaluationTrace if uses_l2 => {
                akita_challenges::selective_l2_challenge_config(ring_dimension).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "{context} has no selective L2 challenge config for D={ring_dimension}"
                    ))
                })?
            }
            akita_types::OpeningMethod::EvaluationTrace => {
                ring_challenge_config(ring_dimension)?
            }
        };
        if actual != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "{context} challenge config does not match the trusted runtime hook for D={ring_dimension}"
            )));
        }
        Ok(())
    };

    let root = &schedule.root.params;
    validate(
        root.fold_challenge_config(),
        root.opening_method(),
        root.d_a(),
        matches!(
            root.inner().matrix.security_route(),
            akita_types::InnerCommitSecurityRoute::L2 { .. }
        ),
        "root fold",
    )?;
    for (index, group) in root.precommitted_groups().iter().enumerate() {
        validate(
            group.fold_challenge_config(),
            group.opening_method(),
            group.inner_commit_matrix_params().ring_dimension(),
            matches!(
                group.inner_commit_matrix_params().security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            &format!("root precommitted group {index}"),
        )?;
    }
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        validate(
            step.params.fold_challenge_config(),
            step.params.opening_method(),
            step.params.d_a(),
            matches!(
                step.params.inner().matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            &format!("recursive fold {index}"),
        )?;
    }
    validate(
        schedule.terminal.fold_challenge_config,
        akita_types::OpeningMethod::EvaluationTrace,
        schedule.terminal.d_a(),
        matches!(
            schedule.terminal.inner.matrix.security_route(),
            akita_types::InnerCommitSecurityRoute::L2 { .. }
        ),
        "terminal fold",
    )
}

#[cfg(all(test, feature = "fp128-dense"))]
mod tests {
    use super::*;
    use crate::generated::fp128_dense_table;
    use crate::resolve::trusted_catalog_from_generated;
    fn policy() -> PlannerPolicy {
        let identity = fp128_dense_table().identity;
        PlannerPolicy {
            cost_model: identity.cost_model,
            selective_l2_response_model: identity.selective_l2_response_model,
            selection_policy: identity.selection_policy,
            recursive_split_search_policy: identity.recursive_split_search_policy,
            setup_field_budget: identity.setup_field_budget,
            min_offloaded_witness_contraction: identity.min_offloaded_witness_contraction,
            ring_dimension_schedule_mode: identity.ring_dimension_schedule_mode,
            decomposition: identity.decomposition,
            sis_modulus_profile: identity.sis_modulus_profile,
            sis_security_policy: identity.sis_security_policy,
            sis_table_digest: identity.sis_table_digest,
            sis_l2_table_digest: identity.sis_l2_table_digest,
            claim_ext_degree: identity.claim_ext_degree,
            chal_ext_degree: identity.chal_ext_degree,
            inner_basis_range: identity.inner_basis_range,
            opening_basis_range: identity.opening_basis_range,
            witness_chunk: identity.witness_chunk,
            recursive_setup_planning: identity.recursive_setup_planning,
        }
    }

    #[test]
    fn artifact_round_trip_preserves_catalog_identity_and_selection() {
        let policy = policy();
        let challenge = |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        };
        let catalog = trusted_catalog_from_generated(fp128_dense_table(), &policy, challenge)
            .expect("materialize generated catalog");
        let bytes = catalog.to_artifact_bytes().expect("encode artifact");
        let decoded = TrustedScheduleCatalog::from_artifact_bytes(
            &bytes,
            catalog.family_name(),
            &policy,
            challenge,
        )
        .expect("decode artifact");
        assert_eq!(decoded.family_name(), catalog.family_name());
        assert_eq!(decoded.catalog_digest(), catalog.catalog_digest());
        assert_eq!(decoded.len(), catalog.len());
        let selection = catalog.rows_by_digest[0].selection();
        assert_eq!(
            decoded
                .resolve_selection(selection)
                .expect("resolve row")
                .selection(),
            selection
        );
    }

    #[test]
    fn catalog_rejects_duplicate_prover_lookup_keys() {
        let policy = policy();
        let challenge = |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        };
        let catalog = trusted_catalog_from_generated(fp128_dense_table(), &policy, challenge)
            .expect("materialize generated catalog");
        let row = catalog.rows().next().expect("generated catalog row");
        let duplicate = (row.profiles().clone(), row.schedule().clone());
        let error = TrustedScheduleCatalog::try_new(
            "duplicate-key-test",
            [duplicate.clone(), duplicate],
            &policy,
            challenge,
        )
        .expect_err("a prover lookup key must identify exactly one row");
        assert!(error.to_string().contains("duplicate prover lookup key"));
    }

    #[test]
    fn artifact_rejects_trailing_bytes() {
        let artifact = ScheduleCatalogArtifactV1 {
            magic: ARTIFACT_MAGIC,
            version: ARTIFACT_VERSION,
            protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            policy_digest: policy_digest(&policy()),
            family_name: "test".to_string(),
            rows: Vec::new(),
        };
        let mut bytes = serde_json::to_vec(&artifact).expect("encode fixture");
        bytes.push(0);
        assert!(
            TrustedScheduleCatalog::from_artifact_bytes(&bytes, "test", &policy(), |_| {
                Ok(SparseChallengeConfig::pm1_only(1))
            })
            .is_err()
        );
    }
}
