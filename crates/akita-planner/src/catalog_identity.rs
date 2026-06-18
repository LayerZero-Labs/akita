//! Catalog identity validation for generated schedule tables.
//!
//! Each shipped table embeds a [`GeneratedScheduleCatalogIdentity`] that must
//! match the runtime [`PlannerPolicy`] and hook closures before lookup proceeds.
//! Identity mismatch is a hard error; a row miss after validation falls back to
//! the offline DP search.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::AkitaScheduleInputs;

use crate::generated::{
    GeneratedDirectStep, GeneratedScheduleCatalogIdentity, GeneratedScheduleKey,
    GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::PlannerPolicy;

/// Fixed-width digest of an identity for wiring guards (not a security primitive).
pub fn identity_digest(identity: &GeneratedScheduleCatalogIdentity) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut h = Fnv64::new();
    h.write_bytes(identity.family_name.as_bytes());
    h.write_u64(u64::from(identity.zk_enabled));
    h.write_u64(sis_family_tag(identity.sis_family));
    h.write_u64(identity.ring_dimension as u64);
    write_decomposition(&mut h, identity.decomposition);
    h.write_u64(u64::from(identity.ring_subfield_norm_bound));
    h.write_u64(identity.claim_ext_degree as u64);
    h.write_u64(identity.chal_ext_degree as u64);
    h.write_u64(u64::from(identity.basis_range.0));
    h.write_u64(u64::from(identity.basis_range.1));
    h.write_u64(identity.onehot_chunk_size as u64);
    h.write_u64(u64::from(identity.tiered));
    h.write_u64(match identity.root_fold_shape {
        TensorChallengeShape::Flat => 0,
        TensorChallengeShape::Tensor => 1,
    });
    h.write_u64(identity.ring_dimensions.len() as u64);
    for &d in identity.ring_dimensions {
        h.write_u64(d as u64);
    }
    h.write_u64(identity.ring_challenge_config_digest);
    h.write_u64(identity.key_count as u64);
    h.write_u64(identity.key_digest);
    let digest = h.finish();
    out[..8].copy_from_slice(&digest.to_le_bytes());
    out
}

fn sis_family_tag(family: akita_types::SisModulusFamily) -> u64 {
    match family {
        akita_types::SisModulusFamily::Q32 => 0,
        akita_types::SisModulusFamily::Q64 => 1,
        akita_types::SisModulusFamily::Q128 => 2,
    }
}

/// Derive the expected catalog identity for `policy` and `entries` under the
/// runtime hooks. Used by tests and the table emitter.
pub fn expected_catalog_identity(
    family_name: &'static str,
    policy: &PlannerPolicy,
    entries: &[GeneratedScheduleTableEntry],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedScheduleCatalogIdentity, AkitaError> {
    let sample_key = entries
        .first()
        .map(|e| e.key)
        .ok_or_else(|| AkitaError::InvalidSetup("empty schedule catalog".to_string()))?;
    let root_fold_shape = root_fold_shape_for_key(sample_key, &fold_shape)?;
    let ring_dimensions = collect_ring_dimensions(entries);
    let ring_challenge_config_digest =
        ring_challenge_config_digest(&ring_dimensions, &ring_challenge_config)?;
    let keys: Vec<GeneratedScheduleKey> = entries.iter().map(|e| e.key).collect();
    Ok(GeneratedScheduleCatalogIdentity {
        family_name,
        zk_enabled: cfg!(feature = "zk"),
        sis_family: policy.sis_family,
        ring_dimension: policy.ring_dimension,
        decomposition: policy.decomposition,
        ring_subfield_norm_bound: policy.ring_subfield_norm_bound,
        claim_ext_degree: policy.claim_ext_degree,
        chal_ext_degree: policy.chal_ext_degree,
        basis_range: policy.basis_range,
        onehot_chunk_size: policy.onehot_chunk_size,
        tiered: policy.tiered,
        root_fold_shape,
        ring_dimensions: &[],
        ring_challenge_config_digest,
        key_count: keys.len(),
        key_digest: key_digest(&keys),
    })
}

/// Validate that `catalog`'s embedded identity matches the runtime policy and hooks.
pub fn validate_catalog_identity(
    catalog: &GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_shape: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    let Some(embedded) = catalog.identity else {
        return Ok(());
    };
    let expected = expected_catalog_identity(
        embedded.family_name,
        policy,
        catalog.entries,
        ring_challenge_config,
        fold_shape,
    )?;
    let ring_dimensions = collect_ring_dimensions(catalog.entries);
    if embedded.family_name != expected.family_name
        || embedded.zk_enabled != expected.zk_enabled
        || embedded.sis_family != expected.sis_family
        || embedded.ring_dimension != expected.ring_dimension
        || embedded.decomposition != expected.decomposition
        || embedded.ring_subfield_norm_bound != expected.ring_subfield_norm_bound
        || embedded.claim_ext_degree != expected.claim_ext_degree
        || embedded.chal_ext_degree != expected.chal_ext_degree
        || embedded.basis_range != expected.basis_range
        || embedded.onehot_chunk_size != expected.onehot_chunk_size
        || embedded.tiered != expected.tiered
        || embedded.root_fold_shape != expected.root_fold_shape
        || embedded.ring_dimensions != ring_dimensions.as_slice()
        || embedded.ring_challenge_config_digest != expected.ring_challenge_config_digest
        || embedded.key_count != expected.key_count
        || embedded.key_digest != expected.key_digest
    {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule catalog identity mismatch for family {}",
            embedded.family_name
        )));
    }
    Ok(())
}

fn root_fold_shape_for_key(
    key: GeneratedScheduleKey,
    fold_shape: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<TensorChallengeShape, AkitaError> {
    let current_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    Ok(fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: 0,
        current_w_len,
    }))
}

fn collect_ring_dimensions(entries: &[GeneratedScheduleTableEntry]) -> Vec<usize> {
    let mut dims = Vec::new();
    for entry in entries {
        for step in entry.steps {
            match step {
                GeneratedStep::Fold(f) => push_unique(&mut dims, f.ring_d as usize),
                GeneratedStep::Direct(GeneratedDirectStep { commit: Some(c) }) => {
                    push_unique(&mut dims, c.ring_d as usize);
                }
                GeneratedStep::Direct(GeneratedDirectStep { commit: None }) => {}
            }
        }
    }
    dims.sort_unstable();
    dims
}

fn push_unique(dims: &mut Vec<usize>, d: usize) {
    if !dims.contains(&d) {
        dims.push(d);
    }
}

pub fn key_digest(keys: &[GeneratedScheduleKey]) -> u64 {
    let mut sorted: Vec<GeneratedScheduleKey> = keys.to_vec();
    sorted.sort_by_key(|k| {
        (
            k.num_vars,
            k.num_commitment_groups,
            k.num_t_vectors,
            k.num_w_vectors,
            k.num_z_vectors,
        )
    });
    let mut h = Fnv64::new();
    for k in sorted {
        h.write_u64(k.num_vars as u64);
        h.write_u64(k.num_commitment_groups as u64);
        h.write_u64(k.num_t_vectors as u64);
        h.write_u64(k.num_w_vectors as u64);
        h.write_u64(k.num_z_vectors as u64);
    }
    h.finish()
}

pub fn ring_challenge_config_digest(
    ring_dimensions: &[usize],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<u64, AkitaError> {
    let mut h = Fnv64::new();
    for &d in ring_dimensions {
        h.write_u64(d as u64);
        let cfg = ring_challenge_config(d)?;
        encode_sparse_challenge_config(&mut h, &cfg);
    }
    Ok(h.finish())
}

fn write_decomposition(h: &mut Fnv64, d: akita_types::DecompositionParams) {
    h.write_u64(u64::from(d.log_basis));
    h.write_u64(u64::from(d.log_commit_bound));
    match d.log_open_bound {
        Some(v) => {
            h.write_u64(1);
            h.write_u64(u64::from(v));
        }
        None => h.write_u64(0),
    }
}

fn encode_sparse_challenge_config(h: &mut Fnv64, cfg: &SparseChallengeConfig) {
    match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => {
            h.write_u64(0);
            h.write_u64(*weight as u64);
            h.write_u64(nonzero_coeffs.len() as u64);
            for c in nonzero_coeffs {
                h.write_u64(i64::from(*c) as u64);
            }
        }
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
            operator_norm_threshold,
        } => {
            h.write_u64(1);
            h.write_u64(*count_mag1 as u64);
            h.write_u64(*count_mag2 as u64);
            h.write_u64(u64::from(*operator_norm_threshold));
        }
        SparseChallengeConfig::BoundedL1Norm => {
            h.write_u64(2);
            h.write_u64(32);
            h.write_u64(121);
            h.write_u64(8);
        }
    }
}

struct Fnv64 {
    state: u64,
}

impl Fnv64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self {
            state: Self::OFFSET,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.state ^= u64::from(*b);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{DecompositionParams, SisModulusFamily};

    fn flat_fold(_: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    fn sample_policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            },
            sis_family: SisModulusFamily::Q128,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 4,
            chal_ext_degree: 4,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
            tiered: false,
        }
    }

    fn sample_entry() -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            key: GeneratedScheduleKey {
                num_vars: 16,
                num_commitment_groups: 1,
                num_t_vectors: 1,
                num_w_vectors: 1,
                num_z_vectors: 1,
            },
            steps: &[],
        }
    }

    fn sample_entries() -> &'static [GeneratedScheduleTableEntry] {
        Box::leak(Box::new([sample_entry()]))
    }

    #[test]
    fn identity_mismatch_returns_error() {
        let policy = sample_policy();
        let entries = sample_entries();
        let ring_challenge_config = |_: usize| -> Result<SparseChallengeConfig, AkitaError> {
            Ok(SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            })
        };
        let expected = expected_catalog_identity(
            "fp128_d64_onehot",
            &policy,
            &entries,
            ring_challenge_config,
            flat_fold,
        )
        .expect("expected identity");
        let mut wrong = expected;
        wrong.ring_dimension = 128;
        let catalog = GeneratedScheduleTable {
            sis_family: policy.sis_family,
            entries,
            identity: Some(wrong),
        };
        let err = validate_catalog_identity(&catalog, &policy, ring_challenge_config, flat_fold)
            .expect_err("mismatch should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn missing_identity_skips_validation() {
        let policy = sample_policy();
        let entries = sample_entries();
        let catalog = GeneratedScheduleTable {
            sis_family: policy.sis_family,
            entries,
            identity: None,
        };
        validate_catalog_identity(&catalog, &policy, |_| unreachable!(), flat_fold)
            .expect("None identity skips validation");
    }
}
