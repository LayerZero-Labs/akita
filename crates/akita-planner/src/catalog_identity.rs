//! Catalog identity validation for generated schedule tables.
//!
//! Each shipped table embeds a [`GeneratedScheduleCatalogIdentity`] that must
//! match the runtime [`PlannerPolicy`] and hook closures before lookup proceeds.
//! Identity mismatch is a hard error; a row miss after validation falls back to
//! the offline DP search.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::AkitaScheduleInputs;

use crate::generated::{
    catalog_key_cmp, GeneratedDirectStep, GeneratedScheduleCatalogIdentity, GeneratedScheduleKey,
    GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::PlannerPolicy;

static VALIDATED_CATALOGS: LazyLock<Mutex<HashSet<CatalogValidationCacheKey>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn lock_validated_catalogs(
) -> Result<std::sync::MutexGuard<'static, HashSet<CatalogValidationCacheKey>>, AkitaError> {
    VALIDATED_CATALOGS
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("catalog validation cache poisoned".to_string()))
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CatalogValidationCacheKey {
    entries_ptr: usize,
    entries_len: usize,
    identity_digest: [u8; 32],
    policy_digest: [u8; 32],
}

/// Fixed-width digest of a [`PlannerPolicy`] for catalog validation caching.
pub fn policy_digest(policy: &PlannerPolicy) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut h = Fnv64::new();
    h.write_u64(sis_family_tag(policy.sis_family));
    h.write_u64(policy.ring_dimension as u64);
    write_decomposition(&mut h, policy.decomposition);
    h.write_u64(u64::from(policy.ring_subfield_norm_bound));
    h.write_u64(policy.claim_ext_degree as u64);
    h.write_u64(policy.chal_ext_degree as u64);
    h.write_u64(u64::from(policy.basis_range.0));
    h.write_u64(u64::from(policy.basis_range.1));
    h.write_u64(policy.onehot_chunk_size as u64);
    h.write_u64(u64::from(policy.tiered));
    let digest = h.finish();
    out[..8].copy_from_slice(&digest.to_le_bytes());
    out
}

/// Fixed-width digest of an identity for wiring guards (not a security primitive).
pub fn identity_digest(identity: &GeneratedScheduleCatalogIdentity) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut h = Fnv64::new();
    h.write_bytes(identity.family_name.as_bytes());
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
    h.write_u64(identity.witness_chunk.num_chunks as u64);
    h.write_u64(identity.witness_chunk.num_activated_levels as u64);
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

/// Fields derived from policy, entries, and runtime hooks for identity checks.
///
/// The owned (non-`'static`) mirror of [`GeneratedScheduleCatalogIdentity`]; the
/// derived equality is the single identity guard, so adding a field to either
/// type (both are built with struct literals) is automatically covered by the
/// comparison in [`validate_catalog_identity_impl`].
#[derive(Clone, Debug, Eq, PartialEq)]
struct CatalogIdentityExpectation {
    family_name: &'static str,
    sis_family: akita_types::SisModulusFamily,
    ring_dimension: usize,
    decomposition: akita_types::DecompositionParams,
    ring_subfield_norm_bound: u32,
    claim_ext_degree: usize,
    chal_ext_degree: usize,
    basis_range: (u32, u32),
    onehot_chunk_size: usize,
    tiered: bool,
    witness_chunk: akita_types::ChunkedWitnessCfg,
    root_fold_shape: TensorChallengeShape,
    ring_dimensions: Vec<usize>,
    ring_challenge_config_digest: u64,
    key_count: usize,
    key_digest: u64,
}

impl CatalogIdentityExpectation {
    /// The owned mirror of a shipped catalog's embedded identity.
    fn from_embedded(identity: &GeneratedScheduleCatalogIdentity) -> Self {
        Self {
            family_name: identity.family_name,
            sis_family: identity.sis_family,
            ring_dimension: identity.ring_dimension,
            decomposition: identity.decomposition,
            ring_subfield_norm_bound: identity.ring_subfield_norm_bound,
            claim_ext_degree: identity.claim_ext_degree,
            chal_ext_degree: identity.chal_ext_degree,
            basis_range: identity.basis_range,
            onehot_chunk_size: identity.onehot_chunk_size,
            tiered: identity.tiered,
            witness_chunk: identity.witness_chunk,
            root_fold_shape: identity.root_fold_shape,
            ring_dimensions: identity.ring_dimensions.to_vec(),
            ring_challenge_config_digest: identity.ring_challenge_config_digest,
            key_count: identity.key_count,
            key_digest: identity.key_digest,
        }
    }
}

fn intern_ring_dimensions(dimensions: Vec<usize>) -> &'static [usize] {
    Box::leak(dimensions.into_boxed_slice())
}

fn catalog_identity_expectation(
    family_name: &'static str,
    policy: &PlannerPolicy,
    entries: &[GeneratedScheduleTableEntry],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<CatalogIdentityExpectation, AkitaError> {
    let root_fold_shape = root_fold_shape_for_entries(entries, &fold_challenge_shape_at_level)?;
    let ring_dimensions = collect_ring_dimensions(entries);
    let ring_challenge_config_digest =
        ring_challenge_config_digest(&ring_dimensions, &ring_challenge_config)?;
    let keys: Vec<GeneratedScheduleKey> = entries.iter().map(|e| e.key).collect();
    Ok(CatalogIdentityExpectation {
        family_name,
        sis_family: policy.sis_family,
        ring_dimension: policy.ring_dimension,
        decomposition: policy.decomposition,
        ring_subfield_norm_bound: policy.ring_subfield_norm_bound,
        claim_ext_degree: policy.claim_ext_degree,
        chal_ext_degree: policy.chal_ext_degree,
        basis_range: policy.basis_range,
        onehot_chunk_size: policy.onehot_chunk_size,
        tiered: policy.tiered,
        witness_chunk: policy.witness_chunk,
        root_fold_shape,
        ring_dimensions,
        ring_challenge_config_digest,
        key_count: keys.len(),
        key_digest: key_digest(&keys),
    })
}

/// Derive the expected catalog identity for `policy` and `entries` under the
/// runtime hooks. Used by tests and the table emitter.
pub fn expected_catalog_identity(
    family_name: &'static str,
    policy: &PlannerPolicy,
    entries: &[GeneratedScheduleTableEntry],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedScheduleCatalogIdentity, AkitaError> {
    let expected = catalog_identity_expectation(
        family_name,
        policy,
        entries,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;
    Ok(GeneratedScheduleCatalogIdentity {
        family_name: expected.family_name,
        sis_family: expected.sis_family,
        ring_dimension: expected.ring_dimension,
        decomposition: expected.decomposition,
        ring_subfield_norm_bound: expected.ring_subfield_norm_bound,
        claim_ext_degree: expected.claim_ext_degree,
        chal_ext_degree: expected.chal_ext_degree,
        basis_range: expected.basis_range,
        onehot_chunk_size: expected.onehot_chunk_size,
        tiered: expected.tiered,
        witness_chunk: expected.witness_chunk,
        root_fold_shape: expected.root_fold_shape,
        ring_dimensions: intern_ring_dimensions(expected.ring_dimensions),
        ring_challenge_config_digest: expected.ring_challenge_config_digest,
        key_count: expected.key_count,
        key_digest: expected.key_digest,
    })
}

/// Validate that `catalog`'s embedded identity matches the runtime policy and hooks.
pub fn validate_catalog_identity(
    catalog: &GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    let cache_key = CatalogValidationCacheKey {
        entries_ptr: catalog.entries.as_ptr() as usize,
        entries_len: catalog.entries.len(),
        identity_digest: identity_digest(&catalog.identity),
        policy_digest: policy_digest(policy),
    };
    if lock_validated_catalogs()?.contains(&cache_key) {
        return verify_runtime_hooks_on_cache_hit(
            catalog,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }

    validate_catalog_identity_impl(
        catalog,
        policy,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;

    lock_validated_catalogs()?.insert(cache_key);
    Ok(())
}

fn validate_catalog_identity_impl(
    catalog: &GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    validate_catalog_keys(catalog.entries)?;
    let embedded = catalog.identity;
    let expected = catalog_identity_expectation(
        embedded.family_name,
        policy,
        catalog.entries,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;
    if CatalogIdentityExpectation::from_embedded(&embedded) != expected {
        return Err(catalog_identity_mismatch_error(
            embedded.family_name,
            "policy or runtime-hook drift",
        ));
    }
    Ok(())
}

fn verify_runtime_hooks_on_cache_hit(
    catalog: &GeneratedScheduleTable,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<(), AkitaError> {
    verify_ring_challenge_config_digest_on_cache_hit(&catalog.identity, ring_challenge_config)?;
    let root_fold_shape =
        root_fold_shape_for_entries(catalog.entries, &fold_challenge_shape_at_level)?;
    if root_fold_shape != catalog.identity.root_fold_shape {
        return Err(catalog_identity_mismatch_error(
            catalog.identity.family_name,
            "root_fold_shape",
        ));
    }
    Ok(())
}

fn verify_ring_challenge_config_digest_on_cache_hit(
    identity: &GeneratedScheduleCatalogIdentity,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    let recomputed = ring_challenge_config_digest(identity.ring_dimensions, ring_challenge_config)?;
    if recomputed != identity.ring_challenge_config_digest {
        return Err(catalog_identity_mismatch_error(
            identity.family_name,
            "ring_challenge_config_digest",
        ));
    }
    Ok(())
}

fn validate_catalog_keys(entries: &[GeneratedScheduleTableEntry]) -> Result<(), AkitaError> {
    for pair in entries.windows(2) {
        match catalog_key_cmp(pair[0].key, pair[1].key) {
            Ordering::Less => {}
            Ordering::Equal => {
                return Err(AkitaError::InvalidSetup(format!(
                    "schedule catalog contains duplicate key {:?}",
                    pair[0].key
                )));
            }
            Ordering::Greater => {
                return Err(AkitaError::InvalidSetup(
                    "schedule catalog entries are not sorted for binary lookup \
                     (num_polynomials, then num_vars)"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn catalog_identity_mismatch_error(family_name: &str, field: &str) -> AkitaError {
    AkitaError::InvalidSetup(format!(
        "schedule catalog identity mismatch for family {family_name}: {field}"
    ))
}

fn root_fold_shape_for_key(
    key: GeneratedScheduleKey,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<TensorChallengeShape, AkitaError> {
    let current_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    Ok(fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars,
        level: 0,
        current_w_len,
    }))
}

fn root_fold_shape_for_entries(
    entries: &[GeneratedScheduleTableEntry],
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<TensorChallengeShape, AkitaError> {
    let first = entries
        .first()
        .map(|e| e.key)
        .ok_or_else(|| AkitaError::InvalidSetup("empty schedule catalog".to_string()))?;
    let expected = root_fold_shape_for_key(first, fold_challenge_shape_at_level)?;
    for entry in entries.iter().skip(1) {
        let actual = root_fold_shape_for_key(entry.key, fold_challenge_shape_at_level)?;
        if actual != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule catalog family has mixed root fold shapes: first key {:?} uses {:?}, key {:?} uses {:?}",
                first, expected, entry.key, actual
            )));
        }
    }
    Ok(expected)
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
    sorted.sort_by_key(|k| (k.num_vars, k.num_polynomials));
    let mut h = Fnv64::new();
    for k in sorted {
        h.write_u64(k.num_vars as u64);
        h.write_u64(k.num_polynomials as u64);
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
    h.write_bytes(&cfg.domain_separator_bytes());
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
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        }
    }

    fn sample_entry() -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            key: GeneratedScheduleKey {
                num_vars: 16,
                num_polynomials: 1,
            },
            steps: &[],
        }
    }

    fn sample_entries() -> &'static [GeneratedScheduleTableEntry] {
        Box::leak(Box::new([sample_entry()]))
    }

    fn sample_ring_challenge_config(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn expected_identity(
        policy: &PlannerPolicy,
        entries: &'static [GeneratedScheduleTableEntry],
    ) -> GeneratedScheduleCatalogIdentity {
        expected_catalog_identity(
            "fp128_d64_onehot",
            policy,
            entries,
            sample_ring_challenge_config,
            flat_fold,
        )
        .expect("expected identity")
    }

    #[test]
    fn catalog_identity_cache_hit_revalidates_runtime_hooks() {
        let policy = sample_policy();
        let entries = sample_entries();
        let catalog = GeneratedScheduleTable {
            entries,
            identity: expected_identity(&policy, entries),
        };
        validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
            .expect("first validation");
        validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
            .expect("cached validation");
    }

    #[test]
    fn catalog_identity_cache_hit_rejects_mismatched_root_fold_shape() {
        let policy = sample_policy();
        let entries = sample_entries();
        let catalog = GeneratedScheduleTable {
            entries,
            identity: expected_identity(&policy, entries),
        };
        validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
            .expect("prime cache");
        let tensor_fold = |_: AkitaScheduleInputs| TensorChallengeShape::Tensor;
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, tensor_fold)
                .expect_err("cached path must reject fold-shape hook drift");
        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("root_fold_shape")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn identity_mismatch_returns_error() {
        let policy = sample_policy();
        let entries = sample_entries();
        let expected = expected_identity(&policy, entries);
        let mut wrong = expected;
        wrong.ring_dimension = 128;
        let catalog = GeneratedScheduleTable {
            entries,
            identity: wrong,
        };
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
                .expect_err("mismatch should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn ring_challenge_digest_mismatch_returns_error() {
        let policy = sample_policy();
        let entries = sample_entries();
        let mut wrong = expected_identity(&policy, entries);
        wrong.ring_challenge_config_digest ^= 1;
        let catalog = GeneratedScheduleTable {
            entries,
            identity: wrong,
        };
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
                .expect_err("ring challenge digest mismatch should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn key_digest_mismatch_returns_error() {
        let policy = sample_policy();
        let entries = sample_entries();
        let mut wrong = expected_identity(&policy, entries);
        wrong.key_digest ^= 1;
        let catalog = GeneratedScheduleTable {
            entries,
            identity: wrong,
        };
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
                .expect_err("key digest mismatch should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn duplicate_catalog_keys_are_rejected() {
        let policy = sample_policy();
        let entries = Box::leak(Box::new([sample_entry(), sample_entry()]));
        let catalog = GeneratedScheduleTable {
            entries,
            identity: expected_identity(&policy, entries),
        };
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
                .expect_err("duplicate keys should error");
        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("duplicate key")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unsorted_catalog_keys_are_rejected() {
        let policy = sample_policy();
        let mut batched = sample_entry();
        let mut singleton = sample_entry();
        batched.key.num_polynomials = 4;
        batched.key.num_vars = 20;
        singleton.key.num_polynomials = 1;
        singleton.key.num_vars = 30;
        // Binary lookup order is (num_polynomials, num_vars); batched before singleton.
        let entries = Box::leak(Box::new([batched, singleton]));
        let catalog = GeneratedScheduleTable {
            entries,
            identity: expected_identity(&policy, entries),
        };
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
                .expect_err("unsorted keys should error");
        assert!(
            matches!(err, AkitaError::InvalidSetup(ref msg) if msg.contains("not sorted for binary lookup")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn mixed_root_fold_shapes_are_rejected() {
        let policy = sample_policy();
        let mut second = sample_entry();
        second.key.num_vars += 1;
        let entries = Box::leak(Box::new([sample_entry(), second]));
        let fold_shape = |inputs: AkitaScheduleInputs| {
            if inputs.num_vars == second.key.num_vars {
                TensorChallengeShape::Tensor
            } else {
                TensorChallengeShape::Flat
            }
        };

        let err = expected_catalog_identity(
            "fp128_d64_onehot",
            &policy,
            entries,
            sample_ring_challenge_config,
            fold_shape,
        )
        .expect_err("mixed root fold shapes should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
