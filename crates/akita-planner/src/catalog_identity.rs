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
use akita_types::{AkitaScheduleInputs, PolynomialGroupLayout, PrecommittedGroupDescriptor};

use crate::generated::{
    generated_schedule_key_cmp, GeneratedBlockGeometry, GeneratedCommittedGroup,
    GeneratedFoldScheduleEntry, GeneratedOpenCommitMatrix, GeneratedScheduleCatalogIdentity,
    GeneratedScheduleTable, GeneratedWitnessPartition,
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
    h.write_u64(sis_modulus_profile_tag(policy.sis_modulus_profile));
    h.write_u64(u64::from(policy.sis_security_policy.tag()));
    h.write_bytes(&policy.sis_table_digest.0);
    h.write_u64(policy.ring_dimension as u64);
    write_decomposition(&mut h, policy.decomposition);
    h.write_u64(u64::from(policy.ring_subfield_norm_bound));
    h.write_u64(policy.claim_ext_degree as u64);
    h.write_u64(policy.chal_ext_degree as u64);
    h.write_u64(u64::from(policy.basis_range.0));
    h.write_u64(u64::from(policy.basis_range.1));
    h.write_u64(policy.onehot_chunk_size as u64);
    h.write_u64(policy.witness_chunk.num_chunks as u64);
    h.write_u64(policy.witness_chunk.num_activated_levels as u64);
    h.write_u64(u64::from(policy.recursive_setup_planning));
    let digest = h.finish();
    out[..8].copy_from_slice(&digest.to_le_bytes());
    out
}

/// Fixed-width digest of an identity for wiring guards (not a security primitive).
pub fn identity_digest(identity: &GeneratedScheduleCatalogIdentity) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut h = Fnv64::new();
    h.write_bytes(identity.family_name.as_bytes());
    h.write_u64(sis_modulus_profile_tag(identity.sis_modulus_profile));
    h.write_u64(u64::from(identity.sis_security_policy.tag()));
    h.write_bytes(&identity.sis_table_digest.0);
    h.write_u64(identity.ring_dimension as u64);
    write_decomposition(&mut h, identity.decomposition);
    h.write_u64(u64::from(identity.ring_subfield_norm_bound));
    h.write_u64(identity.claim_ext_degree as u64);
    h.write_u64(identity.chal_ext_degree as u64);
    h.write_u64(u64::from(identity.basis_range.0));
    h.write_u64(u64::from(identity.basis_range.1));
    h.write_u64(identity.onehot_chunk_size as u64);
    h.write_u64(identity.witness_chunk.num_chunks as u64);
    h.write_u64(identity.witness_chunk.num_activated_levels as u64);
    h.write_u64(u64::from(identity.recursive_setup_planning));

    match identity.root_fold_shape {
        TensorChallengeShape::Flat => h.write_u64(0),
        TensorChallengeShape::Tensor { fold_low_len } => {
            h.write_u64(1);
            h.write_u64(fold_low_len as u64);
        }
    }
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

fn sis_modulus_profile_tag(family: akita_types::SisModulusProfileId) -> u64 {
    match family {
        akita_types::SisModulusProfileId::Q32Offset99 => 0,
        akita_types::SisModulusProfileId::Q64Offset59 => 1,
        akita_types::SisModulusProfileId::Q128OffsetA7F7 => 2,
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
    sis_modulus_profile: akita_types::SisModulusProfileId,
    sis_security_policy: akita_types::SisSecurityPolicyId,
    sis_table_digest: akita_types::SisTableDigest,
    ring_dimension: usize,
    decomposition: akita_types::DecompositionParams,
    ring_subfield_norm_bound: u32,
    claim_ext_degree: usize,
    chal_ext_degree: usize,
    basis_range: (u32, u32),
    onehot_chunk_size: usize,
    witness_chunk: akita_types::ChunkedWitnessCfg,
    recursive_setup_planning: bool,

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
            sis_modulus_profile: identity.sis_modulus_profile,
            sis_security_policy: identity.sis_security_policy,
            sis_table_digest: identity.sis_table_digest,
            ring_dimension: identity.ring_dimension,
            decomposition: identity.decomposition,
            ring_subfield_norm_bound: identity.ring_subfield_norm_bound,
            claim_ext_degree: identity.claim_ext_degree,
            chal_ext_degree: identity.chal_ext_degree,
            basis_range: identity.basis_range,
            onehot_chunk_size: identity.onehot_chunk_size,
            witness_chunk: identity.witness_chunk,
            recursive_setup_planning: identity.recursive_setup_planning,

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
    entries: &[GeneratedFoldScheduleEntry],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<CatalogIdentityExpectation, AkitaError> {
    let root_fold_shape = root_fold_shape_for_entries(entries, &fold_challenge_shape_at_level)?;
    let ring_dimensions = collect_ring_dimensions(entries);
    let ring_challenge_config_digest =
        ring_challenge_config_digest(&ring_dimensions, &ring_challenge_config)?;
    Ok(CatalogIdentityExpectation {
        family_name,
        sis_modulus_profile: policy.sis_modulus_profile,
        sis_security_policy: policy.sis_security_policy,
        sis_table_digest: policy.sis_table_digest,
        ring_dimension: policy.ring_dimension,
        decomposition: policy.decomposition,
        ring_subfield_norm_bound: policy.ring_subfield_norm_bound,
        claim_ext_degree: policy.claim_ext_degree,
        chal_ext_degree: policy.chal_ext_degree,
        basis_range: policy.basis_range,
        onehot_chunk_size: policy.onehot_chunk_size,
        witness_chunk: policy.witness_chunk,
        recursive_setup_planning: policy.recursive_setup_planning,

        root_fold_shape,
        ring_dimensions,
        ring_challenge_config_digest,
        key_count: entries.len(),
        key_digest: entries_key_digest(entries),
    })
}

/// Derive the expected catalog identity for `policy` and `entries` under the
/// runtime hooks. Used by tests and the table emitter.
pub fn expected_catalog_identity(
    family_name: &'static str,
    policy: &PlannerPolicy,
    entries: &[GeneratedFoldScheduleEntry],
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
        sis_modulus_profile: expected.sis_modulus_profile,
        sis_security_policy: expected.sis_security_policy,
        sis_table_digest: expected.sis_table_digest,
        ring_dimension: expected.ring_dimension,
        decomposition: expected.decomposition,
        ring_subfield_norm_bound: expected.ring_subfield_norm_bound,
        claim_ext_degree: expected.claim_ext_degree,
        chal_ext_degree: expected.chal_ext_degree,
        basis_range: expected.basis_range,
        onehot_chunk_size: expected.onehot_chunk_size,
        witness_chunk: expected.witness_chunk,
        recursive_setup_planning: expected.recursive_setup_planning,

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

fn validate_catalog_keys(entries: &[GeneratedFoldScheduleEntry]) -> Result<(), AkitaError> {
    for pair in entries.windows(2) {
        match generated_schedule_key_cmp(&pair[0], &pair[1]) {
            Ordering::Less => {}
            Ordering::Equal => {
                return Err(AkitaError::InvalidSetup(format!(
                    "schedule catalog contains duplicate key {:?}",
                    pair[0]
                )));
            }
            Ordering::Greater => {
                return Err(AkitaError::InvalidSetup(
                    "schedule catalog entries are not sorted for binary lookup \
                     (final_group num_vars/num_polynomials, then precommitted layout)"
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
    key: PolynomialGroupLayout,
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<TensorChallengeShape, AkitaError> {
    let input_witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    Ok(fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len,
    }))
}

fn root_fold_shape_for_entries(
    entries: &[GeneratedFoldScheduleEntry],
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<TensorChallengeShape, AkitaError> {
    let first = entries
        .first()
        .map(|e| e.root.final_group.layout)
        .ok_or_else(|| AkitaError::InvalidSetup("empty schedule catalog".to_string()))?;
    let expected = root_fold_shape_for_key(first, fold_challenge_shape_at_level)?;
    for entry in entries.iter().skip(1) {
        let actual =
            root_fold_shape_for_key(entry.root.final_group.layout, fold_challenge_shape_at_level)?;
        if actual != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule catalog family has mixed root fold shapes: first key {:?} uses {:?}, key {:?} uses {:?}",
                first, expected, entry.root.final_group.layout, actual
            )));
        }
    }
    Ok(expected)
}

fn collect_ring_dimensions(entries: &[GeneratedFoldScheduleEntry]) -> Vec<usize> {
    let mut dims = Vec::new();
    for entry in entries {
        collect_group_ring_dimensions(entry.root.final_group.commitment, &mut dims);
        push_unique(
            &mut dims,
            entry.root.open_commit_matrix.ring_dimension as usize,
        );
        for group in entry.root.precommitted_groups {
            collect_group_ring_dimensions(group.commitment, &mut dims);
        }
        for fold in entry.recursive_folds {
            collect_group_ring_dimensions(fold.witness, &mut dims);
            push_unique(&mut dims, fold.open_commit_matrix.ring_dimension as usize);
            if let Some(prefix) = fold.incoming_setup_prefix {
                collect_group_ring_dimensions(prefix.commitment, &mut dims);
                push_unique(&mut dims, prefix.d_setup as usize);
            }
        }
        push_unique(
            &mut dims,
            entry.terminal.inner_commit_matrix.ring_dimension as usize,
        );
    }
    dims.sort_unstable();
    dims
}

fn collect_group_ring_dimensions(group: GeneratedCommittedGroup, dims: &mut Vec<usize>) {
    push_unique(dims, group.inner_commit_matrix.ring_dimension as usize);
    push_unique(dims, group.outer_commit_matrix.ring_dimension as usize);
}

fn push_unique(dims: &mut Vec<usize>, d: usize) {
    if !dims.contains(&d) {
        dims.push(d);
    }
}

pub fn key_digest(keys: &[PolynomialGroupLayout]) -> u64 {
    let mut sorted: Vec<PolynomialGroupLayout> = keys.to_vec();
    sorted.sort_by_key(|k| (k.num_vars(), k.num_polynomials()));
    let mut h = Fnv64::new();
    for k in sorted {
        h.write_u64(k.num_vars() as u64);
        h.write_u64(k.num_polynomials() as u64);
    }
    h.finish()
}

fn entries_key_digest(entries: &[GeneratedFoldScheduleEntry]) -> u64 {
    let mut entries = entries.to_vec();
    entries.sort_by(generated_schedule_key_cmp);
    let mut h = Fnv64::new();
    for entry in entries {
        write_generated_schedule_key(&mut h, entry.root.final_group.layout);
        write_generated_group(&mut h, entry.root.final_group.commitment);
        h.write_u64(entry.root.precommitted_groups.len() as u64);
        for group in entry.root.precommitted_groups {
            write_generated_precommitted_group_key(&mut h, &group.descriptor);
            write_generated_group(&mut h, group.commitment);
        }
        write_generated_open_matrix(&mut h, entry.root.open_commit_matrix);
        write_generated_partition(&mut h, entry.root.witness_partition);
        h.write_u64(entry.recursive_folds.len() as u64);
        for fold in entry.recursive_folds {
            write_generated_group(&mut h, fold.witness);
            write_generated_open_matrix(&mut h, fold.open_commit_matrix);
            write_generated_partition(&mut h, fold.witness_partition);
            h.write_u64(u64::from(fold.incoming_setup_prefix.is_some()));
            if let Some(prefix) = fold.incoming_setup_prefix {
                h.write_u64(prefix.natural_len);
                h.write_u64(u64::from(prefix.d_setup));
                write_generated_group(&mut h, prefix.commitment);
            }
        }
        write_generated_geometry(&mut h, entry.terminal.geometry);
        h.write_u64(u64::from(entry.terminal.inner_commit_matrix.ring_dimension));
        h.write_u64(u64::from(entry.terminal.inner_commit_matrix.log_basis));
    }
    h.finish()
}

fn write_generated_geometry(h: &mut Fnv64, value: GeneratedBlockGeometry) {
    h.write_u64(value.live_ring_elements_per_claim);
    h.write_u64(value.positions_per_block);
    h.write_u64(value.live_blocks);
}

fn write_generated_group(h: &mut Fnv64, value: GeneratedCommittedGroup) {
    write_generated_geometry(h, value.geometry);
    h.write_u64(u64::from(value.inner_commit_matrix.ring_dimension));
    h.write_u64(u64::from(value.inner_commit_matrix.log_basis));
    h.write_u64(u64::from(value.outer_commit_matrix.ring_dimension));
    h.write_u64(u64::from(value.outer_commit_matrix.log_basis));
    h.write_u64(u64::from(value.outer_commit_matrix.slice_count));
}

fn write_generated_open_matrix(h: &mut Fnv64, value: GeneratedOpenCommitMatrix) {
    h.write_u64(u64::from(value.ring_dimension));
    h.write_u64(u64::from(value.log_basis));
    h.write_u64(u64::from(value.slice_count));
}

fn write_generated_partition(h: &mut Fnv64, value: GeneratedWitnessPartition) {
    match value {
        GeneratedWitnessPartition::Single => h.write_u64(1),
        GeneratedWitnessPartition::Distributed { num_chunks } => {
            h.write_u64(2);
            h.write_u64(u64::from(num_chunks));
        }
    }
}

fn write_generated_schedule_key(h: &mut Fnv64, key: PolynomialGroupLayout) {
    h.write_u64(key.num_vars() as u64);
    h.write_u64(key.num_polynomials() as u64);
}

fn write_generated_precommitted_group_key(h: &mut Fnv64, key: &PrecommittedGroupDescriptor) {
    write_generated_schedule_key(h, key.group);
    h.write_u64(key.num_live_ring_elements_per_claim as u64);
    h.write_u64(key.num_positions_per_block as u64);
    h.write_u64(key.num_live_blocks as u64);
    h.write_u64(u64::from(key.log_basis_inner));
    h.write_u64(u64::from(key.log_basis_outer));
    h.write_u64(key.n_a as u64);
    h.write_bytes(&key.a_coeff_linf_bound.to_le_bytes());
    h.write_u64(key.n_b as u64);
    h.write_bytes(&key.b_coeff_linf_bound.to_le_bytes());
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
