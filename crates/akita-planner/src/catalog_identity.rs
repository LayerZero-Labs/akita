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
    GeneratedDirectStep, GeneratedGroupBatchScheduleKey, GeneratedGroupBatchScheduleTableEntry,
    GeneratedPrecommittedGroupKey, GeneratedScheduleCatalogIdentity, GeneratedScheduleKey,
    GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::PlannerPolicy;

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
    root_fold_shape: TensorChallengeShape,
    ring_dimensions: Vec<usize>,
    ring_challenge_config_digest: u64,
    key_count: usize,
    key_digest: u64,
}

fn intern_ring_dimensions(dimensions: Vec<usize>) -> &'static [usize] {
    Box::leak(dimensions.into_boxed_slice())
}

fn catalog_identity_expectation(
    family_name: &'static str,
    policy: &PlannerPolicy,
    entries: &[GeneratedScheduleTableEntry],
    group_batch_entries: &[GeneratedGroupBatchScheduleTableEntry],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<CatalogIdentityExpectation, AkitaError> {
    let root_fold_shape =
        root_fold_shape_for_entries(entries, group_batch_entries, &fold_challenge_shape_at_level)?;
    let ring_dimensions = collect_ring_dimensions(entries, group_batch_entries);
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
        root_fold_shape,
        ring_dimensions,
        ring_challenge_config_digest,
        key_count: keys.len() + group_batch_entries.len(),
        key_digest: combined_key_digest(&keys, group_batch_entries),
    })
}

/// Derive the expected catalog identity for `policy` and `entries` under the
/// runtime hooks. Used by tests and the table emitter.
pub fn expected_catalog_identity(
    family_name: &'static str,
    policy: &PlannerPolicy,
    entries: &[GeneratedScheduleTableEntry],
    group_batch_entries: &[GeneratedGroupBatchScheduleTableEntry],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<GeneratedScheduleCatalogIdentity, AkitaError> {
    let expected = catalog_identity_expectation(
        family_name,
        policy,
        entries,
        group_batch_entries,
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
    let embedded = catalog.identity;
    let expected = catalog_identity_expectation(
        embedded.family_name,
        policy,
        catalog.entries,
        catalog.group_batch_entries,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )?;
    macro_rules! check_field {
        ($field:ident) => {
            if embedded.$field != expected.$field {
                return Err(catalog_identity_mismatch_error(
                    embedded.family_name,
                    stringify!($field),
                ));
            }
        };
    }

    check_field!(family_name);
    check_field!(sis_family);
    check_field!(ring_dimension);
    check_field!(decomposition);
    check_field!(ring_subfield_norm_bound);
    check_field!(claim_ext_degree);
    check_field!(chal_ext_degree);
    check_field!(basis_range);
    check_field!(onehot_chunk_size);
    check_field!(tiered);
    check_field!(root_fold_shape);
    if embedded.ring_dimensions != expected.ring_dimensions.as_slice() {
        return Err(catalog_identity_mismatch_error(
            embedded.family_name,
            "ring_dimensions",
        ));
    }
    check_field!(ring_challenge_config_digest);
    check_field!(key_count);
    check_field!(key_digest);
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
    group_batch_entries: &[GeneratedGroupBatchScheduleTableEntry],
    fold_challenge_shape_at_level: &impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<TensorChallengeShape, AkitaError> {
    let first = entries
        .first()
        .map(|e| e.key)
        .or_else(|| group_batch_entries.first().map(|e| e.key.main))
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
    for entry in group_batch_entries {
        let actual = root_fold_shape_for_key(entry.key.main, fold_challenge_shape_at_level)?;
        if actual != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule catalog family has mixed root fold shapes: first key {:?} uses {:?}, grouped key {:?} uses {:?}",
                first, expected, entry.key.main, actual
            )));
        }
    }
    Ok(expected)
}

fn collect_ring_dimensions(
    entries: &[GeneratedScheduleTableEntry],
    group_batch_entries: &[GeneratedGroupBatchScheduleTableEntry],
) -> Vec<usize> {
    let mut dims = Vec::new();
    for entry in entries {
        collect_step_ring_dimensions(entry.steps, &mut dims);
    }
    for entry in group_batch_entries {
        collect_step_ring_dimensions(entry.steps, &mut dims);
    }
    dims.sort_unstable();
    dims
}

fn collect_step_ring_dimensions(steps: &[GeneratedStep], dims: &mut Vec<usize>) {
    for step in steps {
        match step {
            GeneratedStep::Fold(f) => push_unique(dims, f.ring_d as usize),
            GeneratedStep::Direct(GeneratedDirectStep { commit: Some(c) }) => {
                push_unique(dims, c.ring_d as usize);
            }
            GeneratedStep::Direct(GeneratedDirectStep { commit: None }) => {}
        }
    }
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

fn combined_key_digest(
    keys: &[GeneratedScheduleKey],
    group_batch_entries: &[GeneratedGroupBatchScheduleTableEntry],
) -> u64 {
    let mut scalar = keys.to_vec();
    scalar.sort_by_key(|k| (k.num_vars, k.num_polynomials));
    let mut grouped: Vec<GeneratedGroupBatchScheduleKey> =
        group_batch_entries.iter().map(|entry| entry.key).collect();
    grouped.sort_by(group_batch_key_cmp);

    let mut h = Fnv64::new();
    for k in scalar {
        h.write_u64(0);
        write_generated_schedule_key(&mut h, k);
    }
    for k in grouped {
        h.write_u64(1);
        write_generated_schedule_key(&mut h, k.main);
        h.write_u64(k.precommitteds.len() as u64);
        for group in k.precommitteds {
            write_generated_precommitted_group_key(&mut h, group);
        }
    }
    h.finish()
}

fn group_batch_key_cmp(
    left: &GeneratedGroupBatchScheduleKey,
    right: &GeneratedGroupBatchScheduleKey,
) -> std::cmp::Ordering {
    let left_main = (left.main.num_vars, left.main.num_polynomials);
    let right_main = (right.main.num_vars, right.main.num_polynomials);
    left_main
        .cmp(&right_main)
        .then_with(|| left.precommitteds.len().cmp(&right.precommitteds.len()))
        .then_with(|| {
            left.precommitteds
                .iter()
                .map(precommitted_group_sort_key)
                .cmp(right.precommitteds.iter().map(precommitted_group_sort_key))
        })
}

fn precommitted_group_sort_key(
    key: &GeneratedPrecommittedGroupKey,
) -> (usize, usize, usize, usize, u32, usize, usize) {
    (
        key.key.num_vars,
        key.key.num_polynomials,
        key.m_vars,
        key.r_vars,
        key.log_basis,
        key.n_a,
        key.conservative_n_b,
    )
}

fn write_generated_schedule_key(h: &mut Fnv64, key: GeneratedScheduleKey) {
    h.write_u64(key.num_vars as u64);
    h.write_u64(key.num_polynomials as u64);
}

fn write_generated_precommitted_group_key(h: &mut Fnv64, key: &GeneratedPrecommittedGroupKey) {
    write_generated_schedule_key(h, key.key);
    h.write_u64(key.m_vars as u64);
    h.write_u64(key.r_vars as u64);
    h.write_u64(u64::from(key.log_basis));
    h.write_u64(key.n_a as u64);
    h.write_u64(key.conservative_n_b as u64);
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

    fn sample_group_batch_entries() -> &'static [GeneratedGroupBatchScheduleTableEntry] {
        &[]
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
            sample_group_batch_entries(),
            sample_ring_challenge_config,
            flat_fold,
        )
        .expect("expected identity")
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
            group_batch_entries: sample_group_batch_entries(),
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
            group_batch_entries: sample_group_batch_entries(),
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
            group_batch_entries: sample_group_batch_entries(),
            identity: wrong,
        };
        let err =
            validate_catalog_identity(&catalog, &policy, sample_ring_challenge_config, flat_fold)
                .expect_err("key digest mismatch should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
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
            sample_group_batch_entries(),
            sample_ring_challenge_config,
            fold_shape,
        )
        .expect_err("mixed root fold shapes should error");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
