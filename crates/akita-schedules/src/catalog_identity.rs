//! Catalog identity validation for generated schedule tables.
//!
//! Each generated table embeds a [`GeneratedScheduleCatalogIdentity`] that must
//! match the runtime [`PlannerPolicy`] and hook closures before lookup proceeds.
//! Identity mismatch is a hard error; a row miss after validation falls back to
//! the offline DP search.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{LazyLock, Mutex};

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::instance_descriptor::AKITA_INSTANCE_DESCRIPTOR_VERSION;
use akita_types::{
    CommitmentRingDims, GroupCommitPhaseParams, PolynomialGroupLayout, SETUP_PREFIX_CONTENT_TAG,
};

use crate::generated::{
    generated_schedule_key_cmp, GeneratedFoldCore, GeneratedFoldScheduleEntry,
    GeneratedFrozenGroup, GeneratedGroup, GeneratedMatrix, GeneratedRecursiveFold,
    GeneratedRootFold, GeneratedScheduleCatalogIdentity, GeneratedScheduleTable,
};
use crate::{PlannerPolicy, RingDimensionScheduleMode};

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
    h.write_bytes(&policy.sis_l2_table_digest.0);
    h.write_u64(u64::from(policy.selective_l2_response_model.tag()));
    write_ring_dimension_schedule_mode(&mut h, policy.ring_dimension_schedule_mode);
    write_decomposition(&mut h, policy.decomposition);
    h.write_u64(policy.claim_ext_degree as u64);
    h.write_u64(policy.chal_ext_degree as u64);
    h.write_u64(u64::from(policy.inner_basis_range.0));
    h.write_u64(u64::from(policy.inner_basis_range.1));
    h.write_u64(u64::from(policy.opening_basis_range.0));
    h.write_u64(u64::from(policy.opening_basis_range.1));
    h.write_u64(policy.witness_chunk.num_chunks as u64);
    h.write_u64(policy.witness_chunk.num_activated_levels as u64);
    h.write_u64(u64::from(policy.recursive_setup_planning));
    h.write_u64(u64::from(policy.cost_model.tag()));
    h.write_u64(u64::from(policy.selection_policy.tag()));
    h.write_u64(u64::from(policy.recursive_split_search_policy.tag()));
    h.write_u64(u64::from(policy.recursive_setup_search_policy.tag()));
    write_optional_usize(&mut h, policy.setup_field_budget);
    h.write_u64(policy.min_offloaded_witness_contraction as u64);
    let digest = h.finish();
    out[..8].copy_from_slice(&digest.to_le_bytes());
    out
}

/// Fixed-width digest of an identity for wiring guards (not a security primitive).
pub fn identity_digest(identity: &GeneratedScheduleCatalogIdentity) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut h = Fnv64::new();
    h.write_bytes(identity.family_name.as_bytes());
    h.write_u64(u64::from(identity.protocol_epoch));
    h.write_u64(sis_modulus_profile_tag(identity.sis_modulus_profile));
    h.write_u64(u64::from(identity.sis_security_policy.tag()));
    h.write_bytes(&identity.sis_table_digest.0);
    h.write_bytes(&identity.sis_l2_table_digest.0);
    h.write_u64(u64::from(identity.selective_l2_response_model.tag()));
    write_decomposition(&mut h, identity.decomposition);
    h.write_u64(identity.claim_ext_degree as u64);
    h.write_u64(identity.chal_ext_degree as u64);
    h.write_u64(u64::from(identity.inner_basis_range.0));
    h.write_u64(u64::from(identity.inner_basis_range.1));
    h.write_u64(u64::from(identity.opening_basis_range.0));
    h.write_u64(u64::from(identity.opening_basis_range.1));
    h.write_u64(identity.witness_chunk.num_chunks as u64);
    h.write_u64(identity.witness_chunk.num_activated_levels as u64);
    h.write_u64(u64::from(identity.recursive_setup_planning));
    h.write_u64(u64::from(identity.cost_model.tag()));
    h.write_u64(u64::from(identity.selection_policy.tag()));
    h.write_u64(u64::from(identity.recursive_split_search_policy.tag()));
    h.write_u64(u64::from(identity.recursive_setup_search_policy.tag()));
    write_optional_usize(&mut h, identity.setup_field_budget);
    h.write_u64(identity.min_offloaded_witness_contraction as u64);

    h.write_u64(identity.ring_dimensions.len() as u64);
    for &d in identity.ring_dimensions {
        h.write_u64(d as u64);
    }
    write_ring_dimension_schedule_mode(&mut h, identity.ring_dimension_schedule_mode);
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
    protocol_epoch: u32,
    cost_model: crate::PlannerCostModelId,
    selective_l2_response_model: crate::SelectiveL2ResponseModelId,
    selection_policy: crate::SelectionPolicyId,
    recursive_split_search_policy: crate::RecursiveSplitSearchPolicy,
    recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy,
    setup_field_budget: Option<usize>,
    min_offloaded_witness_contraction: usize,
    sis_modulus_profile: akita_types::SisModulusProfileId,
    sis_security_policy: akita_types::SisSecurityPolicyId,
    sis_table_digest: akita_types::SisTableDigest,
    sis_l2_table_digest: akita_types::SisL2TableDigest,
    decomposition: akita_types::DecompositionParams,
    claim_ext_degree: usize,
    chal_ext_degree: usize,
    inner_basis_range: (u32, u32),
    opening_basis_range: (u32, u32),
    witness_chunk: akita_types::ChunkedWitnessCfg,
    recursive_setup_planning: bool,

    ring_dimension_schedule_mode: RingDimensionScheduleMode,
    ring_dimensions: Vec<usize>,
    ring_challenge_config_digest: u64,
    key_count: usize,
    key_digest: u64,
}

impl CatalogIdentityExpectation {
    /// The owned mirror of a generated catalog's embedded identity.
    fn from_embedded(identity: &GeneratedScheduleCatalogIdentity) -> Self {
        Self {
            family_name: identity.family_name,
            protocol_epoch: identity.protocol_epoch,
            cost_model: identity.cost_model,
            selective_l2_response_model: identity.selective_l2_response_model,
            selection_policy: identity.selection_policy,
            recursive_split_search_policy: identity.recursive_split_search_policy,
            recursive_setup_search_policy: identity.recursive_setup_search_policy,
            setup_field_budget: identity.setup_field_budget,
            min_offloaded_witness_contraction: identity.min_offloaded_witness_contraction,
            sis_modulus_profile: identity.sis_modulus_profile,
            sis_security_policy: identity.sis_security_policy,
            sis_table_digest: identity.sis_table_digest,
            sis_l2_table_digest: identity.sis_l2_table_digest,
            decomposition: identity.decomposition,
            claim_ext_degree: identity.claim_ext_degree,
            chal_ext_degree: identity.chal_ext_degree,
            inner_basis_range: identity.inner_basis_range,
            opening_basis_range: identity.opening_basis_range,
            witness_chunk: identity.witness_chunk,
            recursive_setup_planning: identity.recursive_setup_planning,

            ring_dimension_schedule_mode: identity.ring_dimension_schedule_mode,
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
) -> Result<CatalogIdentityExpectation, AkitaError> {
    validate_entry_dimensions(entries, policy.ring_dimension_schedule_mode)?;
    let ring_dimensions = collect_ring_dimensions(entries);
    let challenge_ring_dimensions = challenge_a_dimensions(policy.ring_dimension_schedule_mode);
    let ring_challenge_config_digest =
        ring_challenge_config_digest(&challenge_ring_dimensions, &ring_challenge_config)?;
    Ok(CatalogIdentityExpectation {
        family_name,
        protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
        cost_model: policy.cost_model,
        selective_l2_response_model: policy.selective_l2_response_model,
        selection_policy: policy.selection_policy,
        recursive_split_search_policy: policy.recursive_split_search_policy,
        recursive_setup_search_policy: policy.recursive_setup_search_policy,
        setup_field_budget: policy.setup_field_budget,
        min_offloaded_witness_contraction: policy.min_offloaded_witness_contraction,
        sis_modulus_profile: policy.sis_modulus_profile,
        sis_security_policy: policy.sis_security_policy,
        sis_table_digest: policy.sis_table_digest,
        sis_l2_table_digest: policy.sis_l2_table_digest,
        decomposition: policy.decomposition,
        claim_ext_degree: policy.claim_ext_degree,
        chal_ext_degree: policy.chal_ext_degree,
        inner_basis_range: policy.inner_basis_range,
        opening_basis_range: policy.opening_basis_range,
        witness_chunk: policy.witness_chunk,
        recursive_setup_planning: policy.recursive_setup_planning,

        ring_dimension_schedule_mode: policy.ring_dimension_schedule_mode,
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
) -> Result<GeneratedScheduleCatalogIdentity, AkitaError> {
    let expected =
        catalog_identity_expectation(family_name, policy, entries, ring_challenge_config)?;
    Ok(GeneratedScheduleCatalogIdentity {
        family_name: expected.family_name,
        protocol_epoch: expected.protocol_epoch,
        cost_model: expected.cost_model,
        selective_l2_response_model: expected.selective_l2_response_model,
        selection_policy: expected.selection_policy,
        recursive_split_search_policy: expected.recursive_split_search_policy,
        recursive_setup_search_policy: expected.recursive_setup_search_policy,
        setup_field_budget: expected.setup_field_budget,
        min_offloaded_witness_contraction: expected.min_offloaded_witness_contraction,
        sis_modulus_profile: expected.sis_modulus_profile,
        sis_security_policy: expected.sis_security_policy,
        sis_table_digest: expected.sis_table_digest,
        sis_l2_table_digest: expected.sis_l2_table_digest,
        decomposition: expected.decomposition,
        claim_ext_degree: expected.claim_ext_degree,
        chal_ext_degree: expected.chal_ext_degree,
        inner_basis_range: expected.inner_basis_range,
        opening_basis_range: expected.opening_basis_range,
        witness_chunk: expected.witness_chunk,
        recursive_setup_planning: expected.recursive_setup_planning,

        ring_dimension_schedule_mode: expected.ring_dimension_schedule_mode,
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
) -> Result<(), AkitaError> {
    let cache_key = CatalogValidationCacheKey {
        entries_ptr: catalog.entries.as_ptr() as usize,
        entries_len: catalog.entries.len(),
        identity_digest: identity_digest(&catalog.identity),
        policy_digest: policy_digest(policy),
    };
    if lock_validated_catalogs()?.contains(&cache_key) {
        return verify_runtime_hooks_on_cache_hit(catalog, ring_challenge_config);
    }

    validate_catalog_identity_impl(catalog, policy, ring_challenge_config)?;

    lock_validated_catalogs()?.insert(cache_key);
    Ok(())
}

fn validate_catalog_identity_impl(
    catalog: &GeneratedScheduleTable,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    validate_catalog_keys(catalog.entries)?;
    let embedded = catalog.identity;
    let expected = catalog_identity_expectation(
        embedded.family_name,
        policy,
        catalog.entries,
        ring_challenge_config,
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
) -> Result<(), AkitaError> {
    verify_ring_challenge_config_digest_on_cache_hit(&catalog.identity, ring_challenge_config)?;
    Ok(())
}

fn verify_ring_challenge_config_digest_on_cache_hit(
    identity: &GeneratedScheduleCatalogIdentity,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    let challenge_ring_dimensions = challenge_a_dimensions(identity.ring_dimension_schedule_mode);
    let recomputed =
        ring_challenge_config_digest(&challenge_ring_dimensions, ring_challenge_config)?;
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
            Ordering::Less | Ordering::Equal => {}
            Ordering::Greater => {
                return Err(AkitaError::InvalidSetup(
                    "schedule catalog entries are not sorted for binary lookup \
                     (final_group num_vars/num_polynomials, source encoding, then exact \
                      precommitted profiles)"
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

fn collect_ring_dimensions(entries: &[GeneratedFoldScheduleEntry]) -> Vec<usize> {
    let mut dims = Vec::new();
    for entry in entries {
        collect_group_ring_dimensions(entry.root.core.group, &mut dims);
        push_unique(
            &mut dims,
            entry.root.core.open_commit_matrix.ring_dimension as usize,
        );
        for group in entry.root.precommitted_groups {
            push_unique(&mut dims, group.group.profile.inner.matrix.ring_dimension());
            push_unique(&mut dims, group.group.profile.outer.matrix.ring_dimension());
        }
        for fold in entry.recursive_folds {
            collect_group_ring_dimensions(fold.core.group, &mut dims);
            push_unique(
                &mut dims,
                fold.core.open_commit_matrix.ring_dimension as usize,
            );
            if let Some(prefix) = fold.setup_prefix {
                push_unique(
                    &mut dims,
                    prefix.group.profile.inner.matrix.ring_dimension(),
                );
                push_unique(
                    &mut dims,
                    prefix.group.profile.outer.matrix.ring_dimension(),
                );
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

fn challenge_a_dimensions(mode: RingDimensionScheduleMode) -> Vec<usize> {
    match mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => vec![ring_dimension],
        RingDimensionScheduleMode::AdaptiveDimension {
            potential_a_dimensions,
            ..
        } => potential_a_dimensions.to_vec(),
    }
}

fn validate_entry_dimensions(
    entries: &[GeneratedFoldScheduleEntry],
    mode: RingDimensionScheduleMode,
) -> Result<(), AkitaError> {
    let dimensions = |group: GeneratedGroup, opening: u32| CommitmentRingDims {
        inner: group.inner_commit_matrix.ring_dimension as usize,
        outer: group.outer_commit_matrix.ring_dimension as usize,
        opening: opening as usize,
    };
    for entry in entries {
        let root = dimensions(
            entry.root.core.group,
            entry.root.core.open_commit_matrix.ring_dimension,
        );
        validate_level_dimensions(mode, 0, root, None, entry.final_group)?;
        let mut previous = root;
        for (index, fold) in entry.recursive_folds.iter().enumerate() {
            let current = dimensions(fold.core.group, fold.core.open_commit_matrix.ring_dimension);
            validate_level_dimensions(mode, index + 1, current, Some(previous), entry.final_group)?;
            previous = current;
        }
        let terminal_d = entry.terminal.inner_commit_matrix.ring_dimension as usize;
        let terminal_level = entry.recursive_folds.len() + 1;
        let terminal_is_admitted = terminal_dimension_is_admitted(mode, terminal_level, terminal_d);
        if !terminal_is_admitted {
            return Err(AkitaError::InvalidSetup(format!(
                "generated terminal D{terminal_d} is outside the policy terminal dimension domain for key {:?}",
                entry.final_group
            )));
        }
        if matches!(mode, RingDimensionScheduleMode::AdaptiveDimension { .. })
            && terminal_d > previous.d_a()
        {
            return Err(AkitaError::InvalidSetup(format!(
                "generated terminal D{terminal_d} exceeds predecessor A dimension D{} for key {:?}",
                previous.d_a(),
                entry.final_group
            )));
        }
    }
    Ok(())
}

fn terminal_dimension_is_admitted(
    mode: RingDimensionScheduleMode,
    terminal_level: usize,
    terminal_d: usize,
) -> bool {
    match mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            terminal_d == ring_dimension
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            ..
        } => {
            if terminal_level < num_search_levels {
                potential_a_dimensions.contains(&terminal_d)
            } else {
                suffix_dimensions.contains(&terminal_d)
            }
        }
    }
}

fn validate_level_dimensions(
    mode: RingDimensionScheduleMode,
    level: usize,
    dimensions: CommitmentRingDims,
    previous: Option<CommitmentRingDims>,
    key: PolynomialGroupLayout,
) -> Result<(), AkitaError> {
    let admitted = match mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            dimensions == CommitmentRingDims::uniform(ring_dimension)
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            if level < num_search_levels {
                potential_a_dimensions.contains(&dimensions.d_a())
                    && potential_b_dimensions.contains(&dimensions.d_b())
                    && potential_d_dimensions.contains(&dimensions.d_d())
                    && previous.is_none_or(|ceiling| {
                        dimensions.d_a() <= ceiling.d_a()
                            && dimensions.d_b() <= ceiling.d_b()
                            && dimensions.d_d() <= ceiling.d_d()
                    })
            } else {
                dimensions.d_a() == dimensions.d_b()
                    && dimensions.d_b() == dimensions.d_d()
                    && suffix_dimensions.contains(&dimensions.d_a())
                    && previous.is_none_or(|ceiling| {
                        dimensions.d_a() <= ceiling.d_a()
                            && dimensions.d_b() <= ceiling.d_b()
                            && dimensions.d_d() <= ceiling.d_d()
                    })
            }
        }
    };
    if !admitted {
        return Err(AkitaError::InvalidSetup(format!(
            "generated dimensions {dimensions:?} at level {level} are outside policy for key {key:?}"
        )));
    }
    Ok(())
}

fn collect_group_ring_dimensions(group: GeneratedGroup, dims: &mut Vec<usize>) {
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

fn write_setup_prefix_content_mode_full_prefix(h: &mut Fnv64) {
    h.write_bytes(SETUP_PREFIX_CONTENT_TAG);
    h.write_bytes(&[0; 4]);
}

fn entries_key_digest(entries: &[GeneratedFoldScheduleEntry]) -> u64 {
    entries_key_digest_with_setup_prefix_content_mode(entries, true)
}

fn entries_key_digest_with_setup_prefix_content_mode(
    entries: &[GeneratedFoldScheduleEntry],
    write_full_prefix_content_mode: bool,
) -> u64 {
    let mut entries = entries.to_vec();
    entries.sort_by(generated_schedule_key_cmp);
    let mut h = Fnv64::new();
    for entry in entries {
        write_generated_schedule_key(&mut h, entry.final_group);
        write_generated_root_fold(&mut h, &entry.root);
        h.write_u64(entry.recursive_folds.len() as u64);
        for fold in entry.recursive_folds {
            write_generated_recursive_fold(&mut h, fold, write_full_prefix_content_mode);
        }
        write_generated_geometry(&mut h, entry.terminal.geometry);
        h.write_u64(u64::from(entry.terminal.inner_commit_matrix.ring_dimension));
        h.write_u64(u64::from(entry.terminal.inner_commit_matrix.log_basis));
        h.write_u64(u64::from(entry.terminal.num_digits_inner));
        h.write_u64(u64::from(entry.terminal.fold_log_basis));
        h.write_u64(u64::from(entry.terminal.fold_digit_count));
        h.write_u64(u64::from(entry.terminal.inner_output_rank));
        h.write_u128(entry.terminal.inner_coeff_linf_bound);
        write_optional_u128(&mut h, entry.terminal.response_l2_sq_cap);
        write_optional_u128(&mut h, entry.terminal.z_linf_cap);
        h.write_u64(u64::from(entry.terminal.z_rice_low_bits));
        h.write_u64(entry.terminal.z_payload_bytes);
    }
    h.finish()
}

fn write_opening_method(h: &mut Fnv64, method: akita_types::OpeningMethod) {
    match method {
        akita_types::OpeningMethod::EvaluationTrace => {}
        akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => {
            h.write_u64(1);
            h.write_u64(challenge_subring_dimension as u64);
        }
    }
}

fn write_generated_geometry(h: &mut Fnv64, value: akita_types::BlockGeometry) {
    // Geometry is hashed in declaration order so every executed count is bound.
    h.write_u64(value.live_ring_elements_per_claim as u64);
    h.write_u64(value.positions_per_block as u64);
    h.write_u64(value.live_blocks as u64);
}

fn write_generated_group(h: &mut Fnv64, value: GeneratedGroup) {
    write_generated_geometry(h, value.geometry);
    h.write_u64(u64::from(value.inner_commit_matrix.ring_dimension));
    h.write_u64(u64::from(value.inner_commit_matrix.log_basis));
    h.write_u64(u64::from(value.outer_commit_matrix.ring_dimension));
    h.write_u64(u64::from(value.outer_commit_matrix.log_basis));
    h.write_u64(u64::from(value.outer_slice_count));
    h.write_u64(u64::from(value.num_digits_fold));
    write_opening_method(h, value.opening_method);
}

fn write_generated_frozen_group(h: &mut Fnv64, value: &GeneratedFrozenGroup) {
    write_generated_precommitted_group_key(h, &value.profile);
    h.write_u64(u64::from(value.num_digits_fold));
    // The opening plan is derived from the consuming fold, so the only
    // group-owned inputs are the method and the fold depth.
    write_opening_method(h, value.opening_method);
}

fn write_generated_fold_core(h: &mut Fnv64, fold: GeneratedFoldCore) {
    write_generated_group(h, fold.group);
    write_generated_open_matrix(h, fold.open_commit_matrix);
    write_generated_partition(h, fold.witness_chunks);
    h.write_u64(u64::from(fold.ring_relation_mode.tag()));
}

fn write_generated_root_fold(h: &mut Fnv64, fold: &GeneratedRootFold) {
    write_generated_fold_core(h, fold.core);
    h.write_u64(u64::from(fold.num_digits_inner));
    h.write_u64(fold.precommitted_groups.len() as u64);
    for group in fold.precommitted_groups {
        write_generated_frozen_group(h, &group.group);
    }
}

fn write_generated_recursive_fold(
    h: &mut Fnv64,
    fold: &GeneratedRecursiveFold,
    full_prefix_content_mode: bool,
) {
    write_generated_fold_core(h, fold.core);
    h.write_u64(u64::from(fold.setup_prefix.is_some()));
    if let Some(prefix) = &fold.setup_prefix {
        if full_prefix_content_mode {
            write_setup_prefix_content_mode_full_prefix(h);
        }
        write_generated_frozen_group(h, &prefix.group);
        h.write_u64(prefix.natural_len);
    }
    h.write_u64(match fold.payload_mode {
        akita_types::CommitmentPayloadMode::Compressed => 0,
        akita_types::CommitmentPayloadMode::Raw => 1,
    });
    write_optional_u128(h, fold.response_l2_sq_cap);
}

fn write_optional_u128(h: &mut Fnv64, value: Option<u128>) {
    h.write_u64(u64::from(value.is_some()));
    h.write_u128(value.unwrap_or(0));
}

fn write_generated_open_matrix(h: &mut Fnv64, value: GeneratedMatrix) {
    h.write_u64(u64::from(value.ring_dimension));
    h.write_u64(u64::from(value.log_basis));
}

fn write_generated_partition(h: &mut Fnv64, witness_chunks: u32) {
    h.write_u64(u64::from(witness_chunks));
}

fn write_generated_schedule_key(h: &mut Fnv64, key: PolynomialGroupLayout) {
    h.write_u64(key.num_vars() as u64);
    h.write_u64(key.num_polynomials() as u64);
}

fn write_generated_precommitted_group_key(h: &mut Fnv64, key: &GroupCommitPhaseParams) {
    h.write_bytes(&key.canonical_descriptor_bytes());
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

fn write_ring_dimension_schedule_mode(h: &mut Fnv64, mode: RingDimensionScheduleMode) {
    match mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            h.write_u64(0);
            h.write_u64(ring_dimension as u64);
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            h.write_u64(1);
            h.write_u64(num_search_levels as u64);
            h.write_u64(suffix_dimensions.len() as u64);
            for &dimension in suffix_dimensions {
                h.write_u64(dimension as u64);
            }
            for dimensions in [
                potential_a_dimensions,
                potential_b_dimensions,
                potential_d_dimensions,
            ] {
                h.write_u64(dimensions.len() as u64);
                for &dimension in dimensions {
                    h.write_u64(dimension as u64);
                }
            }
        }
    }
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

fn write_optional_usize(h: &mut Fnv64, value: Option<usize>) {
    match value {
        Some(value) => {
            h.write_u64(1);
            h.write_u64(value as u64);
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

    fn write_u128(&mut self, v: u128) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn finish(self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod selection_policy_identity_tests {
    use crate::SelectionPolicyId;

    #[test]
    fn root_output_objectives_do_not_reuse_legacy_catalog_tags() {
        assert_eq!(SelectionPolicyId::MinEstimatedProofPayloadV2.tag(), 4);
        assert_eq!(SelectionPolicyId::MinFirstDirectSetupThenPayloadV2.tag(), 5);
        assert_eq!(
            SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3.tag(),
            6
        );
        assert!(![1, 2, 3].contains(&SelectionPolicyId::MinEstimatedProofPayloadV2.tag()));
        assert!(![1, 2, 3].contains(&SelectionPolicyId::MinFirstDirectSetupThenPayloadV2.tag()));
        assert!(![1, 2, 3, 4, 5].contains(
            &SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3.tag()
        ));
    }
}

#[cfg(test)]
mod terminal_dimension_tests {
    use super::*;

    #[test]
    fn adaptive_terminal_uses_potential_dimensions_only_before_suffix_cutover() {
        let mode = RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels: 2,
            suffix_dimensions: &[64],
            potential_a_dimensions: &[64, 128],
            potential_b_dimensions: &[64],
            potential_d_dimensions: &[64],
        };

        assert!(terminal_dimension_is_admitted(mode, 1, 128));
        assert!(!terminal_dimension_is_admitted(mode, 2, 128));
        assert!(terminal_dimension_is_admitted(mode, 2, 64));
    }
}

#[cfg(all(test, feature = "fp128-onehot-recursive"))]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;

    #[test]
    fn full_prefix_catalog_identity_rejects_old_zero_padded_digest() {
        let table = crate::generated::fp128_onehot_recursive_table();
        let old_digest = entries_key_digest_with_setup_prefix_content_mode(table.entries, false);
        assert_ne!(
            old_digest, table.identity.key_digest,
            "full-prefix setup content mode must change the generated key digest"
        );

        let stale = GeneratedScheduleTable {
            identity: GeneratedScheduleCatalogIdentity {
                key_digest: old_digest,
                ..table.identity
            },
            ..table
        };
        let policy = PlannerPolicy {
            cost_model: stale.identity.cost_model,
            selective_l2_response_model: stale.identity.selective_l2_response_model,
            selection_policy: stale.identity.selection_policy,
            recursive_split_search_policy: stale.identity.recursive_split_search_policy,
            recursive_setup_search_policy: stale.identity.recursive_setup_search_policy,
            setup_field_budget: stale.identity.setup_field_budget,
            min_offloaded_witness_contraction: stale.identity.min_offloaded_witness_contraction,
            sis_modulus_profile: stale.identity.sis_modulus_profile,
            sis_security_policy: stale.identity.sis_security_policy,
            sis_table_digest: stale.identity.sis_table_digest,
            sis_l2_table_digest: stale.identity.sis_l2_table_digest,
            decomposition: stale.identity.decomposition,
            claim_ext_degree: stale.identity.claim_ext_degree,
            chal_ext_degree: stale.identity.chal_ext_degree,
            inner_basis_range: stale.identity.inner_basis_range,
            opening_basis_range: stale.identity.opening_basis_range,
            witness_chunk: stale.identity.witness_chunk,
            recursive_setup_planning: stale.identity.recursive_setup_planning,
            ring_dimension_schedule_mode: stale.identity.ring_dimension_schedule_mode,
        };
        let err = validate_catalog_identity(&stale, &policy, |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        })
        .expect_err("old zero-padded catalog identity must reject");
        assert!(err
            .to_string()
            .contains("schedule catalog identity mismatch"));
    }

    #[test]
    fn recursive_setup_search_policy_is_catalog_bound() {
        let table = crate::generated::fp128_onehot_recursive_table();
        let mut policy = PlannerPolicy {
            cost_model: table.identity.cost_model,
            selective_l2_response_model: table.identity.selective_l2_response_model,
            selection_policy: table.identity.selection_policy,
            recursive_split_search_policy: table.identity.recursive_split_search_policy,
            recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::Exhaustive,
            setup_field_budget: table.identity.setup_field_budget,
            min_offloaded_witness_contraction: table.identity.min_offloaded_witness_contraction,
            sis_modulus_profile: table.identity.sis_modulus_profile,
            sis_security_policy: table.identity.sis_security_policy,
            sis_table_digest: table.identity.sis_table_digest,
            sis_l2_table_digest: table.identity.sis_l2_table_digest,
            decomposition: table.identity.decomposition,
            claim_ext_degree: table.identity.claim_ext_degree,
            chal_ext_degree: table.identity.chal_ext_degree,
            inner_basis_range: table.identity.inner_basis_range,
            opening_basis_range: table.identity.opening_basis_range,
            witness_chunk: table.identity.witness_chunk,
            recursive_setup_planning: table.identity.recursive_setup_planning,
            ring_dimension_schedule_mode: table.identity.ring_dimension_schedule_mode,
        };
        assert_ne!(
            policy.recursive_setup_search_policy,
            table.identity.recursive_setup_search_policy
        );
        let err = validate_catalog_identity(&table, &policy, |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        })
        .expect_err("recursive setup search policy mismatch must reject");
        assert!(err
            .to_string()
            .contains("schedule catalog identity mismatch"));

        policy.recursive_setup_search_policy = table.identity.recursive_setup_search_policy;
        validate_catalog_identity(&table, &policy, |d| {
            SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
                AkitaError::InvalidSetup(format!("unsupported test ring dimension {d}"))
            })
        })
        .expect("matching recursive setup search policy");
    }
}
