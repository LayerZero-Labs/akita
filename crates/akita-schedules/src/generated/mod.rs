#![allow(missing_docs)]

pub const MAX_COMMIT_MATRIX_SLICES: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedBlockGeometry {
    pub live_ring_elements_per_claim: u64,
    pub positions_per_block: u64,
    pub live_blocks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedInnerCommitMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedOuterCommitMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
    pub slice_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedOpenCommitMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
    pub slice_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedCommittedGroup {
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedInnerCommitMatrix,
    pub outer_commit_matrix: GeneratedOuterCommitMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootFinalGroup {
    pub layout: akita_types::PolynomialGroupLayout,
    pub num_digits_inner: u32,
    pub num_digits_fold: u32,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootPrecommittedGroup {
    pub descriptor: akita_types::CommittedGroupProfile,
    pub num_digits_fold: u32,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedWitnessPartition {
    Single,
    Distributed { num_chunks: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootFold {
    pub final_group: GeneratedRootFinalGroup,
    pub precommitted_groups: &'static [GeneratedRootPrecommittedGroup],
    pub open_commit_matrix: GeneratedOpenCommitMatrix,
    pub witness_partition: GeneratedWitnessPartition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedSetupPrefixInput {
    pub natural_len: u64,
    pub num_digits_fold: u32,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRecursiveFold {
    pub payload_mode: akita_types::CommitmentPayloadMode,
    pub witness: GeneratedCommittedGroup,
    pub num_digits_fold: u32,
    pub open_commit_matrix: GeneratedOpenCommitMatrix,
    pub incoming_setup_prefix: Option<GeneratedSetupPrefixInput>,
    pub witness_partition: GeneratedWitnessPartition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedTerminalFold {
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedInnerCommitMatrix,
    pub num_digits_inner: u32,
    pub inner_output_rank: u32,
    pub inner_coeff_linf_bound: u128,
    pub z_admission_linf_cap: u128,
    pub z_rice_low_bits: u32,
    pub z_payload_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedPrecommittedProfile {
    pub group: akita_types::PolynomialGroupLayout,
    pub commitment: GeneratedCommittedGroup,
    pub num_digits_inner: u32,
    pub inner_output_rank: u32,
    pub inner_coeff_linf_bound: u128,
    pub num_digits_outer: u32,
    pub outer_output_rank: u32,
    pub outer_coeff_linf_bound: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldScheduleEntry {
    pub root: GeneratedRootFold,
    pub recursive_folds: &'static [GeneratedRecursiveFold],
    pub terminal: GeneratedTerminalFold,
}

impl GeneratedFoldScheduleEntry {
    /// Build the runtime schedule lookup key represented by this generated row.
    pub fn to_runtime_lookup_key(self) -> akita_types::AkitaScheduleLookupKey {
        akita_types::AkitaScheduleLookupKey {
            final_group: self.root.final_group.layout,
            precommitteds: self
                .root
                .precommitted_groups
                .iter()
                .map(|group| group.descriptor)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleCatalogIdentity {
    pub family_name: &'static str,
    pub protocol_epoch: u32,
    pub cost_model: crate::PlannerCostModelId,
    pub selection_policy: crate::SelectionPolicyId,
    pub setup_field_budget: Option<usize>,
    pub min_offloaded_witness_contraction: usize,
    pub sis_modulus_profile: SisModulusProfileId,
    pub sis_security_policy: akita_types::SisSecurityPolicyId,
    pub sis_table_digest: akita_types::SisTableDigest,
    pub uniform_ring_dimension: usize,
    pub setup_prefix_inner_ring_dimension: usize,
    pub decomposition: akita_types::DecompositionParams,
    pub ring_subfield_norm_bound: u32,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    pub basis_range: (u32, u32),
    /// Multi-chunk witness layout this table was emitted under. A chunked policy
    /// never aliases a single-chunk table (and vice versa), even when row keys
    /// match. `ChunkedWitnessCfg::default()` for single-chunk tables.
    pub witness_chunk: akita_types::ChunkedWitnessCfg,
    pub recursive_setup_planning: bool,

    /// Complete uniform or adaptive dimension policy used to generate this catalog.
    pub ring_dimension_schedule_mode: crate::RingDimensionScheduleMode,
    pub ring_dimensions: &'static [usize],
    pub ring_challenge_config_digest: u64,
    pub key_count: usize,
    pub key_digest: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratedScheduleTable {
    pub entries: &'static [GeneratedFoldScheduleEntry],
    pub precommitted_profiles: &'static [GeneratedPrecommittedProfile],
    pub identity: GeneratedScheduleCatalogIdentity,
}

pub mod expand;
pub mod validate;
pub(crate) mod walk;
pub use crate::{
    ChunkedWitnessCfg, CommitmentRingDims, DecompositionParams, PlannerCostModelId,
    RingDimensionScheduleMode, SelectionPolicyId, SisSecurityPolicyId,
};
pub use akita_types::{
    CommitmentPayloadMode, CommittedGroupProfile, InnerCommitMatrixParams, OuterCommitMatrixParams,
    PolynomialGroupLayout,
};
pub use akita_types::{SisModulusProfileId, SisTableDigest};
pub use validate::{validate_generated_schedule_entry, validate_generated_schedule_table};

/// Returns true when `entries` are ordered for [`table_entry`] binary search.
pub fn catalog_entries_sorted_for_lookup(entries: &[GeneratedFoldScheduleEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| !generated_schedule_key_cmp(&window[0], &window[1]).is_gt())
}

pub fn table_entry_range(
    table: GeneratedScheduleTable,
    key: &akita_types::AkitaScheduleLookupKey,
) -> std::ops::Range<usize> {
    let start = table
        .entries
        .partition_point(|entry| generated_schedule_key_cmp_runtime(entry, key).is_lt());
    let end = table
        .entries
        .partition_point(|entry| !generated_schedule_key_cmp_runtime(entry, key).is_gt());
    start..end
}

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Option<&'static GeneratedFoldScheduleEntry> {
    let range = table_entry_range(table, key);
    if range.is_empty() {
        None
    } else {
        table.entries.get(range.start)
    }
}

pub fn generated_schedule_key_cmp(
    left: &GeneratedFoldScheduleEntry,
    right: &GeneratedFoldScheduleEntry,
) -> std::cmp::Ordering {
    let left_main = (
        left.root.final_group.layout.num_vars(),
        left.root.final_group.layout.num_polynomials(),
    );
    let right_main = (
        right.root.final_group.layout.num_vars(),
        right.root.final_group.layout.num_polynomials(),
    );
    left_main
        .cmp(&right_main)
        .then_with(|| {
            left.root
                .precommitted_groups
                .len()
                .cmp(&right.root.precommitted_groups.len())
        })
        .then_with(|| {
            left.root
                .precommitted_groups
                .iter()
                .map(|group| precommitted_group_sort_key(&group.descriptor))
                .cmp(
                    right
                        .root
                        .precommitted_groups
                        .iter()
                        .map(|group| precommitted_group_sort_key(&group.descriptor)),
                )
        })
}

pub fn generated_schedule_key_cmp_runtime(
    generated: &GeneratedFoldScheduleEntry,
    runtime: &akita_types::AkitaScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (
        generated.root.final_group.layout.num_vars(),
        generated.root.final_group.layout.num_polynomials(),
    );
    let right_main = (
        runtime.final_group.num_vars(),
        runtime.final_group.num_polynomials(),
    );
    left_main
        .cmp(&right_main)
        .then_with(|| {
            generated
                .root
                .precommitted_groups
                .len()
                .cmp(&runtime.precommitteds.len())
        })
        .then_with(|| {
            let generated = generated
                .root
                .precommitted_groups
                .iter()
                .map(|group| &group.descriptor);
            generated
                .zip(&runtime.precommitteds)
                .map(|(left, right)| {
                    precommitted_group_sort_key(left).cmp(&precommitted_group_sort_key(right))
                })
                .find(|ord| *ord != std::cmp::Ordering::Equal)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Sort order for runtime keys; matches [`generated_schedule_key_cmp`].
pub fn runtime_schedule_key_cmp(
    left: &akita_types::AkitaScheduleLookupKey,
    right: &akita_types::AkitaScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (
        left.final_group.num_vars(),
        left.final_group.num_polynomials(),
    );
    let right_main = (
        right.final_group.num_vars(),
        right.final_group.num_polynomials(),
    );
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

fn precommitted_group_sort_key(key: &akita_types::CommittedGroupProfile) -> Vec<u8> {
    key.canonical_descriptor_bytes()
}

fn schedule_key_eq(
    generated: &GeneratedFoldScheduleEntry,
    key: &akita_types::AkitaScheduleLookupKey,
) -> bool {
    generated.root.final_group.layout == key.final_group
        && generated.root.precommitted_groups.len() == key.precommitteds.len()
        && generated
            .root
            .precommitted_groups
            .iter()
            .zip(&key.precommitteds)
            .all(|(generated, layout)| precommitted_group_key_eq(&generated.descriptor, layout))
}

fn precommitted_group_key_eq(
    generated: &akita_types::CommittedGroupProfile,
    layout: &akita_types::CommittedGroupProfile,
) -> bool {
    generated == layout
}

#[cfg(test)]
mod mixed_dimension_key_tests {
    use super::{precommitted_group_key_eq, precommitted_group_sort_key};
    use akita_types::{
        CommittedGroupProfile, InnerCommitMatrixParams, OuterCommitMatrixParams,
        PolynomialGroupLayout, SisModulusProfileId, SisTableDigest,
    };

    fn descriptor() -> CommittedGroupProfile {
        CommittedGroupProfile {
            version: CommittedGroupProfile::VERSION,
            group: PolynomialGroupLayout::new(12, 1),
            num_live_ring_elements_per_claim: 32,
            num_positions_per_block: 8,
            num_live_blocks: 4,
            log_basis_inner: 4,
            num_digits_inner: 2,
            inner_commit_matrix: InnerCommitMatrixParams::new_unchecked(
                akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q128OffsetA7F7,
                3,
                16,
                7,
                128,
            ),
            log_basis_outer: 5,
            num_digits_outer: 2,
            outer_commit_matrix: OuterCommitMatrixParams::new_unchecked(
                akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q128OffsetA7F7,
                2,
                48,
                11,
                64,
            ),
        }
    }

    #[test]
    fn precommitted_key_identity_includes_both_native_ring_dimensions() {
        let base = descriptor();
        let mut changed_inner = base;
        changed_inner.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
            akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            3,
            16,
            7,
            64,
        );
        let mut changed_outer = base;
        changed_outer.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
            akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            2,
            48,
            11,
            32,
        );
        for changed in [changed_inner, changed_outer] {
            assert!(!precommitted_group_key_eq(&base, &changed));
            assert_ne!(
                precommitted_group_sort_key(&base),
                precommitted_group_sort_key(&changed)
            );
        }
    }
}

/// Returns an error when the generated key does not match the runtime key.
pub(crate) fn validate_entry_key(
    generated: &GeneratedFoldScheduleEntry,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Result<(), akita_field::AkitaError> {
    if schedule_key_eq(generated, key) {
        Ok(())
    } else {
        Err(akita_field::AkitaError::InvalidSetup(
            "generated schedule key mismatch".to_string(),
        ))
    }
}

pub(crate) fn validate_certified_bases(
    log_basis_inner: u32,
    log_basis_outer: u32,
    log_basis_open: u32,
    policy: &crate::PlannerPolicy,
    context: &str,
) -> Result<(), akita_field::AkitaError> {
    let (min, max) = policy.basis_range;
    for (role, basis) in [
        ("inner", log_basis_inner),
        ("outer", log_basis_outer),
        ("open", log_basis_open),
    ] {
        if basis < min || basis > max {
            return Err(akita_field::AkitaError::InvalidSetup(format!(
                "{context} {role} basis {basis} outside policy range [{min}, {max}]"
            )));
        }
    }
    if log_basis_open < log_basis_inner || log_basis_open < log_basis_outer {
        return Err(akita_field::AkitaError::InvalidSetup(format!(
            "{context} certified open basis must dominate inner and outer bases"
        )));
    }
    Ok(())
}

// @generated schedule module wiring begin
#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub mod fp128_d64_onehot_multi_chunk;
#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub mod fp128_d64_onehot_multi_chunk_precommitted;
#[cfg(feature = "fp128-d64-onehot-recursive")]
pub mod fp128_d64_onehot_recursive;
#[cfg(feature = "fp128-d64-onehot-recursive-multi-chunk-w8r2")]
pub mod fp128_d64_onehot_recursive_multi_chunk_w8r2;
#[cfg(feature = "fp128-d64-onehot-recursive-multi-chunk-w8r2")]
pub mod fp128_d64_onehot_recursive_multi_chunk_w8r2_precommitted;
#[cfg(feature = "fp128-d64-onehot-recursive")]
pub mod fp128_d64_onehot_recursive_precommitted;
#[cfg(feature = "fp128-dense")]
pub mod fp128_dense;
#[cfg(feature = "fp128-dense-multi-chunk")]
pub mod fp128_dense_multi_chunk;
#[cfg(feature = "fp128-dense-multi-chunk")]
pub mod fp128_dense_multi_chunk_precommitted;
#[cfg(feature = "fp128-dense")]
pub mod fp128_dense_precommitted;
#[cfg(feature = "fp128-onehot")]
pub mod fp128_onehot;
#[cfg(feature = "fp128-onehot-multi-chunk")]
pub mod fp128_onehot_multi_chunk;
#[cfg(feature = "fp128-onehot-multi-chunk")]
pub mod fp128_onehot_multi_chunk_precommitted;
#[cfg(feature = "fp128-onehot-multi-chunk-w2r2")]
pub mod fp128_onehot_multi_chunk_w2r2;
#[cfg(feature = "fp128-onehot-multi-chunk-w2r2")]
pub mod fp128_onehot_multi_chunk_w2r2_precommitted;
#[cfg(feature = "fp128-onehot-multi-chunk-w4r2")]
pub mod fp128_onehot_multi_chunk_w4r2;
#[cfg(feature = "fp128-onehot-multi-chunk-w4r2")]
pub mod fp128_onehot_multi_chunk_w4r2_precommitted;
#[cfg(feature = "fp128-onehot")]
pub mod fp128_onehot_precommitted;
#[cfg(feature = "fp32-d128-onehot")]
pub mod fp32_d128_onehot;
#[cfg(feature = "fp32-d128-onehot")]
pub mod fp32_d128_onehot_precommitted;
#[cfg(feature = "fp32-d256-onehot")]
pub mod fp32_d256_onehot;
#[cfg(feature = "fp32-d256-onehot")]
pub mod fp32_d256_onehot_precommitted;
#[cfg(feature = "fp64-d128-dense")]
pub mod fp64_d128_dense;
#[cfg(feature = "fp64-d128-dense")]
pub mod fp64_d128_dense_precommitted;
#[cfg(feature = "fp64-d128-onehot")]
pub mod fp64_d128_onehot;
#[cfg(feature = "fp64-d128-onehot")]
pub mod fp64_d128_onehot_precommitted;
#[cfg(feature = "fp64-d256-onehot")]
pub mod fp64_d256_onehot;
#[cfg(feature = "fp64-d256-onehot")]
pub mod fp64_d256_onehot_precommitted;

#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub fn fp128_d64_onehot_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk::FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES,
        precommitted_profiles: fp128_d64_onehot_multi_chunk_precommitted::FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_d64_onehot_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-recursive")]
pub fn fp128_d64_onehot_recursive_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_recursive::FP128_D64_ONEHOT_RECURSIVE_SCHEDULES,
        precommitted_profiles: fp128_d64_onehot_recursive_precommitted::FP128_D64_ONEHOT_RECURSIVE_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_d64_onehot_recursive::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-recursive-multi-chunk-w8r2")]
pub fn fp128_d64_onehot_recursive_multi_chunk_w8r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_recursive_multi_chunk_w8r2::FP128_D64_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES,
        precommitted_profiles: fp128_d64_onehot_recursive_multi_chunk_w8r2_precommitted::FP128_D64_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_d64_onehot_recursive_multi_chunk_w8r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-dense")]
pub fn fp128_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_dense::FP128_DENSE_SCHEDULES,
        precommitted_profiles:
            fp128_dense_precommitted::FP128_DENSE_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-dense-multi-chunk")]
pub fn fp128_dense_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_dense_multi_chunk::FP128_DENSE_MULTI_CHUNK_SCHEDULES,
        precommitted_profiles: fp128_dense_multi_chunk_precommitted::FP128_DENSE_MULTI_CHUNK_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_dense_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot")]
pub fn fp128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot::FP128_ONEHOT_SCHEDULES,
        precommitted_profiles:
            fp128_onehot_precommitted::FP128_ONEHOT_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-multi-chunk")]
pub fn fp128_onehot_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_multi_chunk::FP128_ONEHOT_MULTI_CHUNK_SCHEDULES,
        precommitted_profiles: fp128_onehot_multi_chunk_precommitted::FP128_ONEHOT_MULTI_CHUNK_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_onehot_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-multi-chunk-w2r2")]
pub fn fp128_onehot_multi_chunk_w2r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_multi_chunk_w2r2::FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES,
        precommitted_profiles: fp128_onehot_multi_chunk_w2r2_precommitted::FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_onehot_multi_chunk_w2r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-multi-chunk-w4r2")]
pub fn fp128_onehot_multi_chunk_w4r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_multi_chunk_w4r2::FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES,
        precommitted_profiles: fp128_onehot_multi_chunk_w4r2_precommitted::FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp128_onehot_multi_chunk_w4r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d128-onehot")]
pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d128_onehot::FP32_D128_ONEHOT_SCHEDULES,
        precommitted_profiles:
            fp32_d128_onehot_precommitted::FP32_D128_ONEHOT_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp32_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d256-onehot")]
pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d256_onehot::FP32_D256_ONEHOT_SCHEDULES,
        precommitted_profiles:
            fp32_d256_onehot_precommitted::FP32_D256_ONEHOT_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp32_d256_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128-dense")]
pub fn fp64_d128_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128_dense::FP64_D128_DENSE_SCHEDULES,
        precommitted_profiles:
            fp64_d128_dense_precommitted::FP64_D128_DENSE_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp64_d128_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128-onehot")]
pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128_onehot::FP64_D128_ONEHOT_SCHEDULES,
        precommitted_profiles:
            fp64_d128_onehot_precommitted::FP64_D128_ONEHOT_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp64_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d256-onehot")]
pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d256_onehot::FP64_D256_ONEHOT_SCHEDULES,
        precommitted_profiles:
            fp64_d256_onehot_precommitted::FP64_D256_ONEHOT_SCHEDULES_PRECOMMITTED_PROFILES,
        identity: fp64_d256_onehot::CATALOG_IDENTITY,
    }
}
// @generated schedule module wiring end
