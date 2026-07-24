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
pub enum GeneratedRootSource {
    Dense { coefficient_bits: u32 },
    OneHot { chunk_size: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedRootFinalChallenge {
    Flat,
    Tensor { fold_low_len: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootFinalGroup {
    pub layout: akita_types::PolynomialGroupLayout,
    pub source: GeneratedRootSource,
    pub challenge: GeneratedRootFinalChallenge,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootPrecommittedGroup {
    pub descriptor: akita_types::PrecommittedGroupDescriptor,
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
    pub d_setup: u32,
    pub commitment: GeneratedCommittedGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRecursiveFold {
    pub witness: GeneratedCommittedGroup,
    pub open_commit_matrix: GeneratedOpenCommitMatrix,
    pub incoming_setup_prefix: Option<GeneratedSetupPrefixInput>,
    pub witness_partition: GeneratedWitnessPartition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedTerminalFold {
    pub geometry: GeneratedBlockGeometry,
    pub inner_commit_matrix: GeneratedInnerCommitMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldScheduleEntry {
    pub root: GeneratedRootFold,
    pub recursive_folds: &'static [GeneratedRecursiveFold],
    pub terminal: GeneratedTerminalFold,
}

impl GeneratedFoldScheduleEntry {
    /// Build the runtime schedule lookup key represented by this generated row.
    pub(crate) fn to_runtime_lookup_key(self) -> akita_types::AkitaScheduleLookupKey {
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
    pub max_setup_envelope_field_elements: usize,
    pub min_offloaded_witness_contraction: usize,
    pub sis_modulus_profile: SisModulusProfileId,
    pub sis_security_policy: akita_types::SisSecurityPolicyId,
    pub sis_table_digest: akita_types::SisTableDigest,
    pub ring_dimension: usize,
    pub decomposition: akita_types::DecompositionParams,
    pub ring_subfield_norm_bound: u32,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    pub basis_range: (u32, u32),
    pub onehot_chunk_size: usize,
    /// Multi-chunk witness layout this table was emitted under. A chunked policy
    /// never aliases a single-chunk table (and vice versa), even when row keys
    /// match. `ChunkedWitnessCfg::default()` for single-chunk tables.
    pub witness_chunk: akita_types::ChunkedWitnessCfg,
    pub recursive_setup_planning: bool,

    pub root_fold_shape: akita_challenges::TensorChallengeShape,
    pub ring_dimensions: &'static [usize],
    pub ring_challenge_config_digest: u64,
    pub key_count: usize,
    pub key_digest: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratedScheduleTable {
    pub entries: &'static [GeneratedFoldScheduleEntry],
    pub identity: GeneratedScheduleCatalogIdentity,
}

pub mod expand;
pub mod validate;
pub(crate) mod walk;
pub use crate::{
    ChunkedWitnessCfg, DecompositionParams, PlannerCostModelId, SelectionPolicyId,
    SisSecurityPolicyId, TensorChallengeShape,
};
pub use akita_types::{PolynomialGroupLayout, PrecommittedGroupDescriptor};
pub use akita_types::{SisModulusProfileId, SisTableDigest};
pub use validate::{validate_generated_schedule_entry, validate_generated_schedule_table};

/// Returns true when `entries` are ordered for [`table_entry`] binary search.
pub fn catalog_entries_sorted_for_lookup(entries: &[GeneratedFoldScheduleEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| generated_schedule_key_cmp(&window[0], &window[1]).is_lt())
}

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Option<&'static GeneratedFoldScheduleEntry> {
    debug_assert!(catalog_entries_sorted_for_lookup(table.entries));
    table
        .entries
        .binary_search_by(|entry| generated_schedule_key_cmp_runtime(entry, key))
        .ok()
        .map(|idx| &table.entries[idx])
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

fn precommitted_group_sort_key(
    key: &akita_types::PrecommittedGroupDescriptor,
) -> (usize, usize, usize, usize, usize, u32, u32, usize, usize) {
    (
        key.group.num_vars(),
        key.group.num_polynomials(),
        key.num_live_ring_elements_per_claim,
        key.num_positions_per_block,
        key.num_live_blocks,
        key.log_basis_inner,
        key.log_basis_outer,
        key.n_a,
        key.n_b,
    )
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
    generated: &akita_types::PrecommittedGroupDescriptor,
    layout: &akita_types::PrecommittedGroupDescriptor,
) -> bool {
    generated.group == layout.group
        && generated.num_live_ring_elements_per_claim == layout.num_live_ring_elements_per_claim
        && generated.num_positions_per_block == layout.num_positions_per_block
        && generated.num_live_blocks == layout.num_live_blocks
        && generated.log_basis_inner == layout.log_basis_inner
        && generated.log_basis_outer == layout.log_basis_outer
        && generated.n_a == layout.n_a
        && generated.a_coeff_linf_bound == layout.a_coeff_linf_bound
        && generated.n_b == layout.n_b
        && generated.b_coeff_linf_bound == layout.b_coeff_linf_bound
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
#[cfg(feature = "fp128-d128-dense")]
pub mod fp128_d128_dense;
#[cfg(feature = "fp128-d128-onehot")]
pub mod fp128_d128_onehot;
#[cfg(feature = "fp128-d64-dense")]
pub mod fp128_d64_dense;
#[cfg(feature = "fp128-d64-dense-multi-chunk")]
pub mod fp128_d64_dense_multi_chunk;
#[cfg(feature = "fp128-d64-onehot")]
pub mod fp128_d64_onehot;
#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub mod fp128_d64_onehot_multi_chunk;
#[cfg(feature = "fp128-d64-onehot-multi-chunk-w2r2")]
pub mod fp128_d64_onehot_multi_chunk_w2r2;
#[cfg(feature = "fp128-d64-onehot-multi-chunk-w4r2")]
pub mod fp128_d64_onehot_multi_chunk_w4r2;
#[cfg(feature = "fp128-d64-onehot-recursive")]
pub mod fp128_d64_onehot_recursive;
#[cfg(feature = "fp128-d64-onehot-recursive-multi-chunk-w8r2")]
pub mod fp128_d64_onehot_recursive_multi_chunk_w8r2;
#[cfg(feature = "fp128-d64-onehot-tensor")]
pub mod fp128_d64_onehot_tensor;
#[cfg(feature = "fp32-d128-onehot")]
pub mod fp32_d128_onehot;
#[cfg(feature = "fp32-d256-onehot")]
pub mod fp32_d256_onehot;
#[cfg(feature = "fp64-d128-dense")]
pub mod fp64_d128_dense;
#[cfg(feature = "fp64-d128-onehot")]
pub mod fp64_d128_onehot;
#[cfg(feature = "fp64-d256-onehot")]
pub mod fp64_d256_onehot;

#[cfg(feature = "fp128-d128-dense")]
pub fn fp128_d128_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_dense::FP128_D128_DENSE_SCHEDULES,
        identity: fp128_d128_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d128-onehot")]
pub fn fp128_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d128_onehot::FP128_D128_ONEHOT_SCHEDULES,
        identity: fp128_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-dense")]
pub fn fp128_d64_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_dense::FP128_D64_DENSE_SCHEDULES,
        identity: fp128_d64_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-dense-multi-chunk")]
pub fn fp128_d64_dense_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_dense_multi_chunk::FP128_D64_DENSE_MULTI_CHUNK_SCHEDULES,
        identity: fp128_d64_dense_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot")]
pub fn fp128_d64_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot::FP128_D64_ONEHOT_SCHEDULES,
        identity: fp128_d64_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk")]
pub fn fp128_d64_onehot_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk::FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES,
        identity: fp128_d64_onehot_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk-w2r2")]
pub fn fp128_d64_onehot_multi_chunk_w2r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk_w2r2::FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES,
        identity: fp128_d64_onehot_multi_chunk_w2r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-multi-chunk-w4r2")]
pub fn fp128_d64_onehot_multi_chunk_w4r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_multi_chunk_w4r2::FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES,
        identity: fp128_d64_onehot_multi_chunk_w4r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-recursive")]
pub fn fp128_d64_onehot_recursive_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_recursive::FP128_D64_ONEHOT_RECURSIVE_SCHEDULES,
        identity: fp128_d64_onehot_recursive::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-recursive-multi-chunk-w8r2")]
pub fn fp128_d64_onehot_recursive_multi_chunk_w8r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_recursive_multi_chunk_w8r2::FP128_D64_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES,
        identity: fp128_d64_onehot_recursive_multi_chunk_w8r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-d64-onehot-tensor")]
pub fn fp128_d64_onehot_tensor_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_d64_onehot_tensor::FP128_D64_ONEHOT_TENSOR_SCHEDULES,
        identity: fp128_d64_onehot_tensor::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d128-onehot")]
pub fn fp32_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d128_onehot::FP32_D128_ONEHOT_SCHEDULES,
        identity: fp32_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-d256-onehot")]
pub fn fp32_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_d256_onehot::FP32_D256_ONEHOT_SCHEDULES,
        identity: fp32_d256_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128-dense")]
pub fn fp64_d128_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128_dense::FP64_D128_DENSE_SCHEDULES,
        identity: fp64_d128_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d128-onehot")]
pub fn fp64_d128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d128_onehot::FP64_D128_ONEHOT_SCHEDULES,
        identity: fp64_d128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-d256-onehot")]
pub fn fp64_d256_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_d256_onehot::FP64_D256_ONEHOT_SCHEDULES,
        identity: fp64_d256_onehot::CATALOG_IDENTITY,
    }
}
// @generated schedule module wiring end
