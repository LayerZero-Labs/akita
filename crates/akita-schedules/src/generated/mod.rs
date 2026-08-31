#![allow(missing_docs)]

/// Compact planner input for any commitment matrix.
///
/// The containing field identifies whether this is an inner, outer, opening,
/// or terminal matrix. Expansion re-derives its rank against the checked-in SIS
/// table, so generated rows only store the chosen ring dimension and basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedMatrix {
    pub ring_dimension: u32,
    pub log_basis: u32,
}

/// One group a fold derives and commits itself.
///
/// The root-only inner digit depth lives on [`GeneratedRootFold`]. Recursive
/// folds derive it from their incoming witness, so it cannot be set here.
///
/// The group's polynomial layout is deliberately *not* here. It is a property
/// of the row's lookup key, not of a group, so it lives on
/// `GeneratedFoldScheduleEntry` where every row has exactly one and no position
/// has to default it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedGroup {
    pub geometry: akita_types::BlockGeometry,
    pub inner_commit_matrix: GeneratedMatrix,
    pub outer_commit_matrix: GeneratedMatrix,
    pub outer_slice_count: u32,
    pub num_digits_fold: u32,
    pub opening_method: akita_types::OpeningMethod,
}

/// One group's exact frozen commit and opening inputs.
///
/// This stays frozen rather than using compact inputs because
/// `to_runtime_lookup_key` has no `PlannerPolicy` and drives catalog binary
/// search: deriving the profile would make key comparison depend on expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFrozenGroup {
    pub profile: akita_types::GroupCommitPhaseParams,
    /// Procedure the consuming fold uses to open this group.
    pub opening_method: akita_types::OpeningMethod,
    /// Folded-witness digit depth for this group.
    pub num_digits_fold: u32,
}

/// A group committed before the root fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedPrecommittedGroup {
    pub group: GeneratedFrozenGroup,
}

/// A setup output consumed as the first group of a recursive fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedSetupPrefix {
    pub group: GeneratedFrozenGroup,
    pub natural_len: u64,
}

/// Fields executed by every nonterminal fold level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldCore {
    pub group: GeneratedGroup,
    pub open_commit_matrix: GeneratedMatrix,
    pub witness_chunks: u32,
    pub ring_relation_mode: akita_types::RingRelationMode,
}

/// Generated fields that are legal only at the root fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRootFold {
    pub core: GeneratedFoldCore,
    pub num_digits_inner: u32,
    pub precommitted_groups: &'static [GeneratedPrecommittedGroup],
}

/// Generated fields that are legal only at a recursive fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedRecursiveFold {
    pub core: GeneratedFoldCore,
    pub setup_prefix: Option<GeneratedSetupPrefix>,
    pub payload_mode: akita_types::CommitmentPayloadMode,
    pub response_l2_sq_cap: Option<u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedTerminalFold {
    pub geometry: akita_types::BlockGeometry,
    pub inner_commit_matrix: GeneratedMatrix,
    pub num_digits_inner: u32,
    pub fold_log_basis: u32,
    pub fold_digit_count: u32,
    pub inner_output_rank: u32,
    pub inner_coeff_linf_bound: u128,
    pub response_l2_sq_cap: Option<u128>,
    pub z_linf_cap: Option<u128>,
    pub z_rice_low_bits: u32,
    pub z_payload_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldScheduleEntry {
    /// Layout of this row's final group: the row's own lookup key.
    pub final_group: akita_types::PolynomialGroupLayout,
    pub root: GeneratedRootFold,
    pub recursive_folds: &'static [GeneratedRecursiveFold],
    pub terminal: GeneratedTerminalFold,
}

impl GeneratedFoldScheduleEntry {
    /// Build the runtime schedule lookup key represented by this generated row.
    pub fn to_runtime_lookup_key(self) -> akita_types::AkitaScheduleLookupKey {
        akita_types::AkitaScheduleLookupKey {
            final_group: self.final_group,
            precommitteds: self
                .root
                .precommitted_groups
                .iter()
                .map(|group| group.group.profile)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleCatalogIdentity {
    pub family_name: &'static str,
    pub protocol_epoch: u32,
    pub cost_model: crate::PlannerCostModelId,
    pub selective_l2_response_model: crate::SelectiveL2ResponseModelId,
    pub selection_policy: crate::SelectionPolicyId,
    pub recursive_split_search_policy: crate::RecursiveSplitSearchPolicy,
    pub recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy,
    pub setup_field_budget: Option<usize>,
    pub min_offloaded_witness_contraction: usize,
    pub sis_modulus_profile: SisModulusProfileId,
    pub sis_security_policy: akita_types::SisSecurityPolicyId,
    pub sis_table_digest: akita_types::SisTableDigest,
    pub sis_l2_table_digest: akita_types::SisL2TableDigest,
    pub decomposition: akita_types::DecompositionParams,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    pub inner_basis_range: (u32, u32),
    pub opening_basis_range: (u32, u32),
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
    pub identity: GeneratedScheduleCatalogIdentity,
}

pub mod expand;
pub mod validate;
pub(crate) mod walk;
pub use crate::{
    ChunkedWitnessCfg, CommitmentRingDims, DecompositionParams, PlannerCostModelId,
    RecursiveSetupSearchPolicy, RecursiveSplitSearchPolicy, RingDimensionScheduleMode,
    SelectionPolicyId, SelectiveL2ResponseModelId, SisSecurityPolicyId,
};
pub use akita_types::{
    BlockGeometry, CommitmentPayloadMode, GroupCommitPhaseParams, InnerCommitMatrixParams,
    OuterCommitMatrixParams, PolynomialGroupLayout,
};
pub use akita_types::{SisL2TableDigest, SisModulusProfileId, SisTableDigest};
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
        left.final_group.num_vars(),
        left.final_group.num_polynomials(),
    );
    let right_main = (
        right.final_group.num_vars(),
        right.final_group.num_polynomials(),
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
                .map(|group| precommitted_group_sort_key(&group.group.profile))
                .cmp(
                    right
                        .root
                        .precommitted_groups
                        .iter()
                        .map(|group| precommitted_group_sort_key(&group.group.profile)),
                )
        })
}

pub fn generated_schedule_key_cmp_runtime(
    generated: &GeneratedFoldScheduleEntry,
    runtime: &akita_types::AkitaScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (
        generated.final_group.num_vars(),
        generated.final_group.num_polynomials(),
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
                .map(|group| &group.group.profile);
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

fn precommitted_group_sort_key(key: &akita_types::GroupCommitPhaseParams) -> Vec<u8> {
    key.canonical_descriptor_bytes()
}

fn schedule_key_eq(
    generated: &GeneratedFoldScheduleEntry,
    key: &akita_types::AkitaScheduleLookupKey,
) -> bool {
    generated.final_group == key.final_group
        && generated.root.precommitted_groups.len() == key.precommitteds.len()
        && generated
            .root
            .precommitted_groups
            .iter()
            .zip(&key.precommitteds)
            .all(|(generated, layout)| precommitted_group_key_eq(&generated.group.profile, layout))
}

fn precommitted_group_key_eq(
    generated: &akita_types::GroupCommitPhaseParams,
    layout: &akita_types::GroupCommitPhaseParams,
) -> bool {
    generated == layout
}

/// Returns an error when the generated key does not match the runtime key.
pub(crate) fn validate_entry_key(
    generated: &GeneratedFoldScheduleEntry,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Result<(), akita_error::AkitaError> {
    if schedule_key_eq(generated, key) {
        Ok(())
    } else {
        Err(akita_error::AkitaError::InvalidSetup(
            "generated schedule key mismatch".to_string(),
        ))
    }
}

// @generated schedule module wiring begin
#[cfg(feature = "fp128-dense")]
pub mod fp128_dense;
#[cfg(feature = "fp128-dense-bounded")]
pub mod fp128_dense_bounded;
#[cfg(feature = "fp128-dense-multi-chunk")]
pub mod fp128_dense_multi_chunk;
#[cfg(feature = "fp128-onehot")]
pub mod fp128_onehot;
#[cfg(feature = "fp128-onehot-multi-chunk")]
pub mod fp128_onehot_multi_chunk;
#[cfg(feature = "fp128-onehot-multi-chunk-w2r2")]
pub mod fp128_onehot_multi_chunk_w2r2;
#[cfg(feature = "fp128-onehot-multi-chunk-w4r2")]
pub mod fp128_onehot_multi_chunk_w4r2;
#[cfg(feature = "fp128-onehot-recursive")]
pub mod fp128_onehot_recursive;
#[cfg(feature = "fp128-onehot-recursive-multi-chunk-w8r2")]
pub mod fp128_onehot_recursive_multi_chunk_w8r2;
#[cfg(feature = "fp32-dense")]
pub mod fp32_dense;
#[cfg(feature = "fp32-onehot")]
pub mod fp32_onehot;
#[cfg(feature = "fp64-dense")]
pub mod fp64_dense;
#[cfg(feature = "fp64-onehot")]
pub mod fp64_onehot;

#[cfg(feature = "fp128-dense")]
pub fn fp128_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_dense::FP128_DENSE_SCHEDULES,
        identity: fp128_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-dense-bounded")]
pub fn fp128_dense_bounded_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_dense_bounded::FP128_DENSE_BOUNDED_SCHEDULES,
        identity: fp128_dense_bounded::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-dense-multi-chunk")]
pub fn fp128_dense_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_dense_multi_chunk::FP128_DENSE_MULTI_CHUNK_SCHEDULES,
        identity: fp128_dense_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot")]
pub fn fp128_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot::FP128_ONEHOT_SCHEDULES,
        identity: fp128_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-multi-chunk")]
pub fn fp128_onehot_multi_chunk_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_multi_chunk::FP128_ONEHOT_MULTI_CHUNK_SCHEDULES,
        identity: fp128_onehot_multi_chunk::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-multi-chunk-w2r2")]
pub fn fp128_onehot_multi_chunk_w2r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_multi_chunk_w2r2::FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES,
        identity: fp128_onehot_multi_chunk_w2r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-multi-chunk-w4r2")]
pub fn fp128_onehot_multi_chunk_w4r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_multi_chunk_w4r2::FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES,
        identity: fp128_onehot_multi_chunk_w4r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-recursive")]
pub fn fp128_onehot_recursive_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_recursive::FP128_ONEHOT_RECURSIVE_SCHEDULES,
        identity: fp128_onehot_recursive::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp128-onehot-recursive-multi-chunk-w8r2")]
pub fn fp128_onehot_recursive_multi_chunk_w8r2_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp128_onehot_recursive_multi_chunk_w8r2::FP128_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES,
        identity: fp128_onehot_recursive_multi_chunk_w8r2::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-dense")]
pub fn fp32_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_dense::FP32_DENSE_SCHEDULES,
        identity: fp32_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp32-onehot")]
pub fn fp32_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp32_onehot::FP32_ONEHOT_SCHEDULES,
        identity: fp32_onehot::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-dense")]
pub fn fp64_dense_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_dense::FP64_DENSE_SCHEDULES,
        identity: fp64_dense::CATALOG_IDENTITY,
    }
}

#[cfg(feature = "fp64-onehot")]
pub fn fp64_onehot_table() -> GeneratedScheduleTable {
    GeneratedScheduleTable {
        entries: fp64_onehot::FP64_ONEHOT_SCHEDULES,
        identity: fp64_onehot::CATALOG_IDENTITY,
    }
}
// @generated schedule module wiring end

#[cfg(test)]
mod mixed_dimension_key_tests {
    use super::{precommitted_group_key_eq, precommitted_group_sort_key};
    use akita_types::{
        GroupCommitPhaseParams, InnerCommitMatrixParams, OuterCommitMatrixParams,
        PolynomialGroupLayout, SisModulusProfileId, SisTableDigest,
    };

    fn descriptor() -> GroupCommitPhaseParams {
        GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::new(12, 1),
            blocks: akita_types::BlockGeometry::new(32, 8, 4),
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            inner: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(4, 2),
                InnerCommitMatrixParams::new_unchecked(
                    akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
                    SisTableDigest::CURRENT,
                    SisModulusProfileId::Q128OffsetA7F7,
                    3,
                    16,
                    7,
                    128,
                ),
            ),
            outer: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(5, 2),
                OuterCommitMatrixParams::new_unchecked(
                    akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
                    SisTableDigest::CURRENT,
                    SisModulusProfileId::Q128OffsetA7F7,
                    2,
                    48,
                    11,
                    64,
                ),
            ),
        }
    }

    #[test]
    fn precommitted_key_identity_includes_both_native_ring_dimensions() {
        let base = descriptor();
        let mut changed_inner = base;
        changed_inner.inner.matrix = InnerCommitMatrixParams::new_unchecked(
            akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            3,
            16,
            7,
            64,
        );
        let mut changed_outer = base;
        changed_outer.outer.matrix = OuterCommitMatrixParams::new_unchecked(
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
