#![allow(missing_docs)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    /// Stored first-tier `B` rank.
    pub n_b: u32,
    pub n_d: u32,
}

/// Terminal direct-send step in a generated schedule.
///
/// `commit` is `Some` only for a **root-direct** entry (a schedule whose
/// single step is this `Direct`): it carries the brute-forced root commit
/// layout — the same 7-field shape as a fold step — so the runtime can
/// expand it into the committed `LevelParams` via
/// [`GeneratedFoldStep::expand_to_level_params`] without re-running the
/// offline SIS derivation.
///
/// Terminal-direct steps that follow one or more folds ship the cleartext
/// witness without committing, so they carry `commit: None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedDirectStep {
    pub commit: Option<GeneratedFoldStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedStep {
    Fold(GeneratedFoldStep),
    Direct(GeneratedDirectStep),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedCommitmentGroupScheduleKey {
    pub num_vars: usize,
    pub num_polynomials: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedCommitmentGroupLayout {
    pub key: GeneratedCommitmentGroupScheduleKey,
    pub m_vars: usize,
    pub r_vars: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub conservative_n_b: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleLookupKey {
    /// Final group shape for the grouped root commitment.
    pub final_group: GeneratedCommitmentGroupScheduleKey,
    pub precommitteds: &'static [GeneratedCommitmentGroupLayout],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedGroupBatchScheduleTableEntry {
    pub key: GeneratedScheduleLookupKey,
    pub steps: &'static [GeneratedStep],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleTableEntry {
    pub key: GeneratedCommitmentGroupScheduleKey,
    pub steps: &'static [GeneratedStep],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleCatalogIdentity {
    pub family_name: &'static str,
    pub sis_family: SisModulusFamily,
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

    pub root_fold_shape: akita_challenges::TensorChallengeShape,
    pub ring_dimensions: &'static [usize],
    pub ring_challenge_config_digest: u64,
    pub key_count: usize,
    pub key_digest: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratedScheduleTable {
    pub entries: &'static [GeneratedScheduleTableEntry],
    pub group_batch_entries: &'static [GeneratedGroupBatchScheduleTableEntry],
    pub identity: GeneratedScheduleCatalogIdentity,
}

pub mod expand;
pub mod validate;
pub(crate) mod walk;
pub use akita_types::SisModulusFamily;
pub use validate::{
    validate_generated_group_batch_schedule_entry, validate_generated_schedule_entry,
    validate_generated_schedule_table,
};

use core::cmp::Ordering;

/// Lexicographic order used by shipped catalog emission: `num_polynomials`, then `num_vars`.
#[inline]
pub fn catalog_key_cmp(
    a: GeneratedCommitmentGroupScheduleKey,
    b: GeneratedCommitmentGroupScheduleKey,
) -> Ordering {
    a.num_polynomials
        .cmp(&b.num_polynomials)
        .then_with(|| a.num_vars.cmp(&b.num_vars))
}

/// Returns true when `entries` are ordered for [`table_entry`] binary search.
pub fn catalog_entries_sorted_for_lookup(entries: &[GeneratedScheduleTableEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| catalog_key_cmp(window[0].key, window[1].key).is_lt())
}

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: GeneratedCommitmentGroupScheduleKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    debug_assert!(catalog_entries_sorted_for_lookup(table.entries));
    table
        .entries
        .binary_search_by(|entry| catalog_key_cmp(entry.key, key))
        .ok()
        .map(|idx| &table.entries[idx])
}

pub fn group_batch_table_entry(
    table: GeneratedScheduleTable,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Option<&'static GeneratedGroupBatchScheduleTableEntry> {
    debug_assert!(catalog_group_batch_entries_sorted_for_lookup(
        table.group_batch_entries
    ));
    table
        .group_batch_entries
        .binary_search_by(|entry| generated_group_batch_key_cmp_runtime(&entry.key, key))
        .ok()
        .map(|idx| &table.group_batch_entries[idx])
}

/// Returns true when grouped rows are ordered for [`group_batch_table_entry`]
/// binary search.
pub fn catalog_group_batch_entries_sorted_for_lookup(
    entries: &[GeneratedGroupBatchScheduleTableEntry],
) -> bool {
    entries
        .windows(2)
        .all(|window| generated_group_batch_key_cmp(&window[0].key, &window[1].key).is_lt())
}

pub fn generated_group_batch_key_cmp(
    left: &GeneratedScheduleLookupKey,
    right: &GeneratedScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (left.final_group.num_vars, left.final_group.num_polynomials);
    let right_main = (
        right.final_group.num_vars,
        right.final_group.num_polynomials,
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

pub fn generated_group_batch_key_cmp_runtime(
    generated: &GeneratedScheduleLookupKey,
    runtime: &akita_types::AkitaScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (
        generated.final_group.num_vars,
        generated.final_group.num_polynomials,
    );
    let right_main = (
        runtime.final_group.num_vars,
        runtime.final_group.num_polynomials,
    );
    left_main
        .cmp(&right_main)
        .then_with(|| {
            generated
                .precommitteds
                .len()
                .cmp(&runtime.precommitteds.len())
        })
        .then_with(|| precommitted_groups_cmp(generated.precommitteds, &runtime.precommitteds))
}

/// Sort order for runtime grouped keys; matches [`generated_group_batch_key_cmp`].
pub fn runtime_group_batch_key_cmp(
    left: &akita_types::AkitaScheduleLookupKey,
    right: &akita_types::AkitaScheduleLookupKey,
) -> std::cmp::Ordering {
    let left_main = (left.final_group.num_vars, left.final_group.num_polynomials);
    let right_main = (
        right.final_group.num_vars,
        right.final_group.num_polynomials,
    );
    left_main
        .cmp(&right_main)
        .then_with(|| left.precommitteds.len().cmp(&right.precommitteds.len()))
        .then_with(|| {
            left.precommitteds
                .iter()
                .map(runtime_precommitted_group_sort_key)
                .cmp(
                    right
                        .precommitteds
                        .iter()
                        .map(runtime_precommitted_group_sort_key),
                )
        })
}

fn runtime_precommitted_group_sort_key(
    key: &akita_types::CommitmentGroupLayout,
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

fn precommitted_groups_cmp(
    generated: &[GeneratedCommitmentGroupLayout],
    runtime: &[akita_types::CommitmentGroupLayout],
) -> std::cmp::Ordering {
    generated
        .iter()
        .zip(runtime)
        .map(|(left, right)| {
            precommitted_group_sort_key(left).cmp(&(
                right.key.num_vars,
                right.key.num_polynomials,
                right.m_vars,
                right.r_vars,
                right.log_basis,
                right.n_a,
                right.conservative_n_b,
            ))
        })
        .find(|ord| *ord != std::cmp::Ordering::Equal)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn precommitted_group_sort_key(
    key: &GeneratedCommitmentGroupLayout,
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

fn group_batch_key_eq(
    generated: &GeneratedScheduleLookupKey,
    key: &akita_types::AkitaScheduleLookupKey,
) -> bool {
    generated.final_group
        == GeneratedCommitmentGroupScheduleKey {
            num_vars: key.final_group.num_vars,
            num_polynomials: key.final_group.num_polynomials,
        }
        && generated.precommitteds.len() == key.precommitteds.len()
        && generated
            .precommitteds
            .iter()
            .zip(&key.precommitteds)
            .all(|(generated, layout)| precommitted_group_key_eq(generated, layout))
}

fn precommitted_group_key_eq(
    generated: &GeneratedCommitmentGroupLayout,
    layout: &akita_types::CommitmentGroupLayout,
) -> bool {
    generated.key
        == GeneratedCommitmentGroupScheduleKey {
            num_vars: layout.key.num_vars,
            num_polynomials: layout.key.num_polynomials,
        }
        && generated.m_vars == layout.m_vars
        && generated.r_vars == layout.r_vars
        && generated.log_basis == layout.log_basis
        && generated.n_a == layout.n_a
        && generated.conservative_n_b == layout.conservative_n_b
}

/// Returns an error when the generated grouped key does not match the runtime key.
pub(crate) fn validate_group_batch_entry_key(
    generated: &GeneratedScheduleLookupKey,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Result<(), akita_field::AkitaError> {
    if group_batch_key_eq(generated, key) {
        Ok(())
    } else {
        Err(akita_field::AkitaError::InvalidSetup(
            "generated grouped schedule key mismatch".to_string(),
        ))
    }
}

/// Build a runtime grouped key from a generated catalog row key.
pub(crate) fn runtime_key_from_generated(
    key: &GeneratedScheduleLookupKey,
) -> akita_types::AkitaScheduleLookupKey {
    use akita_types::{AkitaScheduleLookupKey, CommitmentGroupLayout, CommitmentGroupScheduleKey};

    AkitaScheduleLookupKey {
        final_group: CommitmentGroupScheduleKey::new(
            key.final_group.num_vars,
            key.final_group.num_polynomials,
        ),
        precommitteds: key
            .precommitteds
            .iter()
            .map(|group| CommitmentGroupLayout {
                key: CommitmentGroupScheduleKey::new(group.key.num_vars, group.key.num_polynomials),
                m_vars: group.m_vars,
                r_vars: group.r_vars,
                log_basis: group.log_basis,
                n_a: group.n_a,
                conservative_n_b: group.conservative_n_b,
            })
            .collect(),
    }
}
