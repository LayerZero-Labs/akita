#![allow(missing_docs)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStep {
    pub ring_d: u32,
    pub log_basis: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    /// Stored first-tier `B` rank. This is the actual committed rank: the shrunk
    /// `B'` rank when the step is tiered (`tier_split.is_some()`), and the full
    /// `B` rank otherwise.
    pub n_b: u32,
    pub n_d: u32,
    /// Tiered split factor `f`. `None` for single-tier steps; `Some(f)` when the
    /// step reuses a smaller `B'` across `f` column-slices (paired with `n_f`).
    pub tier_split: Option<u32>,
    /// Second-tier `F` rank. `None` for single-tier steps; `Some` iff
    /// `tier_split` is `Some`. Expansion sizes `F` from `tier_split`, `n_b`, and
    /// the level's `num_digits_open`.
    pub n_f: Option<u32>,
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
pub struct GeneratedScheduleKey {
    pub num_vars: usize,
    pub num_polynomials: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedPrecommittedGroupKey {
    pub key: GeneratedScheduleKey,
    pub m_vars: usize,
    pub r_vars: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub conservative_n_b: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedGroupBatchScheduleKey {
    /// Main group shape for the final commitment.
    pub main: akita_types::AkitaScheduleLookupKey,
    pub precommitteds: &'static [GeneratedPrecommittedGroupKey],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleTableEntry {
    pub key: GeneratedScheduleKey,
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
    pub tiered: bool,
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
    pub identity: GeneratedScheduleCatalogIdentity,
}

pub mod expand;
pub mod validate;
pub(crate) mod walk;
pub use akita_types::SisModulusFamily;
pub use validate::{validate_generated_schedule_entry, validate_generated_schedule_table};

use core::cmp::Ordering;

/// Lexicographic order used by shipped catalog emission: `num_polynomials`, then `num_vars`.
#[inline]
pub fn catalog_key_cmp(a: GeneratedScheduleKey, b: GeneratedScheduleKey) -> Ordering {
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
    key: GeneratedScheduleKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    debug_assert!(catalog_entries_sorted_for_lookup(table.entries));
    table
        .entries
        .binary_search_by(|entry| catalog_key_cmp(entry.key, key))
        .ok()
        .map(|idx| &table.entries[idx])
}
