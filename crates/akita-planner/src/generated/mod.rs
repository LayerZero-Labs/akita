#![allow(missing_docs)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedSetupPrefixGroup {
    pub natural_len: u32,
    pub m_vars: u32,
    pub r_vars: u32,
    pub n_a: u32,
    pub n_b: u32,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedFoldStepWithSetupMetadata {
    pub fold: GeneratedFoldStep,
    pub setup_prefix_group: Option<GeneratedSetupPrefixGroup>,
    pub setup_contribution_mode: akita_types::SetupContributionMode,
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
    FoldWithSetupMetadata(GeneratedFoldStepWithSetupMetadata),
    Direct(GeneratedDirectStep),
}

impl GeneratedStep {
    pub fn fold_step(&self) -> Option<&GeneratedFoldStep> {
        match self {
            Self::Fold(step) => Some(step),
            Self::FoldWithSetupMetadata(step) => Some(&step.fold),
            Self::Direct(_) => None,
        }
    }

    pub fn fold_step_mut(&mut self) -> Option<&mut GeneratedFoldStep> {
        match self {
            Self::Fold(step) => Some(step),
            Self::FoldWithSetupMetadata(step) => Some(&mut step.fold),
            Self::Direct(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleTableEntry {
    pub final_group: akita_types::PolynomialGroupLayout,
    pub precommitteds: &'static [akita_types::PrecommittedGroupParams],
    pub steps: &'static [GeneratedStep],
}

impl GeneratedScheduleTableEntry {
    /// Build the runtime schedule lookup key represented by this generated row.
    pub(crate) fn to_runtime_lookup_key(self) -> akita_types::AkitaScheduleLookupKey {
        akita_types::AkitaScheduleLookupKey {
            final_group: self.final_group,
            precommitteds: self.precommitteds.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratedScheduleCatalogIdentity {
    pub family_name: &'static str,
    pub sis_family: SisModulusFamily,
    pub min_sis_security_bits: u16,
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
    pub entries: &'static [GeneratedScheduleTableEntry],
    pub identity: GeneratedScheduleCatalogIdentity,
}

pub mod expand;
pub mod validate;
pub(crate) mod walk;
pub use akita_types::SisModulusFamily;
pub use akita_types::{PolynomialGroupLayout, PrecommittedGroupParams, SetupContributionMode};
pub use validate::{validate_generated_schedule_entry, validate_generated_schedule_table};

/// Returns true when `entries` are ordered for [`table_entry`] binary search.
pub fn catalog_entries_sorted_for_lookup(entries: &[GeneratedScheduleTableEntry]) -> bool {
    entries
        .windows(2)
        .all(|window| generated_schedule_key_cmp(&window[0], &window[1]).is_lt())
}

pub fn table_entry(
    table: GeneratedScheduleTable,
    key: &akita_types::AkitaScheduleLookupKey,
) -> Option<&'static GeneratedScheduleTableEntry> {
    debug_assert!(catalog_entries_sorted_for_lookup(table.entries));
    table
        .entries
        .binary_search_by(|entry| generated_schedule_key_cmp_runtime(entry, key))
        .ok()
        .map(|idx| &table.entries[idx])
}

pub fn generated_schedule_key_cmp(
    left: &GeneratedScheduleTableEntry,
    right: &GeneratedScheduleTableEntry,
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

pub fn generated_schedule_key_cmp_runtime(
    generated: &GeneratedScheduleTableEntry,
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
                .precommitteds
                .len()
                .cmp(&runtime.precommitteds.len())
        })
        .then_with(|| precommitted_groups_cmp(generated.precommitteds, &runtime.precommitteds))
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

fn precommitted_groups_cmp(
    generated: &[akita_types::PrecommittedGroupParams],
    runtime: &[akita_types::PrecommittedGroupParams],
) -> std::cmp::Ordering {
    generated
        .iter()
        .zip(runtime)
        .map(|(left, right)| {
            precommitted_group_sort_key(left).cmp(&precommitted_group_sort_key(right))
        })
        .find(|ord| *ord != std::cmp::Ordering::Equal)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn precommitted_group_sort_key(
    key: &akita_types::PrecommittedGroupParams,
) -> (usize, usize, usize, usize, u32, usize, usize) {
    (
        key.group.num_vars(),
        key.group.num_polynomials(),
        key.m_vars,
        key.r_vars,
        key.log_basis,
        key.n_a,
        key.conservative_n_b,
    )
}

fn schedule_key_eq(
    generated: &GeneratedScheduleTableEntry,
    key: &akita_types::AkitaScheduleLookupKey,
) -> bool {
    generated.final_group == key.final_group
        && generated.precommitteds.len() == key.precommitteds.len()
        && generated
            .precommitteds
            .iter()
            .zip(&key.precommitteds)
            .all(|(generated, layout)| precommitted_group_key_eq(generated, layout))
}

fn precommitted_group_key_eq(
    generated: &akita_types::PrecommittedGroupParams,
    layout: &akita_types::PrecommittedGroupParams,
) -> bool {
    generated.group == layout.group
        && generated.m_vars == layout.m_vars
        && generated.r_vars == layout.r_vars
        && generated.log_basis == layout.log_basis
        && generated.n_a == layout.n_a
        && generated.conservative_n_b == layout.conservative_n_b
}

/// Returns an error when the generated key does not match the runtime key.
pub(crate) fn validate_entry_key(
    generated: &GeneratedScheduleTableEntry,
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
