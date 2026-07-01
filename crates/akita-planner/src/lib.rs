//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! This crate is a **pure, `Cfg`-free DP library**. The single entry point
//! is [`find_schedule`], which runs an exhaustive dynamic program to
//! minimize proof size for a schedule lookup key. Every per-preset input is
//! carried by the plain-value [`PlannerPolicy`] plus a `ring_challenge_config` /
//! `fold_challenge_shape_at_level` closure pair, so the planner names no `CommitmentConfig`
//! types and depends only on `akita-types` / `akita-challenges` /
//! `akita-field`.
//!
//! The preset family list, the `gen_schedule_tables` binary, and the
//! `policy_of::<Cfg>()` bridge that derives a [`PlannerPolicy`] from a preset
//! live in `akita-config`, the only crate that can name the presets.

pub use akita_types::{DecompositionParams, SisModulusFamily};

pub mod catalog_identity;
pub mod emit;
pub mod generated;
mod group_batch;
mod resolve;
pub mod schedule_params;

pub use akita_challenges::TensorChallengeShape;
pub use catalog_identity::{
    expected_catalog_identity, identity_digest, key_digest, policy_digest,
    ring_challenge_config_digest, validate_catalog_identity,
};
pub use emit::{
    refresh_generated_wiring, run_regen_fmt, write_family_module, write_group_batch_family_module,
    EmitSpec,
};
pub use generated::{
    catalog_entries_sorted_for_lookup, validate_generated_schedule_entry,
    validate_generated_schedule_table, GeneratedScheduleCatalogIdentity, GeneratedScheduleTable,
};
pub use group_batch::find_group_batch_schedule;
pub use resolve::{
    estimate_proof_bytes, generated_schedule_lookup_key, resolve_group_batch_schedule,
    resolve_schedule, schedule_from_entry,
};
pub use schedule_params::find_schedule;

/// Plain-value brute-force inputs the planner DP needs.
///
/// This is the `Cfg`-free projection of a `CommitmentConfig` preset that
/// the DP and SIS sizing read. `akita-config` derives it from a preset via
/// its `policy_of::<Cfg>()` bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannerPolicy {
    /// Ring degree `D` (`Cfg::D`).
    pub ring_dimension: usize,
    /// Gadget base + coefficient bounds (`Cfg::decomposition()`).
    pub decomposition: DecompositionParams,
    /// SIS modulus family (`Cfg::sis_modulus_family()`).
    pub sis_family: SisModulusFamily,
    /// `psi`-embedding infinity-norm expansion
    /// (`Cfg::ring_subfield_embedding_norm_bound()`).
    pub ring_subfield_norm_bound: u32,
    /// Opening-reduction extension width (`Cfg::EXT_DEGREE`).
    pub claim_ext_degree: usize,
    /// Fiat-Shamir scalar extension width (`Cfg::EXT_DEGREE`).
    pub chal_ext_degree: usize,
    /// Inclusive `(min, max)` log-basis search range (`Cfg::basis_range()`).
    pub basis_range: (u32, u32),
    /// One-hot chunk size `K` (`Cfg::onehot_chunk_size()`).
    ///
    /// Used to bound the committed one-hot witness L1 mass per ring element
    /// (`nonzeros = ceil(D/K)`) for the weak-binding collision norm and the
    /// folded-witness digit count. Only consulted at a root level whose
    /// `log_commit_bound == 1`; dense levels use `nonzeros = D`.
    pub onehot_chunk_size: usize,
    /// Enable the tiered second commitment matrix `F` (`Cfg::TIERED_COMMITMENT`).
    ///
    /// When `true`, [`schedule_params`] is allowed to reuse a smaller first-tier
    /// matrix `B` across `f` witness slices and size a second-tier matrix `F`
    /// per level whose first-tier footprint would otherwise exceed `A`. When
    /// `false`, every level emits `tier_split == 1` / `f_key == None` and the
    /// layout is identical to the historical single-tier scheme. Also keys the
    /// tiered schedule catalog so a tiered policy never aliases a
    /// non-tiered table.
    pub tiered: bool,
}
