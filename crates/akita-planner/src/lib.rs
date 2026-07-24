//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! This crate is a **pure, `Cfg`-free DP library**. The DP entry point
//! is [`find_group_batch_schedule`], which runs an exhaustive dynamic program to
//! minimize proof size for a schedule lookup key. Every per-preset input is
//! carried by the plain-value [`PlannerPolicy`] plus a `ring_challenge_config` /
//! `fold_challenge_shape_at_level` closure pair, so the planner names no `CommitmentConfig`
//! types and depends only on `akita-schedules` / `akita-types` /
//! `akita-challenges` / `akita-field`.
//!
//! With the `catalog-gen` feature enabled, this crate also owns the offline
//! generated-table family list and `gen_schedule_tables` binary. That feature
//! is allowed to name `akita-config` presets; normal planner use remains
//! preset-free.

pub use akita_schedules::{
    ChunkedWitnessCfg, DecompositionParams, PlannerCostModelId, PlannerPolicy, SelectionPolicyId,
    SisModulusProfileId, SisSecurityPolicyId, DEFAULT_SIS_SECURITY_POLICY,
};

pub mod emit;
#[cfg(feature = "catalog-gen")]
pub mod generated_families;
mod group_batch;
pub mod schedule_params;

pub use akita_challenges::TensorChallengeShape;
pub use akita_schedules::{
    catalog_entries_sorted_for_lookup, estimate_proof_bytes, expected_catalog_identity,
    identity_digest, key_digest, policy_digest, resolve_group_batch_schedule, resolve_schedule,
    ring_challenge_config_digest, runtime_schedule_key_cmp, schedule_from_entry,
    validate_catalog_identity, validate_generated_schedule_entry,
    validate_generated_schedule_table, GeneratedScheduleCatalogIdentity, GeneratedScheduleTable,
};
pub use emit::{refresh_generated_wiring, run_regen_fmt, write_family_module, EmitSpec};
pub use group_batch::find_group_batch_schedule;
pub use schedule_params::{find_schedule, suffix_opening_layout};
