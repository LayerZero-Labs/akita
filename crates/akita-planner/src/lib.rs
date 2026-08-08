//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! This crate is a **pure, `Cfg`-free DP library**. The DP entry point
//! is [`find_schedule`], which runs an exhaustive dynamic program to
//! minimize proof size for a schedule lookup key. Every per-preset input is
//! carried by the plain-value [`PlannerPolicy`] plus a `ring_challenge_config` /
//! ring-challenge closure, so the planner names no `CommitmentConfig`
//! types and depends only on `akita-schedules` / `akita-types` /
//! `akita-challenges` / `akita-field`.
//! Scalar and mixed-D planning are selected internally by the grouped gate from
//! the policy-bound ring-dimension domain.
//!
//! With the `catalog-gen` feature enabled, this crate also owns the offline
//! generated-table family list and `gen_schedule_tables` binary. That feature
//! is allowed to name `akita-config` presets; normal planner use remains
//! preset-free.

pub use akita_schedules::{
    ChunkedWitnessCfg, DecompositionParams, InnerBasisSource, PlannerCostModelId, PlannerPolicy,
    SelectionPolicyId, SisModulusProfileId, SisSecurityPolicyId, DEFAULT_SIS_SECURITY_POLICY,
};

pub mod emit;
#[cfg(feature = "catalog-gen")]
pub mod generated_families;
mod planner;
pub mod schedule_params;

pub use akita_schedules::{
    catalog_entries_sorted_for_lookup, estimate_proof_bytes, expected_catalog_identity,
    identity_digest, key_digest, policy_digest, resolve_group_batch_schedule, resolve_schedule,
    ring_challenge_config_digest, runtime_schedule_key_cmp, schedule_from_entry,
    validate_catalog_identity, validate_generated_schedule_entry,
    validate_generated_schedule_table, GeneratedScheduleCatalogIdentity, GeneratedScheduleTable,
};
pub use emit::{publish_generated_outputs, render_generated_outputs, EmitSpec};
pub use planner::find_schedule;
pub use schedule_params::{
    plan_standalone_precommit, suffix_opening_layout, RingDimensionSearchDomain,
    StandalonePrecommitCandidate, StandalonePrecommitPlan, StandalonePrecommitSelectionPolicy,
};
