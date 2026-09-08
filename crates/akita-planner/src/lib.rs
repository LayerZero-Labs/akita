//! Offline schedule planner for the Akita polynomial commitment scheme.
//!
//! This crate is a **pure, `Cfg`-free DP library**. The DP entry point
//! is [`find_schedule`], which runs an exhaustive dynamic program to
//! optimize a schedule lookup key under its catalog-bound selection policy.
//! Every per-preset input is carried by the plain-value [`PlannerPolicy`] plus a `ring_challenge_config` /
//! ring-challenge closure, so the planner names no `CommitmentConfig`
//! types and depends only on `akita-schedules` / `akita-types` /
//! `akita-challenges` / `akita-error`.
//! Scalar and mixed-D planning are selected internally by the grouped gate from
//! the policy-bound ring-dimension domain.
//!
//! With the `catalog-gen` feature enabled, this crate also owns the offline
//! artifact-family list and `gen_schedule_artifacts` binary. That feature
//! is allowed to name `akita-config` presets; normal planner use remains
//! preset-free.

pub use akita_schedules::{
    ChunkedWitnessCfg, DecompositionParams, PlannerCostModelId, PlannerPolicy,
    RecursiveSetupSearchPolicy, RecursiveSplitSearchPolicy, RingDimensionScheduleMode,
    SelectionPolicyId, SelectiveL2ResponseModelId, SisModulusProfileId, SisSecurityPolicyId,
    DEFAULT_SIS_SECURITY_POLICY,
};

mod diagnostics;
pub mod emit;
#[cfg(feature = "catalog-gen")]
pub mod generated_families;
mod planner;
mod policy;
mod response_model;
pub mod schedule_params;

pub use akita_schedules::policy_digest;
pub use emit::{
    publish_artifact_outputs, render_schedule_artifact_outputs_with_validation, ArtifactOutput,
    EmitSpec, MaterializationDiagnostics,
};
pub use planner::find_schedule;
#[cfg(feature = "test-support")]
pub use planner::find_schedule_for_test_relation_mode;
pub use policy::InnerBasisSource;
pub use schedule_params::suffix_opening_layout;
#[cfg(feature = "test-support")]
pub use schedule_params::TestRelationModeFilter;
