//! Runtime schedule catalogs and strict generated schedule resolution.

pub mod catalog_identity;
pub mod generated;
mod group_batch;
mod resolve;
mod runtime;

pub use akita_challenges::TensorChallengeShape;
pub use akita_types::{
    ChunkedWitnessCfg, DecompositionParams, SisModulusProfileId, SisSecurityPolicyId,
    DEFAULT_SIS_SECURITY_POLICY,
};
pub use catalog_identity::{
    expected_catalog_identity, identity_digest, key_digest, policy_digest,
    ring_challenge_config_digest, validate_catalog_identity,
};
pub use generated::*;
pub use resolve::{
    estimate_proof_bytes, resolve_group_batch_schedule, resolve_schedule, schedule_from_entry,
};
pub use runtime::{
    default_sis_security_policy, suffix_opening_layout, PlannerCostModelId, PlannerPolicy,
    RuntimeSchedulePolicy, SelectionPolicyId,
};
