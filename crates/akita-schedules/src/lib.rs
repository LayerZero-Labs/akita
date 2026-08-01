//! Runtime schedule catalogs and strict generated schedule resolution.

pub mod catalog_identity;
pub mod generated;
mod group_batch;
mod resolve;
mod runtime;

pub use akita_challenges::TensorChallengeShape;
pub use akita_types::{
    ChunkedWitnessCfg, CommitmentRingDims, DecompositionParams, SisModulusProfileId,
    SisSecurityPolicyId, DEFAULT_SIS_SECURITY_POLICY,
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

/// Shared schedule-construction primitives used by offline search and generated-row replay.
#[doc(hidden)]
pub mod planner_support {
    pub use crate::runtime::{
        checked_power_of_two_vars, grouped_segment_rings, materialize_candidate_schedule,
        optimize_fold_challenge_shape, planned_next_witness_len,
        stage3_payload_bytes_for_successor, CandidateFoldStep, CandidateTerminalResponse,
    };
}
