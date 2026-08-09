//! Runtime schedule catalogs and strict generated schedule resolution.

mod audit;
mod candidate;
pub mod catalog_identity;
pub mod generated;
mod group_batch;
mod resolve;
mod runtime;

pub use akita_types::{
    suffix_opening_layout, ChunkedWitnessCfg, CommitmentRingDims, DecompositionParams,
    SisModulusProfileId, SisSecurityPolicyId, DEFAULT_SIS_SECURITY_POLICY,
};
pub use catalog_identity::{
    expected_catalog_identity, identity_digest, key_digest, policy_digest,
    ring_challenge_config_digest, validate_catalog_identity,
};
pub use generated::*;
pub use resolve::{
    estimate_proof_bytes, resolve_generated_precommitted_group_profile,
    resolve_generated_schedule_selection, resolve_group_batch_schedule, resolve_schedule,
    schedule_from_entry, select_generated_schedule_row, select_generated_schedule_row_for_profiles,
    ResolvedScheduleRow,
};
pub use runtime::{
    default_sis_security_policy, InnerBasisSource, PlannerCostModelId, PlannerPolicy,
    RingDimensionScheduleMode, RuntimeSchedulePolicy, SelectionPolicyId, ADAPTIVE_SEARCH_LEVELS,
};

/// Shared schedule-construction primitives used by offline search and generated-row replay.
#[doc(hidden)]
pub mod planner_support {
    pub use crate::candidate::{
        projected_collision_role_price, sis_key_at_dimension, RingDimensionCandidate,
    };
    pub use crate::runtime::{
        grouped_segment_rings, materialize_candidate_schedule, planned_next_witness_len,
        stage3_payload_bytes_for_successor, validate_policy, CandidateFoldStep,
        CandidateTerminalResponse, MAX_RECURSION_DEPTH,
    };
}
