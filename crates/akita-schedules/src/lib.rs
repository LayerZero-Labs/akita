//! Runtime schedule catalogs and strict schedule resolution.

mod artifact;
mod audit;
mod candidate;
mod policy_digest;
mod resolve;
mod runtime;
mod traversal;

pub use akita_types::{
    suffix_opening_layout, ChunkedWitnessCfg, CommitmentRingDims, DecompositionParams,
    SisModulusProfileId, SisSecurityPolicyId, DEFAULT_SIS_SECURITY_POLICY,
};
pub use artifact::{
    TrustedScheduleCatalog, MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
    MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES,
};
pub use policy_digest::policy_digest;
pub use resolve::ResolvedScheduleRow;
pub use runtime::{
    default_sis_security_policy, expanded_schedule_proof_payload_bytes, validate_policy,
    PlannerCostModelId, PlannerPolicy, RecursiveSetupSearchPolicy, RecursiveSplitSearchPolicy,
    RingDimensionScheduleMode, RuntimeSchedulePolicy, SelectionPolicyId,
    SelectiveL2ResponseModelId, ADAPTIVE_SEARCH_LEVELS,
};

/// Shared schedule-construction primitives used by offline search and artifact validation.
#[doc(hidden)]
pub mod planner_support {
    pub use crate::candidate::{
        projected_collision_role_price, selective_l2_inner_matrix, sis_key_at_dimension,
        SelectiveL2CandidateGeometry,
    };
    pub use crate::runtime::{
        candidate_grinding_nonce_bits, first_direct_setup_capacity_for_schedule,
        first_direct_setup_field_len_for_schedule, materialize_candidate_schedule,
        nonterminal_level_payload_bytes, planned_next_witness_len,
        stage3_payload_bytes_for_successor, validate_policy, CandidateFoldStep,
        CandidateTerminalResponse, NonterminalLevelPayloadBytes, MAX_RECURSION_DEPTH,
    };
}
