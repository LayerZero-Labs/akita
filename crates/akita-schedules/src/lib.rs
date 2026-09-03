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
    estimate_proof_bytes, resolve_generated_catalog_row_for_key,
    resolve_generated_catalog_row_for_profiles, resolve_generated_schedule_selection,
    schedule_from_entry, ResolvedScheduleRow,
};
pub use runtime::{
    default_sis_security_policy, expanded_schedule_proof_payload_bytes, validate_policy,
    PlannerCostModelId, PlannerPolicy, RecursiveSetupSearchPolicy, RecursiveSplitSearchPolicy,
    RingDimensionScheduleMode, RuntimeSchedulePolicy, SelectionPolicyId,
    SelectiveL2ResponseModelId, ADAPTIVE_SEARCH_LEVELS,
};

/// Shared schedule-construction primitives used by offline search and generated-row replay.
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
