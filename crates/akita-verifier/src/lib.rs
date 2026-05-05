//! Verifier-facing API surface for the Akita PCS.
//!
//! This crate owns verifier replay for already-selected Akita proof schedules.
//! It deliberately avoids prover polynomial backends, commit hints, recursive
//! witness construction, and planner search.
//!
//! Downstream verifier-only integrations should pair this crate with
//! `akita-types` for proof/setup/claim shapes and `akita-config` for concrete
//! runtime schedule policy. The broader `akita-pcs` crate is an umbrella for
//! end-to-end examples and also re-exports prover-facing APIs.

pub mod proof;
pub mod protocol;
pub mod stages;

pub use akita_types::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
pub use proof::{
    direct_witness_field_elements, direct_witness_opening_matches, prepare_verifier_claims,
    verify_root_direct_openings, PreparedVerifierClaims,
};
pub use protocol::{
    prepare_batched_verifier_schedule_context, prepare_m_eval, ring_switch_verifier,
    verify_batched_proof_with_schedule, verify_batched_recursive_suffix,
    verify_batched_with_policy, verify_fold_batched_proof, verify_one_level, verify_root_level,
    BatchedVerifierScheduleContext, FoldVerifierLayouts, PreparedMEval, RecursiveVerifierState,
    RingSwitchVerifyOutput,
};
pub use stages::{
    derive_stage1_challenges, AkitaStage1Verifier, AkitaStage2Verifier, Stage2MEvalSource,
};
