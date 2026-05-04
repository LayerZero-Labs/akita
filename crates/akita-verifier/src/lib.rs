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

pub mod batched;
pub mod claims;
pub mod direct;
pub mod levels;
pub mod ring_switch;
pub mod stage1;
pub mod stage2;

pub use akita_types::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
pub use batched::{
    prepare_batched_verifier_schedule_context, verify_batched_proof_with_schedule,
    verify_batched_with_policy, BatchedVerifierScheduleContext, FoldVerifierLayouts,
};
pub use claims::{prepare_verifier_claims, PreparedVerifierClaims};
pub use direct::{
    direct_witness_field_elements, direct_witness_opening_matches, verify_root_direct_openings,
};
pub use levels::{
    verify_batched_recursive_suffix, verify_fold_batched_proof, verify_one_level,
    verify_root_level, RecursiveVerifierState,
};
pub use ring_switch::{
    prepare_m_eval, ring_switch_verifier, PreparedMEval, RingSwitchVerifyOutput,
};
pub use stage1::{derive_stage1_challenges, HachiStage1Verifier};
pub use stage2::{HachiStage2Verifier, Stage2MEvalSource};
