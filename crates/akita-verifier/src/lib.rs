//! Verifier-facing API surface for the Akita PCS.
//!
//! This crate owns verifier replay for already-selected Akita proof schedules.
//! It deliberately avoids prover polynomial backends, commit hints, recursive
//! witness construction, and planner search.

pub mod batched;
pub mod claims;
pub mod direct;
pub mod levels;
pub mod ring_switch;
pub mod stage1;
pub mod stage2;

pub use batched::{
    verify_batched_proof_with_schedule, BatchedVerifierScheduleContext, FoldVerifierLayouts,
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
