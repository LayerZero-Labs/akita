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
    derive_tiered_setup_material_for_verifier, materialize_setup_claim_tables,
    prepare_batched_verifier_schedule_context, prepare_m_eval, prepopulate_tiered_s_cache,
    ring_switch_verifier, verify_batched_proof_with_schedule, verify_batched_recursive_suffix,
    verify_batched_with_policy, verify_fold_batched_proof, verify_one_level, verify_root_level,
    verify_setup_claim_reduction, verify_stage2_with_setup_claim_reduction,
    BatchedVerifierScheduleContext, FoldVerifierLayouts, PreparedMEval, RecursiveVerifierState,
    RingSwitchVerifyOutput,
};
pub use stages::{
    derive_stage1_challenges, AkitaStage1Verifier, AkitaStage2Verifier, Stage2MEvalSource,
};

/// Cross-check shim for tests: compute the per-claim opening of a
/// dense ring polynomial at the routed setup opening point under
/// `claim_lp`'s shape, exactly as
/// [`crate::protocol::levels::expand_tiered_setup_claims`] does for
/// chunks and meta. Used by integration tests to verify the
/// verifier's chunk-opening reconstruction matches the prover's
/// `DensePoly::evaluate_and_fold` path without spinning up a full
/// end-to-end run.
///
/// # Errors
///
/// Returns whatever
/// [`protocol::levels::dense_ring_opening_at_point`] returns.
#[doc(hidden)]
pub fn __test_dense_ring_opening_at_point<F, const D: usize>(
    coeffs: &[akita_algebra::CyclotomicRing<F, D>],
    opening_point: &[F],
    claim_lp: &akita_types::LevelParams,
    alpha_bits: usize,
) -> Result<F, akita_field::AkitaError>
where
    F: akita_field::FieldCore + akita_field::CanonicalField,
{
    protocol::levels::dense_ring_opening_at_point::<F, D>(
        coeffs,
        opening_point,
        claim_lp,
        alpha_bits,
    )
}
