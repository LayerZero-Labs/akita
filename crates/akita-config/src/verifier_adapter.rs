//! Config-owned verifier adapter without prover/setup dependencies.

use crate::{
    bind_instance_descriptor_for_config, validate_field_roles_for_config, CommitmentConfig,
};
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use akita_types::{
    scheduled_next_level_params, AkitaBatchedProof, AkitaVerifierSetup, BasisMode, RingCommitment,
    RingSubfieldEncoding,
};
use akita_verifier::{
    verify_batched_with_policy, verify_root_direct_commitments_with_params, VerifierClaims,
};

/// Timer-free config-owned batched verifier entrypoint.
///
/// This is the canonical verifier-policy adapter used by both the public scheme
/// wrapper and environments such as the Jolt guest where `std::time::Instant`
/// is unavailable.
///
/// # Errors
///
/// Returns an error if field-role validation, descriptor binding, direct-root
/// checks, or proof replay fails.
#[allow(clippy::type_complexity)]
pub fn batched_verify_with_config<'a, F, T, const D: usize, Cfg>(
    proof: &AkitaBatchedProof<F, Cfg::ChallengeField>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    claims: VerifierClaims<'a, Cfg::ClaimField, RingCommitment<F, D>>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HasUnreducedOps
        + HalvingField
        + FromPrimitiveInt
        + PseudoMersenneField
        + Valid
        + AkitaSerialize,
    T: Transcript<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField:
        RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    validate_field_roles_for_config::<F, D, Cfg>()?;
    verify_batched_with_policy::<F, Cfg::ClaimField, Cfg::ChallengeField, T, D, _, _, _, _, _>(
        proof,
        setup,
        transcript,
        claims,
        basis,
        |incidence_summary| Cfg::get_params_for_prove(incidence_summary),
        |schedule, next_inputs| {
            scheduled_next_level_params(schedule, 1, next_inputs, Cfg::level_params_with_log_basis)
        },
        Cfg::get_params_for_batched_commitment,
        |transcript, incidence_summary, schedule, basis| {
            bind_instance_descriptor_for_config::<Cfg, T, D>(
                transcript,
                setup.expanded.seed(),
                incidence_summary,
                schedule,
                basis,
            )
        },
        |witnesses, setup, commitments, incidence_summary, params, direct_commitment_payload| {
            verify_root_direct_commitments_with_params::<F, D>(
                witnesses,
                setup,
                commitments,
                incidence_summary,
                params,
                direct_commitment_payload,
            )
        },
    )
}
