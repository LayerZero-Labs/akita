//! Top-level batched verifier orchestration once a schedule is selected.

use crate::proof::claims::{prepare_verifier_claims, PreparedVerifierClaims};
use crate::proof::direct::verify_zero_fold_openings_with_incidence;
use crate::protocol::levels::verify_fold_batched_proof;
use crate::protocol::root_direct::verify_root_direct_commitments_with_params;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{
    folded_root_supports_opening_shape, root_direct_schedule, root_tensor_projection_enabled,
    schedule_is_root_direct, schedule_root_fold_step, scheduled_next_level_params,
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaProofStep, AkitaVerifierSetup, BasisMode,
    RingCommitment, RingSubfieldEncoding, Schedule, SetupContributionMode, Step, VerifierClaims,
};

/// Structural slice of `<AkitaBatchedProof as Valid>::check`, inlined to avoid
/// requiring `F: Valid + L: Valid` at the verifier entrypoint.
fn check_batched_proof_step_shape<F, L>(proof: &AkitaBatchedProof<F, L>) -> Result<(), AkitaError>
where
    F: FieldCore,
    L: FieldCore,
{
    match &proof.root {
        AkitaBatchedRootProof::Fold(_) => {
            let Some((last, rest)) = proof.steps.split_last() else {
                return Err(AkitaError::InvalidProof);
            };
            if !matches!(last, AkitaProofStep::Terminal(_))
                || rest
                    .iter()
                    .any(|step| !matches!(step, AkitaProofStep::Intermediate(_)))
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        AkitaBatchedRootProof::Terminal(_) => {
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
        }
        AkitaBatchedRootProof::ZeroFold { .. } => {
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(())
}

fn select_batched_verifier_schedule<'a, Cfg, const D: usize>(
    prepared_claims: &PreparedVerifierClaims<'a, Cfg::ClaimField, RingCommitment<Cfg::Field, D>>,
) -> Result<Schedule, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field> + ExtField<Cfg::ClaimField>,
{
    let num_vars = prepared_claims.incidence_summary.num_vars();
    let mut schedule = Cfg::get_params_for_prove(&prepared_claims.incidence_summary)
        .map_err(|_| AkitaError::InvalidProof)?;
    if let Some(root_step) = schedule_root_fold_step(&schedule) {
        let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if !folded_root_supports_opening_shape::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
            &prepared_claims.opening_points,
            &root_step.params,
            alpha_bits,
        ) && !root_tensor_projection_enabled::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
            num_vars,
        ) {
            let params = Cfg::get_params_for_batched_commitment(&prepared_claims.incidence_summary)
                .map_err(|_| AkitaError::InvalidProof)?;
            schedule =
                root_direct_schedule(num_vars, params).map_err(|_| AkitaError::InvalidProof)?;
        }
    }

    Ok(schedule)
}

fn validate_schedule_onehot_chunk_size<Cfg: CommitmentConfig>(
    schedule: &Schedule,
) -> Result<(), AkitaError> {
    let expected = Cfg::onehot_chunk_size();
    if Cfg::decomposition().log_commit_bound != 1 || expected <= 1 {
        return Ok(());
    }
    let root_params = match schedule.steps.first() {
        Some(akita_types::Step::Fold(root)) => Some(&root.params),
        Some(akita_types::Step::Direct(root)) => root.params.as_ref(),
        None => None,
    }
    .ok_or(AkitaError::InvalidProof)?;
    if root_params.onehot_chunk_size != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Verify a batched proof under config `Cfg`.
///
/// This is the verifier crate's top-level orchestration entrypoint. It owns
/// public claim normalization, schedule selection (from `Cfg`), the root-direct
/// rewrite, transcript instance-descriptor binding, root-direct and folded-root
/// dispatch, and recursive verifier replay.
///
/// The root-direct branch recomputes commitments with the same root commitment
/// layout the prover used at commit time (`Cfg::get_params_for_batched_commitment`
/// for the same incidence); a mismatching layout would cause
/// [`verify_root_direct_commitments_with_params`] to reject a correctly
/// produced proof.
///
/// # Errors
///
/// Returns an error if public claims are malformed, schedule/layout policy
/// rejects the proof shape, root-direct commitment recomputation rejects, or
/// proof replay fails.
pub fn verify_batched<'a, Cfg, T, const D: usize>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ChallengeField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: VerifierClaims<'a, Cfg::ClaimField, RingCommitment<Cfg::Field, D>>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + PseudoMersenneField,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>
        + ExtField<Cfg::ClaimField>
        + FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + AkitaSerialize,
    T: Transcript<Cfg::Field>,
{
    // Reject malformed step shapes that the downstream `fold_levels()` filter
    // would silently skip past.
    check_batched_proof_step_shape(proof)?;

    let prepared_claims = prepare_verifier_claims(&setup.expanded, &claims)?;
    let schedule = select_batched_verifier_schedule::<Cfg, D>(&prepared_claims)?;
    validate_schedule_onehot_chunk_size::<Cfg>(&schedule)?;

    bind_transcript_instance_descriptor::<Cfg::Field, T, D, Cfg>(
        &setup.expanded,
        &prepared_claims.incidence_summary,
        &schedule,
        basis,
        transcript,
    )?;

    let PreparedVerifierClaims {
        opening_points,
        commitments,
        openings,
        incidence_summary,
    } = prepared_claims;

    match &proof.root {
        AkitaBatchedRootProof::ZeroFold { witnesses, .. } => {
            #[cfg(feature = "zk")]
            if !proof.zk_hiding.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            if !proof.steps.is_empty() {
                return Err(AkitaError::InvalidProof);
            }
            let Some(Step::Direct(direct)) = schedule.steps.first() else {
                return Err(AkitaError::InvalidProof);
            };
            if !schedule_is_root_direct(&schedule) {
                return Err(AkitaError::InvalidProof);
            }
            let params = direct.params.as_ref().ok_or(AkitaError::InvalidProof)?;
            verify_zero_fold_openings_with_incidence(
                witnesses,
                &opening_points,
                &openings,
                &incidence_summary,
                basis,
            )?;
            #[cfg(feature = "zk")]
            let direct_commitment_payload = proof
                .root
                .direct_b_blinding_digits()
                .ok_or(AkitaError::InvalidProof)?;
            verify_root_direct_commitments_with_params::<Cfg::Field, D>(
                witnesses,
                setup,
                &commitments,
                &incidence_summary,
                params,
                #[cfg(feature = "zk")]
                direct_commitment_payload,
            )?;
        }
        AkitaBatchedRootProof::Fold(_) | AkitaBatchedRootProof::Terminal(_) => {
            let Some(Step::Fold(root_step)) = schedule.steps.first() else {
                return Err(AkitaError::InvalidProof);
            };
            let next_level_params =
                scheduled_next_level_params(&schedule, 1).map_err(|_| AkitaError::InvalidProof)?;
            verify_fold_batched_proof::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, T, D>(
                proof,
                setup,
                transcript,
                &opening_points,
                &openings,
                &commitments,
                &incidence_summary,
                basis,
                &schedule,
                &root_step.params,
                &next_level_params,
                &next_level_params,
                setup_contribution_mode,
            )?;
        }
    }

    Ok(())
}
