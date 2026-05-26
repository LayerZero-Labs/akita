//! End-to-end Akita PCS scheme orchestration.

use akita_config::CommitmentConfig;
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
#[allow(unused_imports)]
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_prover::kernels::crt_ntt::NttSlotCache;
use akita_prover::{
    batched_commit, commit, prove_batched, AkitaPolyOps, AkitaProverSetup, CommitmentProver,
    ProverClaims, RootTensorProjectionPoly,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use akita_types::AkitaVerifierSetup;
use akita_types::{
    root_tensor_projection_enabled, schedule_root_fold_step, AkitaBatchedProof,
    AkitaCommitmentHint, ClaimIncidenceSummary, RingCommitment,
};
use akita_types::{validate_ring_subfield_role, BasisMode, RingSubfieldEncoding};
use akita_verifier::{verify_batched, CommitmentVerifier, VerifierClaims};
use std::marker::PhantomData;
use std::time::Instant;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

fn validate_field_roles_for_ring<F, const D: usize, Cfg>() -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>,
{
    validate_ring_subfield_role::<F, Cfg::ClaimField, D>("claim field")?;
    validate_ring_subfield_role::<F, Cfg::ChallengeField, D>("challenge field")?;
    let relative_degree =
        <Cfg::ChallengeField as akita_field::ExtField<Cfg::ClaimField>>::EXT_DEGREE;
    let expected_challenge_degree = Cfg::CLAIM_EXT_DEGREE
        .checked_mul(relative_degree)
        .ok_or_else(|| AkitaError::InvalidSetup("field tower degree overflow".to_string()))?;
    if Cfg::CHAL_EXT_DEGREE != expected_challenge_degree {
        return Err(AkitaError::InvalidSetup(format!(
            "challenge field degree {} does not match claim degree {} times relative degree {}",
            Cfg::CHAL_EXT_DEGREE,
            Cfg::CLAIM_EXT_DEGREE,
            relative_degree
        )));
    }
    Ok(())
}

fn should_transform_root_commitment<F, const D: usize, Cfg>(
    incidence: &ClaimIncidenceSummary,
) -> Result<bool, AkitaError>
where
    F: FieldCore,
    Cfg: CommitmentConfig<Field = F>,
{
    if !root_tensor_projection_enabled::<F, Cfg::ClaimField, Cfg::ChallengeField, D>(
        incidence.num_vars(),
    ) {
        return Ok(false);
    }
    let schedule = Cfg::get_params_for_prove(incidence)?;
    Ok(schedule_root_fold_step(&schedule).is_some())
}

impl<F, const D: usize, Cfg> CommitmentProver<F, D> for AkitaCommitmentScheme<D, Cfg>
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
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>
        + FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + AkitaSerialize,
{
    type ProverSetup = AkitaProverSetup<F, D>;
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type ClaimField = Cfg::ClaimField;
    type CommitHint = AkitaCommitmentHint<F, D>;
    type BatchedProof = AkitaBatchedProof<F, Cfg::ChallengeField>;

    fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_point: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup {
        validate_field_roles_for_ring::<F, D, Cfg>().expect("invalid Akita field tower");
        akita_setup::new_prover_setup::<F, D, Cfg>(
            max_num_vars,
            max_num_polys_per_point,
            max_num_points,
        )
        .expect("commitment setup failed")
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.verifier_setup()
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit")]
    fn commit<P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError> {
        if let Some(first) = polys.first() {
            let incidence = ClaimIncidenceSummary::same_point(first.num_vars(), polys.len())?;
            if should_transform_root_commitment::<F, D, Cfg>(&incidence)? {
                let transformed = polys
                    .iter()
                    .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ChallengeField>())
                    .collect::<Result<Vec<RootTensorProjectionPoly<F, D>>, _>>()?;
                return commit::<F, Cfg, RootTensorProjectionPoly<F, D>, D>(&transformed, setup);
            }
        }
        commit::<F, Cfg, P, D>(polys, setup)
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
    fn batched_commit<P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        polys_per_point: &[&[P]],
        setup: &Self::ProverSetup,
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError> {
        let incidence =
            akita_prover::prepare_batched_commit_inputs::<F, D, P>(polys_per_point, setup)?;
        if should_transform_root_commitment::<F, D, Cfg>(&incidence)? {
            let transformed: Vec<Vec<RootTensorProjectionPoly<F, D>>> = polys_per_point
                .iter()
                .map(|polys| {
                    polys
                        .iter()
                        .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ChallengeField>())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
            let transformed_refs: Vec<&[RootTensorProjectionPoly<F, D>]> =
                transformed.iter().map(Vec::as_slice).collect();
            return batched_commit::<F, Cfg, RootTensorProjectionPoly<F, D>, D>(
                &transformed_refs,
                setup,
            );
        }
        batched_commit::<F, Cfg, P, D>(polys_per_point, setup)
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    fn batched_prove<'a, T: Transcript<F>, P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>>(
        setup: &Self::ProverSetup,
        claims: ProverClaims<'a, Self::ClaimField, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError> {
        let t_prove_total = Instant::now();
        validate_field_roles_for_ring::<F, D, Cfg>()?;
        let proof = prove_batched::<F, Cfg, T, P, D>(setup, claims, transcript, basis)?;

        tracing::info!(
            levels = proof.num_fold_levels() + usize::from(proof.root.as_fold().is_some()),
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "akita batched prove complete"
        );

        Ok(proof)
    }
}

impl<F, const D: usize, Cfg> CommitmentVerifier<F, D> for AkitaCommitmentScheme<D, Cfg>
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
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField:
        RingSubfieldEncoding<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type ClaimField = Cfg::ClaimField;
    type BatchedProof = AkitaBatchedProof<F, Cfg::ChallengeField>;

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, Self::ClaimField, Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        let t_verify_akita = Instant::now();
        validate_field_roles_for_ring::<F, D, Cfg>()?;
        verify_batched::<F, Cfg, T, D>(proof, setup, transcript, claims, basis)?;

        tracing::info!(
            levels = proof.num_fold_levels() + 1,
            elapsed_s = t_verify_akita.elapsed().as_secs_f64(),
            "akita batched verify complete"
        );

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Akita"
    }
}

#[cfg(test)]
mod tests;
