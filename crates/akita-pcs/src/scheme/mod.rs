//! End-to-end Akita PCS scheme orchestration.

use akita_config::CommitmentConfig;
use akita_field::fields::wide::{HasOptimizedFold, HasWide};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    HasUnreducedOps, PseudoMersenneField, RandomSampling,
};
use akita_prover::{
    AkitaPolyOps, AkitaProverSetup, CommitmentComputeBackend, CommitmentProver, ProverClaims,
    ProverComputeBackend,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use akita_types::AkitaVerifierSetup;
use akita_types::{validate_ring_subfield_role, BasisMode, RingSubfieldEncoding};
use akita_types::{AkitaBatchedProof, AkitaCommitmentHint, RingCommitment};
use akita_verifier::{CommitmentVerifier, VerifierClaims};
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

impl<F, const D: usize, Cfg> CommitmentProver<F, D> for AkitaCommitmentScheme<D, Cfg>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
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
        + HasOptimizedFold
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
    ) -> Result<Self::ProverSetup, AkitaError> {
        validate_field_roles_for_ring::<F, D, Cfg>()?;
        akita_setup::new_prover_setup::<F, D, Cfg>(
            max_num_vars,
            max_num_polys_per_point,
            max_num_points,
        )
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup.verifier_setup()
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit")]
    fn commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys: &[P],
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitmentComputeBackend<F>,
    {
        akita_prover::commit::<Cfg, D, P, B>(polys, setup.expanded.as_ref(), backend, prepared)
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys_per_point: &[&[P]],
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError>
    where
        P: AkitaPolyOps<F, D>,
        B: CommitmentComputeBackend<F>,
    {
        akita_prover::batched_commit::<Cfg, D, P, B>(
            polys_per_point,
            setup.expanded.as_ref(),
            backend,
            prepared,
        )
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        claims: ProverClaims<'a, Self::ClaimField, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F>,
        P: AkitaPolyOps<F, D>,
        B: ProverComputeBackend<F>,
    {
        let t_prove_total = Instant::now();
        validate_field_roles_for_ring::<F, D, Cfg>()?;
        let proof = akita_prover::prove_batched::<Cfg, T, P, B, D>(
            &setup.expanded,
            backend,
            prepared,
            claims,
            transcript,
            basis,
        )?;

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
        akita_verifier::verify_batched::<Cfg, T, D>(proof, setup, transcript, claims, basis)?;

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
