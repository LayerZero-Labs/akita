//! End-to-end Akita PCS scheme orchestration.

use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, PseudoMersenneField, RandomSampling,
};
use akita_prover::compute::{
    ComputeBackendSetup, LevelProveStacks, RecursiveProveBackend, RootCommitBackend,
    RootCommitPoly, RootProvePoly, UniformProverStack,
};
use akita_prover::ProverOpeningData;
use akita_prover::ProverTranscriptGrind;
use akita_prover::{AkitaProverSetup, CommitmentProver};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use akita_types::{
    validate_ring_subfield_role, BasisMode, FpExtEncoding, PolynomialGroupLayout,
    SetupContributionMode,
};
use akita_types::{AkitaBatchedProof, AkitaCommitmentHint, RingCommitment};
use akita_types::{AkitaVerifierSetup, OpeningClaims};
use akita_verifier::CommitmentVerifier;
use std::marker::PhantomData;
use std::time::Instant;

/// End-to-end PCS wrapper, generic over ring degree `D` and config `Cfg`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<const D: usize, Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

type CommitmentWithHint<F, const D: usize> = (RingCommitment<F, D>, AkitaCommitmentHint<F, D>);

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
    Cfg::ExtField: FpExtEncoding<F>,
    Cfg::ExtField: FrobeniusExtField<F>
        + FromPrimitiveInt
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize,
{
    type ProverSetup = AkitaProverSetup<F, D>;
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type ExtField = Cfg::ExtField;
    type CommitHint = AkitaCommitmentHint<F, D>;
    type BatchedProof = AkitaBatchedProof<F, Cfg::ExtField>;

    fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
    ) -> Result<Self::ProverSetup, AkitaError> {
        validate_ring_subfield_role::<F, Cfg::ExtField, D>("extension field")?;
        akita_setup::new_prover_setup::<F, D, Cfg>(max_num_vars, max_num_polys_per_commitment_group)
    }

    fn setup_prover_recursion(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
    ) -> Result<Self::ProverSetup, AkitaError> {
        validate_ring_subfield_role::<F, Cfg::ExtField, D>("extension field")?;
        akita_setup::new_prover_setup_recursion::<F, D, Cfg>(
            max_num_vars,
            max_num_polys_per_commitment_group,
        )
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        setup
            .verifier_setup()
            .expect("prover setup must convert to verifier setup")
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        polys: &[P],
        stack: &UniformProverStack<'_, F, B, D>,
    ) -> Result<CommitmentWithHint<F, D>, AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::ExtField, D>,
    {
        akita_prover::batched_commit::<Cfg, D, P, B>(polys, setup.expanded.as_ref(), stack)
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit_final_group")]
    fn commit_final_group<P, B>(
        setup: &Self::ProverSetup,
        polys: &[P],
        stack: &UniformProverStack<'_, F, B, D>,
        precommitteds: Vec<PolynomialGroupLayout>,
    ) -> Result<CommitmentWithHint<F, D>, AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::ExtField, D>,
    {
        akita_prover::commit_final_group::<Cfg, D, P, B>(
            polys,
            setup.expanded.as_ref(),
            stack,
            precommitteds,
        )
    }

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        claims: ProverOpeningData<'a, Self::ExtField, P, F, D>,
        stacks: &'a impl LevelProveStacks<'a, F, D, Commit = B, Opening = B, Tensor = B, RingSwitch = B>,
        transcript: &mut T,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F> + ProverTranscriptGrind<F>,
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
        P: RootProvePoly<F, D>,
        B: RecursiveProveBackend<F, P, Self::ExtField, D> + ComputeBackendSetup<F> + 'a,
        <B as ComputeBackendSetup<F>>::PreparedSetup<D>: 'a,
    {
        let t_prove_total = Instant::now();
        validate_ring_subfield_role::<F, Cfg::ExtField, D>("extension field")?;
        let proof = akita_prover::batched_prove::<Cfg, T, P, B, B, B, B, D>(
            &setup.expanded,
            &setup.prefix_slots,
            &setup.fold_a_ones,
            stacks,
            claims,
            transcript,
            basis,
            setup_contribution_mode,
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
    Cfg::ExtField: FpExtEncoding<F>,
    Cfg::ExtField: FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    type VerifierSetup = AkitaVerifierSetup<F>;
    type Commitment = RingCommitment<F, D>;
    type ExtField = Cfg::ExtField;
    type BatchedProof = AkitaBatchedProof<F, Cfg::ExtField>;

    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: OpeningClaims<'_, Self::ExtField, &Self::Commitment>,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<(), AkitaError> {
        let t_verify_akita = Instant::now();
        validate_ring_subfield_role::<F, Cfg::ExtField, D>("extension field")?;
        akita_verifier::batched_verify::<Cfg, T, D>(
            proof,
            setup,
            transcript,
            claims,
            basis,
            setup_contribution_mode,
        )?;

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
