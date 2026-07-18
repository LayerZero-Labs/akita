//! End-to-end Akita PCS scheme orchestration.

use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, PseudoMersenneField, RandomSampling,
};
use akita_prover::compute::{
    ComputeBackendSetup, LevelProveStacks, RecursiveProveBackend, RuntimeRootCommitBackend,
    RuntimeRootCommitPoly, RuntimeRootProvePoly, UniformProverStack,
};
use akita_prover::ProverOpeningData;
use akita_prover::ProverTranscriptGrind;
use akita_prover::{AkitaProverSetup, CommittedGroupWithHint};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::Transcript;
use akita_types::{
    dispatch_for_field, validate_ring_subfield_role, BasisMode, Commitment, FpExtEncoding,
    PolynomialGroupLayout,
};
use akita_types::{AkitaBatchedProof, AkitaCommitmentHint, SetupContributionMode};
use akita_types::{AkitaVerifierSetup, OpeningClaims};
use std::marker::PhantomData;
use std::time::Instant;

type CommitmentWithHint<F> = (Commitment<F>, AkitaCommitmentHint<F>);

/// End-to-end PCS wrapper, generic over commitment config `Cfg`.
///
/// Root ring degree is derived from `Cfg`'s schedule policy at setup time
/// (`policy_of::<Cfg>().ring_dimension`, equal to `Cfg::D` for uniform-D presets).
/// Per-level suffix folds dispatch on each step's schedule `ring_dimension`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<Cfg> AkitaCommitmentScheme<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + PseudoMersenneField
        + Valid
        + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FrobeniusExtField<Cfg::Field>
        + FromPrimitiveInt
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize,
{
    /// Build prover setup for the config's generation ring dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested capacity, field tower, or generated setup is invalid.
    pub fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
    ) -> Result<AkitaProverSetup<Cfg::Field>, AkitaError> {
        let ring_d = akita_config::policy_of::<Cfg>().ring_dimension;
        dispatch_for_field!(ProtocolDispatchSlot::Envelope, Cfg::Field, ring_d, |D| {
            validate_ring_subfield_role::<Cfg::Field, Cfg::ExtField, D>("extension field")?;
            akita_setup::new_prover_setup::<Cfg::Field, Cfg>(
                max_num_vars,
                max_num_polys_per_commitment_group,
            )
        })
    }

    /// Derive verifier setup from prover setup.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when setup conversion fails.
    pub fn setup_verifier(
        setup: &AkitaProverSetup<Cfg::Field>,
    ) -> Result<AkitaVerifierSetup<Cfg::Field>, AkitaError> {
        setup.verifier_setup()
    }

    /// Validate the field tower against the config schedule policy ring dimension.
    fn validate_cfg_ring_policy() -> Result<usize, AkitaError> {
        let ring_d = akita_config::policy_of::<Cfg>().ring_dimension;
        dispatch_for_field!(ProtocolDispatchSlot::Envelope, Cfg::Field, ring_d, |D| {
            validate_ring_subfield_role::<Cfg::Field, Cfg::ExtField, D>("extension field")
        })?;
        Ok(ring_d)
    }

    /// Validate policy ring dimension and setup envelope generation degree.
    fn validate_policy_ring_dim(setup: &AkitaProverSetup<Cfg::Field>) -> Result<(), AkitaError> {
        let ring_d = Self::validate_cfg_ring_policy()?;
        setup.ensure_root_ring_dim(ring_d)
    }

    fn validate_verifier_policy_ring_dim(
        setup: &AkitaVerifierSetup<Cfg::Field>,
    ) -> Result<(), AkitaError> {
        let ring_d = Self::validate_cfg_ring_policy()?;
        if setup.expanded.seed().gen_ring_dim != ring_d {
            return Err(AkitaError::InvalidInput(format!(
                "setup gen_ring_dim={} does not match scheme root ring degree {ring_d}",
                setup.expanded.seed().gen_ring_dim
            )));
        }
        Ok(())
    }

    /// Commit a single opening-point bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit")]
    pub fn commit<P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        polys: &[P],
        stack: &UniformProverStack<'_, Cfg::Field, B>,
    ) -> Result<CommitmentWithHint<Cfg::Field>, AkitaError>
    where
        Cfg::Field: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
        P: RuntimeRootCommitPoly<Cfg::Field>,
        B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
    {
        Self::validate_policy_ring_dim(setup)?;
        akita_prover::commit::<Cfg, P, B>(polys, setup.expanded.as_ref(), stack)
    }

    /// Commit the polynomial bundle used by a batched prove.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, layout selection, or any per-point commitment fails.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
    pub fn batched_commit<P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        polys: &[P],
        stack: &UniformProverStack<'_, Cfg::Field, B>,
    ) -> Result<CommitmentWithHint<Cfg::Field>, AkitaError>
    where
        Cfg::Field: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
        P: RuntimeRootCommitPoly<Cfg::Field>,
        B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
    {
        Self::validate_policy_ring_dim(setup)?;
        akita_prover::batched_commit::<Cfg, P, B>(polys, setup.expanded.as_ref(), stack)
    }

    /// Commit one standalone one-hot commitment group.
    ///
    /// # Errors
    ///
    /// Returns an error if the group is empty, dense, exceeds setup capacity, or cannot be planned.
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit_group")]
    pub fn commit_group<P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        polys: &[P],
        stack: &UniformProverStack<'_, Cfg::Field, B>,
    ) -> Result<CommittedGroupWithHint<Cfg::Field>, AkitaError>
    where
        Cfg::Field: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
        P: RuntimeRootCommitPoly<Cfg::Field>,
        B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
    {
        Self::validate_policy_ring_dim(setup)?;
        akita_prover::commit_group::<Cfg, P, B>(polys, setup.expanded.as_ref(), stack)
    }

    /// Commit the final polynomial bundle for a multi-group root commitment.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, multi-group layout selection, or
    /// commitment execution fails.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit_final_group")]
    pub fn commit_final_group<P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        polys: &[P],
        stack: &UniformProverStack<'_, Cfg::Field, B>,
        precommitteds: Vec<PolynomialGroupLayout>,
    ) -> Result<CommitmentWithHint<Cfg::Field>, AkitaError>
    where
        Cfg::Field: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
        P: RuntimeRootCommitPoly<Cfg::Field>,
        B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
    {
        Self::validate_policy_ring_dim(setup)?;
        akita_prover::commit_final_group::<Cfg, P, B>(
            polys,
            setup.expanded.as_ref(),
            stack,
            precommitteds,
        )
    }

    /// Produce a fused batched opening proof for one shared opening point.
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    pub fn batched_prove<'a, T, P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        claims: ProverOpeningData<'a, Cfg::ExtField, P, Cfg::Field>,
        stacks: &'a impl LevelProveStacks<
            'a,
            Cfg::Field,
            Commit = B,
            Opening = B,
            Tensor = B,
            RingSwitch = B,
        >,
        transcript: &mut T,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, AkitaError>
    where
        T: Transcript<Cfg::Field> + ProverTranscriptGrind<Cfg::Field>,
        Cfg::Field: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field> + AdditiveGroup,
        P: RuntimeRootProvePoly<Cfg::Field>,
        B: RecursiveProveBackend<Cfg::Field, P, Cfg::ExtField>
            + ComputeBackendSetup<Cfg::Field>
            + 'a,
        <B as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    {
        let t_prove_total = Instant::now();
        Self::validate_policy_ring_dim(setup)?;
        let proof = akita_prover::batched_prove::<Cfg, T, P, B, B, B, B>(
            &setup.expanded,
            &setup.prefix_slots,
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

    /// Verify a fused batched opening proof at one shared opening point.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
    pub fn batched_verify<T: Transcript<Cfg::Field>>(
        proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
        setup: &AkitaVerifierSetup<Cfg::Field>,
        transcript: &mut T,
        claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<(), AkitaError> {
        Self::validate_verifier_policy_ring_dim(setup)?;
        batched_verify_inner::<Cfg, T>(
            proof,
            setup,
            transcript,
            claims,
            basis,
            setup_contribution_mode,
        )
    }

    /// Protocol identifier.
    #[must_use]
    pub fn protocol_name() -> &'static [u8] {
        PROTOCOL_NAME
    }
}

fn batched_verify_inner<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
    basis: BasisMode,
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + HasWide
        + HalvingField
        + FromPrimitiveInt
        + PseudoMersenneField
        + Valid
        + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: FrobeniusExtField<Cfg::Field> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<Cfg::Field>,
{
    let t_verify_akita = Instant::now();
    akita_verifier::batched_verify::<Cfg, T>(
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

const PROTOCOL_NAME: &[u8] = b"Akita";

#[cfg(test)]
mod tests;
