//! End-to-end Akita PCS scheme orchestration.

use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_prover::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, LevelProveStacks,
    RuntimeCoefficientPackingBackendFor, RuntimeCommitBackendFor, RuntimeCommitSource,
    RuntimeOpeningProveBackendFor, RuntimeRingSwitchProveBackend, RuntimeTensorBackendFor,
    SuffixOpeningProveBackend, SuffixTensorProveBackend, UniformProverStack,
};
use akita_prover::{AkitaProverSetup, CommitOutput, GroupContext};
use akita_prover::{PreparedGroupProveOps, RecursiveFoldSource, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_transcript::{Transcript, TranscriptChallengePreview};
use akita_types::AkitaBatchedProof;
use akita_types::AkitaVerifierSetup;
use akita_types::{
    BasisMode, FoldSchedule, FpExtEncoding, GroupBatchStatement, OpeningClaimsLayout,
    SetupMatrixCapacity,
};
use jolt_field::{AdditiveGroup, CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced};
use std::marker::PhantomData;
use std::time::Instant;

/// End-to-end PCS wrapper, generic over commitment config `Cfg`.
///
/// Every concrete ring degree is derived from the selected schedule. Setup is
/// flat and does not carry a nominal ring dimension.
#[derive(Clone, Copy, Debug, Default)]
pub struct AkitaCommitmentScheme<Cfg: CommitmentConfig> {
    _cfg: PhantomData<Cfg>,
}

impl<Cfg> AkitaCommitmentScheme<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: ExtField<Cfg::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    /// Build a flat prover setup for the config's provisioning policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested capacity, field tower, or generated setup is invalid.
    pub fn setup_prover(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
    ) -> Result<AkitaProverSetup<Cfg::Field>, AkitaError>
    where
        Cfg::Field: AkitaDeserialize<Context = ()>,
    {
        akita_setup::new_prover_setup::<Cfg::Field, Cfg>(
            max_num_vars,
            max_num_polys_per_commitment_group,
        )
    }

    /// Derive a verifier setup that preserves the prover's full matrix prefix.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when setup conversion fails.
    pub fn setup_verifier(
        setup: &AkitaProverSetup<Cfg::Field>,
    ) -> Result<AkitaVerifierSetup<Cfg::Field>, AkitaError> {
        let capacity = SetupMatrixCapacity {
            num_field_elements: setup.expanded.shared_matrix().num_field_elements(),
        };
        setup.to_verifier_setup(capacity)
    }

    /// Derive a verifier setup narrowed to one resolved schedule and root
    /// opening layout.
    ///
    /// Offloaded setup-contribution producers do not retain their natural
    /// public-matrix prefixes. The first direct producer after an offloaded
    /// chain and the terminal matrix still do.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the schedule is malformed or
    /// its verifier matrix requirement exceeds the prover setup.
    pub fn setup_verifier_for_schedule(
        setup: &AkitaProverSetup<Cfg::Field>,
        schedule: &FoldSchedule,
        root_layout: &OpeningClaimsLayout,
    ) -> Result<AkitaVerifierSetup<Cfg::Field>, AkitaError> {
        let capacity =
            akita_types::verifier_setup_matrix_capacity_for_schedule(schedule, root_layout)?;
        setup.to_verifier_setup(capacity)
    }

    /// Commit one polynomial group in its complete parameter context.
    ///
    /// # Errors
    ///
    /// Returns an error when the group is malformed, its scheduled S/G or
    /// explicit parameters are unsupported, setup capacity is insufficient,
    /// or commitment execution fails.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit")]
    pub fn commit<P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        polys: &[P],
        stack: &UniformProverStack<'_, Cfg::Field, B>,
        context: GroupContext<'_>,
    ) -> Result<CommitOutput<Cfg::Field>, AkitaError>
    where
        Cfg::Field: Ring + Unreduced + Field + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
        P: RuntimeCommitSource<Cfg::Field>,
        B: RuntimeCommitBackendFor<Cfg::Field, P>,
    {
        akita_config::validate_config_policy::<Cfg>()?;
        akita_prover::commit::<Cfg, P, B>(polys, setup.expanded.as_ref(), stack, context)
    }

    /// Produce a fused batched opening proof over ordered commitment groups.
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation fails.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    pub fn batched_prove<'a, T, P, B>(
        setup: &AkitaProverSetup<Cfg::Field>,
        opening: SelectedProverOpeningData<'a, Cfg::ExtField, P, Cfg::Field>,
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
    ) -> Result<AkitaBatchedProof<Cfg::Field, Cfg::ExtField>, AkitaError>
    where
        T: Transcript<Cfg::Field> + TranscriptChallengePreview,
        Cfg::Field: Ring + Unreduced + Field + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
        P: PreparedGroupProveOps<Cfg::Field, Cfg::ExtField, B>,
        B: ComputeBackendSetup<Cfg::Field>
            + RuntimeCommitBackendFor<Cfg::Field, akita_prover::RecursiveWitnessFlat>
            + RuntimeOpeningProveBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>>
            + RuntimeCoefficientPackingBackendFor<
                Cfg::Field,
                RecursiveFoldSource<Cfg::Field>,
                Cfg::ExtField,
            > + SuffixOpeningProveBackend<Cfg::Field>
            + DigitRowsComputeBackend<Cfg::Field>
            + RuntimeTensorBackendFor<Cfg::Field, RecursiveFoldSource<Cfg::Field>, Cfg::ExtField>
            + SuffixTensorProveBackend<Cfg::Field, Cfg::ExtField>
            + RuntimeRingSwitchProveBackend<Cfg::Field>
            + 'a,
        <B as ComputeBackendSetup<Cfg::Field>>::PreparedSetup: 'a,
    {
        let t_prove_total = Instant::now();
        akita_config::validate_config_policy::<Cfg>()?;
        let proof = akita_prover::batched_prove::<Cfg, T, P, B, B, B, B>(
            &setup.expanded,
            &setup.prefix_slots,
            stacks,
            opening,
            transcript,
            basis,
        )?;

        tracing::info!(
            levels = proof.num_fold_levels(),
            elapsed_s = t_prove_total.elapsed().as_secs_f64(),
            "akita batched prove complete"
        );

        Ok(proof)
    }

    /// Verify a fused batched opening proof over ordered commitment groups.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
    pub fn batched_verify<T: Transcript<Cfg::Field>>(
        proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
        setup: &AkitaVerifierSetup<Cfg::Field>,
        transcript: &mut T,
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        akita_config::validate_config_policy::<Cfg>()?;
        batched_verify_inner::<Cfg, T>(proof, setup, transcript, statement, basis)
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
    statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
    basis: BasisMode,
) -> Result<(), AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field:
        Field + CanonicalEncoding + Unreduced + Ring + PseudoMersenne + Valid + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
    T: Transcript<Cfg::Field>,
{
    let t_verify_akita = Instant::now();
    akita_verifier::batched_verify::<Cfg, T>(proof, setup, transcript, statement, basis)?;

    tracing::info!(
        levels = proof.num_fold_levels(),
        elapsed_s = t_verify_akita.elapsed().as_secs_f64(),
        "akita batched verify complete"
    );

    Ok(())
}

const PROTOCOL_NAME: &[u8] = b"Akita";

#[cfg(test)]
mod tests;
