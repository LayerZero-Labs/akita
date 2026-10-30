//! End-to-end Akita PCS scheme orchestration.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{AkitaProverSetup, CommitmentHandle, CpuBackend, PinnedProverSetup};
use akita_error::AkitaError;
use akita_params::{BasisMode, FoldSchedule, OpeningClaimsLayout, SetupMatrixCapacity};
use akita_prover::{BackendRegistry, FixedFoldRoute, ProverBackend, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_transcript::ProverChannel;
use akita_types::AkitaVerifierSetup;
use akita_types::FpExtEncoding;
use akita_verifier::{AkitaVerifier, PinnedVerifierSetup};
use jolt_field::{AdditiveGroup, CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced, WithCommitAccumulator};
use std::time::Instant;

/// End-to-end PCS wrapper, generic over commitment config `Cfg`.
///
/// Every concrete ring degree is derived from the selected schedule. Setup is
/// flat and does not carry a nominal ring dimension.
#[derive(Clone, Debug)]
pub struct AkitaCommitmentScheme<Cfg: CommitmentConfig> {
    schedules: TrustedScheduleCatalog<Cfg>,
}

impl<Cfg> AkitaCommitmentScheme<Cfg>
where
    Cfg: CommitmentConfig + 'static,
    Cfg::Field: Field + CanonicalEncoding + Unreduced + PseudoMersenne + Valid + AkitaSerialize,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    Cfg::ExtField: ExtField<Cfg::Field> + Ring + Unreduced + Fold + AkitaSerialize,
{
    /// Construct a scheme from a catalog already bound to `Cfg`.
    #[must_use]
    pub fn new(schedules: TrustedScheduleCatalog<Cfg>) -> Self {
        Self { schedules }
    }

    /// Decode a trusted schedule artifact and bind it to this scheme instance.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding, row audit, or config binding fails.
    pub fn from_schedule_artifact(
        bytes: &[u8],
        chunking: akita_params::ChunkedWitnessCfg,
    ) -> Result<Self, AkitaError> {
        Ok(Self::new(
            TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(bytes, chunking)?,
        ))
    }

    /// The single validated catalog used by setup, commitment, proving, and verification.
    pub fn schedules(&self) -> &TrustedScheduleCatalog<Cfg> {
        &self.schedules
    }

    /// Build a flat prover setup for the config's provisioning policy.
    ///
    /// `max_num_batched_polys` bounds the total polynomial count across the
    /// final group and every precommitted group in one opening batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested capacity, field tower, catalog, or setup is invalid.
    pub fn setup_prover(
        &self,
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<AkitaProverSetup<Cfg::Field>, AkitaError>
    where
        Cfg::Field: AkitaDeserialize<Context = ()> + WithCommitAccumulator,
    {
        let requirements = akita_config::SetupRequirements::from_catalog::<Cfg>(
            &self.schedules,
            max_num_vars,
            max_num_batched_polys,
        )?;
        akita_setup::new_prover_setup::<Cfg::Field>(&requirements)
    }

    /// Build a scheme over the catalog identity of a pinned prover setup.
    #[must_use]
    pub fn from_pinned_prover_setup(pinned: &PinnedProverSetup<Cfg>) -> Self {
        Self::new(pinned.catalog().clone())
    }

    /// Build a verifier from a loader-issued setup, which must carry this
    /// scheme's catalog identity.
    ///
    /// This replaces [`Self::verifier`] for production verifiers.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the pinned catalog is another
    /// catalog or preparing an admitted row fails.
    pub fn pinned_verifier(
        &self,
        pinned: PinnedVerifierSetup<Cfg>,
    ) -> Result<AkitaVerifier<Cfg>, AkitaError> {
        if pinned.catalog().catalog_digest() != self.schedules.catalog_digest() {
            return Err(AkitaError::InvalidSetup(
                "pinned verifier setup carries another catalog identity".into(),
            ));
        }
        AkitaVerifier::from_pinned(pinned)
    }

    /// Derive a verifier setup that preserves the prover's full matrix prefix.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when setup conversion fails.
    ///
    /// # Safety
    ///
    /// Soundness contract of [`AkitaProverSetup::to_verifier_setup`]; a
    /// production verifier loads a `PinnedVerifierSetup` instead.
    pub unsafe fn setup_verifier(
        &self,
        setup: &AkitaProverSetup<Cfg::Field>,
    ) -> Result<AkitaVerifierSetup<Cfg::Field>, AkitaError> {
        let capacity = SetupMatrixCapacity {
            num_field_elements: setup.expanded().shared_matrix().num_field_elements(),
        };
        // SAFETY: forwarded to the caller.
        unsafe { setup.to_verifier_setup(capacity) }
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
    ///
    /// # Safety
    ///
    /// As [`Self::setup_verifier`].
    pub unsafe fn setup_verifier_for_schedule(
        &self,
        setup: &AkitaProverSetup<Cfg::Field>,
        schedule: &FoldSchedule,
        root_layout: &OpeningClaimsLayout,
    ) -> Result<AkitaVerifierSetup<Cfg::Field>, AkitaError> {
        let capacity =
            akita_params::verifier_setup_matrix_capacity_for_schedule(schedule, root_layout)?;
        // SAFETY: forwarded to the caller.
        unsafe { setup.to_verifier_setup(capacity) }
    }

    /// Produce a fused batched opening proof from retained commitments, in a
    /// standalone transcript under `session`.
    ///
    /// Each handle retains its exact source. The backend validates public claims
    /// against that commitment before creating independent proof state.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched ownership, setup, claims, or an invalid
    /// protocol transition.
    #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
    #[allow(clippy::type_complexity)]
    pub fn batched_prove<'a>(
        &self,
        setup: &AkitaProverSetup<Cfg::Field>,
        opening: SelectedProverOpeningData<
            'a,
            Cfg::ExtField,
            CommitmentHandle<Cfg::Field, Cfg::ExtField>,
            Cfg::Field,
        >,
        backend: &CpuBackend<Cfg::Field, Cfg::ExtField>,
        session: &[u8],
        basis: BasisMode,
    ) -> Result<Vec<u8>, AkitaError>
    where
        Cfg::Field: WithCommitAccumulator + 'static,
        Cfg::ExtField: jolt_field::MulBaseUnreduced<Cfg::Field> + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
        CpuBackend<Cfg::Field, Cfg::ExtField>: ProverBackend<
            Cfg::Field,
            Cfg::ExtField,
            CommitmentHandle = CommitmentHandle<Cfg::Field, Cfg::ExtField>,
        >,
    {
        let started = Instant::now();
        let mut channel = akita_transcript::new_prover_channel(session);
        self.batched_prove_embedded(setup, opening, backend, &mut channel, basis)?;
        let proof = channel.narg_string().to_vec();
        tracing::info!(
            proof_bytes = proof.len(),
            elapsed_s = started.elapsed().as_secs_f64(),
            "akita batched prove complete"
        );
        Ok(proof)
    }

    /// Produce a fused batched opening proof inside the caller's transcript.
    ///
    /// The proof bytes are appended to `channel`'s proof string. The caller
    /// must have absorbed the [`akita_transcript::CommitmentDigest`] of every
    /// group it opens before this call.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched ownership, setup, claims, or an invalid
    /// protocol transition. The channel is then unusable for a proof.
    #[allow(clippy::type_complexity)]
    pub fn batched_prove_embedded<'a>(
        &self,
        setup: &AkitaProverSetup<Cfg::Field>,
        opening: SelectedProverOpeningData<
            'a,
            Cfg::ExtField,
            CommitmentHandle<Cfg::Field, Cfg::ExtField>,
            Cfg::Field,
        >,
        backend: &CpuBackend<Cfg::Field, Cfg::ExtField>,
        channel: &mut ProverChannel,
        basis: BasisMode,
    ) -> Result<(), AkitaError>
    where
        Cfg::Field: WithCommitAccumulator + 'static,
        Cfg::ExtField: jolt_field::MulBaseUnreduced<Cfg::Field> + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
        CpuBackend<Cfg::Field, Cfg::ExtField>: ProverBackend<
            Cfg::Field,
            Cfg::ExtField,
            CommitmentHandle = CommitmentHandle<Cfg::Field, Cfg::ExtField>,
        >,
    {
        let resolved = self.schedules.resolve_selection(opening.selection())?;
        let required_prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening.opening_layout(),
        )?;
        let prefix_slots =
            backend.import_setup_prefixes(setup.prefix_slots(), &required_prefix_ids)?;
        let mut registry = BackendRegistry::<Cfg>::new()?;
        let root = registry.register(backend, &prefix_slots)?;
        let mut route = FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
        akita_prover::batched_prove_embedded::<Cfg, _>(
            setup.expanded().descriptor(),
            &self.schedules,
            &registry,
            opening,
            channel,
            basis,
            &mut route,
        )
    }

    /// Build a verifier for `setup` over this scheme's schedule catalog.
    ///
    /// The verifier admits every catalog row `setup` supports and prepares
    /// their terminal matrices once; see [`AkitaVerifier::new`].
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when sizing or preparing an
    /// admitted row fails.
    ///
    /// # Safety
    ///
    /// Soundness contract of [`AkitaVerifier::new`]; a production verifier
    /// uses [`Self::pinned_verifier`].
    pub unsafe fn verifier(
        &self,
        setup: AkitaVerifierSetup<Cfg::Field>,
    ) -> Result<AkitaVerifier<Cfg>, AkitaError> {
        // SAFETY: forwarded to the caller.
        unsafe { AkitaVerifier::new(setup, self.schedules.clone()) }
    }

    /// Protocol identifier.
    #[must_use]
    pub fn protocol_name() -> &'static [u8] {
        PROTOCOL_NAME
    }
}

const PROTOCOL_NAME: &[u8] = b"Akita";

#[cfg(test)]
mod tests;
