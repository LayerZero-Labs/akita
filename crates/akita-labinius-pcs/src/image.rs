//! Commit the clear image in its canonical padded layout and open one evaluation.

mod binding;

use akita_algebra::binary::field_switch::SwitchField;
use akita_config::TrustedScheduleCatalog;
use akita_cpu_backend::{AkitaProverSetup, CommitmentHandle, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_labinius_prover::{commit_binary_clear_prepared, lowered::flatten_image};
use akita_labinius_verifier::{
    channel::ClearChannel, lowered::LoweredRootLayout, BinaryClearCommitment,
};
use akita_params::{sis::labinius::LabiniusDigitBase, BasisMode};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    backend::{ProofAdmission, ProofScope},
    SelectedProverOpeningData,
};
use akita_serialization::Valid;
use akita_types::{AkitaVerifierSetup, CommittedGroup, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::AkitaVerifier;

use crate::{ImageConfig, PreparedMatrix, RootSetup, F};
use binding::bind_image_statement;

/// Public table commitment, retained CPU source, and original clear image.
pub struct ImageCommitOutput {
    pub committed_group: CommittedGroup<F>,
    pub private_handle: CommitmentHandle<F, F>,
    pub image: BinaryClearCommitment<F, 648, akita_algebra::MinusTrinomial>,
}

/// One coefficient-field evaluation of the padded image multilinear extension.
#[derive(Clone, Copy)]
pub struct ImageEvaluation<'a> {
    pub point: &'a [F],
    pub value: F,
}

/// Caller-trusted catalog, Akita setup and owning CPU backend for image openings.
pub struct ImageProver {
    pub(crate) scheme: AkitaCommitmentScheme<ImageConfig>,
    pub(crate) setup: AkitaProverSetup<F>,
    pub(crate) backend: CpuBackend<F, F>,
}

impl ImageProver {
    /// Prepare a prover without loading artifacts or searching for schedules.
    pub fn new(
        schedules: TrustedScheduleCatalog<ImageConfig>,
        setup: AkitaProverSetup<F>,
    ) -> Result<Self, AkitaError> {
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let backend = CpuBackend::new(setup.expanded.clone())?;
        Ok(Self {
            scheme: AkitaCommitmentScheme::new(schedules),
            setup,
            backend,
        })
    }

    /// Compute the prepared clear image and commit its padded dense Lagrange table.
    ///
    /// `H::Source: Sync` is required by the existing prepared commitment kernel,
    /// including when this crate's `parallel` feature is disabled.
    pub fn commit<H: SwitchField>(
        &self,
        admitted: &RootSetup,
        prepared: &PreparedMatrix,
        source: &[H::Source],
    ) -> Result<ImageCommitOutput, AkitaError>
    where
        H::Source: Sync,
    {
        let image = commit_binary_clear_prepared::<H, F, 648, akita_algebra::MinusTrinomial>(
            prepared,
            admitted.setup(),
            source,
        )?;
        let layout =
            LoweredRootLayout::new(admitted.setup(), admitted.shape(), LabiniusDigitBase::Bits2)?;
        let table = flatten_image(&layout, &image)?;
        let polynomial = DensePoly::<F>::from_field_evals(layout.image_log_len(), table)?;
        let mut polynomials = Vec::new();
        polynomials
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidInput("image-source allocation failed".into()))?;
        polynomials.push(polynomial);
        let source = self.backend.import_source(polynomials)?;
        let output = self.backend.commit(
            self.scheme.schedules(),
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )?;
        Ok(ImageCommitOutput {
            committed_group: output.committed_group,
            private_handle: output.private_handle,
            image,
        })
    }

    /// Bind the statement and prove inside the caller's active clear channel.
    ///
    /// The caller owns the enclosing session, continuation and EOF check.
    pub fn open_on_channel<H: SwitchField, S: ClearChannel>(
        &self,
        admitted: &RootSetup,
        committed: &ImageCommitOutput,
        evaluation: ImageEvaluation<'_>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let row = self
            .scheme
            .schedules()
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                committed.committed_group.profile().group,
            ))?;
        akita_config::ensure_prover_schedule_fits_setup::<ImageConfig>(
            &self.setup.expanded,
            row.schedule(),
            &row.profiles().opening_layout()?,
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        values.push(evaluation.value);
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        groups.push(PolynomialGroupClaims::new(
            evaluation.point,
            values,
            committed.committed_group.clone(),
        )?);
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        handles.push(committed.private_handle.clone());
        let opening = SelectedProverOpeningData::from_committed_claims::<ImageConfig>(
            OpeningClaims::from_groups(groups)?,
            handles,
            self.scheme.schedules(),
        )?;
        // Exercise the backend's canonical ownership/setup boundary before
        // advancing the parent transcript. The actual proof creates its own scope.
        let scope = self.backend.begin_proof(
            self.setup.expanded.descriptor(),
            self.scheme.schedules(),
            row.schedule(),
            opening.opening_layout(),
        )?;
        let guard = ProofScope::admitted(&self.backend, scope);
        let context = self.backend.proof_context(guard.session(), 0)?.for_group(0);
        self.backend.validate_commitment(
            guard.session(),
            &context,
            &committed.private_handle,
            committed.committed_group.profile(),
            committed.committed_group.commitment(),
        )?;
        guard.finish()?;
        let bound = bind_image_statement::<H, S>(
            admitted,
            self.setup.expanded.descriptor(),
            self.scheme.schedules(),
            &committed.committed_group,
            evaluation,
            channel,
        )?;
        bound.session.prove(channel, |session| {
            self.scheme.batched_prove(
                &self.setup,
                opening,
                &self.backend,
                session,
                BasisMode::Lagrange,
            )
        })
    }
}

/// Caller-trusted catalog and prepared Akita verifier for image openings.
pub struct ImageVerifier {
    verifier: AkitaVerifier<ImageConfig>,
}

impl ImageVerifier {
    /// Prepare catalog admission and terminal matrices before reading any proof.
    pub fn new(
        schedules: TrustedScheduleCatalog<ImageConfig>,
        setup: AkitaVerifierSetup<F>,
    ) -> Result<Self, AkitaError> {
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(Self {
            verifier: AkitaVerifier::new(setup, schedules)?,
        })
    }

    /// Bind the same statement, check bounded framing and replay the inner proof.
    pub fn verify_on_channel<H: SwitchField, S: ClearChannel>(
        &self,
        admitted: &RootSetup,
        commitment: &CommittedGroup<F>,
        evaluation: ImageEvaluation<'_>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let row =
            self.verifier
                .schedules()
                .resolve_key(&akita_params::ScheduleLookupKey::single(
                    commitment.profile().group,
                ))?;
        if !self.verifier.admits(row.selection().row_digest) {
            return Err(AkitaError::InvalidSetup(
                "image row does not fit verifier setup".into(),
            ));
        }
        let bound = bind_image_statement::<H, S>(
            admitted,
            self.verifier.setup().expanded().descriptor(),
            self.verifier.schedules(),
            commitment,
            evaluation,
            channel,
        )?;
        bound.session.verify(channel, |proof, session| {
            self.verifier
                .batched_verify(proof, session, bound.statement, BasisMode::Lagrange)
        })
    }
}
