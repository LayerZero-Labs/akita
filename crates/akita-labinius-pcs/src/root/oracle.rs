//! The three ordered root oracle calls over the shared Akita backend.

use akita_config::ResolvedScheduleRow;
use akita_cpu_backend::{CommitmentHandle, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{RootEvaluationClaims, RootProverOracle, RootVerifierOracle},
};
use akita_params::{BasisMode, PrecommittedGroupProfiles};
use akita_prover::{
    backend::{ProofAdmission, ProofScope},
    SelectedProverOpeningData,
};
use akita_serialization::AkitaDeserialize;
use akita_types::CommittedGroup;
use jolt_field::Ring;

use super::{
    binding::{
        self, bind_opening, commitment_size, grouped_claims, resolve_rows, validate_image, Order,
    },
    RootPcsProver, RootPcsVerifier,
};
use crate::{config::DigitConfig, session::canonical_bytes, ImageCommitOutput, F};

/// Single-use Akita-backed prover oracle, borrowing the image's owning backend.
pub(super) struct RootPcsProverOracle<'a, C: DigitConfig> {
    owner: &'a RootPcsProver<C>,
    image: &'a ImageCommitOutput,
    image_row: &'a ResolvedScheduleRow,
    row: &'a ResolvedScheduleRow,
    response: Option<(CommittedGroup<F>, CommitmentHandle<F, F>)>,
    order: Order,
}
impl<'a, C: DigitConfig> RootPcsProverOracle<'a, C> {
    pub(super) fn new(
        owner: &'a RootPcsProver<C>,
        image: &'a ImageCommitOutput,
    ) -> Result<Self, AkitaError> {
        let (image_row, row) = resolve_rows(
            &owner.admitted,
            owner.image.scheme.schedules(),
            owner.digits.schedules(),
        )?;
        validate_image(&image.committed_group, image_row)?;
        // Admit the retained Y handle before any parent transcript operation.
        let backend = &owner.image.backend;
        let scope = backend.begin_proof(
            owner.image.setup.expanded.descriptor(),
            owner.digits.schedules(),
            row.schedule(),
            &row.profiles().opening_layout()?,
        )?;
        let guard = ProofScope::admitted(backend, scope);
        let context = backend.proof_context(guard.session(), 0)?.for_group(0);
        backend.validate_commitment(
            guard.session(),
            &context,
            &image.private_handle,
            image.committed_group.profile(),
            image.committed_group.commitment(),
        )?;
        guard.finish()?;
        Ok(Self {
            owner,
            image,
            image_row,
            row,
            response: None,
            order: Order::default(),
        })
    }
}
impl<C: DigitConfig> RootProverOracle<F> for RootPcsProverOracle<'_, C> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(0)?;
        binding::bind_image::<C, S>(
            layout,
            &self.owner.admitted,
            self.owner.image.setup.expanded.descriptor(),
            self.image_row,
            &self.image.committed_group,
            channel,
        )
    }
    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(1)?;
        if *layout
            != LoweredRootLayout::new(
                self.owner.admitted.setup(),
                self.owner.admitted.shape(),
                C::BASE,
            )?
            || digits.len() != layout.witness_len()
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(digits.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        fields.extend(digits.iter().map(|&digit| F::from_u64(u64::from(digit))));
        let polynomial = DensePoly::<F>::from_field_evals(layout.witness_log_len(), fields)?;
        let mut polynomials = Vec::new();
        polynomials
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        polynomials.push(polynomial);
        let backend = &self.owner.image.backend;
        let source = backend.import_source(polynomials)?;
        let mut profiles = Vec::new();
        profiles
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidProof)?;
        profiles.push(*self.image.committed_group.profile());
        let precommitteds = PrecommittedGroupProfiles::from_profiles(profiles)?;
        let output = backend.commit(
            self.owner.digits.schedules(),
            &source,
            GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )?;
        if *output.committed_group.profile() != self.row.profiles().final_group {
            return Err(AkitaError::InvalidProof);
        }
        let mut bytes = canonical_bytes(&output.committed_group)?;
        if bytes.len() != commitment_size(self.row.profiles().final_group)? {
            return Err(AkitaError::InvalidProof);
        }
        let mut length = u64::try_from(bytes.len())
            .map_err(|_| AkitaError::InvalidProof)?
            .to_le_bytes();
        channel.message(&mut length)?;
        channel.message(&mut bytes)?;
        self.response = Some((output.committed_group, output.private_handle));
        Ok(())
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<F>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(2)?;
        let (response, response_handle) = self.response.as_ref().ok_or(AkitaError::InvalidProof)?;
        let (_, session) = bind_opening::<C, S>(
            self.owner.image.setup.expanded.descriptor(),
            self.row,
            claims,
            &self.image.committed_group,
            response,
            channel,
        )?;
        let groups = grouped_claims(claims, self.image.committed_group.clone(), response.clone())?;
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(2)
            .map_err(|_| AkitaError::InvalidProof)?;
        handles.push(self.image.private_handle.clone());
        handles.push(response_handle.clone());
        let opening = SelectedProverOpeningData::from_committed_claims::<C>(
            groups,
            handles,
            self.owner.digits.schedules(),
        )?;
        session.prove(channel, |session| {
            self.owner.digits.batched_prove(
                &self.owner.image.setup,
                opening,
                &self.owner.image.backend,
                session,
                BasisMode::Lagrange,
            )
        })
    }
}

/// Single-use verifier oracle; all rows are resolved before proof input is read.
pub(super) struct RootPcsVerifierOracle<'a, C: DigitConfig> {
    owner: &'a RootPcsVerifier<C>,
    image: &'a CommittedGroup<F>,
    image_row: &'a ResolvedScheduleRow,
    row: &'a ResolvedScheduleRow,
    response: Option<CommittedGroup<F>>,
    order: Order,
}
impl<'a, C: DigitConfig> RootPcsVerifierOracle<'a, C> {
    pub(super) fn new(
        owner: &'a RootPcsVerifier<C>,
        image: &'a CommittedGroup<F>,
    ) -> Result<Self, AkitaError> {
        let (image_row, row) = resolve_rows(
            &owner.admitted,
            &owner.image_schedules,
            &owner.digit_schedules,
        )?;
        validate_image(image, image_row)?;
        Ok(Self {
            owner,
            image,
            image_row,
            row,
            response: None,
            order: Order::default(),
        })
    }
}
impl<C: DigitConfig> RootVerifierOracle<F> for RootPcsVerifierOracle<'_, C> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(0)?;
        binding::bind_image::<C, S>(
            layout,
            &self.owner.admitted,
            self.owner.verifier.setup().expanded().descriptor(),
            self.image_row,
            self.image,
            channel,
        )
    }
    fn bind_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(1)?;
        if *layout
            != LoweredRootLayout::new(
                self.owner.admitted.setup(),
                self.owner.admitted.shape(),
                C::BASE,
            )?
        {
            return Err(AkitaError::InvalidProof);
        }
        let expected = commitment_size(self.row.profiles().final_group)?;
        let mut length = [0u8; 8];
        channel
            .message(&mut length)
            .map_err(|_| AkitaError::InvalidProof)?;
        let length =
            usize::try_from(u64::from_le_bytes(length)).map_err(|_| AkitaError::InvalidProof)?;
        if length != expected {
            return Err(AkitaError::InvalidProof);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| AkitaError::InvalidProof)?;
        bytes.resize(length, 0);
        channel
            .message(&mut bytes)
            .map_err(|_| AkitaError::InvalidProof)?;
        let response = CommittedGroup::<F>::deserialize_compressed_exact(&bytes, &())
            .map_err(|_| AkitaError::InvalidProof)?;
        if *response.profile() != self.row.profiles().final_group
            || canonical_bytes(&response)? != bytes
        {
            return Err(AkitaError::InvalidProof);
        }
        self.response = Some(response);
        Ok(())
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<F>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(2)?;
        let response = self.response.as_ref().ok_or(AkitaError::InvalidProof)?;
        let (statement, session) = bind_opening::<C, S>(
            self.owner.verifier.setup().expanded().descriptor(),
            self.row,
            claims,
            self.image,
            response,
            channel,
        )?;
        session.verify(channel, |proof, session| {
            self.owner.verifier.verify(proof, session, statement)
        })
    }
}
