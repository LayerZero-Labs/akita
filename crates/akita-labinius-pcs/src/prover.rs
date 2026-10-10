//! Commit a binary message and open binary, prime, or both claims.

use crate::{
    binding::{
        bind_image, bind_opening, ensure_setup_prefix_coverage, exchange_commitment,
        grouped_claims, opening_statement, response_row, root_layout, Order,
    },
    derive_root_setup,
    family::{prime::flatten_prime_table, FieldFamily},
    shipped::SupportedGeometry,
    RootSetup,
};
use akita_algebra::binary::field_switch::SwitchField;
use akita_config::{
    ensure_prover_schedule_fits_setup, CommitmentConfig, ResolvedScheduleRow,
    TrustedScheduleCatalog,
};
use akita_cpu_backend::{
    AkitaProverSetup, CommitOutput, CommitmentHandle, CpuBackend, DensePoly, GroupContext,
};
use akita_error::{checked, AkitaError};
use akita_labinius_prover::{
    commit_binary_clear_prepared, lowered::encode_image, prove_root_reduction_bytes,
    PreparedLimbCommitMatrix,
};
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{RootEvaluationClaims, RootProverOracle},
    BinaryClearCommitment, RootOpeningMode, RootStatement,
};
use akita_params::{BasisMode, PrecommittedGroupProfiles};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_serialization::Valid;
use akita_types::CommittedGroup;
use jolt_field::{ExtField, Ring};
use tracing::info_span;

/// A committed binary message: its public commitment and the prover's state.
pub struct Committed<P: FieldFamily> {
    /// The public commitment, the nested Akita commitment to the image digit
    /// table.
    pub commitment: CommittedGroup<P::Base>,
    /// The clear root commitment whose image digit table `commitment` binds.
    /// The root reduction takes it as the prover's witness.
    pub clear: BinaryClearCommitment,
    handle: CommitmentHandle<P::Base, P::Challenge>,
}

/// The prover of one root geometry over one proof-field family.
///
/// It owns the root matrix in its prepared form, the nested Akita setup and
/// the CPU backend that commits every table. A committed table is owned by
/// its commitment handle, not by the backend.
pub struct Prover<P: FieldFamily> {
    root: RootSetup,
    layout: LoweredRootLayout,
    prepared: PreparedLimbCommitMatrix,
    scheme: AkitaCommitmentScheme<P::Digits>,
    setup: AkitaProverSetup<P::Base>,
    elements: Option<TrustedScheduleCatalog<P::Elements>>,
    backend: CpuBackend<P::Base, P::Challenge>,
}

impl<P: FieldFamily> Prover<P> {
    /// Derive the root matrix from the nested setup's seed and admit the
    /// geometry's grouped row against that setup.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the setup is malformed, the
    /// catalog lacks the geometry's rows, or the setup does not cover the
    /// grouped row; and the admission error of the geometry or the field pair.
    pub fn new(
        geometry: SupportedGeometry,
        catalog: TrustedScheduleCatalog<P::Digits>,
        elements: Option<TrustedScheduleCatalog<P::Elements>>,
        setup: AkitaProverSetup<P::Base>,
    ) -> Result<Self, AkitaError> {
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let root = derive_root_setup(geometry, &setup.expanded.descriptor().setup_seed)?;
        let layout = root_layout::<P>(&root)?;
        for selected in std::iter::once(None).chain(elements.as_ref().map(Some)) {
            let row = response_row::<P>(&layout, &catalog, selected)?;
            ensure_setup_prefix_coverage(row, |id| setup.prefix_slots.get(id).is_some())?;
            ensure_prover_schedule_fits_setup::<P::Digits>(
                &setup.expanded,
                row.schedule(),
                &row.profiles().opening_layout()?,
            )?;
        }
        let prepared = PreparedLimbCommitMatrix::prepare_for_setup(root.setup())?;
        let backend = CpuBackend::new(setup.expanded.clone())?;
        Ok(Self {
            root,
            layout,
            prepared,
            scheme: AkitaCommitmentScheme::new(catalog),
            setup,
            elements,
            backend,
        })
    }

    /// The admitted root setup, for a caller that runs the reduction itself.
    pub fn root(&self) -> &RootSetup {
        &self.root
    }

    /// Commit the message: its clear root commitment, then the image digit
    /// table of that commitment under the nested scheme.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSize`] when `source` does not have the
    /// geometry's length, and any commitment error.
    pub fn commit<H: SwitchField>(&self, source: &[H::Source]) -> Result<Committed<P>, AkitaError>
    where
        H::Source: Sync,
    {
        let clear = info_span!("pcs_clear_commitment").in_scope(|| {
            commit_binary_clear_prepared::<H>(&self.prepared, self.root.setup(), source)
        })?;
        let CommitOutput {
            committed_group: commitment,
            private_handle: handle,
        } = info_span!("pcs_image_digit_commitment").in_scope(|| {
            let digits = encode_image(&self.layout, &clear)?;
            self.commit_table(
                self.scheme.schedules(),
                self.layout.image_log_len(),
                digit_table::<P>(&digits)?,
                GroupContext::scheduler_without_precommitted_groups(),
            )
        })?;
        Ok(Committed {
            commitment,
            clear,
            handle,
        })
    }

    /// Open the statement's binary claim, prime claim, or both on `source`.
    ///
    /// The proof is one self-delimiting byte string: the root reduction, with
    /// the response commitment and the grouped Akita proof in line.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement is malformed, `committed` does not
    /// commit `source`, the claim is false, or proving fails.
    pub fn open<H: SwitchField>(
        &self,
        source: &[H::Source],
        committed: &Committed<P>,
        statement: &RootStatement<'_, H, P::Challenge>,
    ) -> Result<Vec<u8>, AkitaError> {
        let (proof, _) = prove_root_reduction_bytes::<H, P::Base, P::Challenge, _, _, _>(
            &self.root,
            source,
            &committed.clear,
            statement,
            &mut self.oracle(committed, statement.mode()?)?,
        )?;
        Ok(proof)
    }

    /// A single-use oracle for a root reduction of `committed` over
    /// [`Self::root`].
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the catalog no longer
    /// resolves the geometry's grouped row.
    pub fn oracle<'a>(
        &'a self,
        committed: &'a Committed<P>,
        mode: RootOpeningMode,
    ) -> Result<impl RootProverOracle<P::Challenge> + 'a, AkitaError> {
        let elements = if mode.has_prime() {
            Some(self.elements.as_ref().ok_or_else(|| {
                AkitaError::InvalidSetup("prime element catalog is missing".into())
            })?)
        } else {
            None
        };
        Ok(ProverOracle {
            owner: self,
            image: committed,
            row: response_row::<P>(&self.layout, self.scheme.schedules(), elements)?,
            prime: None,
            response: None,
            mode,
            order: Order::default(),
        })
    }

    /// Commit one dense base-field table under its owning configuration.
    fn commit_table<C: CommitmentConfig<Field = P::Base, ExtField = P::Challenge>>(
        &self,
        catalog: &TrustedScheduleCatalog<C>,
        log_len: usize,
        table: Vec<P::Base>,
        context: GroupContext<'_>,
    ) -> Result<CommitOutput<P::Base, P::Challenge>, AkitaError> {
        let mut polynomials = Vec::new();
        polynomials
            .try_reserve_exact(1)
            .map_err(|_| AkitaError::InvalidInput("table allocation failed".into()))?;
        polynomials.push(DensePoly::<P::Base>::from_field_evals(log_len, table)?);
        let source = self.backend.import_source(polynomials)?;
        self.backend.commit(catalog, &source, context)
    }
}

/// Lift stored digits into the nested commitment's base field.
fn digit_table<P: FieldFamily>(digits: &[u8]) -> Result<Vec<P::Base>, AkitaError> {
    let mut table = Vec::new();
    table
        .try_reserve_exact(digits.len())
        .map_err(|_| AkitaError::InvalidInput("digit table allocation failed".into()))?;
    table.extend(
        digits
            .iter()
            .map(|&digit| P::Base::from_u64(u64::from(digit))),
    );
    Ok(table)
}

/// The nested scheme behind the reduction's ordered prover calls.
struct ProverOracle<'a, P: FieldFamily> {
    owner: &'a Prover<P>,
    image: &'a Committed<P>,
    row: &'a ResolvedScheduleRow,
    prime: Option<CommitOutput<P::Base, P::Challenge>>,
    mode: RootOpeningMode,
    response: Option<CommitOutput<P::Base, P::Challenge>>,
    order: Order,
}

impl<P: FieldFamily> RootProverOracle<P::Challenge> for ProverOracle<'_, P> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(Order::Image)?;
        bind_image::<P, S>(
            &self.owner.layout,
            layout,
            self.owner.setup.expanded.descriptor(),
            self.row,
            &self.image.commitment,
            channel,
        )?;
        self.order = if self.mode.has_prime() {
            Order::Prime
        } else {
            Order::Response
        };
        Ok(())
    }

    fn commit_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        table: &[P::Challenge],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(Order::Prime)?;
        if !self.mode.has_prime() || *layout != self.owner.layout {
            return Err(AkitaError::InvalidProof);
        }
        let profile = *self
            .row
            .profiles()
            .precommitteds
            .get(1)
            .ok_or(AkitaError::InvalidProof)?;
        let log_len = checked::sum([
            layout.prime_log_len(),
            usize::from(P::Challenge::DEGREE == 2),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let catalog = self
            .owner
            .elements
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?;
        let output = self.owner.commit_table(
            catalog,
            log_len,
            flatten_prime_table::<P>(layout, table)?,
            GroupContext::scheduler_without_precommitted_groups(),
        )?;
        exchange_commitment::<P, S>(profile, Some(&output.committed_group), channel)?;
        self.prime = Some(output);
        self.order = Order::Response;
        Ok(())
    }

    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(Order::Response)?;
        if *layout != self.owner.layout || digits.len() != layout.witness_len() {
            return Err(AkitaError::InvalidProof);
        }
        let mut profiles = Vec::new();
        profiles
            .try_reserve_exact(if self.mode.has_prime() { 2 } else { 1 })
            .map_err(|_| AkitaError::InvalidProof)?;
        profiles.push(*self.image.commitment.profile());
        if let Some(prime) = &self.prime {
            profiles.push(*prime.committed_group.profile());
        }
        let precommitteds = PrecommittedGroupProfiles::from_profiles(profiles)?;
        let output = self.owner.commit_table(
            self.owner.scheme.schedules(),
            layout.witness_log_len(),
            digit_table::<P>(digits)?,
            GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )?;
        exchange_commitment::<P, S>(
            self.row.profiles().final_group,
            Some(&output.committed_group),
            channel,
        )?;
        self.response = Some(output);
        self.order = Order::Discharge;
        Ok(())
    }

    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<P::Challenge>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(Order::Discharge)?;
        if claims.prime.is_some() != self.mode.has_prime() {
            return Err(AkitaError::InvalidProof);
        }
        let response = self.response.as_ref().ok_or(AkitaError::InvalidProof)?;
        let session = bind_opening::<P, S>(
            self.owner.setup.expanded.descriptor(),
            self.row,
            &opening_statement::<P>(
                &self.owner.layout,
                self.row,
                claims,
                &self.image.commitment,
                self.prime.as_ref().map(|output| &output.committed_group),
                &response.committed_group,
            )?,
            channel,
        )?;
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(if self.mode.has_prime() { 3 } else { 2 })
            .map_err(|_| AkitaError::InvalidProof)?;
        handles.push(self.image.handle.clone());
        if let Some(prime) = &self.prime {
            handles.push(prime.private_handle.clone());
        }
        handles.push(response.private_handle.clone());
        let opening = SelectedProverOpeningData::from_committed_claims::<P::Digits>(
            grouped_claims::<P, _>(
                &self.owner.layout,
                claims,
                self.image.commitment.clone(),
                self.prime
                    .as_ref()
                    .map(|output| output.committed_group.clone()),
                response.committed_group.clone(),
            )?,
            handles,
            self.owner.scheme.schedules(),
        )?;
        session.prove(channel, |session| {
            self.owner.scheme.batched_prove(
                &self.owner.setup,
                opening,
                &self.owner.backend,
                session,
                BasisMode::Lagrange,
            )
        })
    }
}
