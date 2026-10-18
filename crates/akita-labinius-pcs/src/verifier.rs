//! Verify binary, prime, or both claims on one committed binary message.

use crate::{
    binding::{
        bind_image, bind_opening, ensure_setup_prefix_coverage, exchange_commitment,
        opening_statement, response_row, root_layout, Order,
    },
    derive_root_setup,
    family::FieldFamily,
    shipped::SupportedGeometry,
    RootSetup,
};
use akita_algebra::binary::field_switch::SwitchField;
use akita_config::{ResolvedScheduleRow, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{verify_root_reduction_bytes, RootEvaluationClaims, RootVerifierOracle},
    RootOpeningMode, RootStatement,
};
use akita_params::BasisMode;
use akita_serialization::Valid;
use akita_types::{AkitaVerifierSetup, CommittedGroup};
use akita_verifier::AkitaVerifier;

/// The verifier of one root geometry over one proof-field family.
pub struct Verifier<P: FieldFamily> {
    root: RootSetup,
    layout: LoweredRootLayout,
    nested: AkitaVerifier<P::Digits>,
    elements: Option<TrustedScheduleCatalog<P::Elements>>,
}

impl<P: FieldFamily> Verifier<P> {
    /// Derive the root matrix from the nested setup's seed and admit the
    /// geometry's grouped row against that setup, before any proof is read.
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
        setup: AkitaVerifierSetup<P::Base>,
    ) -> Result<Self, AkitaError> {
        setup
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let root = derive_root_setup(geometry, &setup.expanded().descriptor().setup_seed)?;
        let layout = root_layout::<P>(&root)?;
        let nested = AkitaVerifier::new(setup, catalog)?;
        for selected in std::iter::once(None).chain(elements.as_ref().map(Some)) {
            let row = response_row::<P>(&layout, nested.schedules(), selected)?;
            ensure_setup_prefix_coverage(row, |id| {
                nested.setup().prefix_slots().get(id).is_some()
            })?;
            if !nested.admits(row.selection().row_digest) {
                return Err(AkitaError::InvalidSetup(
                    "the grouped row does not fit the verifier setup".into(),
                ));
            }
        }
        Ok(Self {
            root,
            layout,
            nested,
            elements,
        })
    }

    /// The admitted root setup, for a caller that runs the reduction itself.
    pub fn root(&self) -> &RootSetup {
        &self.root
    }

    /// Verify all claims in `statement` against `commitment`.
    /// The whole of `proof` must be consumed.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement is malformed or the proof is
    /// rejected.
    pub fn verify<H: SwitchField>(
        &self,
        commitment: &CommittedGroup<P::Base>,
        statement: &RootStatement<'_, H, P::Challenge>,
        proof: &[u8],
    ) -> Result<(), AkitaError> {
        verify_root_reduction_bytes::<H, P::Base, P::Challenge, _, _, _>(
            &self.root,
            statement,
            &mut self.oracle(commitment, statement.mode()?)?,
            proof,
        )?;
        Ok(())
    }

    /// A single-use oracle for a root reduction of `commitment` over
    /// [`Self::root`].
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the catalog no longer
    /// resolves the geometry's grouped row.
    pub fn oracle<'a>(
        &'a self,
        commitment: &'a CommittedGroup<P::Base>,
        mode: RootOpeningMode,
    ) -> Result<impl RootVerifierOracle<P::Challenge> + 'a, AkitaError> {
        let elements = if mode.has_prime() {
            Some(self.elements.as_ref().ok_or_else(|| {
                AkitaError::InvalidSetup("prime element catalog is missing".into())
            })?)
        } else {
            None
        };
        Ok(VerifierOracle {
            owner: self,
            image: commitment,
            row: response_row::<P>(&self.layout, self.nested.schedules(), elements)?,
            prime: None,
            response: None,
            mode,
            order: Order::default(),
        })
    }
}

/// The nested scheme behind the reduction's ordered verifier calls.
struct VerifierOracle<'a, P: FieldFamily> {
    owner: &'a Verifier<P>,
    image: &'a CommittedGroup<P::Base>,
    row: &'a ResolvedScheduleRow,
    prime: Option<CommittedGroup<P::Base>>,
    mode: RootOpeningMode,
    response: Option<CommittedGroup<P::Base>>,
    order: Order,
}

impl<P: FieldFamily> RootVerifierOracle<P::Challenge> for VerifierOracle<'_, P> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(Order::Image)?;
        bind_image::<P, S>(
            &self.owner.layout,
            layout,
            self.owner.nested.setup().expanded().descriptor(),
            self.row,
            self.image,
            channel,
        )?;
        self.order = if self.mode.has_prime() {
            Order::Prime
        } else {
            Order::Response
        };
        Ok(())
    }

    fn bind_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
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
        self.prime = Some(exchange_commitment::<P, S>(profile, None, channel)?);
        self.order = Order::Response;
        Ok(())
    }

    fn bind_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.order.take(Order::Response)?;
        if *layout != self.owner.layout {
            return Err(AkitaError::InvalidProof);
        }
        self.response = Some(exchange_commitment::<P, S>(
            self.row.profiles().final_group,
            None,
            channel,
        )?);
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
        let statement = opening_statement::<P>(
            &self.owner.layout,
            self.row,
            claims,
            self.image,
            self.prime.as_ref(),
            response,
        )?;
        let session = bind_opening::<P, S>(
            self.owner.nested.setup().expanded().descriptor(),
            self.row,
            &statement,
            channel,
        )?;
        session.verify(channel, |proof, session| {
            self.owner
                .nested
                .batched_verify(proof, session, statement, BasisMode::Lagrange)
        })
    }
}
