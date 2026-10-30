//! External setup obligations. Successful reduction remains conditional until its accumulator is decided.
mod query;
pub mod transcript;

use crate::{AkitaExpandedSetup, AkitaSetupSeed, FpExtEncoding};
use akita_error::{checked, AkitaError};
use akita_params::dispatch::field_modulus_be_bytes;
use akita_serialization::AkitaSerialize;
use jolt_field::{CanonicalEncoding, ExtField, Field};
pub use query::{SetupQuery, SetupWeightTerm};

/// Exact flat setup polynomial; the rectangular split is intentionally not its identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupPolynomialId {
    seed: AkitaSetupSeed,
    modulus: [u8; 32],
    extension_degree: usize,
    field_len: usize,
}

impl SetupPolynomialId {
    /// Select an actual power-of-two prefix of an authenticated setup.
    pub fn new<F, E>(setup: &AkitaExpandedSetup<F>, field_len: usize) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + ExtField<F>,
    {
        if !field_len.is_power_of_two() || field_len > setup.shared_matrix().num_field_elements() {
            return Err(AkitaError::InvalidInput(
                "invalid external setup domain".into(),
            ));
        }
        Ok(Self {
            seed: setup.descriptor().setup_seed.clone(),
            modulus: field_modulus_be_bytes::<F>()?,
            extension_degree: E::DEGREE,
            field_len,
        })
    }

    pub fn validate<F, E>(&self, setup: &AkitaExpandedSetup<F>) -> Result<(), AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + ExtField<F>,
    {
        if Self::new::<F, E>(setup, self.field_len)? != *self {
            return Err(AkitaError::InvalidSetup(
                "external setup identity mismatch".into(),
            ));
        }
        Ok(())
    }

    pub const fn field_len(&self) -> usize {
        self.field_len
    }
    pub fn num_variables(&self) -> usize {
        self.field_len.trailing_zeros() as usize
    }

    /// One split contract for both native preparation and the bound batch statement.
    pub(super) fn validate_split(&self, coefficient_len: usize) -> Result<(), AkitaError> {
        if !coefficient_len.is_power_of_two()
            || coefficient_len > self.field_len
            || (coefficient_len == 1 && self.field_len != 1)
        {
            return Err(AkitaError::InvalidInput("invalid setup split".into()));
        }
        Ok(())
    }

    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), AkitaError> {
        bytes.extend_from_slice(b"akita/setup-claim-batch/v1/flat-lsb-first");
        self.seed
            .serialize_compressed(&mut *bytes)
            .map_err(|_| AkitaError::InvalidProof)?;
        bytes.extend_from_slice(&self.modulus);
        self.extension_degree
            .serialize_compressed(&mut *bytes)
            .map_err(|_| AkitaError::InvalidProof)?;
        self.field_len
            .serialize_compressed(bytes)
            .map_err(|_| AkitaError::InvalidProof)
    }
}

/// One exact query and candidate answer. Constructed from verifier replay, never a host-selected log.
#[derive(Clone)]
pub struct SetupClaim<E: Field> {
    polynomial: SetupPolynomialId,
    query: SetupQuery<E>,
    value: E,
}

impl<E: Field> SetupClaim<E> {
    pub fn new(
        polynomial: SetupPolynomialId,
        query: SetupQuery<E>,
        value: E,
    ) -> Result<Self, AkitaError> {
        query.validate(&polynomial)?;
        Ok(Self {
            polynomial,
            query,
            value,
        })
    }
    pub const fn polynomial(&self) -> &SetupPolynomialId {
        &self.polynomial
    }
    pub const fn query(&self) -> &SetupQuery<E> {
        &self.query
    }
    pub const fn value(&self) -> E {
        self.value
    }
}

/// Checked ordered statement for one compatible setup domain and coefficient-first split.
#[derive(Clone)]
pub struct SetupBatchStatement<E: Field> {
    polynomial: SetupPolynomialId,
    coefficient_len: usize,
    claims: Vec<SetupClaim<E>>,
}

impl<E: Field> SetupBatchStatement<E> {
    pub fn new(
        polynomial: SetupPolynomialId,
        coefficient_len: usize,
        claims: Vec<SetupClaim<E>>,
    ) -> Result<Self, AkitaError> {
        if claims.len() > 65536 {
            return Err(AkitaError::InvalidInput("invalid setup batch shape".into()));
        }
        polynomial.validate_split(coefficient_len)?;
        for claim in &claims {
            if claim.polynomial != polynomial {
                return Err(AkitaError::InvalidInput("incompatible setup claims".into()));
            }
            claim.query.validate(&polynomial)?;
            claim.query.validate_split(coefficient_len)?;
        }
        Ok(Self {
            polynomial,
            coefficient_len,
            claims,
        })
    }
    pub const fn polynomial(&self) -> &SetupPolynomialId {
        &self.polynomial
    }
    pub const fn coefficient_len(&self) -> usize {
        self.coefficient_len
    }
    pub fn claims(&self) -> &[SetupClaim<E>] {
        &self.claims
    }
    pub fn input_claim(&self, coefficients: &[E]) -> Result<E, AkitaError> {
        if coefficients.len() != self.claims.len() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(self
            .claims
            .iter()
            .zip(coefficients)
            .fold(E::zero(), |sum, (claim, &coefficient)| {
                sum + coefficient * claim.value
            }))
    }

    /// Evaluate the combined query at the shared sumcheck endpoint.
    pub fn evaluate_weight(&self, coefficients: &[E], point: &[E]) -> Result<E, AkitaError> {
        if coefficients.len() != self.claims.len() {
            return Err(AkitaError::InvalidProof);
        }
        self.claims
            .iter()
            .zip(coefficients)
            .try_fold(E::zero(), |sum, (claim, &coefficient)| {
                Ok(sum + coefficient * claim.query.evaluate_weight(&self.polynomial, point)?)
            })
    }

    /// Canonical mathematical statement, excluding equality caches and backend handles.
    pub fn descriptor_bytes<F>(&self) -> Result<Vec<u8>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        if self.polynomial.modulus != field_modulus_be_bytes::<F>()?
            || self.polynomial.extension_degree != E::DEGREE
        {
            return Err(AkitaError::InvalidInput(
                "setup batch field mismatch".into(),
            ));
        }
        let mut bytes = Vec::new();
        self.polynomial.encode(&mut bytes)?;
        self.coefficient_len
            .serialize_compressed(&mut bytes)
            .map_err(|_| AkitaError::InvalidProof)?;
        self.claims
            .len()
            .serialize_compressed(&mut bytes)
            .map_err(|_| AkitaError::InvalidProof)?;
        for claim in &self.claims {
            claim.query.append_descriptor_bytes::<F>(&mut bytes)?;
            for value in claim.value.to_base_vec() {
                bytes.extend_from_slice(&value.to_bytes_le_vec());
            }
        }
        Ok(bytes)
    }
}

/// One unresolved evaluation of the identified setup polynomial.
#[must_use = "a setup accumulator must be inherited or discharged before accepting"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupAccumulator<E: Field> {
    polynomial: SetupPolynomialId,
    point: Vec<E>,
    value: E,
}

impl<E: Field> SetupAccumulator<E> {
    pub fn new(polynomial: SetupPolynomialId, point: Vec<E>, value: E) -> Result<Self, AkitaError> {
        if point.len() != polynomial.num_variables() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self {
            polynomial,
            point,
            value,
        })
    }
    pub const fn polynomial(&self) -> &SetupPolynomialId {
        &self.polynomial
    }
    pub fn point(&self) -> &[E] {
        &self.point
    }
    pub const fn value(&self) -> E {
        self.value
    }
    pub fn into_claim(self) -> Result<SetupClaim<E>, AkitaError> {
        SetupClaim::new(
            self.polynomial,
            SetupQuery::evaluation(self.point)?,
            self.value,
        )
    }

    /// Embed a smaller prefix evaluation using zero selectors on new high coordinates.
    pub fn into_claim_for(
        mut self,
        polynomial: SetupPolynomialId,
    ) -> Result<SetupClaim<E>, AkitaError> {
        if self.polynomial.seed != polynomial.seed
            || self.polynomial.modulus != polynomial.modulus
            || self.polynomial.extension_degree != polynomial.extension_degree
            || self.polynomial.field_len > polynomial.field_len
        {
            return Err(AkitaError::InvalidInput(
                "incompatible setup accumulator embedding".into(),
            ));
        }
        self.point.resize(polynomial.num_variables(), E::zero());
        self.polynomial = polynomial;
        self.into_claim()
    }
}

/// Authoritative standalone Spongefish argument bytes.
pub struct SetupBatchProof {
    argument: Vec<u8>,
}
impl SetupBatchProof {
    pub fn from_argument(argument: Vec<u8>) -> Self {
        Self { argument }
    }
    pub fn argument(&self) -> &[u8] {
        &self.argument
    }
    pub fn into_argument(self) -> Vec<u8> {
        self.argument
    }
}

/// Native factor retention policy; regeneration trades index storage for preparation work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupBatchMemory {
    Retain,
    Regenerate,
}

pub(super) fn point_domain(point_len: usize) -> Result<usize, AkitaError> {
    checked::pow2(point_len)
        .ok_or_else(|| AkitaError::InvalidInput("setup point dimension overflow".into()))
}

impl<E: Field> SetupAccumulator<E> {
    /// Authenticate the closing commitment from the installed registry and produce an ordinary opening claim.
    /// Pass this as a precommitted group to a compatible admitted schedule, with Lagrange basis.
    pub fn prefix_opening_claim<F>(
        &self,
        setup: &crate::AkitaVerifierSetup<F>,
        slot: &akita_params::SetupPrefixSlotId,
    ) -> Result<crate::PolynomialGroupClaims<'static, E, crate::CommittedGroup<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: FpExtEncoding<F> + ExtField<F>,
    {
        self.polynomial.validate::<F, E>(setup.expanded())?;
        let prefix = setup.prefix_slots().get(slot).ok_or_else(|| {
            AkitaError::InvalidSetup("missing accumulator prefix commitment".into())
        })?;
        if prefix.id.n_prefix()? != self.polynomial.field_len
            || self.point.len() != self.polynomial.num_variables()
        {
            return Err(AkitaError::InvalidInput(
                "accumulator prefix opening domain mismatch".into(),
            ));
        }
        let [payload] = prefix.commitment.rows.as_slice() else {
            return Err(AkitaError::InvalidSetup(
                "invalid accumulator prefix commitment payload".into(),
            ));
        };
        crate::PolynomialGroupClaims::new(
            self.point.clone(),
            vec![self.value],
            crate::CommittedGroup::new(
                prefix.id.commitment_profile,
                crate::Commitment::new(payload.clone()),
            ),
        )
    }
}

mod encoding;
