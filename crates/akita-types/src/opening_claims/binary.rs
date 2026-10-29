//! Binary word claims inside the ordinary grouped opening interface.

use super::{GroupClaimKind, PolynomialGroupClaims};
use crate::binary_source::BinaryWordClaim;
use crate::proof::commitment::CommittedGroupKind;
use crate::CommittedGroup;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

impl<'a, E: Field, F: Field + CanonicalEncoding> PolynomialGroupClaims<'a, E, CommittedGroup<F>> {
    /// Select one logical word claim over GF(2^128).
    ///
    /// The ordinary point and evaluation vectors are zero carrier geometry for
    /// suffix schedule selection. Only the typed word claim is the binary claim.
    /// Use `map_commitment` to borrow the same frozen claim for verification.
    /// The opening runs the field switch and authenticates its carrier terminal.
    /// Multiple points on one polynomial must be reduced by the application;
    /// the opening rejects a second claim with the same polynomial identifier.
    pub fn new_binary(
        commitment: CommittedGroup<F>,
        claim: BinaryWordClaim,
    ) -> Result<Self, AkitaError> {
        let CommittedGroupKind::Binary(manifest) = commitment.kind() else {
            return Err(AkitaError::InvalidInput(
                "binary claim requires a binary commitment".into(),
            ));
        };
        claim.validate(manifest.descriptor())?;
        commitment
            .profile()
            .validate_frozen_precommit(F::MODULUS_BITS)?;
        manifest.validate_carrier(commitment.profile())?;
        let manifest = manifest.clone();
        let layout = commitment.profile().group;
        let mut point = Vec::new();
        point
            .try_reserve_exact(layout.num_vars())
            .map_err(|_| AkitaError::InvalidProof)?;
        point.resize(layout.num_vars(), E::zero());
        let mut evaluations = Vec::new();
        evaluations
            .try_reserve_exact(layout.num_polynomials())
            .map_err(|_| AkitaError::InvalidProof)?;
        evaluations.resize(layout.num_polynomials(), E::zero());
        Ok(Self {
            point: point.into(),
            evaluations,
            commitment,
            kind: GroupClaimKind::Binary {
                manifest: Box::new(manifest),
                claim,
            },
        })
    }
}
