//! Binary terminal claims inside the ordinary grouped opening interface.

use super::{GroupClaimKind, PolynomialGroupClaims};
use crate::binary_source::BinaryTerminalClaim;
use crate::proof::commitment::CommittedGroupKind;
use crate::CommittedGroup;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

impl<'a, E: Field, F: Field + CanonicalEncoding> PolynomialGroupClaims<'a, E, CommittedGroup<F>> {
    /// Select one binary terminal claim; the future field switch produces it.
    ///
    /// The ordinary point and evaluation vectors are zero carrier geometry for
    /// suffix schedule selection. Only the typed terminal is the binary claim.
    /// Use `map_commitment` to borrow the same frozen claim for verification.
    pub fn new_binary(
        commitment: CommittedGroup<F>,
        claim: BinaryTerminalClaim,
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
