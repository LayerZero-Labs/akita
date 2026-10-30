//! Explicit construction of the binary group kind in the canonical commitment.

use super::{Commitment, CommittedGroup, CommittedGroupKind};
use crate::binary_source::{BinaryCommittedGroupManifest, BINARY_COMMITTED_PAYLOAD_COEFFICIENTS};
use akita_error::AkitaError;
use akita_params::GroupCommitPhaseParams;
use jolt_field::{CanonicalEncoding, Field};

impl<F: Field + CanonicalEncoding> CommittedGroup<F> {
    /// Select the experimental binary root while retaining an ordinary suffix carrier.
    ///
    /// The carrier profile never describes the binary A/B commitment. Their real
    /// contracts are frozen in the typed manifest. This admission has no security claim.
    pub fn new_binary(
        profile: GroupCommitPhaseParams,
        commitment: Commitment<F>,
        manifest: BinaryCommittedGroupManifest,
    ) -> Result<Self, AkitaError> {
        profile.validate_frozen_precommit(F::MODULUS_BITS)?;
        if akita_params::field_modulus::<F>()? != akita_params::binary_setup::BINARY_SETUP_P {
            return Err(AkitaError::InvalidInput(
                "binary commitment requires the admitted proof prime".into(),
            ));
        }
        manifest.validate_carrier(&profile)?;
        if commitment.rows().coeff_len() != BINARY_COMMITTED_PAYLOAD_COEFFICIENTS {
            return Err(AkitaError::InvalidInput(
                "binary commitment payload shape mismatch".into(),
            ));
        }
        Ok(Self {
            profile,
            commitment,
            kind: CommittedGroupKind::Binary(manifest),
        })
    }
}
