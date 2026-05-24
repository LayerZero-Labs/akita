//! Config-owned verifier policy helpers shared by scheme wrappers and profiles.

use crate::CommitmentConfig;
use akita_field::{AkitaError, ExtField, FieldCore, FromPrimitiveInt};
use akita_types::{validate_ring_subfield_role, RingSubfieldEncoding};

/// Validate that a config's claim/challenge field roles are compatible with
/// the active ring dimension.
///
/// # Errors
///
/// Returns an error if the field tower degrees or ring-subfield roles are
/// inconsistent.
pub fn validate_field_roles_for_config<F, const D: usize, Cfg>() -> Result<(), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: RingSubfieldEncoding<F>,
    Cfg::ChallengeField: RingSubfieldEncoding<F>,
{
    if D != Cfg::D {
        return Err(AkitaError::InvalidSetup(format!(
            "ring dimension D={D} does not match config D={}",
            Cfg::D
        )));
    }
    validate_ring_subfield_role::<F, Cfg::ClaimField, D>("claim field")?;
    validate_ring_subfield_role::<F, Cfg::ChallengeField, D>("challenge field")?;
    let relative_degree = <Cfg::ChallengeField as ExtField<Cfg::ClaimField>>::EXT_DEGREE;
    let expected_challenge_degree = Cfg::CLAIM_EXT_DEGREE
        .checked_mul(relative_degree)
        .ok_or_else(|| AkitaError::InvalidSetup("field tower degree overflow".to_string()))?;
    if Cfg::CHAL_EXT_DEGREE != expected_challenge_degree {
        return Err(AkitaError::InvalidSetup(format!(
            "challenge field degree {} does not match claim degree {} times relative degree {}",
            Cfg::CHAL_EXT_DEGREE,
            Cfg::CLAIM_EXT_DEGREE,
            relative_degree
        )));
    }
    Ok(())
}
