//! Setup admission, binding, and field lifting for small-modulus commitments.

use akita_algebra::{
    binary::field_switch::SwitchField, MinusTrinomial, SmoothFftField, TrinomialLimbDomain,
    TrinomialRing,
};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::canonical_coefficient, BinaryClearCommitment, BinaryClearSetup,
};

use super::{commit_binary_clear_limb_prepared, PreparedLimbCommitMatrix, DEGREE};
use crate::commit_kernel::check_source_len;

impl PreparedLimbCommitMatrix {
    /// Prepare and bind a reduced matrix to a degree-648 small-modulus setup.
    ///
    /// Reject shared-prime setups, primes the limb domain does not admit, and
    /// matrix coefficients outside `[0, q0)` with `InvalidSetup`. Only values
    /// built by this constructor can serve the setup-bound commitment path.
    pub fn prepare_for_setup<F: SmoothFftField>(
        setup: &BinaryClearSetup<F, 648, MinusTrinomial>,
    ) -> Result<Self, AkitaError> {
        let q0 = setup.commitment_modulus().small_modulus().ok_or_else(|| {
            AkitaError::InvalidSetup("limb commitment requires a small modulus".into())
        })?;
        let domain = TrinomialLimbDomain::new(q0)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let elements = setup.matrix().iter().map(|element| {
            let mut values = [0; DEGREE];
            for (out, &coefficient) in values.iter_mut().zip(element.coefficients()) {
                let canonical = canonical_coefficient(coefficient)?;
                if canonical >= u128::from(q0) {
                    return Err(AkitaError::InvalidSetup(
                        "matrix coefficient must be canonical below the limb prime".into(),
                    ));
                }
                *out = u32::try_from(canonical).map_err(|_| {
                    AkitaError::InvalidSetup("limb matrix coefficient exceeds u32".into())
                })?;
            }
            Ok(values)
        });
        let mut prepared = Self::prepare_elements(domain, setup.n_a(), setup.m(), elements)?;
        prepared.setup_digest = Some(*setup.matrix_view_digest());
        Ok(prepared)
    }
}

/// Commit with the 32-bit-lane kernel for a degree-648 small-modulus setup.
///
/// Source length is checked before allocating images. Rank, width, modulus and
/// matrix-view digest must match the setup used by
/// [`PreparedLimbCommitMatrix::prepare_for_setup`]; raw prepared matrices and
/// mismatches return `InvalidSetup`. Images retain column/row/coefficient order
/// and lift canonical residues directly into the coefficient field.
pub fn commit_binary_clear_small_modulus_prepared<H, F>(
    prepared: &PreparedLimbCommitMatrix,
    setup: &BinaryClearSetup<F, 648, MinusTrinomial>,
    source: &[H::Source],
) -> Result<BinaryClearCommitment<F, 648, MinusTrinomial>, AkitaError>
where
    H: SwitchField,
    H::Source: Sync,
    F: SmoothFftField,
{
    check_source_len(setup.source_len(), source.len())?;
    if prepared.n_a != setup.n_a()
        || prepared.m != setup.m()
        || setup.commitment_modulus().small_modulus() != Some(prepared.domain.prime())
        || prepared.setup_digest.as_ref() != Some(setup.matrix_view_digest())
    {
        return Err(AkitaError::InvalidSetup(
            "prepared limb matrix does not match setup".into(),
        ));
    }
    let image_count = checked::product([setup.columns(), setup.n_a()])
        .ok_or_else(|| AkitaError::InvalidSetup("limb commitment size overflow".into()))?;
    let coefficients = commit_binary_clear_limb_prepared::<H>(prepared, setup.columns(), source)?;
    let mut images = Vec::new();
    images
        .try_reserve_exact(image_count)
        .map_err(|_| AkitaError::InvalidInput("limb matrix image allocation failed".into()))?;
    for element in coefficients.chunks_exact(DEGREE) {
        let mut lifted = [F::zero(); DEGREE];
        for (out, &coefficient) in lifted.iter_mut().zip(element) {
            *out = F::from_u64(u64::from(coefficient));
        }
        images.push(
            TrinomialRing::from_coefficients(lifted)
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?,
        );
    }
    Ok(BinaryClearCommitment { images })
}
