//! Setup binding for limb-transform commitments.

use akita_algebra::TrinomialLimbDomain;
use akita_algebra::{binary::field_switch::SwitchField, MinusTrinomial};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{BinaryClearCommitment, BinaryClearSetup};

use super::{commit_binary_clear_limb_prepared, PreparedLimbCommitMatrix, DEGREE};

impl PreparedLimbCommitMatrix {
    /// Prepare and bind the matrix of a degree-648 setup.
    ///
    /// Reject commitment primes the limb domain does not admit with
    /// `InvalidSetup`. Only values built by this constructor can serve the
    /// setup-bound commitment path.
    pub fn prepare_for_setup(
        setup: &BinaryClearSetup<DEGREE, MinusTrinomial>,
    ) -> Result<Self, AkitaError> {
        let domain = TrinomialLimbDomain::new(setup.modulus())
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let entries = checked::product([setup.n_a(), setup.m()]).ok_or(AkitaError::InvalidProof)?;
        let mut prepared = Self::prepare_elements(
            domain,
            setup.n_a(),
            setup.m(),
            (0..entries).map(|index| {
                let coefficients = setup
                    .matrix_element(index / setup.m(), index % setup.m())
                    .ok_or(AkitaError::InvalidProof)?;
                <[u32; DEGREE]>::try_from(coefficients).map_err(|_| AkitaError::InvalidProof)
            }),
        )?;
        prepared.setup_digest = Some(*setup.matrix_view_digest());
        Ok(prepared)
    }
}

/// Commit with the 32-bit-lane kernel for a degree-648 setup.
///
/// Source length is checked before allocating images. Rank, width, modulus and
/// matrix-view digest must match the setup used by
/// [`PreparedLimbCommitMatrix::prepare_for_setup`]; raw prepared matrices and
/// mismatches return `InvalidSetup`. Images are the canonical residues in
/// column/row/coefficient order, equal to [`crate::commit_binary_clear`].
pub fn commit_binary_clear_prepared<H>(
    prepared: &PreparedLimbCommitMatrix,
    setup: &BinaryClearSetup<DEGREE, MinusTrinomial>,
    source: &[H::Source],
) -> Result<BinaryClearCommitment, AkitaError>
where
    H: SwitchField,
    H::Source: Sync,
{
    if source.len() != setup.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: setup.source_len(),
            actual: source.len(),
        });
    }
    if prepared.n_a != setup.n_a()
        || prepared.m != setup.m()
        || setup.modulus() != prepared.domain.prime()
        || prepared.setup_digest.as_ref() != Some(setup.matrix_view_digest())
    {
        return Err(AkitaError::InvalidSetup(
            "prepared limb matrix does not match setup".into(),
        ));
    }
    Ok(BinaryClearCommitment {
        images: commit_binary_clear_limb_prepared::<H>(prepared, setup.columns(), source)?,
    })
}
