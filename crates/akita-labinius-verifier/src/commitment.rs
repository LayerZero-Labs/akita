//! Commitment data and the canonical explicit matrix action.

use akita_algebra::fft::SmoothFftField;
use akita_algebra::ring::{TrinomialModulus, TrinomialNttDomain, TrinomialRing};
use akita_error::{checked, AkitaError};

use crate::profile::BinaryClearSetup;

/// Public images, ordered column first and then Ajtai matrix row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryClearCommitment<F, const D: usize, M: TrinomialModulus> {
    /// `images[column * n_a + matrix_row]`.
    pub images: Vec<TrinomialRing<F, D, M>>,
}

/// Apply the public row-major Ajtai matrix with trinomial NTT products.
pub fn apply_matrix<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    vector: &[TrinomialRing<F, D, M>],
) -> Result<Vec<TrinomialRing<F, D, M>>, AkitaError> {
    if vector.len() != setup.m() {
        return Err(AkitaError::InvalidInput(
            "matrix vector length mismatch".into(),
        ));
    }
    let domain = TrinomialNttDomain::<F, D, M>::new()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    let mut workspace = domain.workspace();
    let mut result = Vec::new();
    result
        .try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidInput("matrix image allocation failed".into()))?;
    for row in 0..setup.n_a() {
        let start = checked::product([row, setup.m()]).ok_or(AkitaError::InvalidProof)?;
        let range = checked::range(start, setup.m()).ok_or(AkitaError::InvalidProof)?;
        let matrix_row = setup.matrix().get(range).ok_or(AkitaError::InvalidProof)?;
        let mut accumulator =
            TrinomialRing::zero().map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        for (matrix_element, vector_element) in matrix_row.iter().zip(vector) {
            accumulator +=
                domain.multiply_with_workspace(matrix_element, vector_element, &mut workspace);
        }
        result.push(accumulator);
    }
    Ok(result)
}
