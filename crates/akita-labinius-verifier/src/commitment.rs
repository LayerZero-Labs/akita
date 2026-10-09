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

/// Canonical integer representative of a supported coefficient field element.
pub fn canonical_coefficient<F: SmoothFftField>(coefficient: F) -> Result<u128, AkitaError> {
    let mut bytes = [0u8; 16];
    let output = bytes
        .get_mut(..F::NUM_BYTES)
        .ok_or_else(|| AkitaError::InvalidSetup("coefficient field exceeds 128 bits".into()))?;
    coefficient.to_bytes_le(output);
    Ok(u128::from_le_bytes(bytes))
}

/// Unique centred representative in `[-(P-1)/2, (P-1)/2]`.
///
/// The caller must admit the integer bound before using this as an exact lift.
pub fn centered_coefficient<F: SmoothFftField>(coefficient: F) -> Result<i128, AkitaError> {
    let canonical = canonical_coefficient(coefficient)?;
    let modulus = match F::MODULUS_BITS {
        64 => (1u128 << 64).checked_sub(F::OFFSET),
        128 => u128::MAX
            .checked_sub(F::OFFSET)
            .and_then(|value| value.checked_add(1)),
        _ => None,
    }
    .ok_or_else(|| AkitaError::InvalidSetup("unsupported coefficient field".into()))?;
    if canonical <= modulus / 2 {
        i128::try_from(canonical).map_err(|_| AkitaError::InvalidProof)
    } else {
        i128::try_from(modulus - canonical)
            .map(|magnitude| -magnitude)
            .map_err(|_| AkitaError::InvalidProof)
    }
}

/// Centre the exact binary-source matrix product, then reduce modulo q0.
///
/// With reduced A and binary packing coefficients in `[-1,1]`, every
/// coefficient has magnitude at most `3*m*D*(q0-1)`. Setup admission requires
/// twice this bound below P before the existing P-field transform is used.
pub fn reduce_commitment_image<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    images: &mut [TrinomialRing<F, D, M>],
) -> Result<(), AkitaError> {
    let Some(q0) = setup.commitment_modulus().small_modulus() else {
        return Ok(());
    };
    for image in images {
        let mut coefficients = [F::zero(); D];
        for (target, &coefficient) in coefficients.iter_mut().zip(image.coefficients()) {
            let reduced = centered_coefficient(coefficient)?.rem_euclid(i128::from(q0));
            *target = F::from_u64(u64::try_from(reduced).map_err(|_| AkitaError::InvalidProof)?);
        }
        *image = TrinomialRing::from_coefficients(coefficients)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    }
    Ok(())
}
