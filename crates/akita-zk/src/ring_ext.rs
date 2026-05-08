//! Thin adapters between Akita sparse challenges and cyclotomic rings.

use crate::error::ZkResult;
use crate::norm::field_from_centered_i128;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::{AkitaError, CanonicalField, FieldCore};

fn validate_sparse_challenge<const D: usize>(challenge: &SparseChallenge) -> ZkResult<()> {
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "sparse challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    for (&position, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        if position as usize >= D {
            return Err(AkitaError::InvalidInput(format!(
                "sparse challenge position {position} out of range for D={D}"
            )));
        }
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "sparse challenge coefficients must be non-zero".to_string(),
            ));
        }
    }
    Ok(())
}

/// Convert a sparse challenge into coefficient-form ring representation.
///
/// # Errors
///
/// Returns an error if the challenge shape is invalid for ring degree `D`.
pub fn sparse_challenge_to_ring<F, const D: usize>(
    challenge: &SparseChallenge,
) -> ZkResult<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    validate_sparse_challenge::<D>(challenge)?;
    let mut out = CyclotomicRing::zero();
    for (&position, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        out.coefficients_mut()[position as usize] = field_from_centered_i128(coeff as i128)?;
    }
    Ok(out)
}

/// Multiply a ring element by a sparse challenge.
///
/// # Errors
///
/// Returns an error if the challenge shape is invalid for ring degree `D`.
pub fn mul_sparse_challenge<F, const D: usize>(
    challenge: &SparseChallenge,
    value: &CyclotomicRing<F, D>,
) -> ZkResult<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
{
    validate_sparse_challenge::<D>(challenge)?;
    let mut out = CyclotomicRing::zero();
    for (&position, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let scale = field_from_centered_i128(coeff as i128)?;
        value.shift_scale_accumulate_into(&mut out, position as usize, scale);
    }
    Ok(out)
}

/// Multiply every element of a ring vector by a sparse challenge.
///
/// # Errors
///
/// Returns an error if the challenge shape is invalid for ring degree `D`.
pub fn mul_sparse_challenge_vec<F, const D: usize>(
    challenge: &SparseChallenge,
    values: &[CyclotomicRing<F, D>],
) -> ZkResult<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
{
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(mul_sparse_challenge(challenge, value)?);
    }
    Ok(out)
}

/// Add two ring vectors of equal length.
///
/// # Errors
///
/// Returns an error if the vectors have different lengths.
pub fn add_ring_vecs<F, const D: usize>(
    lhs: &[CyclotomicRing<F, D>],
    rhs: &[CyclotomicRing<F, D>],
) -> ZkResult<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore,
{
    if lhs.len() != rhs.len() {
        return Err(AkitaError::InvalidInput(format!(
            "ring vector length mismatch: {} != {}",
            lhs.len(),
            rhs.len()
        )));
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(a, b)| *a + *b).collect())
}

/// Subtract two ring vectors of equal length.
///
/// # Errors
///
/// Returns an error if the vectors have different lengths.
pub fn sub_ring_vecs<F, const D: usize>(
    lhs: &[CyclotomicRing<F, D>],
    rhs: &[CyclotomicRing<F, D>],
) -> ZkResult<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore,
{
    if lhs.len() != rhs.len() {
        return Err(AkitaError::InvalidInput(format!(
            "ring vector length mismatch: {} != {}",
            lhs.len(),
            rhs.len()
        )));
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(a, b)| *a - *b).collect())
}
