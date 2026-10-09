use akita_algebra::{fft::SmoothFftField, ring::TrinomialModulus};
use akita_error::{checked, AkitaError};

use super::{
    horner, image_weights_dense, signed_field, witness_weights_dense, LoweredPublic,
    LoweredRootLayout,
};
use crate::{endpoint::pack_response, BinaryClearCommitment, BinaryClearSetup};

/// Check the full digit alphabet and the one batched clear linear relation.
/// In-alphabet tail digits and image padding have zero public weights.
pub fn check_lowered_clear<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<F, D, M>,
    witness_digits: &[u8],
    image: &[F],
) -> Result<(), AkitaError> {
    public.validate_layout(layout)?;
    let alphabet = 1u8
        .checked_shl(layout.encoding().base().bits())
        .ok_or(AkitaError::InvalidProof)?;
    if witness_digits.len() != layout.witness_len()
        || image.len() != layout.image_len()
        || witness_digits.iter().any(|&digit| digit >= alphabet)
    {
        return Err(AkitaError::InvalidProof);
    }
    let witness_weights = witness_weights_dense(layout, public, setup)?;
    let image_weights = image_weights_dense(layout, public)?;
    let witness_sum = witness_digits
        .iter()
        .zip(&witness_weights)
        .fold(F::zero(), |acc, (&digit, &weight)| {
            acc + F::from_u64(u64::from(digit)) * weight
        });
    let image_sum = image
        .iter()
        .zip(&image_weights)
        .fold(F::zero(), |acc, (&value, &weight)| acc + value * weight);
    if witness_sum + image_sum != public.c_pub() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// One remainder A-row identity evaluated at alpha, on decoded scalar integers.
pub fn a_row_residual<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    response: &[[i64; 162]],
    row: usize,
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    layout.validate_setup(setup)?;
    if row >= layout.n_a()
        || commitment.images.len() != layout.image_count()
        || response.len() != layout.scalar_rows()
    {
        return Err(AkitaError::InvalidProof);
    }
    let alpha = public.challenges.alpha;
    let packed = pack_response(setup, response)?;
    let mut remainder = akita_algebra::ring::TrinomialRing::<F, D, M>::zero()
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    for (j, p) in packed.iter().enumerate() {
        let index = checked::mul_add(row, layout.m(), j).ok_or(AkitaError::InvalidProof)?;
        let a = setup.matrix().get(index).ok_or(AkitaError::InvalidProof)?;
        remainder += a
            .schoolbook_mul(p)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    }
    for (col, challenge) in public.embedded_challenges.iter().enumerate() {
        let index = checked::mul_add(col, layout.n_a(), row).ok_or(AkitaError::InvalidProof)?;
        let image = commitment
            .images
            .get(index)
            .ok_or(AkitaError::InvalidProof)?;
        let challenge = akita_algebra::ring::TrinomialRing::from_coefficients(
            challenge
                .as_slice()
                .try_into()
                .map_err(|_| AkitaError::InvalidProof)?,
        )
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        remainder -= challenge
            .schoolbook_mul(image)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    }
    let mut residual = horner(remainder.coefficients(), alpha);
    if layout.encoding().a_carry().is_some() {
        residual -= *public
            .a_carry_terms
            .get(row)
            .ok_or(AkitaError::InvalidProof)?;
    }
    Ok(residual)
}

/// Unreduced parity identity evaluated at xi, on decoded scalar integers.
pub fn parity_row_residual<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    response: &[[i64; 162]],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    if response.len() != layout.scalar_rows() {
        return Err(AkitaError::InvalidProof);
    }
    let mut residual = -public.parity_rhs;
    for (&weight, row) in public.binary_rows.iter().zip(response) {
        let evaluation = row.iter().rev().fold(F::zero(), |acc, &value| {
            acc * public.challenges.xi + signed_field::<F>(i128::from(value))
        });
        residual += weight * evaluation;
    }
    Ok(residual)
}
