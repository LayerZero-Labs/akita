//! Frozen scalar encoding and parity arithmetic from 1c46446bc5f486ef5488fda1c081d07a93bd8d45.

#![cfg(feature = "labinius")]
#![allow(dead_code)]

use akita_algebra::{binary::BinaryField162, SmoothFftField, TrinomialModulus};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::BinaryClearCommitment, frontend::BinaryEvaluationClaim, lowered::LoweredRootLayout,
    profile::BinaryClearSetup, source::equality_weights,
};

/// Encode the exact accepted scalar interval in the canonical response table.
///
/// Sign-flipped packed coefficients use offset `off - 1`; hence their digits
/// complement the ordinary signed encoding. Honest coefficient tails are zero.
pub(crate) fn encode_witness(
    layout: &LoweredRootLayout,
    response: &[[i64; 162]],
) -> Result<Vec<u8>, AkitaError> {
    if response.len() != layout.scalar_rows() {
        return Err(AkitaError::InvalidProof);
    }
    let encoding = layout.encoding();
    let range = encoding.response();
    let (lower, upper) = range.interval();
    let digit_bits = encoding.base().bits();
    let mask = 1u128
        .checked_shl(digit_bits)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    let limit = 1u128
        .checked_shl(range.bits())
        .ok_or(AkitaError::InvalidProof)?;
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.witness_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    table.resize(layout.witness_len(), 0);
    for element in 0..layout.m() {
        for coefficient in 0..layout.degree() {
            let scalar_coefficient = coefficient / layout.k();
            let component = coefficient % layout.k();
            let row =
                checked::mul_add(element, layout.k(), component).ok_or(AkitaError::InvalidProof)?;
            let value = i128::from(
                *response
                    .get(row)
                    .and_then(|values| values.get(scalar_coefficient))
                    .ok_or(AkitaError::InvalidProof)?,
            );
            if value < lower || value > upper {
                return Err(AkitaError::InvalidProof);
            }
            let offset = layout.off(coefficient)?;
            let packed = value
                .checked_mul(layout.sigma(coefficient)?)
                .and_then(|value| value.checked_add(offset))
                .ok_or(AkitaError::InvalidProof)?;
            let mut unsigned = u128::try_from(packed).map_err(|_| AkitaError::InvalidProof)?;
            if unsigned >= limit {
                return Err(AkitaError::InvalidProof);
            }
            for digit in 0..range.digit_count() {
                let address = layout
                    .response_layout()
                    .address(element, digit, coefficient)?;
                *table.get_mut(address).ok_or(AkitaError::InvalidProof)? =
                    u8::try_from(unsigned & mask).map_err(|_| AkitaError::InvalidProof)?;
                unsigned = unsigned
                    .checked_shr(digit_bits)
                    .ok_or(AkitaError::InvalidProof)?;
            }
            if unsigned != 0 {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(table)
}

/// Flatten commitment coefficients into the admitted padded image domain.
pub(crate) fn flatten_image<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    commitment: &BinaryClearCommitment<F, D, M>,
) -> Result<Vec<F>, AkitaError> {
    if D != layout.degree() || commitment.images.len() != layout.image_count() {
        return Err(AkitaError::InvalidProof);
    }
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.image_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    table.resize(layout.image_len(), F::zero());
    for (element, image) in commitment.images.iter().enumerate() {
        for (coefficient, &value) in image.coefficients().iter().enumerate() {
            let address = layout.image_address(element, coefficient)?;
            *table.get_mut(address).ok_or(AkitaError::InvalidProof)? = value;
        }
    }
    Ok(table)
}

/// Divide the unreduced integer parity residual by `Z^162 + Z^81 + 1`.
///
/// The quotient has 161 coefficients. The 162 remainder coefficients must
/// be even; their exact halves form the carry. Every integer accumulation and
/// reduction is checked before any reduction into the coefficient field.
pub(crate) fn parity_quotient_and_carry<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<F, D, M>,
    claim: &BinaryEvaluationClaim,
    u: &[BinaryField162],
    fold_challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
) -> Result<(Vec<i128>, Vec<i128>), AkitaError> {
    if claim.point.len() != setup.num_vars()
        || u.len() != setup.columns()
        || fold_challenges.len() != setup.columns()
        || response.len() != setup.scalar_rows()
        || response
            .iter()
            .flatten()
            .any(|&value| value < setup.lower() || value > setup.upper())
    {
        return Err(AkitaError::InvalidProof);
    }
    for challenge in fold_challenges {
        challenge
            .validate(setup.profile())
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    let row_point = claim
        .point
        .get(..setup.row_vars())
        .ok_or(AkitaError::InvalidProof)?;
    let row_weights = equality_weights(row_point)?;
    if row_weights.len() != response.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut residual = [0i128; 323];
    for (weight, values) in row_weights.iter().zip(response) {
        let bits = weight.to_bytes();
        for bit in 0..162 {
            let byte = bits.get(bit / 8).ok_or(AkitaError::InvalidProof)?;
            if byte & (1u8 << (bit % 8)) == 0 {
                continue;
            }
            for (degree, &value) in values.iter().enumerate() {
                let index = checked::sum([bit, degree]).ok_or(AkitaError::InvalidProof)?;
                let accumulator = residual.get_mut(index).ok_or(AkitaError::InvalidProof)?;
                *accumulator = accumulator
                    .checked_add(i128::from(value))
                    .ok_or(AkitaError::InvalidProof)?;
            }
        }
    }
    for (value, challenge) in u.iter().zip(fold_challenges) {
        let bits = value.to_bytes();
        for bit in 0..162 {
            let byte = bits.get(bit / 8).ok_or(AkitaError::InvalidProof)?;
            if byte & (1u8 << (bit % 8)) == 0 {
                continue;
            }
            for term in challenge.terms() {
                let index = checked::sum([bit, usize::from(term.position)])
                    .ok_or(AkitaError::InvalidProof)?;
                let accumulator = residual.get_mut(index).ok_or(AkitaError::InvalidProof)?;
                *accumulator = accumulator
                    .checked_sub(i128::from(term.coefficient))
                    .ok_or(AkitaError::InvalidProof)?;
            }
        }
    }
    let mut quotient = Vec::new();
    quotient
        .try_reserve_exact(161)
        .map_err(|_| AkitaError::InvalidProof)?;
    quotient.resize(161, 0);
    for degree in (162..323).rev() {
        let leading = *residual.get(degree).ok_or(AkitaError::InvalidProof)?;
        let reduced = degree.checked_sub(162).ok_or(AkitaError::InvalidProof)?;
        *quotient.get_mut(reduced).ok_or(AkitaError::InvalidProof)? = leading;
        *residual.get_mut(degree).ok_or(AkitaError::InvalidProof)? = 0;
        let middle = checked::sum([reduced, 81]).ok_or(AkitaError::InvalidProof)?;
        for index in [reduced, middle] {
            let accumulator = residual.get_mut(index).ok_or(AkitaError::InvalidProof)?;
            *accumulator = accumulator
                .checked_sub(leading)
                .ok_or(AkitaError::InvalidProof)?;
        }
    }
    let mut carry = Vec::new();
    carry
        .try_reserve_exact(162)
        .map_err(|_| AkitaError::InvalidProof)?;
    for &coefficient in residual.get(..162).ok_or(AkitaError::InvalidProof)? {
        if coefficient.rem_euclid(2) != 0 {
            return Err(AkitaError::InvalidProof);
        }
        carry.push(coefficient.checked_div(2).ok_or(AkitaError::InvalidProof)?);
    }
    Ok((quotient, carry))
}
