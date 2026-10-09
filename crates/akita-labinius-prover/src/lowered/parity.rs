use akita_algebra::{binary::BinaryField162, SmoothFftField, TrinomialModulus};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    frontend::BinaryEvaluationClaim, profile::BinaryClearSetup, source::equality_weights,
};

/// Divide the unreduced integer parity residual by `Z^162 + Z^81 + 1`.
///
/// The quotient has 161 coefficients. The 162 remainder coefficients must
/// be even; their exact halves form the carry. Every integer accumulation and
/// reduction is checked before any reduction into the coefficient field.
pub fn parity_quotient_and_carry<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
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
