use akita_algebra::{binary::BinaryField162, TrinomialModulus};
use akita_challenges::BinaryChallenge;
use akita_error::AkitaError;
use jolt_algebra::checked;
use akita_labinius_verifier::{
    frontend::BinaryEvaluationClaim, profile::BinaryClearSetup, source::equality_weights,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Divide the unreduced integer parity residual by `Z^162 + Z^81 + 1`.
///
/// The quotient has 161 coefficients. The 162 remainder coefficients must
/// be even; their exact halves form the carry. Integer accumulation uses a
/// proven i64 bound or checked i128 arithmetic; division remains checked i128
/// before any reduction into the coefficient field.
pub fn parity_quotient_and_carry<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
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
    let mut residual = binary_weighted_integer_convolution(&row_weights, response)?;
    for (&value, challenge) in u.iter().zip(fold_challenges) {
        for bit in set_bits(value) {
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

/// Sum unreduced integer convolutions of binary weights and signed scalar rows.
///
/// A weight is lifted coefficient-wise to zero or one, without reduction modulo
/// two. Enumerating its set bits costs exactly `162 * sum(popcount(weights))`
/// coefficient additions and no integer multiplications. Each result has 323
/// coefficients, before division by the scalar trinomial.
///
/// The absolute-sum bound `rows * 162 * max_abs(values)` covers every worker
/// subset and every reduction order. When it fits i64, workers reuse the narrow
/// accumulator; otherwise the same set-bit arithmetic uses checked i128 sums.
/// The output remains an integer polynomial and has no transcript semantics.
pub fn binary_weighted_integer_convolution<I: Copy + Into<i64> + Sync>(
    weights: &[BinaryField162],
    values: &[[I; 162]],
) -> Result<[i128; 323], AkitaError> {
    if weights.len() != values.len() {
        return Err(AkitaError::InvalidProof);
    }
    let value_bound = values
        .iter()
        .flatten()
        .map(|&value| {
            let value: i64 = value.into();
            value.unsigned_abs()
        })
        .max()
        .unwrap_or(0);
    let narrow = (values.len() as u128)
        .checked_mul(162)
        .and_then(|terms| terms.checked_mul(u128::from(value_bound)))
        .is_some_and(|bound| bound <= i64::MAX as u128);
    if narrow {
        let mut residual = [0i64; 323];
        // Heap-backed worker totals avoid carrying a degree-sized array through
        // Rayon's split recursion, including nested steals on a worker stack.
        #[cfg(feature = "parallel")]
        {
            let total = weights
                .par_iter()
                .zip(values.par_iter())
                .try_fold(Vec::new, |mut total: Vec<i64>, (&weight, values)| {
                    if total.is_empty() {
                        total
                            .try_reserve_exact(residual.len())
                            .map_err(|_| AkitaError::InvalidProof)?;
                        total.resize(residual.len(), 0);
                    }
                    accumulate_row_narrow(&mut total, weight, values)?;
                    Ok::<_, AkitaError>(total)
                })
                .try_reduce(Vec::new, |mut left, right| {
                    if left.is_empty() {
                        return Ok(right);
                    }
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    Ok(left)
                })?;
            for (coefficient, value) in residual.iter_mut().zip(total) {
                *coefficient = value;
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (&weight, values) in weights.iter().zip(values) {
            accumulate_row_narrow(&mut residual, weight, values)?;
        }
        Ok(residual.map(i128::from))
    } else {
        let mut residual = [0i128; 323];
        for (&weight, values) in weights.iter().zip(values) {
            for bit in set_bits(weight) {
                let end = checked::sum([bit, 162]).ok_or(AkitaError::InvalidProof)?;
                let coefficients = residual.get_mut(bit..end).ok_or(AkitaError::InvalidProof)?;
                for (accumulator, &value) in coefficients.iter_mut().zip(values) {
                    let value: i64 = value.into();
                    *accumulator = accumulator
                        .checked_add(i128::from(value))
                        .ok_or(AkitaError::InvalidProof)?;
                }
            }
        }
        Ok(residual)
    }
}

fn accumulate_row_narrow<I: Copy + Into<i64>>(
    residual: &mut [i64],
    weight: BinaryField162,
    values: &[I; 162],
) -> Result<(), AkitaError> {
    for bit in set_bits(weight) {
        let end = checked::sum([bit, 162]).ok_or(AkitaError::InvalidProof)?;
        let coefficients = residual.get_mut(bit..end).ok_or(AkitaError::InvalidProof)?;
        for (accumulator, &value) in coefficients.iter_mut().zip(values) {
            let value: i64 = value.into();
            *accumulator += value;
        }
    }
    Ok(())
}

fn set_bits(value: BinaryField162) -> impl Iterator<Item = usize> {
    value
        .to_words()
        .into_iter()
        .enumerate()
        .flat_map(|(word_index, mut word)| {
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                Some(word_index * 64 + bit)
            })
        })
}
