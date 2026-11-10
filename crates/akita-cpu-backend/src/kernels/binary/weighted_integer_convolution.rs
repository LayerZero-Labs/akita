//! Unreduced binary-weighted signed integer convolution.

use akita_algebra::binary::BinaryField162;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

/// Enumerate the nonzero coordinates of a binary scalar weight.
pub fn set_bits(value: BinaryField162) -> impl Iterator<Item = usize> {
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
