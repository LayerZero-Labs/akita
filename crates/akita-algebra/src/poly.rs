//! Multilinear polynomial evaluation utilities.

#[cfg(feature = "parallel")]
use super::eq_poly::EqPolynomial;
use crate::Field;
use akita_error::AkitaError;
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::Fold;

/// Evaluate a multilinear polynomial (given by boolean-hypercube evaluations in
/// little-endian bit order) at an arbitrary point via iterated folding.
///
/// # Errors
///
/// Returns an error if the evaluation table length is not a power of two or
/// does not match `2^point.len()`.
pub fn multilinear_eval<E: Field>(evals: &[E], point: &[E]) -> Result<E, AkitaError> {
    let point_len = u32::try_from(point.len()).map_err(|_| AkitaError::InvalidSize {
        expected: usize::BITS as usize,
        actual: point.len(),
    })?;
    let expected = 1usize
        .checked_shl(point_len)
        .ok_or(AkitaError::InvalidSize {
            expected: usize::MAX,
            actual: evals.len(),
        })?;
    if !evals.len().is_power_of_two() {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: evals.len(),
        });
    }
    if evals.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: evals.len(),
        });
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const PARALLEL_THRESHOLD: usize = 14;
        if point.len() > PARALLEL_THRESHOLD
            && evals.len() <= akita_serialization::DEFAULT_MAX_SEQUENCE_LEN
        {
            let eq_table = EqPolynomial::evals_parallel(point, None)?;
            return Ok(evals
                .par_iter()
                .zip(eq_table.par_iter())
                .fold(|| E::zero(), |acc, (e, eq)| acc + *e * *eq)
                .reduce(|| E::zero(), |a, b| a + b));
        }
    }

    Ok(multilinear_eval_ref(evals, point))
}

#[inline]
fn multilinear_eval_ref<E: Field>(evals: &[E], point: &[E]) -> E {
    match point.split_last() {
        None => {
            debug_assert_eq!(evals.len(), 1);
            evals[0]
        }
        Some((&r, rest)) => {
            let half = evals.len() / 2;
            let lo = multilinear_eval_ref(&evals[..half], rest);
            let hi = multilinear_eval_ref(&evals[half..], rest);
            lo + r * (hi - lo)
        }
    }
}

/// Fold a nonempty evaluation prefix in place by binding its first variable
/// to `r`, treating a missing final odd entry as implicit zero-padding.
///
/// # Panics
///
/// Panics if the evaluation prefix is empty. This is a prover-only helper where
/// the caller guarantees a nonempty live domain.
#[tracing::instrument(skip_all, name = "fold_evals_in_place")]
pub fn fold_evals_in_place<E: Fold>(evals: &mut Vec<E>, r: E) {
    assert!(!evals.is_empty(), "evaluation prefix must be nonempty");
    let next_len = evals.len().div_ceil(2);
    let ctx = E::precompute(r);

    // Output `i` reads inputs `2i` and `2i + 1`, so outputs `[s, 2s)` read
    // `[2s, 4s)`: such a wave writes below everything it reads, and its
    // entries are independent. Waves run in increasing order because wave `s`
    // overwrites the inputs of wave `s / 2`. Long waves run on the pool in
    // place; an allocation for the folded table would cost a page fault per
    // 16 KiB on top of the fold.
    #[cfg(feature = "parallel")]
    {
        /// Pairs in a wave before it goes to the pool.
        const PAR_WAVE_PAIRS: usize = 1 << 12;
        /// Fewest pairs per pool task.
        const MIN_TASK_PAIRS: usize = 1 << 10;
        if next_len >= 2 * PAR_WAVE_PAIRS {
            let right = evals.get(1).copied().unwrap_or_else(E::zero);
            evals[0] = E::fold_one(&ctx, evals[0], right);
            let mut start = 1;
            while start < next_len {
                let end = (2 * start).min(next_len);
                let (outputs, inputs) = evals.split_at_mut(2 * start);
                let outputs = &mut outputs[start..end];
                let fold = |(output, pair): (&mut E, &[E])| {
                    let right = pair.get(1).copied().unwrap_or_else(E::zero);
                    *output = E::fold_one(&ctx, pair[0], right);
                };
                if outputs.len() >= PAR_WAVE_PAIRS {
                    outputs
                        .par_iter_mut()
                        .zip(inputs.par_chunks(2))
                        .with_min_len(MIN_TASK_PAIRS)
                        .for_each(fold);
                } else {
                    outputs.iter_mut().zip(inputs.chunks(2)).for_each(fold);
                }
                start = end;
            }
            evals.truncate(next_len);
            // The vector still owns the whole table's pages. The folded half
            // is at least 128 KiB here, where an allocator gives the dead
            // pages back without moving the live ones.
            evals.shrink_to_fit();
            return;
        }
    }

    for target in 0..next_len {
        let source = 2 * target;
        let left = evals[source];
        let right = evals.get(source + 1).copied().unwrap_or_else(E::zero);
        evals[target] = E::fold_one(&ctx, left, right);
    }
    evals.truncate(next_len);
}

/// Remove trailing zero coefficients from a coefficient vector, preserving
/// at least one element.
#[inline]
pub fn trim_trailing_zeros<E: Field>(coeffs: &mut Vec<E>) {
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
}

#[cfg(test)]
mod fold_tests {
    use super::fold_evals_in_place;
    use jolt_field::{Prime64Offset59, Ring, Zero};

    #[test]
    fn fold_matches_independent_pair_evaluation() {
        type F = Prime64Offset59;
        let original = (0..1 << 14)
            .map(|index| F::from_u64((index as u64).wrapping_mul(17).wrapping_add(11)))
            .collect::<Vec<_>>();
        let challenge = F::from_u64(29);
        let expected = original
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect::<Vec<_>>();
        let mut folded = original;

        fold_evals_in_place(&mut folded, challenge);

        assert_eq!(folded, expected);
    }

    /// Lengths around and between powers of two: the last wave is ragged and
    /// the final pair may lack its right entry.
    #[test]
    fn fold_zero_pads_an_odd_live_prefix() {
        type F = Prime64Offset59;
        for len in [
            (1 << 14) + 1,
            3 * (1 << 13) + 5,
            (1 << 15) - 3,
            (1 << 16) + 2,
        ] {
            let original = (0..len)
                .map(|index| F::from_u64(index as u64 + 3))
                .collect::<Vec<_>>();
            let challenge = F::from_u64(13);
            let expected = (0..original.len().div_ceil(2))
                .map(|target| {
                    let left = original[2 * target];
                    let right = original
                        .get(2 * target + 1)
                        .copied()
                        .unwrap_or_else(F::zero);
                    left + challenge * (right - left)
                })
                .collect::<Vec<_>>();
            let mut actual = original;

            fold_evals_in_place(&mut actual, challenge);

            assert_eq!(actual, expected, "length {len}");
        }
    }
}
