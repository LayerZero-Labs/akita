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

    // A parallel in-place loop races because output `i` can clobber an input
    // needed by an earlier output. Large tables therefore use one parallel
    // destination allocation; small and late-round prefixes stay serial and
    // allocation-free to avoid Rayon and allocation overhead.
    #[cfg(feature = "parallel")]
    {
        // Native 128-bit folds need a larger batch to amortize dispatch and
        // allocation. Wider fields and extension contexts keep their crossover.
        let native_scalar = std::mem::size_of::<E>() == 16 && std::mem::size_of::<E::Ctx>() == 16;
        let threshold = if native_scalar { 1 << 15 } else { 1 << 12 };
        let grain = if native_scalar { 1 << 12 } else { 1 };
        if next_len >= threshold {
            let source: &[E] = evals;
            let folded = (0..next_len)
                .into_par_iter()
                .with_min_len(grain)
                .map(|target| {
                    let source_index = 2 * target;
                    let left = source[source_index];
                    let right = source
                        .get(source_index + 1)
                        .copied()
                        .unwrap_or_else(E::zero);
                    E::fold_one(&ctx, left, right)
                })
                .collect();
            *evals = folded;
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
    use jolt_field::{Prime128OffsetA7F7, Prime64Offset59, Ring, Zero};

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

    #[test]
    fn fold_zero_pads_an_odd_live_prefix() {
        type F = Prime64Offset59;
        let original = (0..(1 << 14) + 1)
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

        assert_eq!(actual, expected);
    }

    #[test]
    fn native_128_fold_preserves_padding_across_the_parallel_boundary() {
        type F = Prime128OffsetA7F7;
        let challenge = F::from_u64(29);
        for len in [(1 << 16) - 2, (1 << 16) - 1, 1 << 16, (1 << 16) + 1] {
            let original = (0..len)
                .map(|index| F::from_u64(index as u64 * 17 + 11))
                .collect::<Vec<_>>();
            let expected = original
                .chunks(2)
                .map(|pair| {
                    let odd = pair.get(1).copied().unwrap_or_else(F::zero);
                    pair[0] + challenge * (odd - pair[0])
                })
                .collect::<Vec<_>>();
            let mut actual = original;
            fold_evals_in_place(&mut actual, challenge);
            assert_eq!(actual, expected);
        }
    }
}
