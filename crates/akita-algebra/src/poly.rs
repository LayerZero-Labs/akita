//! Polynomial containers and evaluation utilities.

use super::eq_poly::EqPolynomial;
use crate::{cfg_fold_reduce, Field, Ring, Zero};
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{Fold, Unreduced};
use std::io::{Read, Write};
use std::ops::{Add, Neg, Sub};

/// A degree-<D polynomial over `F`, stored as coefficients `[a0, a1, ..., a_{D-1}]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poly<F: Field, const D: usize>(pub [F; D]);

impl<F: Field, const D: usize> Poly<F, D> {
    /// Construct the zero polynomial.
    pub fn zero() -> Self {
        Self([F::zero(); D])
    }
}

impl<F: Field, const D: usize> Add for Poly<F, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst += *src;
        }
        Self(out)
    }
}

impl<F: Field, const D: usize> Sub for Poly<F, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst -= *src;
        }
        Self(out)
    }
}

impl<F: Field, const D: usize> Neg for Poly<F, D> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = self.0;
        for coeff in &mut out {
            *coeff = -*coeff;
        }
        Self(out)
    }
}

impl<F: Field + Valid, const D: usize> Valid for Poly<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize, const D: usize> AkitaSerialize for Poly<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.iter().map(|x| x.serialized_size(compress)).sum()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>, const D: usize> AkitaDeserialize
    for Poly<F, D>
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut arr = [F::zero(); D];
        for coeff in &mut arr {
            *coeff = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        }
        let out = Self(arr);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Evaluate the range-check polynomial `Π_{k=−b/2}^{b/2−1} (w − k)`.
///
/// This polynomial vanishes exactly on the balanced-digit set `{−b/2, …, b/2−1}`,
/// matching the output of `balanced_decompose_pow2`.
/// Total degree in `w` is `b`.
pub fn range_check_eval<E: Field + Ring>(w: E, b: usize) -> E {
    let half = (b / 2) as i64;
    let mut acc = E::one();
    for k in -half..half {
        acc *= w - E::from_i64(k);
    }
    acc
}

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
        const PAR_FOLD_THRESHOLD: usize = 1 << 12;
        if next_len >= PAR_FOLD_THRESHOLD {
            let source: &[E] = evals;
            let folded = (0..next_len)
                .into_par_iter()
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

/// Evaluate a multilinear polynomial with small integer evaluations at a
/// field point, using the split-eq structure with unreduced accumulation.
///
/// Uses `Unreduced::scale_wide` in the inner loop: each eq table entry
/// is widened, scaled by the small witness value, and accumulated without
/// reduction. The inner sum is reduced once per outer iteration, then
/// multiplied by the outer eq factor and accumulated again in wide form.
///
/// Overflow budget: each inner accumulation adds at most `0xFFFF * |small|`
/// to each i32 limb. For `|small| ≤ 128` (b ≤ 256), we can safely
/// accumulate 256 products before an i32 limb overflows.
///
/// # Errors
///
/// Returns an error if the table length does not match `2^point.len()`.
#[tracing::instrument(skip_all, name = "multilinear_eval_small")]
pub fn multilinear_eval_small<E: Field + Unreduced + Ring>(
    evals_small: &[i8],
    point: &[E],
) -> Result<E, AkitaError> {
    let n = point.len();
    let expected_len = 1usize
        .checked_shl(u32::try_from(n).map_err(|_| AkitaError::InvalidSize {
            expected: usize::BITS as usize,
            actual: n,
        })?)
        .ok_or_else(|| {
            AkitaError::InvalidInput("small MLE table dimension overflow".to_string())
        })?;
    if evals_small.len() != expected_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_len,
            actual: evals_small.len(),
        });
    }
    if n == 0 {
        return Ok(E::from_i64(evals_small[0] as i64));
    }

    let m = n / 2;
    let (r_first, r_second) = point.split_at(m);
    let eq_first = EqPolynomial::evals(r_first)?;
    let eq_second = EqPolynomial::evals(r_second)?;
    let in_len = eq_first.len();

    // Max safe accumulations per chunk before i32 overflow.
    // Limbs are 16-bit (0..0xFFFF), scaled by |small| ≤ 128 → 23-bit products.
    // i32::MAX / (0xFFFF * 128) ≈ 256.
    const CHUNK: usize = 256;

    let outer_accum = cfg_fold_reduce!(
        0..eq_second.len(),
        E::Wide::zero,
        |acc, x_out| {
            let base = x_out * in_len;
            let mut inner_field = E::zero();
            for chunk_start in (0..in_len).step_by(CHUNK) {
                let chunk_end = (chunk_start + CHUNK).min(in_len);
                let mut chunk_acc = E::Wide::zero();
                for x_in in chunk_start..chunk_end {
                    chunk_acc += eq_first[x_in].scale_wide(evals_small[base + x_in] as i32);
                }
                inner_field += E::reduce_wide(chunk_acc);
            }

            acc + E::Wide::from(eq_second[x_out] * inner_field)
        },
        |a, b| a + b
    );
    Ok(E::reduce_wide(outer_accum))
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
}
