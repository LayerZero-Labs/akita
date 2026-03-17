//! Polynomial containers and evaluation utilities.

use crate::algebra::fields::wide::{HasWide, ReduceTo};
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::poly::EqPolynomial;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{cfg_fold_reduce, AdditiveGroup, FieldCore, FromSmallInt};
use std::io::{Read, Write};
use std::ops::{Add, Neg, Sub};

/// A degree-<D polynomial over `F`, stored as coefficients `[a0, a1, ..., a_{D-1}]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poly<F: FieldCore, const D: usize>(pub [F; D]);

impl<F: FieldCore, const D: usize> Poly<F, D> {
    /// Construct the zero polynomial.
    pub fn zero() -> Self {
        Self([F::zero(); D])
    }
}

impl<F: FieldCore, const D: usize> Add for Poly<F, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst += *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const D: usize> Sub for Poly<F, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst -= *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const D: usize> Neg for Poly<F, D> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = self.0;
        for coeff in &mut out {
            *coeff = -*coeff;
        }
        Self(out)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for Poly<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for Poly<F, D> {
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

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for Poly<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut arr = [F::zero(); D];
        for coeff in &mut arr {
            *coeff = F::deserialize_with_mode(&mut reader, compress, validate)?;
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
pub fn range_check_eval<E: FieldCore + FromSmallInt>(w: E, b: usize) -> E {
    let half = (b / 2) as i64;
    let mut acc = E::one();
    for k in -half..half {
        acc = acc * (w - E::from_i64(k));
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
pub fn multilinear_eval<E: FieldCore>(evals: &[E], point: &[E]) -> Result<E, HachiError> {
    if !evals.len().is_power_of_two() {
        return Err(HachiError::InvalidSize {
            expected: 1 << point.len(),
            actual: evals.len(),
        });
    }
    if evals.len() != 1 << point.len() {
        return Err(HachiError::InvalidSize {
            expected: 1 << point.len(),
            actual: evals.len(),
        });
    }
    Ok(multilinear_eval_ref(evals, point))
}

#[inline]
fn multilinear_eval_ref<E: FieldCore>(evals: &[E], point: &[E]) -> E {
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

/// Fold an evaluation table in place by binding its first variable to `r`,
/// halving the table size.
///
/// # Panics
///
/// Panics if the evaluation table length is not a power of two or has fewer
/// than 2 elements. This is a prover-only helper where the caller guarantees
/// well-formed input.
#[tracing::instrument(skip_all, name = "fold_evals_in_place")]
pub fn fold_evals_in_place<E: FieldCore>(evals: &mut Vec<E>, r: E) {
    assert!(
        evals.len().is_power_of_two(),
        "evals length must be a power of two"
    );
    assert!(evals.len() >= 2, "evals must have at least 2 elements");
    let half = evals.len() / 2;
    for i in 0..half {
        evals[i] = evals[2 * i] + r * (evals[2 * i + 1] - evals[2 * i]);
    }
    evals.truncate(half);
}

/// Evaluate a multilinear polynomial with small integer evaluations at a
/// field point, using the split-eq structure with unreduced accumulation.
///
/// Uses `HasWide::mul_small_to_wide` in the inner loop: each eq table entry
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
pub fn multilinear_eval_small<E: FieldCore + HasWide + FromSmallInt>(
    evals_small: &[i8],
    point: &[E],
) -> Result<E, HachiError> {
    let n = point.len();
    if evals_small.len() != 1 << n {
        return Err(HachiError::InvalidSize {
            expected: 1 << n,
            actual: evals_small.len(),
        });
    }
    if n == 0 {
        return Ok(E::from_i64(evals_small[0] as i64));
    }

    let m = n / 2;
    let (r_first, r_second) = point.split_at(m);
    let eq_first = EqPolynomial::evals(r_first);
    let eq_second = EqPolynomial::evals(r_second);
    let in_len = eq_first.len();

    // Max safe accumulations per chunk before i32 overflow.
    // Limbs are 16-bit (0..0xFFFF), scaled by |small| ≤ 128 → 23-bit products.
    // i32::MAX / (0xFFFF * 128) ≈ 256.
    const CHUNK: usize = 256;

    let outer_accum = cfg_fold_reduce!(
        0..eq_second.len(),
        || E::Wide::ZERO,
        |acc, x_out| {
            let base = x_out * in_len;
            let mut inner_field = E::zero();
            for chunk_start in (0..in_len).step_by(CHUNK) {
                let chunk_end = (chunk_start + CHUNK).min(in_len);
                let mut chunk_acc = E::Wide::ZERO;
                for x_in in chunk_start..chunk_end {
                    chunk_acc += eq_first[x_in].mul_small_to_wide(evals_small[base + x_in] as i32);
                }
                inner_field += <E::Wide as ReduceTo<E>>::reduce(chunk_acc);
            }

            acc + E::Wide::from(eq_second[x_out] * inner_field)
        },
        |a, b| a + b
    );
    Ok(<E::Wide as ReduceTo<E>>::reduce(outer_accum))
}
