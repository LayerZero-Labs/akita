//! Polynomial containers and evaluation utilities.

use crate::cfg_into_iter;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use crate::FromSmallInt;
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
            *dst = *dst + *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const D: usize> Sub for Poly<F, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst = *dst - *src;
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

/// Evaluate the range-check polynomial `w · Π_{k=1}^{b−1} (w − k)(w + k)`.
///
/// This polynomial vanishes exactly when `w ∈ {−(b−1), …, b−1}`.
/// Total degree in `w` is `2b − 1`.
pub fn range_check_eval<E: FieldCore + FromSmallInt>(w: E, b: usize) -> E {
    let mut acc = w;
    for k in 1..b {
        let k_e = E::from_u64(k as u64);
        acc = acc * (w - k_e) * (w + k_e);
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
    let mut current = evals.to_vec();
    for &r in point {
        let half = current.len() / 2;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            next.push(current[2 * i] + r * (current[2 * i + 1] - current[2 * i]));
        }
        current = next;
    }
    Ok(current[0])
}

/// Fold an evaluation table in place by binding its first variable to `r`,
/// halving the table size.
///
/// # Panics
///
/// Panics if the evaluation table length is not a power of two or has fewer
/// than 2 elements. This is a prover-only helper where the caller guarantees
/// well-formed input.
pub fn fold_evals_in_place<E: FieldCore>(evals: &mut Vec<E>, r: E) {
    assert!(
        evals.len().is_power_of_two(),
        "evals length must be a power of two"
    );
    assert!(evals.len() >= 2, "evals must have at least 2 elements");
    let half = evals.len() / 2;
    let folded: Vec<E> = cfg_into_iter!(0..half)
        .map(|i| evals[2 * i] + r * (evals[2 * i + 1] - evals[2 * i]))
        .collect();
    *evals = folded;
}
