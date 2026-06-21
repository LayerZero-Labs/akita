//! Dense polynomial storage and constructors.

use crate::backend::poly_helpers::try_small_i8_cache_from_ring_coeffs;
use crate::kernels::linear::try_centered_i8;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::{tensor_opening_split, TensorColumnSource};
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseDigitCache<const D: usize> {
    num_digits: usize,
    log_basis: u32,
    planes: Vec<[i8; D]>,
}

/// Dense polynomial: all ring coefficients materialized in memory.
#[derive(Debug)]
pub struct DensePoly<F: FieldCore, const D: usize> {
    /// Actual multilinear variable count of the source witness.
    pub(super) num_vars: usize,
    /// Ring coefficients in sequential block order.
    pub coeffs: Vec<CyclotomicRing<F, D>>,
    pub(super) small_i8_coeffs: Option<Vec<[i8; D]>>,
    digit_cache: OnceLock<DenseDigitCache<D>>,
}

impl<F: FieldCore + Clone, const D: usize> Clone for DensePoly<F, D> {
    fn clone(&self) -> Self {
        Self {
            num_vars: self.num_vars,
            coeffs: self.coeffs.clone(),
            small_i8_coeffs: self.small_i8_coeffs.clone(),
            digit_cache: OnceLock::new(),
        }
    }
}

impl<F: FieldCore + PartialEq, const D: usize> PartialEq for DensePoly<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.num_vars == other.num_vars
            && self.coeffs == other.coeffs
            && self.small_i8_coeffs == other.small_i8_coeffs
    }
}

impl<F: FieldCore + Eq, const D: usize> Eq for DensePoly<F, D> {}

impl<F: FieldCore + CanonicalField, const D: usize> DensePoly<F, D> {
    /// Pack field-element evaluations into ring elements.
    ///
    /// The first `α = log₂(D)` variables become coefficient slots within each
    /// ring element; the remaining variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two or if
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let outer_len = expected_len.div_ceil(D);
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut coeffs = Vec::with_capacity(outer_len);
        let mut small_i8_coeffs = Vec::with_capacity(outer_len);
        let mut all_small_i8 = true;

        for i in 0..outer_len {
            let start = i * D;
            let end = ((i + 1) * D).min(expected_len);
            let slice = &evals[start..end];
            let mut ring = CyclotomicRing::<F, D>::zero();
            for (coeff_idx, coeff) in slice.iter().enumerate() {
                ring.coeffs[coeff_idx] = *coeff;
            }
            coeffs.push(ring);

            if all_small_i8 {
                let mut digits = [0i8; D];
                for (coeff_idx, coeff) in slice.iter().enumerate() {
                    if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                        digits[coeff_idx] = centered;
                    } else {
                        all_small_i8 = false;
                        break;
                    }
                }
                if all_small_i8 {
                    small_i8_coeffs.push(digits);
                }
            }
        }

        Ok(Self {
            num_vars,
            coeffs,
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
            digit_cache: OnceLock::new(),
        })
    }

    /// Wrap an existing vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() * D` overflows `usize`.
    pub fn from_ring_coeffs(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs);
        let total = coeffs
            .len()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        Self {
            num_vars: total.trailing_zeros() as usize,
            coeffs,
            small_i8_coeffs,
            digit_cache: OnceLock::new(),
        }
    }

    pub(super) fn digit_planes_for(&self, num_digits: usize, log_basis: u32) -> Option<&[[i8; D]]> {
        if let Some(cache) = self.digit_cache.get() {
            return (cache.num_digits == num_digits && cache.log_basis == log_basis)
                .then_some(cache.planes.as_slice());
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);
        let mut planes = vec![[0i8; D]; self.coeffs.len() * num_digits];
        cfg_chunks_mut!(planes, num_digits)
            .zip(cfg_iter!(self.coeffs))
            .for_each(|(dst, ring)| {
                ring.balanced_decompose_pow2_i8_into_with_params(dst, &params);
            });
        let _ = self.digit_cache.set(DenseDigitCache {
            num_digits,
            log_basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        (cache.num_digits == num_digits && cache.log_basis == log_basis)
            .then_some(cache.planes.as_slice())
    }

    pub(super) fn live_coeff_len(&self) -> Result<usize, AkitaError> {
        1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            AkitaError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })
    }

    pub(super) fn tensor_shape<E>(
        &self,
        logical_point: Option<&[E]>,
    ) -> Result<(usize, usize), AkitaError>
    where
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        if width > D || !D.is_multiple_of(width) {
            return Err(AkitaError::InvalidInput(format!(
                "extension degree {width} does not evenly pack into dense ring degree {D}"
            )));
        }
        if let Some(point) = logical_point {
            if point.len() != self.num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: self.num_vars,
                    actual: point.len(),
                });
            }
        }
        Ok((split_bits, width))
    }
}

/// Column source over dense ring storage: `row(tail)` is the `width`-length
/// base-field run at flat index `tail*width`. `width` divides `D` and runs are
/// `width`-aligned within a ring, so a run never crosses a ring boundary.
pub(super) struct DenseColumnSource<'a, F: FieldCore, const D: usize> {
    pub(super) coeffs: &'a [CyclotomicRing<F, D>],
    pub(super) width: usize,
}

impl<F: FieldCore, const D: usize> TensorColumnSource<F> for DenseColumnSource<'_, F, D> {
    #[inline]
    fn row(&self, tail: usize) -> &[F] {
        let flat = tail * self.width;
        let ring_idx = flat / D;
        let coeff_idx = flat % D;
        &self.coeffs[ring_idx].coefficients()[coeff_idx..coeff_idx + self.width]
    }
}
