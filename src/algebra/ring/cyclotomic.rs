//! Cyclotomic ring `Z_q[X]/(X^D + 1)` in coefficient form.

use super::sparse_challenge::SparseChallenge;
use crate::algebra::fields::wide::ReduceTo;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{AdditiveGroup, CanonicalField, FieldCore, FieldSampling};
use rand_core::RngCore;
use std::array::from_fn;
use std::io::{Read, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Element of the cyclotomic ring `Z_q[X]/(X^D + 1)`.
///
/// Stored as `D` coefficients in the base field `F`, representing
/// `a_0 + a_1*X + ... + a_{D-1}*X^{D-1}`.
///
/// Multiplication is negacyclic convolution: `X^D = -1`, so a product
/// term at index `i + j >= D` wraps to index `(i + j) - D` with a sign flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CyclotomicRing<F: FieldCore, const D: usize> {
    pub(crate) coeffs: [F; D],
}

impl<F: FieldCore, const D: usize> CyclotomicRing<F, D> {
    /// Construct from a coefficient array.
    #[inline]
    pub fn from_coefficients(coeffs: [F; D]) -> Self {
        Self { coeffs }
    }

    /// Construct from a slice, zero-padding if shorter than `D`.
    ///
    /// Avoids creating a `[F; D]` stack temporary when `D` is large.
    #[inline]
    pub fn from_slice(slice: &[F]) -> Self {
        let mut coeffs = [F::zero(); D];
        let len = slice.len().min(D);
        coeffs[..len].copy_from_slice(&slice[..len]);
        Self { coeffs }
    }

    /// Borrow the coefficient array.
    #[inline]
    pub fn coefficients(&self) -> &[F; D] {
        &self.coeffs
    }

    /// Mutably borrow the coefficient array.
    #[inline]
    pub fn coefficients_mut(&mut self) -> &mut [F; D] {
        &mut self.coeffs
    }

    /// The additive identity (all-zero polynomial).
    #[inline]
    pub fn zero() -> Self {
        Self {
            coeffs: [F::zero(); D],
        }
    }

    /// The multiplicative identity (`1 + 0*X + ... + 0*X^{D-1}`).
    #[inline]
    pub fn one() -> Self {
        let mut coeffs = [F::zero(); D];
        coeffs[0] = F::one();
        Self { coeffs }
    }

    /// The monomial `X` (i.e., `[0, 1, 0, ..., 0]`).
    ///
    /// # Panics
    ///
    /// Panics if `D < 2`.
    #[inline]
    pub fn x() -> Self {
        assert!(D >= 2, "ring degree must be at least 2");
        let mut coeffs = [F::zero(); D];
        coeffs[1] = F::one();
        Self { coeffs }
    }

    /// Scalar multiplication: multiply every coefficient by `k`.
    #[inline]
    pub fn scale(&self, k: &F) -> Self {
        let mut out = self.coeffs;
        for c in &mut out {
            *c = *c * *k;
        }
        Self { coeffs: out }
    }

    /// Apply the cyclotomic automorphism `sigma_k: X -> X^k` for odd `k`.
    ///
    /// In `Z_q[X]/(X^D + 1)`, this permutes/sign-flips coefficients using
    /// exponent reduction modulo `2D`.
    ///
    /// # Panics
    ///
    /// Panics if `D == 0` or `k` is not odd modulo `2D`.
    pub fn sigma(&self, k: usize) -> Self {
        assert!(D > 0, "ring degree must be non-zero");
        let two_d = 2 * D;
        let k_mod = k % two_d;
        assert!(k_mod % 2 == 1, "sigma_k requires odd k in Z_q[X]/(X^D + 1)");

        let mut out = [F::zero(); D];
        for (j, coeff) in self.coeffs.iter().copied().enumerate() {
            let idx = (j * k_mod) % two_d;
            if idx < D {
                out[idx] += coeff;
            } else {
                out[idx - D] -= coeff;
            }
        }
        Self { coeffs: out }
    }

    /// Apply `sigma_{-1}` (`X -> X^{-1} = X^{2D-1}` in this ring).
    ///
    /// # Panics
    ///
    /// Panics if `D == 0`.
    pub fn sigma_m1(&self) -> Self {
        assert!(D > 0, "ring degree must be non-zero");
        self.sigma(2 * D - 1)
    }

    /// Multiply by `X^k` in `Z_q[X]/(X^D + 1)` via O(D) coefficient rotation.
    ///
    /// Since `X^D = -1`, coefficients that wrap past index `D` get negated.
    #[inline]
    pub fn negacyclic_shift(&self, k: usize) -> Self {
        let k = k % D;
        if k == 0 {
            return *self;
        }
        let mut out = [F::zero(); D];
        for i in 0..D {
            let target = i + k;
            if target < D {
                out[target] = self.coeffs[i];
            } else {
                out[target - D] = -self.coeffs[i];
            }
        }
        Self { coeffs: out }
    }

    /// Multiply `self` by a sum of monomials `X^{k_1} + X^{k_2} + ...`
    ///
    /// Each term is a negacyclic shift, so the total cost is
    /// `O(positions.len() * D)` field additions with zero multiplications.
    pub fn mul_by_monomial_sum(&self, nonzero_positions: &[usize]) -> Self {
        let mut result = Self::zero();
        for &k in nonzero_positions {
            self.shift_accumulate_into(&mut result, k);
        }
        result
    }

    /// Fused negacyclic shift + accumulate: `dst += self * X^k`.
    ///
    /// Equivalent to `*dst += self.negacyclic_shift(k)` but avoids
    /// allocating a temporary ring element.
    #[inline]
    pub fn shift_accumulate_into(&self, dst: &mut Self, k: usize) {
        let k = k % D;
        if k == 0 {
            for i in 0..D {
                dst.coeffs[i] += self.coeffs[i];
            }
            return;
        }
        for i in 0..D {
            let target = i + k;
            if target < D {
                dst.coeffs[target] += self.coeffs[i];
            } else {
                dst.coeffs[target - D] -= self.coeffs[i];
            }
        }
    }

    /// Fused negacyclic shift + subtract: `dst -= self * X^k`.
    ///
    /// Equivalent to `*dst -= self.negacyclic_shift(k)` but avoids
    /// allocating a temporary ring element.
    #[inline]
    pub fn shift_sub_into(&self, dst: &mut Self, k: usize) {
        let k = k % D;
        if k == 0 {
            for i in 0..D {
                dst.coeffs[i] -= self.coeffs[i];
            }
            return;
        }
        for i in 0..D {
            let target = i + k;
            if target < D {
                dst.coeffs[target] -= self.coeffs[i];
            } else {
                dst.coeffs[target - D] += self.coeffs[i];
            }
        }
    }

    /// Fused multiply-by-monomial-sum + accumulate:
    /// `dst += self * (X^{k_1} + X^{k_2} + ...)`.
    ///
    /// Equivalent to `*dst += self.mul_by_monomial_sum(positions)` but avoids
    /// all intermediate temporaries.
    pub fn mul_by_monomial_sum_into(&self, dst: &mut Self, nonzero_positions: &[usize]) {
        for &k in nonzero_positions {
            self.shift_accumulate_into(dst, k);
        }
    }

    /// Multiply `self` by a sparse challenge element.
    ///
    /// Cost: `O(omega * D)` field additions instead of `O(D^2)` multiplications.
    /// For `omega=23, D=256` this is 5,888 adds vs 65,536 muls.
    pub fn mul_by_sparse(&self, challenge: &SparseChallenge) -> Self
    where
        F: CanonicalField,
    {
        let mut result = Self::zero();
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            match coeff {
                1 => self.shift_accumulate_into(&mut result, pos as usize),
                -1 => self.shift_sub_into(&mut result, pos as usize),
                c => {
                    let shifted = self.negacyclic_shift(pos as usize);
                    result += shifted.scale(&F::from_i64(c as i64));
                }
            }
        }
        result
    }

    /// Check whether all coefficients are zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    /// Count non-zero coefficients.
    #[inline]
    pub fn hamming_weight(&self) -> usize {
        self.coeffs.iter().filter(|c| !c.is_zero()).count()
    }

    /// Sample a sparse challenge with exactly `omega` non-zeros in `{+1, -1}`.
    ///
    /// # Panics
    ///
    /// Panics if `omega > D` or `D == 0` with non-zero `omega`.
    pub fn sample_sparse_pm1<R: RngCore>(rng: &mut R, omega: usize) -> Self {
        assert!(omega <= D, "omega must be <= ring degree");
        assert!(D > 0 || omega == 0, "ring degree must be non-zero");

        let mut coeffs = [F::zero(); D];
        let mut placed = 0usize;
        while placed < omega {
            let idx = (rng.next_u64() % (D as u64)) as usize;
            if coeffs[idx].is_zero() {
                coeffs[idx] = if (rng.next_u32() & 1) == 0 {
                    F::one()
                } else {
                    -F::one()
                };
                placed += 1;
            }
        }
        Self { coeffs }
    }
}

impl<F: CanonicalField, const D: usize> CyclotomicRing<F, D> {
    /// Balanced decomposition writing directly into a pre-allocated output slice.
    ///
    /// `out` must have length exactly `levels`. Each element receives one digit plane.
    ///
    /// # Panics
    ///
    /// Panics if `log_basis == 0`, `log_basis >= 128`, or `out.len() * log_basis > 128 + log_basis`.
    pub fn balanced_decompose_pow2_into(&self, out: &mut [Self], log_basis: u32) {
        let levels = out.len();
        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );

        let half_b = 1i128 << (log_basis - 1);
        let b = half_b << 1;
        let mask = b - 1;
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;

        for plane in out.iter_mut() {
            *plane = Self::zero();
        }

        for i in 0..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let mut c: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };

            for plane in out.iter_mut() {
                let d = c & mask;
                let balanced = if d >= half_b { d - b } else { d };
                c = (c - balanced) >> log_basis;

                plane.coeffs[i] = if balanced >= 0 {
                    F::from_canonical_u128_reduced(balanced as u128)
                } else {
                    F::from_canonical_u128_reduced(q - ((-balanced) as u128))
                };
            }
        }
    }

    /// Squared Euclidean norm of centered integer coefficients.
    ///
    /// Coefficients are centered into `(-q/2, q/2]` and accumulated as
    /// `sum_i c_i^2`, using saturating arithmetic.
    #[inline]
    pub fn coeff_norm_sq(&self) -> u128
    where
        F: CanonicalField,
    {
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        self.coeffs.iter().fold(0u128, |acc, &coeff| {
            let canonical = coeff.to_canonical_u128();
            let centered: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };
            let abs = centered.unsigned_abs();
            acc.saturating_add(abs.saturating_mul(abs))
        })
    }

    /// Functional gadget recomposition (`G * digits`) for base `2^log_basis`.
    ///
    /// Coefficients from each part are interpreted as one digit plane and
    /// recombined back into canonical integers (then reduced into the field).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis == 0`, `log_basis >= 128`, or `parts.len() * log_basis > 128`.
    pub fn gadget_recompose_pow2(parts: &[Self], log_basis: u32) -> Self {
        if parts.is_empty() {
            return Self::zero();
        }

        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");

        let b = F::from_canonical_u128_reduced(1u128 << log_basis);
        let coeffs = from_fn(|i| {
            let mut acc = F::zero();
            let mut power = F::one();
            for part in parts.iter() {
                acc += part.coeffs[i] * power;
                power = power * b;
            }
            acc
        });
        Self { coeffs }
    }

    /// Recompose from i8 digit planes (output of `balanced_decompose_pow2_i8`).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is zero or >= 128.
    pub fn gadget_recompose_pow2_i8(digits: &[[i8; D]], log_basis: u32) -> Self
    where
        F: CanonicalField,
    {
        if digits.is_empty() {
            return Self::zero();
        }
        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");

        let b = F::from_canonical_u128_reduced(1u128 << log_basis);
        let coeffs = from_fn(|i| {
            let mut acc = F::zero();
            let mut power = F::one();
            for plane in digits {
                acc += F::from_i64(plane[i] as i64) * power;
                power = power * b;
            }
            acc
        });
        Self { coeffs }
    }

    /// Balanced (centered) base-`2^log_basis` gadget decomposition: `G^{-1}`.
    ///
    /// Each coefficient `c` (centered into `(-q/2, q/2]`) is decomposed into
    /// `levels` balanced digits `d_k ∈ [-b/2, b/2)` satisfying
    /// `c ≡ Σ_k d_k · b^k  (mod q)`.
    ///
    /// Negative digits are stored as their field representation (`q + d`).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis == 0`, `log_basis >= 128`, or `levels * log_basis > 128`.
    pub fn balanced_decompose_pow2(&self, levels: usize, log_basis: u32) -> Vec<Self> {
        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );

        let half_b = 1i128 << (log_basis - 1);
        let b = half_b << 1;
        let mask = b - 1;
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;

        let mut digit_planes: Vec<[F; D]> = (0..levels).map(|_| [F::zero(); D]).collect();

        for i in 0..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let mut c: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };

            for plane in digit_planes.iter_mut() {
                let d = c & mask;
                let balanced = if d >= half_b { d - b } else { d };
                c = (c - balanced) >> log_basis;

                plane[i] = if balanced >= 0 {
                    F::from_canonical_u128_reduced(balanced as u128)
                } else {
                    F::from_canonical_u128_reduced(q - ((-balanced) as u128))
                };
            }
        }

        digit_planes
            .into_iter()
            .map(Self::from_coefficients)
            .collect()
    }

    /// Balanced gadget decomposition into native `i8` digits.
    ///
    /// Same semantics as [`balanced_decompose_pow2`](Self::balanced_decompose_pow2)
    /// but stores each digit as `i8` instead of a field element, avoiding
    /// the cost of `F::from_canonical_u128_reduced`.
    ///
    /// Requires `log_basis <= 7` so digits fit in `[-64, 63]` (i8 range).
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is 0 or > 7, or if `levels * log_basis > 128 + log_basis`.
    pub fn balanced_decompose_pow2_i8_into(&self, out: &mut [[i8; D]], log_basis: u32)
    where
        F: CanonicalField,
    {
        let levels = out.len();
        assert!(
            log_basis > 0 && log_basis <= 7,
            "log_basis must be in 1..=7 for i8 output"
        );
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
            "levels * log_basis must be <= 128 + log_basis"
        );

        let half_b = 1i128 << (log_basis - 1);
        let b = half_b << 1;
        let mask = b - 1;
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;

        for i in 0..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let mut c: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };

            for plane in out.iter_mut() {
                let d = c & mask;
                let balanced = if d >= half_b { d - b } else { d };
                c = (c - balanced) >> log_basis;
                plane[i] = balanced as i8;
            }
        }
    }

    /// Allocating variant of [`balanced_decompose_pow2_i8_into`](Self::balanced_decompose_pow2_i8_into).
    pub fn balanced_decompose_pow2_i8(&self, levels: usize, log_basis: u32) -> Vec<[i8; D]>
    where
        F: CanonicalField,
    {
        let mut digit_planes: Vec<[i8; D]> = vec![[0i8; D]; levels];
        self.balanced_decompose_pow2_i8_into(&mut digit_planes, log_basis);
        digit_planes
    }

    /// Balanced decomposition where the last digit carries the remainder.
    ///
    /// The first `levels-1` digits are balanced in `[-b/2, b/2)`, while the
    /// final digit is the remaining (possibly larger) centered value.
    ///
    /// # Panics
    ///
    /// Panics if `levels` is zero, `log_basis` is zero or >= 128, or
    /// `levels * log_basis > 128`.
    pub fn balanced_decompose_pow2_with_carry(&self, levels: usize, log_basis: u32) -> Vec<Self>
    where
        F: CanonicalField,
    {
        assert!(levels > 0, "levels must be positive");
        assert!(log_basis > 0 && log_basis < 128, "invalid log_basis");
        assert!(
            (levels as u32).saturating_mul(log_basis) <= 128,
            "levels * log_basis must be <= 128"
        );

        let b = 1i128 << log_basis;
        let half_b = b / 2;
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;

        let mut digit_planes: Vec<[F; D]> = (0..levels).map(|_| [F::zero(); D]).collect();

        for i in 0..D {
            let canonical = self.coeffs[i].to_canonical_u128();
            let mut c: i128 = if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            };

            for (plane_idx, plane) in digit_planes.iter_mut().enumerate() {
                let balanced = if plane_idx + 1 == levels {
                    c
                } else {
                    let d = c.rem_euclid(b);
                    let digit = if d >= half_b { d - b } else { d };
                    c = (c - digit) / b;
                    digit
                };

                plane[i] = if balanced >= 0 {
                    F::from_canonical_u128_reduced(balanced as u128)
                } else {
                    F::from_canonical_u128_reduced(q - ((-balanced) as u128))
                };
            }
        }

        digit_planes
            .into_iter()
            .map(Self::from_coefficients)
            .collect()
    }
}

impl<F: FieldCore + FieldSampling, const D: usize> CyclotomicRing<F, D> {
    /// Generate a random ring element.
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            coeffs: from_fn(|_| F::sample(rng)),
        }
    }
}

impl<F: FieldCore, const D: usize> AddAssign for CyclotomicRing<F, D> {
    fn add_assign(&mut self, rhs: Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst = *dst + *src;
        }
    }
}

impl<F: FieldCore, const D: usize> SubAssign for CyclotomicRing<F, D> {
    fn sub_assign(&mut self, rhs: Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst = *dst - *src;
        }
    }
}

impl<F: FieldCore, const D: usize> Add for CyclotomicRing<F, D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<F: FieldCore, const D: usize> Sub for CyclotomicRing<F, D> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<F: FieldCore, const D: usize> Neg for CyclotomicRing<F, D> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = self.coeffs;
        for c in &mut out {
            *c = -*c;
        }
        Self { coeffs: out }
    }
}

impl<F: FieldCore, const D: usize> MulAssign for CyclotomicRing<F, D> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore, const D: usize> Add<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl<'a, F: FieldCore, const D: usize> Sub<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl<'a, F: FieldCore, const D: usize> Mul<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self {
        self * *rhs
    }
}

/// Schoolbook negacyclic convolution: O(D^2).
///
/// For each pair `(i, j)`:
/// - If `i + j < D`: accumulate `a_i * b_j` at index `i + j`.
/// - If `i + j >= D`: accumulate `-(a_i * b_j)` at index `(i + j) - D`.
impl<F: FieldCore, const D: usize> Mul for CyclotomicRing<F, D> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = [F::zero(); D];
        for i in 0..D {
            for j in 0..D {
                let product = self.coeffs[i] * rhs.coeffs[j];
                let idx = i + j;
                if idx < D {
                    out[idx] += product;
                } else {
                    out[idx - D] -= product;
                }
            }
        }
        Self { coeffs: out }
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for CyclotomicRing<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.coeffs.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for CyclotomicRing<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for x in self.coeffs.iter() {
            x.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs
            .iter()
            .map(|x| x.serialized_size(compress))
            .sum()
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for CyclotomicRing<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut coeffs = [F::zero(); D];
        for c in &mut coeffs {
            *c = F::deserialize_with_mode(&mut reader, compress, validate)?;
        }
        let out = Self { coeffs };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore, const D: usize> Default for CyclotomicRing<F, D> {
    fn default() -> Self {
        Self::zero()
    }
}

/// Wide (unreduced) cyclotomic ring element for carry-free accumulation.
///
/// Coefficients are wide accumulators (`W: AdditiveGroup`) that support
/// addition/subtraction without modular reduction. After accumulation,
/// call [`reduce`](Self::reduce) to convert back to `CyclotomicRing<F, D>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideCyclotomicRing<W: AdditiveGroup, const D: usize> {
    pub(crate) coeffs: [W; D],
}

impl<W: AdditiveGroup, const D: usize> WideCyclotomicRing<W, D> {
    /// The additive identity (all-zero coefficients).
    pub const ZERO: Self = Self {
        coeffs: [W::ZERO; D],
    };

    /// Returns the zero ring element.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }

    /// Convert a reduced `CyclotomicRing<F, D>` into wide form.
    #[inline]
    pub fn from_ring<F: FieldCore>(ring: &CyclotomicRing<F, D>) -> Self
    where
        W: From<F>,
    {
        Self {
            coeffs: from_fn(|i| W::from(ring.coeffs[i])),
        }
    }

    /// Reduce all coefficients back to canonical field form.
    #[inline]
    pub fn reduce<F: FieldCore>(&self) -> CyclotomicRing<F, D>
    where
        W: ReduceTo<F>,
    {
        CyclotomicRing {
            coeffs: from_fn(|i| self.coeffs[i].reduce()),
        }
    }

    /// Fused negacyclic shift + accumulate: `dst += self * X^k`.
    #[inline]
    pub fn shift_accumulate_into(&self, dst: &mut Self, k: usize) {
        let k = k % D;
        if k == 0 {
            for i in 0..D {
                dst.coeffs[i] += self.coeffs[i];
            }
            return;
        }
        for i in 0..D {
            let target = i + k;
            if target < D {
                dst.coeffs[target] += self.coeffs[i];
            } else {
                dst.coeffs[target - D] -= self.coeffs[i];
            }
        }
    }

    /// Fused negacyclic shift + subtract: `dst -= self * X^k`.
    #[inline]
    pub fn shift_sub_into(&self, dst: &mut Self, k: usize) {
        let k = k % D;
        if k == 0 {
            for i in 0..D {
                dst.coeffs[i] -= self.coeffs[i];
            }
            return;
        }
        for i in 0..D {
            let target = i + k;
            if target < D {
                dst.coeffs[target] -= self.coeffs[i];
            } else {
                dst.coeffs[target - D] += self.coeffs[i];
            }
        }
    }

    /// Fused multiply-by-monomial-sum + accumulate:
    /// `dst += self * (X^{k_1} + X^{k_2} + ...)`.
    pub fn mul_by_monomial_sum_into(&self, dst: &mut Self, nonzero_positions: &[usize]) {
        for &k in nonzero_positions {
            self.shift_accumulate_into(dst, k);
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Add for WideCyclotomicRing<W, D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..D {
            self.coeffs[i] += rhs.coeffs[i];
        }
        self
    }
}

impl<W: AdditiveGroup, const D: usize> AddAssign for WideCyclotomicRing<W, D> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Sub for WideCyclotomicRing<W, D> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..D {
            self.coeffs[i] -= rhs.coeffs[i];
        }
        self
    }
}

impl<W: AdditiveGroup, const D: usize> SubAssign for WideCyclotomicRing<W, D> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Neg for WideCyclotomicRing<W, D> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            coeffs: from_fn(|i| -self.coeffs[i]),
        }
    }
}

impl<W: AdditiveGroup, const D: usize> Default for WideCyclotomicRing<W, D> {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::{Fp128x8i32, Fp64, Fp64x4i32, Prime128M8M4M1M0};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F64 = Fp64<4294967197>;
    type F128 = Prime128M8M4M1M0;
    const D: usize = 64;

    #[test]
    fn wide_shift_accumulate_matches_narrow_fp64() {
        let mut rng = StdRng::seed_from_u64(0x1234);
        let src = CyclotomicRing::<F64, D>::random(&mut rng);
        let initial = CyclotomicRing::<F64, D>::random(&mut rng);

        for k in [0, 1, 7, 31, 63] {
            let mut narrow = initial;
            src.shift_accumulate_into(&mut narrow, k);

            let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
            let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&initial);
            wide_src.shift_accumulate_into(&mut wide_dst, k);
            let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

            assert_eq!(narrow, wide_reduced, "shift_accumulate k={k}");
        }
    }

    #[test]
    fn wide_shift_sub_matches_narrow_fp64() {
        let mut rng = StdRng::seed_from_u64(0x5678);
        let src = CyclotomicRing::<F64, D>::random(&mut rng);
        let initial = CyclotomicRing::<F64, D>::random(&mut rng);

        for k in [0, 1, 15, 32, 63] {
            let mut narrow = initial;
            src.shift_sub_into(&mut narrow, k);

            let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
            let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&initial);
            wide_src.shift_sub_into(&mut wide_dst, k);
            let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

            assert_eq!(narrow, wide_reduced, "shift_sub k={k}");
        }
    }

    #[test]
    fn wide_mul_by_monomial_sum_matches_narrow_fp64() {
        let mut rng = StdRng::seed_from_u64(0xabcd);
        let src = CyclotomicRing::<F64, D>::random(&mut rng);
        let positions = vec![0, 5, 17, 42, 63];

        let mut narrow = CyclotomicRing::<F64, D>::zero();
        src.mul_by_monomial_sum_into(&mut narrow, &positions);

        let wide_src = WideCyclotomicRing::<Fp64x4i32, D>::from_ring(&src);
        let mut wide_dst = WideCyclotomicRing::<Fp64x4i32, D>::zero();
        wide_src.mul_by_monomial_sum_into(&mut wide_dst, &positions);
        let wide_reduced: CyclotomicRing<F64, D> = wide_dst.reduce();

        assert_eq!(narrow, wide_reduced);
    }

    #[test]
    fn wide_many_accumulations_fp128() {
        let mut rng = StdRng::seed_from_u64(0xbeef);
        let src = CyclotomicRing::<F128, D>::random(&mut rng);

        let mut narrow = CyclotomicRing::<F128, D>::zero();
        let wide_src = WideCyclotomicRing::<Fp128x8i32, D>::from_ring(&src);
        let mut wide_dst = WideCyclotomicRing::<Fp128x8i32, D>::zero();

        for k in 0..50 {
            src.shift_accumulate_into(&mut narrow, k % D);
            wide_src.shift_accumulate_into(&mut wide_dst, k % D);
        }
        for k in 0..30 {
            src.shift_sub_into(&mut narrow, k % D);
            wide_src.shift_sub_into(&mut wide_dst, k % D);
        }

        let wide_reduced: CyclotomicRing<F128, D> = wide_dst.reduce();
        assert_eq!(narrow, wide_reduced);
    }
}
