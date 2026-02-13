//! Cyclotomic ring `Z_q[X]/(X^D + 1)` in coefficient form.

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::Field;
use rand_core::RngCore;
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
pub struct CyclotomicRing<F: Field, const D: usize> {
    pub(crate) coeffs: [F; D],
}

impl<F: Field, const D: usize> CyclotomicRing<F, D> {
    /// Construct from a coefficient array.
    #[inline]
    pub fn from_coefficients(coeffs: [F; D]) -> Self {
        Self { coeffs }
    }

    /// Borrow the coefficient array.
    #[inline]
    pub fn coefficients(&self) -> &[F; D] {
        &self.coeffs
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

    /// Generate a random ring element.
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Self {
            coeffs: std::array::from_fn(|_| F::random(rng)),
        }
    }
}

impl<F: Field, const D: usize> AddAssign for CyclotomicRing<F, D> {
    fn add_assign(&mut self, rhs: Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst = *dst + *src;
        }
    }
}

impl<F: Field, const D: usize> SubAssign for CyclotomicRing<F, D> {
    fn sub_assign(&mut self, rhs: Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst = *dst - *src;
        }
    }
}

impl<F: Field, const D: usize> Add for CyclotomicRing<F, D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<F: Field, const D: usize> Sub for CyclotomicRing<F, D> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<F: Field, const D: usize> Neg for CyclotomicRing<F, D> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = self.coeffs;
        for c in &mut out {
            *c = -*c;
        }
        Self { coeffs: out }
    }
}

impl<F: Field, const D: usize> MulAssign for CyclotomicRing<F, D> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: Field, const D: usize> Add<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl<'a, F: Field, const D: usize> Sub<&'a Self> for CyclotomicRing<F, D> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl<'a, F: Field, const D: usize> Mul<&'a Self> for CyclotomicRing<F, D> {
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
impl<F: Field, const D: usize> Mul for CyclotomicRing<F, D> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = [F::zero(); D];
        for i in 0..D {
            for j in 0..D {
                let product = self.coeffs[i] * rhs.coeffs[j];
                let idx = i + j;
                if idx < D {
                    out[idx] = out[idx] + product;
                } else {
                    out[idx - D] = out[idx - D] - product;
                }
            }
        }
        Self { coeffs: out }
    }
}

impl<F: Field + Valid, const D: usize> Valid for CyclotomicRing<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.coeffs.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: Field, const D: usize> HachiSerialize for CyclotomicRing<F, D> {
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

impl<F: Field + Valid, const D: usize> HachiDeserialize for CyclotomicRing<F, D> {
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

impl<F: Field, const D: usize> Default for CyclotomicRing<F, D> {
    fn default() -> Self {
        Self::zero()
    }
}
