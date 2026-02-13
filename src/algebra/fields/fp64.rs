//! A prime field implementation backed by `u64` storage.
//!
//! - Modulus is a `u64` const generic.
//! - Multiplication uses `u128` widening arithmetic.

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore, FieldSampling, Invertible};
use rand_core::RngCore;
use std::io::{Read, Write};

/// Prime field element modulo `MODULUS` stored as `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp64<const MODULUS: u64>(pub(crate) u64);

impl<const MODULUS: u64> Fp64<MODULUS> {
    /// Create an element from a canonical representative in `[0, MODULUS)`.
    #[inline]
    pub fn from_canonical_u64(x: u64) -> Self {
        debug_assert!(x < MODULUS);
        Self(x)
    }

    /// Return the canonical representative in `[0, MODULUS)`.
    #[inline]
    pub fn to_canonical_u64(self) -> u64 {
        self.0
    }

    #[inline]
    fn reduce_u128(x: u128) -> u64 {
        let m = MODULUS as u128;
        let mut rem = 0u128;

        // Division-free fixed-iteration reduction.
        for i in (0..128).rev() {
            rem = (rem << 1) | ((x >> i) & 1);
            let (candidate, borrow) = rem.overflowing_sub(m);
            let mask = 0u128.wrapping_sub((!borrow) as u128);
            rem = (candidate & mask) | (rem & !mask);
        }

        rem as u64
    }

    #[inline]
    fn add_raw(a: u64, b: u64) -> u64 {
        let s = (a as u128) + (b as u128);
        let m = MODULUS as u128;
        let reduced = s.wrapping_sub(m);
        // If s < m, the subtraction underflowed (bit 127 set). Add m back.
        let borrow = reduced >> 127;
        reduced.wrapping_add(borrow.wrapping_neg() & m) as u64
    }

    #[inline]
    fn sub_raw(a: u64, b: u64) -> u64 {
        let diff = (a as u128).wrapping_sub(b as u128);
        // If a < b, the subtraction underflowed (bit 127 set). Add MODULUS.
        let borrow = diff >> 127;
        diff.wrapping_add(borrow.wrapping_neg() & (MODULUS as u128)) as u64
    }

    #[inline]
    fn mul_raw(a: u64, b: u64) -> u64 {
        Self::reduce_u128((a as u128) * (b as u128))
    }

    fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = base * base;
            exp >>= 1;
        }
        acc
    }
}

impl<const MODULUS: u64> std::ops::Add for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}
impl<const MODULUS: u64> std::ops::Sub for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}
impl<const MODULUS: u64> std::ops::Mul for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}
impl<const MODULUS: u64> std::ops::Neg for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<'a, const MODULUS: u64> std::ops::Add<&'a Self> for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, const MODULUS: u64> std::ops::Sub<&'a Self> for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, const MODULUS: u64> std::ops::Mul<&'a Self> for Fp64<MODULUS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const MODULUS: u64> Valid for Fp64<MODULUS> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.0 < MODULUS {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp64 out of range".into()))
        }
    }
}

impl<const MODULUS: u64> HachiSerialize for Fp64<MODULUS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        8
    }
}

impl<const MODULUS: u64> HachiDeserialize for Fp64<MODULUS> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let x = u64::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        if matches!(validate, Validate::Yes) && x >= MODULUS {
            return Err(SerializationError::InvalidData(
                "Fp64 out of range".to_string(),
            ));
        }
        let out = if matches!(validate, Validate::Yes) {
            Self(x)
        } else {
            Self(x % MODULUS)
        };
        Ok(out)
    }
}

impl<const MODULUS: u64> FieldCore for Fp64<MODULUS> {
    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(1 % MODULUS)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }

    fn inv(self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if self.is_zero() {
            None
        } else {
            Some(inv)
        }
    }
}

impl<const MODULUS: u64> Invertible for Fp64<MODULUS> {
    fn inv_or_zero(self) -> Self {
        // Fermat inversion: a^(p-2) mod p (MODULUS should be prime).
        let candidate = self.pow(MODULUS.wrapping_sub(2));
        let mask = nonzero_mask_u64(self.0);
        Self(candidate.0 & mask)
    }
}

impl<const MODULUS: u64> FieldSampling for Fp64<MODULUS> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        // rejection-sample to reduce bias for odd moduli
        loop {
            let x = rng.next_u64();
            let m = MODULUS;
            let t = x % m;
            // Accept if x - t doesn't overflow u64. This is standard trick.
            if x.wrapping_sub(t) <= u64::MAX - (m - 1) {
                return Self(t);
            }
        }
    }
}

impl<const MODULUS: u64> CanonicalField for Fp64<MODULUS> {
    fn from_u64(val: u64) -> Self {
        Self(Self::reduce_u128(val as u128))
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            -Self::from_u64((-val) as u64)
        }
    }

    fn to_canonical_u128(self) -> u128 {
        self.0 as u128
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < MODULUS as u128 {
            Some(Self(val as u64))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        Self((val % (MODULUS as u128)) as u64)
    }
}

#[inline]
fn nonzero_mask_u64(x: u64) -> u64 {
    let nz = ((x | x.wrapping_neg()) >> 63) & 1;
    0u64.wrapping_sub(nz)
}
