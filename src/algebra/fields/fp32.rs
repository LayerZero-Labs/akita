//! A prime field implementation backed by `u32` storage.
//!
//! - Modulus is a `u32` const generic.
//! - Multiplication uses `u64` widening arithmetic.

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::Field;
use rand_core::RngCore;
use std::io::{Read, Write};

/// Prime field element modulo `MODULUS` stored as `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp32<const MODULUS: u32>(pub(crate) u32);

impl<const MODULUS: u32> Fp32<MODULUS> {
    /// Create an element from a canonical representative in `[0, MODULUS)`.
    #[inline]
    pub fn from_canonical_u32(x: u32) -> Self {
        debug_assert!(x < MODULUS);
        Self(x)
    }

    /// Return the canonical representative in `[0, MODULUS)`.
    #[inline]
    pub fn to_canonical_u32(self) -> u32 {
        self.0
    }

    #[inline]
    fn reduce_u64(x: u64) -> u32 {
        let m = MODULUS as u64;
        let mut rem = 0u64;

        // Division-free fixed-iteration reduction.
        for i in (0..64).rev() {
            rem = (rem << 1) | ((x >> i) & 1);
            let (candidate, borrow) = rem.overflowing_sub(m);
            let mask = 0u64.wrapping_sub((!borrow) as u64);
            rem = (candidate & mask) | (rem & !mask);
        }

        rem as u32
    }

    #[inline]
    fn add_raw(a: u32, b: u32) -> u32 {
        let s = (a as u64) + (b as u64);
        let m = MODULUS as u64;
        let reduced = s.wrapping_sub(m);
        // If s < m, the subtraction underflowed (bit 63 set). Add m back.
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & m) as u32
    }

    #[inline]
    fn sub_raw(a: u32, b: u32) -> u32 {
        let diff = (a as u64).wrapping_sub(b as u64);
        // If a < b, the subtraction underflowed (bit 63 set). Add MODULUS.
        let borrow = diff >> 63;
        diff.wrapping_add(borrow.wrapping_neg() & (MODULUS as u64)) as u32
    }

    #[inline]
    fn mul_raw(a: u32, b: u32) -> u32 {
        Self::reduce_u64((a as u64) * (b as u64))
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

impl<const MODULUS: u32> std::ops::Add for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const MODULUS: u32> std::ops::Sub for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const MODULUS: u32> std::ops::Mul for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const MODULUS: u32> std::ops::Neg for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<'a, const MODULUS: u32> std::ops::Add<&'a Self> for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, const MODULUS: u32> std::ops::Sub<&'a Self> for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, const MODULUS: u32> std::ops::Mul<&'a Self> for Fp32<MODULUS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const MODULUS: u32> Valid for Fp32<MODULUS> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.0 < MODULUS {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp32 out of range".into()))
        }
    }
}

impl<const MODULUS: u32> HachiSerialize for Fp32<MODULUS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        4
    }
}

impl<const MODULUS: u32> HachiDeserialize for Fp32<MODULUS> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let x = u32::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        if matches!(validate, Validate::Yes) && x >= MODULUS {
            return Err(SerializationError::InvalidData(
                "Fp32 out of range".to_string(),
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

impl<const MODULUS: u32> Field for Fp32<MODULUS> {
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
        if self.is_zero() {
            None
        } else {
            // Fermat inversion: a^(p-2) mod p (MODULUS should be prime).
            Some(self.pow((MODULUS as u64).wrapping_sub(2)))
        }
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        // Rejection sampling to eliminate modular bias.
        loop {
            let x = rng.next_u32();
            let m = MODULUS;
            let t = x % m;
            if x.wrapping_sub(t) <= u32::MAX - (m - 1) {
                return Self(t);
            }
        }
    }

    fn from_u64(val: u64) -> Self {
        Self(Self::reduce_u64(val))
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            -Self::from_u64((-val) as u64)
        }
    }
}
