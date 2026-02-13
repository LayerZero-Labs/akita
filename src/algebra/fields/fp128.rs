//! A prime field implementation backed by `u128` storage.
//!
//! - Modulus is a `u128` const generic.
//! - Multiplication uses a 256-bit intermediate (see [`crate::algebra::u256`]).
//!
//! This is correctness-first; we can swap in Montgomery/Barrett later.

use super::u256::U256;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore, FieldSampling};
use rand_core::RngCore;
use std::io::{Read, Write};

/// Prime field element modulo `MODULUS` stored as `u128`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp128<const MODULUS: u128>(pub(crate) u128);

impl<const MODULUS: u128> Fp128<MODULUS> {
    /// Create an element from a canonical representative in `[0, MODULUS)`.
    #[inline]
    pub fn from_canonical_u128(x: u128) -> Self {
        debug_assert!(x < MODULUS);
        Self(x)
    }

    /// Return the canonical representative in `[0, MODULUS)`.
    #[inline]
    pub fn to_canonical_u128(self) -> u128 {
        self.0
    }

    #[inline]
    fn add_raw(a: u128, b: u128) -> u128 {
        let (s, carry) = a.overflowing_add(b);
        let (reduced, borrow) = s.overflowing_sub(MODULUS);
        // Need correction (add MODULUS back) only when !carry && borrow,
        // i.e. when the true sum is less than MODULUS.
        let need_correction = (!carry & borrow) as u128;
        reduced.wrapping_add(need_correction.wrapping_neg() & MODULUS)
    }

    #[inline]
    fn sub_raw(a: u128, b: u128) -> u128 {
        let (diff, borrow) = a.overflowing_sub(b);
        // If a < b, borrow is set. Add MODULUS to correct.
        let correction = (borrow as u128).wrapping_neg() & MODULUS;
        diff.wrapping_add(correction)
    }

    #[inline]
    fn reduce_u256(n: U256) -> u128 {
        // Binary long division remainder, maintaining remainder in 129 bits (hi:0/1, lo:u128).
        // Invariant: before each step, remainder < MODULUS.
        let m = MODULUS;
        let mut hi: u8 = 0;
        let mut lo: u128 = 0;

        for i in (0..256).rev() {
            // rem = rem*2 + bit
            let new_hi = (lo >> 127) as u8;
            lo <<= 1;
            lo |= n.bit(i) as u128;
            hi = new_hi;

            // Since rem < 2m after update, subtract at most once.
            if hi == 1 || lo >= m {
                lo = lo.wrapping_sub(m);
                hi = 0;
            }
        }

        debug_assert_eq!(hi, 0);
        lo
    }

    #[inline]
    fn mul_raw(a: u128, b: u128) -> u128 {
        Self::reduce_u256(U256::mul_u128(a, b))
    }

    fn pow_u128(self, mut exp: u128) -> Self {
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

impl<const MODULUS: u128> std::ops::Add for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const MODULUS: u128> std::ops::Sub for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const MODULUS: u128> std::ops::Mul for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const MODULUS: u128> std::ops::Neg for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<'a, const MODULUS: u128> std::ops::Add<&'a Self> for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const MODULUS: u128> std::ops::Sub<&'a Self> for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const MODULUS: u128> std::ops::Mul<&'a Self> for Fp128<MODULUS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const MODULUS: u128> Valid for Fp128<MODULUS> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.0 < MODULUS {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp128 out of range".into()))
        }
    }
}

impl<const MODULUS: u128> HachiSerialize for Fp128<MODULUS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        16
    }
}

impl<const MODULUS: u128> HachiDeserialize for Fp128<MODULUS> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let x = u128::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        if matches!(validate, Validate::Yes) && x >= MODULUS {
            return Err(SerializationError::InvalidData(
                "Fp128 out of range".to_string(),
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

impl<const MODULUS: u128> FieldCore for Fp128<MODULUS> {
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
            Some(self.pow_u128(MODULUS.wrapping_sub(2)))
        }
    }
}

impl<const MODULUS: u128> FieldSampling for Fp128<MODULUS> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        // Rejection-sampling for reduced bias.
        loop {
            let lo = rng.next_u64() as u128;
            let hi = rng.next_u64() as u128;
            let x = lo | (hi << 64);
            let m = MODULUS;
            let t = x % m;
            if x.wrapping_sub(t) <= u128::MAX - (m - 1) {
                return Self(t);
            }
        }
    }
}

impl<const MODULUS: u128> CanonicalField for Fp128<MODULUS> {
    fn from_u64(val: u64) -> Self {
        Self((val as u128) % MODULUS)
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            -Self::from_u64((-val) as u64)
        }
    }

    fn to_canonical_u128(self) -> u128 {
        self.0
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < MODULUS {
            Some(Self(val))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        Self(val % MODULUS)
    }
}
