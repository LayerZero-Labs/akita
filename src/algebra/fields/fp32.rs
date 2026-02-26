//! Prime field for primes of the form `p = 2^k − c` with `c` small, backed
//! by `u32` storage.
//!
//! Uses Solinas-style two-fold reduction: the offset `c` and fold point `k`
//! are computed at compile time from the const-generic modulus `P`.

use std::ops::{Add, Mul, Neg, Sub};

use rand_core::RngCore;

use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{CanonicalField, FieldCore, FieldSampling, Invertible, PseudoMersenneField};
use std::io::{Read, Write};

/// Prime field element for primes `p = 2^k − c` stored as `u32`.
///
/// The fold point `k` and offset `c = 2^k − p` are computed at compile time
/// from the const-generic `P`.  Instantiating with a modulus that does not
/// satisfy the Solinas conditions is a compile-time error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp32<const P: u32>(pub(crate) u32);

impl<const P: u32> Fp32<P> {
    /// Fold point: smallest `k` such that `P ≤ 2^k`.
    const BITS: u32 = 32 - P.leading_zeros();

    /// Offset `c = 2^k − P`.
    const C: u32 = {
        let c = if Self::BITS == 32 {
            0u32.wrapping_sub(P)
        } else {
            (1u32 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            (c as u64) * (c as u64 + 1) < P as u64,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    /// Mask for extracting the low `BITS` bits from a u64.
    const MASK: u64 = if Self::BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << Self::BITS) - 1
    };

    /// Create from a canonical representative in `[0, P)`.
    #[inline]
    pub fn from_canonical_u32(x: u32) -> Self {
        debug_assert!(x < P);
        Self(x)
    }

    /// Return the canonical representative in `[0, P)`.
    #[inline]
    pub fn to_canonical_u32(self) -> u32 {
        self.0
    }

    /// Solinas reduction: fold a u64 at bit `BITS` until the value fits,
    /// then conditionally subtract `P`.
    ///
    /// For multiplication products (< 2^{2·BITS}) exactly 2 folds suffice;
    /// for arbitrary u64 inputs (e.g. `from_u64`) the loop runs at most
    /// `ceil(64 / BITS)` iterations.
    #[inline(always)]
    fn reduce_u64(x: u64) -> u32 {
        let c = Self::C as u64;
        let mut v = x;
        while v >> Self::BITS != 0 {
            v = (v & Self::MASK) + c * (v >> Self::BITS);
        }
        let reduced = v.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    }

    /// Reduce a `u128` to canonical form (for `from_canonical_u128_reduced`).
    #[inline(always)]
    fn reduce_u128(x: u128) -> u32 {
        let c = Self::C as u128;
        let bits = Self::BITS;
        let mask = if bits == 32 {
            u32::MAX as u128
        } else {
            (1u128 << bits) - 1
        };
        let mut v = x;
        while v >> bits != 0 {
            v = (v & mask) + c * (v >> bits);
        }
        let f = v as u64;
        let reduced = f.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    }

    /// Two-fold Solinas reduction for multiplication products.
    ///
    /// Input must be < 2^{2·BITS} (guaranteed for `a*b` where `a,b < P`).
    /// Exactly 2 folds + conditional subtract, no loop.
    #[inline(always)]
    fn reduce_product(x: u64) -> u32 {
        let c = Self::C as u64;
        let f1 = (x & Self::MASK) + c * (x >> Self::BITS);
        let f2 = (f1 & Self::MASK) + c * (f1 >> Self::BITS);
        let reduced = f2.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    }

    #[inline(always)]
    fn add_raw(a: u32, b: u32) -> u32 {
        let s = (a as u64) + (b as u64);
        let reduced = s.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    }

    #[inline(always)]
    fn sub_raw(a: u32, b: u32) -> u32 {
        let diff = (a as u64).wrapping_sub(b as u64);
        let borrow = diff >> 63;
        diff.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    }

    #[inline(always)]
    fn mul_raw(a: u32, b: u32) -> u32 {
        Self::reduce_product((a as u64) * (b as u64))
    }

    #[inline(always)]
    fn sqr_raw(a: u32) -> u32 {
        Self::mul_raw(a, a)
    }

    /// Squaring, equivalent to `self * self`.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self(Self::sqr_raw(self.0))
    }

    fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self::one();
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc * base;
            }
            base = base.square();
            exp >>= 1;
        }
        acc
    }

    /// Extract the canonical value.
    #[inline(always)]
    pub fn to_limbs(self) -> u32 {
        self.0
    }

    /// 32×32 → 64-bit widening multiply, **no reduction**.
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> u64 {
        (self.0 as u64) * (other.0 as u64)
    }

    /// 32×32 → 64-bit widening multiply with a raw `u32` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u32(self, other: u32) -> u64 {
        (self.0 as u64) * (other as u64)
    }

    /// Reduce a u64 value via Solinas folding to a canonical field element.
    #[inline(always)]
    pub fn solinas_reduce(x: u64) -> Self {
        Self(Self::reduce_u64(x))
    }
}

impl<const P: u32> Add for Fp32<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Sub for Fp32<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Mul for Fp32<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Neg for Fp32<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<'a, const P: u32> Add<&'a Self> for Fp32<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u32> Sub<&'a Self> for Fp32<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u32> Mul<&'a Self> for Fp32<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u32> Valid for Fp32<P> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.0 < P {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp32 out of range".into()))
        }
    }
}

impl<const P: u32> HachiSerialize for Fp32<P> {
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

impl<const P: u32> HachiDeserialize for Fp32<P> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let x = u32::deserialize_with_mode(&mut reader, Compress::No, validate)?;
        if matches!(validate, Validate::Yes) && x >= P {
            return Err(SerializationError::InvalidData(
                "Fp32 out of range".to_string(),
            ));
        }
        let out = if matches!(validate, Validate::Yes) {
            Self(x)
        } else {
            Self(Self::reduce_u64(x as u64))
        };
        Ok(out)
    }
}

impl<const P: u32> FieldCore for Fp32<P> {
    fn zero() -> Self {
        Self(0)
    }

    fn one() -> Self {
        Self(if P > 1 { 1 } else { 0 })
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

impl<const P: u32> Invertible for Fp32<P> {
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow((P as u64).wrapping_sub(2));
        let nz = ((self.0 | self.0.wrapping_neg()) >> 31) & 1;
        let mask = 0u32.wrapping_sub(nz);
        Self(candidate.0 & mask)
    }
}

impl<const P: u32> FieldSampling for Fp32<P> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        Self(Self::reduce_u64(rng.next_u64()))
    }
}

impl<const P: u32> CanonicalField for Fp32<P> {
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

    fn to_canonical_u128(self) -> u128 {
        self.0 as u128
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < P as u128 {
            Some(Self(val as u32))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        Self(Self::reduce_u128(val))
    }
}

impl<const P: u32> PseudoMersenneField for Fp32<P> {
    const MODULUS_BITS: u32 = Self::BITS;
    const MODULUS_OFFSET: u128 = Self::C as u128;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp32<251>; // 2^8 - 5

    #[test]
    fn solinas_constants() {
        assert_eq!(F::BITS, 8);
        assert_eq!(F::C, 5);
        assert_eq!(F::MASK, 255);

        type G = Fp32<{ (1u32 << 24) - 3 }>; // 2^24 - 3
        assert_eq!(G::BITS, 24);
        assert_eq!(G::C, 3);
    }

    #[test]
    fn basic_arithmetic() {
        let a = F::from_u64(100);
        let b = F::from_u64(200);
        assert_eq!((a + b).to_canonical_u32(), (100 + 200) % 251);
        assert_eq!((a * b).to_canonical_u32(), (100 * 200) % 251);
        assert_eq!((b - a).to_canonical_u32(), 100);
        assert_eq!((-a).to_canonical_u32(), 251 - 100);
    }

    #[test]
    fn mul_wide_matches_full_mul() {
        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b: F = FieldSampling::sample(&mut rng);
            let expected = a * b;
            let reduced = F::solinas_reduce(a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_u32_matches() {
        let mut rng = StdRng::seed_from_u64(0xabcd_ef01);
        for _ in 0..1000 {
            let a: F = FieldSampling::sample(&mut rng);
            let b = (rng.next_u32() % 251) as u32;
            let expected = a * F::from_canonical_u32(b);
            let reduced = F::solinas_reduce(a.mul_wide_u32(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn reduce_large_values() {
        assert_eq!(
            F::from_u64(u64::MAX).to_canonical_u32(),
            (u64::MAX % 251) as u32
        );
        assert_eq!(F::from_u64(0).to_canonical_u32(), 0);
        assert_eq!(F::from_u64(251).to_canonical_u32(), 0);
        assert_eq!(F::from_u64(252).to_canonical_u32(), 1);
    }

    #[test]
    fn pseudo_mersenne_trait() {
        assert_eq!(<F as PseudoMersenneField>::MODULUS_BITS, 8);
        assert_eq!(<F as PseudoMersenneField>::MODULUS_OFFSET, 5);
    }

    #[test]
    fn cross_prime_32bit() {
        type G = Fp32<{ u32::MAX - 98 }>; // 2^32 - 99
        assert_eq!(G::BITS, 32);
        assert_eq!(G::C, 99);

        let a = G::from_u64(1_000_000);
        let b = G::from_u64(2_000_000);
        let product = (1_000_000u64 * 2_000_000u64) % ((1u64 << 32) - 99);
        assert_eq!((a * b).to_canonical_u32(), product as u32);
    }
}
