//! Prime field for primes of the form `p = 2^k − c` with `c` small, backed
//! by `u16` storage.
//!
//! Uses Solinas-style two-fold reduction: the offset `c` and fold point `k`
//! are computed at compile time from the const-generic modulus `P`.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use jolt_field::{FromPrimitiveInt, Invertible, RandomSampling};
use rand_core::RngCore;

use crate::{BalancedDigitLookup, CanonicalField, HalvingField, PseudoMersenneField};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Prime field element for primes `p = 2^k − c` stored as `u16`.
///
/// The fold point `k` and offset `c = 2^k − p` are computed at compile time
/// from the const-generic `P`.  Instantiating with a modulus that does not
/// satisfy the Solinas conditions is a compile-time error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp16<const P: u32>(pub(crate) u16);

impl<const P: u32> Fp16<P> {
    /// Fold point: smallest `k` such that `P ≤ 2^k`.
    const BITS: u32 = 32 - P.leading_zeros();

    /// Offset `c = 2^k − P`.
    pub const C: u32 = {
        let c = if Self::BITS == 32 {
            0u32.wrapping_sub(P)
        } else {
            (1u32 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(Self::BITS <= 16, "modulus must fit in u16 storage");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            (c as u64) * (c as u64 + 1) < P as u64,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    /// Mask for extracting the low `BITS` bits from a widened product.
    const MASK: u32 = (1u32 << Self::BITS) - 1;

    /// Create from a canonical representative in `[0, P)`.
    #[inline]
    pub fn from_canonical_u16(x: u16) -> Self {
        debug_assert!((x as u32) < P);
        Self(x)
    }

    /// Create from a canonical representative in `[0, P)`.
    #[inline]
    pub fn from_canonical_u32(x: u32) -> Self {
        debug_assert!(x < P);
        Self(x as u16)
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self(0)
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self(if P > 1 { 1 } else { 0 } as u16)
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Multiplicative inverse, or `None` for zero.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        <Self as Invertible>::inverse(self)
    }

    /// Construct from a `u64` reduced modulo the field modulus.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self(Self::reduce_u64(val))
    }

    /// Construct from an `i64` reduced modulo the field modulus.
    #[inline]
    pub fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Self::from_u64(val as u64)
        } else {
            -Self::from_u64(val.unsigned_abs())
        }
    }

    /// Construct from an `i8` reduced modulo the field modulus.
    #[inline]
    pub fn from_i8(val: i8) -> Self {
        Self::from_i64(val as i64)
    }

    /// Return the canonical representative in `[0, P)`.
    #[inline]
    pub fn to_canonical_u16(self) -> u16 {
        self.0
    }

    /// Return the canonical representative in `[0, P)`.
    #[inline]
    pub fn to_canonical_u32(self) -> u32 {
        self.0 as u32
    }

    /// Solinas reduction: fold a u64 at bit `BITS` until the value fits,
    /// then conditionally subtract `P`.
    ///
    /// For multiplication products (< 2^{2·BITS}) exactly 2 folds suffice;
    /// for arbitrary u64 inputs (e.g. `from_u64`) the loop runs at most
    /// `ceil(64 / BITS)` iterations.
    #[inline(always)]
    fn reduce_u64(x: u64) -> u16 {
        let c = Self::C as u64;
        let mut v = x;
        while v >> Self::BITS != 0 {
            v = (v & Self::MASK as u64) + c * (v >> Self::BITS);
        }
        let reduced = v.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u16
    }

    /// Reduce a `u128` to canonical form (for `from_canonical_u128_reduced`).
    #[inline(always)]
    fn reduce_u128(x: u128) -> u16 {
        let c = Self::C as u128;
        let bits = Self::BITS;
        let mask = (1u128 << bits) - 1;
        let mut v = x;
        while v >> bits != 0 {
            v = (v & mask) + c * (v >> bits);
        }
        let f = v as u64;
        let reduced = f.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u16
    }

    /// Two-fold Solinas reduction for multiplication products.
    ///
    /// Input must be < 2^{2·BITS} (guaranteed for `a*b` where `a,b < P`).
    /// Exactly 2 folds + conditional subtract, no loop.
    #[inline(always)]
    fn reduce_product(x: u32) -> u16 {
        let c = Self::C;
        let f1 = (x & Self::MASK) + c * (x >> Self::BITS);
        let f2 = (f1 & Self::MASK) + c * (f1 >> Self::BITS);
        let reduced = f2.wrapping_sub(P);
        let borrow = reduced >> 31;
        reduced.wrapping_add(borrow.wrapping_neg() & P) as u16
    }

    #[inline(always)]
    fn add_raw(a: u16, b: u16) -> u16 {
        let s = (a as u32) + (b as u32);
        let reduced = s.wrapping_sub(P);
        let borrow = reduced >> 31;
        reduced.wrapping_add(borrow.wrapping_neg() & P) as u16
    }

    #[inline(always)]
    fn sub_raw(a: u16, b: u16) -> u16 {
        let diff = (a as u32).wrapping_sub(b as u32);
        let borrow = diff >> 31;
        diff.wrapping_add(borrow.wrapping_neg() & P) as u16
    }

    #[inline(always)]
    fn mul_raw(a: u16, b: u16) -> u16 {
        Self::reduce_product((a as u32) * (b as u32))
    }

    #[inline(always)]
    fn sqr_raw(a: u16) -> u16 {
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
                acc *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        acc
    }

    /// Extract the canonical value.
    #[inline(always)]
    pub fn to_limbs(self) -> u16 {
        self.0
    }

    /// 16×16 → 32-bit widening multiply, **no reduction**.
    #[inline(always)]
    pub fn mul_wide(self, other: Self) -> u32 {
        (self.0 as u32) * (other.0 as u32)
    }

    /// 16×16 → 32-bit widening multiply with a raw `u16` operand,
    /// **no reduction**.
    #[inline(always)]
    pub fn mul_wide_u16(self, other: u16) -> u32 {
        (self.0 as u32) * (other as u32)
    }

    /// Reduce a u32 value via Solinas folding to a canonical field element.
    #[inline(always)]
    pub fn solinas_reduce(x: u32) -> Self {
        Self(Self::reduce_u64(x as u64))
    }
}

impl<const P: u32> Add for Fp16<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Sub for Fp16<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Mul for Fp16<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u32> Neg for Fp16<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(0, self.0))
    }
}

impl<const P: u32> AddAssign for Fp16<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for Fp16<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for Fp16<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, const P: u32> Add<&'a Self> for Fp16<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u32> Sub<&'a Self> for Fp16<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u32> Mul<&'a Self> for Fp16<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u32> Valid for Fp16<P> {
    fn check(&self) -> Result<(), SerializationError> {
        if (self.0 as u32) < P {
            Ok(())
        } else {
            Err(SerializationError::InvalidData("Fp16 out of range".into()))
        }
    }
}

impl<const P: u32> AkitaSerialize for Fp16<P> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(&mut writer, Compress::No)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        2
    }
}

impl<const P: u32> AkitaDeserialize for Fp16<P> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let x = u16::deserialize_with_mode(&mut reader, Compress::No, validate, &())?;
        if matches!(validate, Validate::Yes) && (x as u32) >= P {
            return Err(SerializationError::InvalidData(
                "Fp16 out of range".to_string(),
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

impl<const P: u32> Invertible for Fp16<P> {
    #[inline(always)]
    fn inverse(&self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if self.is_zero() {
            None
        } else {
            Some(inv)
        }
    }

    #[inline(always)]
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow((P as u64).wrapping_sub(2));
        let nz = ((self.0 | self.0.wrapping_neg()) >> 15) & 1;
        let mask = 0u16.wrapping_sub(nz);
        Self(candidate.0 & mask)
    }
}

impl<const P: u32> HalvingField for Fp16<P> {
    #[inline]
    fn half(self) -> Self {
        let x = self.0 as u32;
        Self(((x + (x & 1) * P) >> 1) as u16)
    }
}

impl<const P: u32> RandomSampling for Fp16<P> {
    #[inline(always)]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(Self::reduce_u64(rng.next_u64()))
    }
}

impl<const P: u32> FromPrimitiveInt for Fp16<P> {
    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    #[inline(always)]
    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    #[inline(always)]
    fn from_u128(val: u128) -> Self {
        Self(Self::reduce_u128(val))
    }

    #[inline(always)]
    fn from_i128(val: i128) -> Self {
        if val >= 0 {
            Self::from_u128(val as u128)
        } else {
            -Self::from_u128(val.unsigned_abs())
        }
    }
}

impl<const P: u32> BalancedDigitLookup for Fp16<P> {}

impl<const P: u32> CanonicalField for Fp16<P> {
    fn to_canonical_u128(self) -> u128 {
        self.0 as u128
    }

    fn modulus_bits() -> u32 {
        Self::BITS
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < P as u128 {
            Some(Self(val as u16))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        Self(Self::reduce_u128(val))
    }
}

impl<const P: u32> PseudoMersenneField for Fp16<P> {
    const MODULUS_BITS: u32 = Self::BITS;
    const MODULUS_OFFSET: u128 = Self::C as u128;
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp16<251>; // 2^8 - 5

    #[test]
    fn solinas_constants() {
        assert_eq!(F::BITS, 8);
        assert_eq!(F::C, 5);
        assert_eq!(F::MASK, 255);

        type G = Fp16<65_437>; // 2^16 - 99
        assert_eq!(G::BITS, 16);
        assert_eq!(G::C, 99);
        assert_eq!(G::MASK, 65_535);
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
            let a: F = RandomSampling::random(&mut rng);
            let b: F = RandomSampling::random(&mut rng);
            let expected = a * b;
            let reduced = F::solinas_reduce(a.mul_wide(b));
            assert_eq!(reduced, expected);
        }
    }

    #[test]
    fn mul_wide_u16_matches() {
        let mut rng = StdRng::seed_from_u64(0xabcd_ef01);
        for _ in 0..1000 {
            let a: F = RandomSampling::random(&mut rng);
            let b = (rng.next_u32() % 251) as u16;
            let expected = a * F::from_canonical_u16(b);
            let reduced = F::solinas_reduce(a.mul_wide_u16(b));
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
    fn cross_prime_16bit() {
        type G = Fp16<65_437>; // 2^16 - 99
        assert_eq!(G::BITS, 16);
        assert_eq!(G::C, 99);

        let a = G::from_u64(65_000);
        let b = G::from_u64(64_000);
        let product = (65_000u64 * 64_000u64) % 65_437;
        assert_eq!((a * b).to_canonical_u32(), product as u32);
    }

    #[test]
    fn inverse_and_random_sampling() {
        type G = Fp16<65_437>;
        let a = G::from_u64(12_345);
        let inv = a.inverse().expect("nonzero inverse");
        assert_eq!((a * inv).to_canonical_u32(), 1);
        assert_eq!(G::zero().inverse(), None);

        let mut rng = StdRng::seed_from_u64(0xfeed_beef);
        for _ in 0..1000 {
            let x: G = RandomSampling::random(&mut rng);
            assert!(x.to_canonical_u32() < 65_437);
        }
    }

    #[test]
    fn canonical_serialization_is_two_little_endian_bytes() {
        type G = Fp16<65_437>;
        let zero = G::from_canonical_u16(0);
        let max = G::from_canonical_u16(65_436);

        let mut bytes = Vec::new();
        zero.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        assert_eq!(bytes, [0, 0]);
        assert_eq!(zero.serialized_size(Compress::No), 2);

        bytes.clear();
        max.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        assert_eq!(bytes, [0x9c, 0xff]);
        assert_eq!(max.serialized_size(Compress::No), 2);
    }

    #[test]
    fn validated_deserialization_rejects_noncanonical_u16_values() {
        type G = Fp16<65_437>;
        for x in 65_437u16..=u16::MAX {
            let bytes = x.to_le_bytes();
            let err = G::deserialize_with_mode(&bytes[..], Compress::No, Validate::Yes, &())
                .expect_err("noncanonical fp16 encoding must be rejected");
            assert!(matches!(err, SerializationError::InvalidData(_)));
        }
    }
}
