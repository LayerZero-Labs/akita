#![allow(missing_docs)]

use super::{HachiDeserialize, HachiSerialize};
use rand_core::RngCore;

/// Minimal additive group: add, sub, neg, zero.
///
/// Satisfied by both reduced field elements (`FieldCore`) and wide unreduced
/// accumulators (`Fp128x8i32`, etc.), enabling generic shift-accumulate
/// operations on `WideCyclotomicRing<W, D>`.
pub trait AdditiveGroup:
    Sized
    + Clone
    + Copy
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
{
    /// Additive identity.
    const ZERO: Self;
}

/// Core field operations required across algebra backends.
pub trait FieldCore:
    AdditiveGroup
    + PartialEq
    + HachiSerialize
    + HachiDeserialize<Context = ()>
    + std::ops::Mul<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
{
    /// Additive identity.
    fn zero() -> Self {
        Self::ZERO
    }

    /// Multiplicative identity
    fn one() -> Self;

    /// Check if element is zero
    fn is_zero(&self) -> bool;

    /// Field squaring.
    ///
    /// Default is `self * self`; extension fields override with specialized
    /// formulas that use fewer base-field multiplications.
    fn square(&self) -> Self {
        *self * *self
    }

    /// Field inversion.
    ///
    /// This API may branch on zero-check and is intended for public/non-secret
    /// values. For secret-bearing paths, use [`Invertible::inv_or_zero`].
    fn inv(self) -> Option<Self>;

    /// Multiplicative inverse of 2: `(p + 1) / 2` for odd-characteristic fields.
    const TWO_INV: Self;
}

/// Constant-time inversion helper for secret-bearing code paths.
///
/// Implementations return `0` when the input is `0`, and `x^{-1}` otherwise,
/// without branching on the input value.
pub trait Invertible: FieldCore {
    /// Constant-time inversion with zero-mapping behavior.
    fn inv_or_zero(self) -> Self;
}

/// Embed small integers into a field.
///
/// Every field contains a copy of its prime subfield, and small integers embed
/// into it canonically via reduction modulo the characteristic. This trait is
/// implementable for ALL fields — base and extension alike.
///
/// Only `from_u64` and `from_i64` need concrete implementations; the narrower
/// widths have default impls via lossless widening.
pub trait FromSmallInt: FieldCore {
    /// Embed a `u8` into the field.
    fn from_u8(val: u8) -> Self {
        Self::from_u64(val as u64)
    }

    /// Embed an `i8` into the field.
    fn from_i8(val: i8) -> Self {
        Self::from_i64(val as i64)
    }

    /// Embed a `u16` into the field.
    fn from_u16(val: u16) -> Self {
        Self::from_u64(val as u64)
    }

    /// Embed an `i16` into the field.
    fn from_i16(val: i16) -> Self {
        Self::from_i64(val as i64)
    }

    /// Embed a `u32` into the field.
    fn from_u32(val: u32) -> Self {
        Self::from_u64(val as u64)
    }

    /// Embed an `i32` into the field.
    fn from_i32(val: i32) -> Self {
        Self::from_i64(val as i64)
    }

    /// Embed a `u64` into the field (reduce mod characteristic).
    fn from_u64(val: u64) -> Self;

    /// Embed an `i64` into the field (reduce mod characteristic).
    fn from_i64(val: i64) -> Self;

    /// Embed an `i128` into the field.
    ///
    /// Default implementation splits into u64 limbs with field multiplication
    /// by `2^64`. Override for base fields that have a direct path.
    fn from_i128(val: i128) -> Self {
        if val >= 0 {
            let lo = val as u64;
            let hi = (val >> 64) as u64;
            if hi == 0 {
                Self::from_u64(lo)
            } else {
                let two_64 = Self::from_u64(1u64 << 32) * Self::from_u64(1u64 << 32);
                Self::from_u64(lo) + Self::from_u64(hi) * two_64
            }
        } else {
            -Self::from_i128(-val)
        }
    }

    /// Lookup table mapping balanced digit index → field element.
    ///
    /// For `log_basis` in `1..=5`, returns a 32-entry table where
    /// `table[i]` = `from_i64(i - b/2)` for `i < b = 2^log_basis`,
    /// and zero for `i >= b`.
    ///
    /// Index a digit `d ∈ [-b/2, b/2)` as `table[(d + b/2) as usize]`.
    fn digit_lut(log_basis: u32) -> [Self; 32] {
        debug_assert!(log_basis > 0 && log_basis <= 5);
        let b = 1usize << log_basis;
        let half_b = (b >> 1) as i64;
        std::array::from_fn(|i| {
            if i < b {
                Self::from_i64(i as i64 - half_b)
            } else {
                Self::zero()
            }
        })
    }
}

/// Canonical integer representation for prime (base) field elements.
///
/// Provides a bijection between field elements and integers in `[0, p)`.
/// Only meaningful for base prime fields where elements ARE residues mod p.
/// Extension fields should NOT implement this trait.
pub trait CanonicalField: FromSmallInt {
    /// Return canonical integer representation as `u128`.
    fn to_canonical_u128(self) -> u128;

    /// Construct from canonical value if it is in range.
    fn from_canonical_u128_checked(val: u128) -> Option<Self>;

    /// Construct from canonical value reduced modulo the field modulus.
    fn from_canonical_u128_reduced(val: u128) -> Self;
}

/// Optional sampling support for field elements.
///
/// This is intentionally separate from core field algebra and may evolve.
pub trait FieldSampling: FieldCore {
    /// Generate a sampled field element.
    fn sample<R: RngCore>(rng: &mut R) -> Self;
}

/// Metadata for pseudo-Mersenne style moduli (`2^k - c`).
pub trait PseudoMersenneField: CanonicalField {
    /// Exponent `k` in `2^k - c`.
    const MODULUS_BITS: u32;

    /// Offset `c` in `2^k - c`.
    const MODULUS_OFFSET: u128;
}

/// Module trait for lattice-based algebraic structures
///
/// This trait represents a module over a ring/field, which is fundamental
/// to lattice-based cryptography. Unlike elliptic curve groups, lattice
/// schemes work with module structures.
pub trait Module:
    Sized
    + Clone
    + Copy
    + PartialEq
    + Send
    + Sync
    + HachiSerialize
    + HachiDeserialize
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
{
    /// Scalar type (field/ring elements)
    type Scalar: FieldCore
        + CanonicalField
        + FieldSampling
        + std::ops::Mul<Self, Output = Self>
        + for<'a> std::ops::Mul<&'a Self, Output = Self>;

    /// Zero element
    fn zero() -> Self;

    /// Scalar multiplication
    fn scale(&self, k: &Self::Scalar) -> Self;

    /// Generate random module element
    fn random<R: RngCore>(rng: &mut R) -> Self;
}
