#![allow(missing_docs)]

use super::{HachiDeserialize, HachiSerialize};
use rand_core::RngCore;

/// Core field operations required across algebra backends.
pub trait FieldCore:
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
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
    + for<'a> std::ops::Mul<&'a Self, Output = Self>
{
    /// Additive identity
    fn zero() -> Self;

    /// Multiplicative identity
    fn one() -> Self;

    /// Check if element is zero
    fn is_zero(&self) -> bool;

    /// Field addition
    fn add(&self, rhs: &Self) -> Self;

    /// Field subtraction
    fn sub(&self, rhs: &Self) -> Self;

    /// Field multiplication
    fn mul(&self, rhs: &Self) -> Self;

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

    /// Addition
    fn add(&self, rhs: &Self) -> Self;

    /// Negation
    fn neg(&self) -> Self;

    /// Scalar multiplication
    fn scale(&self, k: &Self::Scalar) -> Self;

    /// Generate random module element
    fn random<R: RngCore>(rng: &mut R) -> Self;
}

pub trait HachiRoutines<M: Module> {}
