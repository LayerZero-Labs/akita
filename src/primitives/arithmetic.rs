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

    /// Field inversion.
    ///
    /// This API may branch on zero-check and is intended for public/non-secret
    /// values. For secret-bearing paths, use [`CtInvertible::inv_or_zero_ct`].
    fn inv(self) -> Option<Self>;
}

/// Constant-time inversion helper for secret-bearing code paths.
///
/// Implementations return `0` when the input is `0`, and `x^{-1}` otherwise,
/// without branching on the input value.
pub trait CtInvertible: FieldCore {
    /// Constant-time inversion with zero-mapping behavior.
    fn inv_or_zero_ct(self) -> Self;
}

/// Canonical conversion operations for field elements.
pub trait CanonicalField: FieldCore {
    /// Convert from `u64`.
    fn from_u64(val: u64) -> Self;

    /// Convert from `i64`.
    fn from_i64(val: i64) -> Self;

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
