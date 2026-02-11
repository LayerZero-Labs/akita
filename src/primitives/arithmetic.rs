#![allow(missing_docs)]

use super::{HachiDeserialize, HachiSerialize};
use rand_core::RngCore;

/// Field trait for lattice-based arithmetic
pub trait Field:
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

    /// Field inversion
    fn inv(self) -> Option<Self>;

    /// Generate random field element
    fn random<R: RngCore>(rng: &mut R) -> Self;

    /// Convert from u64
    fn from_u64(val: u64) -> Self;

    /// Convert from i64
    fn from_i64(val: i64) -> Self;
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
    type Scalar: Field
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
