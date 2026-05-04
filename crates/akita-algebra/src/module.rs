//! Simple module implementations.

use crate::{CanonicalField, FieldCore, FieldSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::ops::{Add, Neg, Sub};

/// Module trait for lattice-based algebraic structures.
///
/// This trait represents a module over a scalar field or ring. It lives in
/// `akita-algebra` because modules are algebraic containers; concrete scalar
/// fields live in `akita-field`.
pub trait Module:
    Sized
    + Clone
    + Copy
    + PartialEq
    + Send
    + Sync
    + AkitaSerialize
    + AkitaDeserialize
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Neg<Output = Self>
    + for<'a> std::ops::Add<&'a Self, Output = Self>
    + for<'a> std::ops::Sub<&'a Self, Output = Self>
{
    /// Scalar type.
    type Scalar: FieldCore + CanonicalField + FieldSampling;

    /// Zero element.
    fn zero() -> Self;

    /// Scalar multiplication.
    fn scale(&self, k: &Self::Scalar) -> Self;

    /// Generate random module element.
    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// Fixed-length vector module over a scalar type `F`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorModule<F: FieldCore, const N: usize>(pub [F; N]);

impl<F: FieldCore, const N: usize> VectorModule<F, N> {
    /// Construct the zero vector.
    #[inline]
    pub fn zero_vec() -> Self {
        Self([F::zero(); N])
    }
}

impl<F: FieldCore, const N: usize> Add for VectorModule<F, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst += *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const N: usize> Sub for VectorModule<F, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = self.0;
        for (dst, src) in out.iter_mut().zip(rhs.0.iter()) {
            *dst -= *src;
        }
        Self(out)
    }
}

impl<F: FieldCore, const N: usize> Neg for VectorModule<F, N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = self.0;
        for coeff in &mut out {
            *coeff = -*coeff;
        }
        Self(out)
    }
}

impl<'a, F: FieldCore, const N: usize> Add<&'a Self> for VectorModule<F, N> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, F: FieldCore, const N: usize> Sub<&'a Self> for VectorModule<F, N> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<F: FieldCore + Valid, const N: usize> Valid for VectorModule<F, N> {
    fn check(&self) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore, const N: usize> AkitaSerialize for VectorModule<F, N> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for x in self.0.iter() {
            x.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.iter().map(|x| x.serialized_size(compress)).sum()
    }
}

impl<F: FieldCore + Valid, const N: usize> AkitaDeserialize for VectorModule<F, N> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut arr = [F::zero(); N];
        for coeff in &mut arr {
            *coeff = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        }
        let out = Self(arr);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F, const N: usize> Module for VectorModule<F, N>
where
    F: FieldCore + CanonicalField + FieldSampling + Valid,
{
    type Scalar = F;

    fn zero() -> Self {
        Self::zero_vec()
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        let mut out = self.0;
        for coeff in &mut out {
            *coeff = *k * *coeff;
        }
        Self(out)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(std::array::from_fn(|_| F::sample(rng)))
    }
}
