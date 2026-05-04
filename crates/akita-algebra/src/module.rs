//! Simple module implementations.

use super::fields::{Fp128, Fp32, Fp64};
use crate::{CanonicalField, FieldCore, FieldSampling, Module};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::ops::{Add, Mul, Neg, Sub};

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
    F: FieldCore
        + CanonicalField
        + FieldSampling
        + Valid
        + Mul<VectorModule<F, N>, Output = VectorModule<F, N>>
        + for<'a> Mul<&'a VectorModule<F, N>, Output = VectorModule<F, N>>,
{
    type Scalar = F;

    fn zero() -> Self {
        Self::zero_vec()
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        // Delegate to Scalar * VectorModule to satisfy the Module trait’s scalar bounds.
        *k * *self
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(std::array::from_fn(|_| F::sample(rng)))
    }
}

// Scalar * VectorModule impls for our local prime field types.

impl<const P: u32, const N: usize> Mul<VectorModule<Fp32<P>, N>> for Fp32<P> {
    type Output = VectorModule<Fp32<P>, N>;
    fn mul(self, rhs: VectorModule<Fp32<P>, N>) -> Self::Output {
        let mut out = rhs.0;
        for coeff in &mut out {
            *coeff = self * *coeff;
        }
        VectorModule(out)
    }
}

impl<'a, const P: u32, const N: usize> Mul<&'a VectorModule<Fp32<P>, N>> for Fp32<P> {
    type Output = VectorModule<Fp32<P>, N>;
    fn mul(self, rhs: &'a VectorModule<Fp32<P>, N>) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u64, const N: usize> Mul<VectorModule<Fp64<P>, N>> for Fp64<P> {
    type Output = VectorModule<Fp64<P>, N>;
    fn mul(self, rhs: VectorModule<Fp64<P>, N>) -> Self::Output {
        let mut out = rhs.0;
        for coeff in &mut out {
            *coeff = self * *coeff;
        }
        VectorModule(out)
    }
}

impl<'a, const P: u64, const N: usize> Mul<&'a VectorModule<Fp64<P>, N>> for Fp64<P> {
    type Output = VectorModule<Fp64<P>, N>;
    fn mul(self, rhs: &'a VectorModule<Fp64<P>, N>) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u128, const N: usize> Mul<VectorModule<Fp128<P>, N>> for Fp128<P> {
    type Output = VectorModule<Fp128<P>, N>;
    fn mul(self, rhs: VectorModule<Fp128<P>, N>) -> Self::Output {
        let mut out = rhs.0;
        for coeff in &mut out {
            *coeff = self * *coeff;
        }
        VectorModule(out)
    }
}

impl<'a, const P: u128, const N: usize> Mul<&'a VectorModule<Fp128<P>, N>> for Fp128<P> {
    type Output = VectorModule<Fp128<P>, N>;
    fn mul(self, rhs: &'a VectorModule<Fp128<P>, N>) -> Self::Output {
        self * *rhs
    }
}
