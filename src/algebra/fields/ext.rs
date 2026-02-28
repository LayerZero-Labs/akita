//! Quadratic and quartic extension fields.

use crate::algebra::module::VectorModule;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::{FieldCore, FieldSampling};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

/// Parameters for an `Fp2` quadratic extension over base field `F`.
pub trait Fp2Config<F: FieldCore> {
    /// Non-residue `NR` such that `u^2 = NR`.
    fn non_residue() -> F;
}

/// Quadratic extension element `c0 + c1 * u` with `u^2 = NR`.
pub struct Fp2<F: FieldCore, C: Fp2Config<F>> {
    /// Constant term.
    pub c0: F,
    /// Coefficient of `u`.
    pub c1: F,
    _cfg: PhantomData<fn() -> C>,
}

impl<F: FieldCore, C: Fp2Config<F>> Fp2<F, C> {
    /// Construct `c0 + c1 * u`.
    #[inline]
    pub fn new(c0: F, c1: F) -> Self {
        Self {
            c0,
            c1,
            _cfg: PhantomData,
        }
    }

    /// Return the conjugate `c0 - c1 * u`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.c0, -self.c1)
    }

    /// Return the norm in the base field: `c0^2 - NR * c1^2`.
    #[inline]
    pub fn norm(self) -> F {
        let nr = C::non_residue();
        (self.c0 * self.c0) - (nr * (self.c1 * self.c1))
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Clone for Fp2<F, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Copy for Fp2<F, C> {}

impl<F: FieldCore, C: Fp2Config<F>> Default for Fp2<F, C> {
    fn default() -> Self {
        Self::new(F::zero(), F::zero())
    }
}

impl<F: FieldCore, C: Fp2Config<F>> PartialEq for Fp2<F, C> {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Add for Fp2<F, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Sub for Fp2<F, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Neg for Fp2<F, C> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.c0, -self.c1)
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Mul for Fp2<F, C> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let nr = C::non_residue();
        let a0 = self.c0;
        let a1 = self.c1;
        let b0 = rhs.c0;
        let b1 = rhs.c1;
        Self::new((a0 * b0) + (nr * (a1 * b1)), (a0 * b1) + (a1 * b0))
    }
}

impl<'a, F: FieldCore, C: Fp2Config<F>> Add<&'a Self> for Fp2<F, C> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, F: FieldCore, C: Fp2Config<F>> Sub<&'a Self> for Fp2<F, C> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, F: FieldCore, C: Fp2Config<F>> Mul<&'a Self> for Fp2<F, C> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid, C: Fp2Config<F>> Valid for Fp2<F, C> {
    fn check(&self) -> Result<(), SerializationError> {
        self.c0.check()?;
        self.c1.check()?;
        Ok(())
    }
}

impl<F: FieldCore, C: Fp2Config<F>> HachiSerialize for Fp2<F, C> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.c0.serialize_with_mode(&mut writer, compress)?;
        self.c1.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.c0.serialized_size(compress) + self.c1.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, C: Fp2Config<F>> HachiDeserialize for Fp2<F, C> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let c0 = F::deserialize_with_mode(&mut reader, compress, validate)?;
        let c1 = F::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self::new(c0, c1);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid, C: Fp2Config<F>> FieldCore for Fp2<F, C> {
    fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    fn one() -> Self {
        Self::new(F::one(), F::zero())
    }

    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
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
            return None;
        }
        let inv_n = self.norm().inv()?;
        Some(Self::new(self.c0 * inv_n, (-self.c1) * inv_n))
    }
}

impl<F: FieldCore + FieldSampling + Valid, C: Fp2Config<F>> FieldSampling for Fp2<F, C> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        Self::new(F::sample(rng), F::sample(rng))
    }
}

/// Parameters for an `Fp4` quadratic extension over `Fp2<F, C2>`.
pub trait Fp4Config<F: FieldCore, C2: Fp2Config<F>> {
    /// Non-residue `NR2` in `Fp2` such that `v^2 = NR2`.
    fn non_residue() -> Fp2<F, C2>;
}

/// Quartic extension element `c0 + c1 * v` over `Fp2`, where `v^2 = NR2`.
pub struct Fp4<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> {
    /// Constant term.
    pub c0: Fp2<F, C2>,
    /// Coefficient of `v`.
    pub c1: Fp2<F, C2>,
    _cfg: PhantomData<fn() -> C4>,
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Fp4<F, C2, C4> {
    /// Construct `c0 + c1 * v`.
    #[inline]
    pub fn new(c0: Fp2<F, C2>, c1: Fp2<F, C2>) -> Self {
        Self {
            c0,
            c1,
            _cfg: PhantomData,
        }
    }

    /// Return the norm in `Fp2`: `c0^2 - NR2 * c1^2`.
    #[inline]
    pub fn norm(self) -> Fp2<F, C2> {
        let nr2 = C4::non_residue();
        (self.c0 * self.c0) - (nr2 * (self.c1 * self.c1))
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Clone for Fp4<F, C2, C4> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Copy for Fp4<F, C2, C4> {}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Default for Fp4<F, C2, C4> {
    fn default() -> Self {
        Self::new(
            Fp2::new(F::zero(), F::zero()),
            Fp2::new(F::zero(), F::zero()),
        )
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> PartialEq for Fp4<F, C2, C4> {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Add for Fp4<F, C2, C4> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Sub for Fp4<F, C2, C4> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Neg for Fp4<F, C2, C4> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.c0, -self.c1)
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Mul for Fp4<F, C2, C4> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let nr2 = C4::non_residue();
        let a0 = self.c0;
        let a1 = self.c1;
        let b0 = rhs.c0;
        let b1 = rhs.c1;
        Self::new((a0 * b0) + (nr2 * (a1 * b1)), (a0 * b1) + (a1 * b0))
    }
}

impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Add<&'a Self> for Fp4<F, C2, C4> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Sub<&'a Self> for Fp4<F, C2, C4> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Mul<&'a Self> for Fp4<F, C2, C4> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Valid for Fp4<F, C2, C4> {
    fn check(&self) -> Result<(), SerializationError> {
        self.c0.check()?;
        self.c1.check()?;
        Ok(())
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> HachiSerialize for Fp4<F, C2, C4> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.c0.serialize_with_mode(&mut writer, compress)?;
        self.c1.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.c0.serialized_size(compress) + self.c1.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> HachiDeserialize
    for Fp4<F, C2, C4>
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let c0 = Fp2::<F, C2>::deserialize_with_mode(&mut reader, compress, validate)?;
        let c1 = Fp2::<F, C2>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self::new(c0, c1);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> FieldCore for Fp4<F, C2, C4> {
    fn zero() -> Self {
        Self::new(Fp2::zero(), Fp2::zero())
    }

    fn one() -> Self {
        Self::new(Fp2::one(), Fp2::zero())
    }

    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
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
            return None;
        }
        let inv_n = self.norm().inv()?;
        Some(Self::new(self.c0 * inv_n, (-self.c1) * inv_n))
    }
}

impl<F: FieldCore + FieldSampling + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> FieldSampling
    for Fp4<F, C2, C4>
{
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        Self::new(Fp2::sample(rng), Fp2::sample(rng))
    }
}

// Scalar * VectorModule impls for extension scalars.

impl<F, C, const N: usize> Mul<VectorModule<Fp2<F, C>, N>> for Fp2<F, C>
where
    F: FieldCore + Valid,
    C: Fp2Config<F>,
{
    type Output = VectorModule<Fp2<F, C>, N>;
    fn mul(self, rhs: VectorModule<Fp2<F, C>, N>) -> Self::Output {
        let mut out = rhs.0;
        for coeff in &mut out {
            *coeff = self * *coeff;
        }
        VectorModule(out)
    }
}

impl<'a, F, C, const N: usize> Mul<&'a VectorModule<Fp2<F, C>, N>> for Fp2<F, C>
where
    F: FieldCore + Valid,
    C: Fp2Config<F>,
{
    type Output = VectorModule<Fp2<F, C>, N>;
    fn mul(self, rhs: &'a VectorModule<Fp2<F, C>, N>) -> Self::Output {
        self * *rhs
    }
}

impl<F, C2, C4, const N: usize> Mul<VectorModule<Fp4<F, C2, C4>, N>> for Fp4<F, C2, C4>
where
    F: FieldCore + Valid,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
{
    type Output = VectorModule<Fp4<F, C2, C4>, N>;
    fn mul(self, rhs: VectorModule<Fp4<F, C2, C4>, N>) -> Self::Output {
        let mut out = rhs.0;
        for coeff in &mut out {
            *coeff = self * *coeff;
        }
        VectorModule(out)
    }
}

impl<'a, F, C2, C4, const N: usize> Mul<&'a VectorModule<Fp4<F, C2, C4>, N>> for Fp4<F, C2, C4>
where
    F: FieldCore + Valid,
    C2: Fp2Config<F>,
    C4: Fp4Config<F, C2>,
{
    type Output = VectorModule<Fp4<F, C2, C4>, N>;
    fn mul(self, rhs: &'a VectorModule<Fp4<F, C2, C4>, N>) -> Self::Output {
        self * *rhs
    }
}
