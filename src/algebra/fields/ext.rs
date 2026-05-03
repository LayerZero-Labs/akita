//! Quadratic and quartic extension fields.

use super::wide::{AccumPair, HasUnreducedOps};
use crate::algebra::module::VectorModule;
use crate::{AdditiveGroup, FieldCore, FieldSampling, FromSmallInt};
use akita_serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};

/// `Fp2Config` with non-residue = -1.
///
/// Valid when `p ≡ 3 (mod 4)`, i.e. -1 is a quadratic non-residue.
pub struct NegOneNr;

impl<F: FieldCore> Fp2Config<F> for NegOneNr {
    const IS_NEG_ONE: bool = true;

    fn non_residue() -> F {
        -F::one()
    }
}

/// `Fp2Config` with non-residue = 2.
///
/// Valid when `p ≡ 5 (mod 8)`, i.e. 2 is a quadratic non-residue.
/// All Hachi pseudo-Mersenne primes (`2^k - c` with `c ≡ 3 mod 8`)
/// satisfy this.
pub struct TwoNr;

impl<F: FieldCore + FromSmallInt> Fp2Config<F> for TwoNr {
    fn non_residue() -> F {
        F::from_u64(2)
    }
}
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// Parameters for an `Fp2` quadratic extension over base field `F`.
pub trait Fp2Config<F: FieldCore> {
    /// Whether the non-residue is -1.
    ///
    /// When `true`, multiplication by the non-residue is a free negation and
    /// the Karatsuba/squaring routines can avoid a base-field multiply.
    const IS_NEG_ONE: bool = false;

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

    /// Multiply a base-field element by the non-residue.
    ///
    /// When `IS_NEG_ONE` is true this is just a negation (no multiply).
    #[inline(always)]
    fn mul_nr(x: F) -> F {
        if C::IS_NEG_ONE {
            -x
        } else {
            C::non_residue() * x
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
        (self.c0 * self.c0) - Self::mul_nr(self.c1 * self.c1)
    }
}

impl<F: FieldCore + std::fmt::Debug, C: Fp2Config<F>> std::fmt::Debug for Fp2<F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp2")
            .field("c0", &self.c0)
            .field("c1", &self.c1)
            .finish()
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

impl<F: FieldCore, C: Fp2Config<F>> Eq for Fp2<F, C> {}

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
impl<F: FieldCore, C: Fp2Config<F>> AddAssign for Fp2<F, C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: FieldCore, C: Fp2Config<F>> SubAssign for Fp2<F, C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Mul for Fp2<F, C> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        Self::new(
            v0 + Self::mul_nr(v1),
            (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1,
        )
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
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let c0 = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let c1 = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::new(c0, c1);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore, C: Fp2Config<F>> AdditiveGroup for Fp2<F, C> {
    const ZERO: Self = Self {
        c0: F::ZERO,
        c1: F::ZERO,
        _cfg: PhantomData,
    };
}

impl<F: FieldCore + Valid, C: Fp2Config<F>> FieldCore for Fp2<F, C> {
    fn one() -> Self {
        Self::new(F::one(), F::zero())
    }

    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    /// Specialized squaring: 2 base-field multiplications instead of 3.
    ///
    /// `(c0 + c1·u)^2 = (c0^2 + NR·c1^2) + (2·c0·c1)·u`
    fn square(&self) -> Self {
        let v0 = self.c0 * self.c0;
        let v1 = self.c1 * self.c1;
        Self::new(v0 + Self::mul_nr(v1), (self.c0 + self.c0) * self.c1)
    }

    fn inv(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inv()?;
        Some(Self::new(self.c0 * inv_n, (-self.c1) * inv_n))
    }

    const TWO_INV: Self = Self {
        c0: F::TWO_INV,
        c1: F::ZERO,
        _cfg: PhantomData,
    };
}

impl<F: FieldCore + FieldSampling + Valid, C: Fp2Config<F>> FieldSampling for Fp2<F, C> {
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        Self::new(F::sample(rng), F::sample(rng))
    }
}

impl<F: FieldCore + FromSmallInt + Valid, C: Fp2Config<F>> FromSmallInt for Fp2<F, C> {
    fn from_u64(val: u64) -> Self {
        Self::new(F::from_u64(val), F::zero())
    }

    fn from_i64(val: i64) -> Self {
        Self::new(F::from_i64(val), F::zero())
    }
}

impl<F: HasUnreducedOps + Valid, C: Fp2Config<F>> HasUnreducedOps for Fp2<F, C> {
    type MulU64Accum = AccumPair<F::MulU64Accum>;
    type ProductAccum = AccumPair<F::ProductAccum>;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> AccumPair<F::MulU64Accum> {
        AccumPair(
            self.c0.mul_u64_unreduced(small),
            self.c1.mul_u64_unreduced(small),
        )
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> AccumPair<F::ProductAccum> {
        // Karatsuba: (c0 + c1·u)(d0 + d1·u) = (c0·d0 + NR·c1·d1) + (c0·d1 + c1·d0)·u
        let v0 = self.c0.mul_to_product_accum(other.c0);
        let v1 = self.c1.mul_to_product_accum(other.c1);
        let cross = (self.c0 + self.c1).mul_to_product_accum(other.c0 + other.c1);

        let nr_v1 = if C::IS_NEG_ONE { -v1 } else { v1 + v1 };
        AccumPair(v0 + nr_v1, cross - v0 - v1)
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: AccumPair<F::MulU64Accum>) -> Self {
        Self::new(
            F::reduce_mul_u64_accum(accum.0),
            F::reduce_mul_u64_accum(accum.1),
        )
    }

    #[inline]
    fn reduce_product_accum(accum: AccumPair<F::ProductAccum>) -> Self {
        Self::new(
            F::reduce_product_accum(accum.0),
            F::reduce_product_accum(accum.1),
        )
    }
}

/// Parameters for an `Fp4` quadratic extension over `Fp2<F, C2>`.
pub trait Fp4Config<F: FieldCore, C2: Fp2Config<F>> {
    /// Non-residue `NR2` in `Fp2` such that `v^2 = NR2`.
    fn non_residue() -> Fp2<F, C2>;
}

/// `Fp4Config` with non-residue `u ∈ Fp2` (the element `(0, 1)`).
///
/// This is the standard tower choice: `Fp4 = Fp2[v] / (v^2 - u)`.
pub struct UnitNr;

impl<F: FieldCore, C2: Fp2Config<F>> Fp4Config<F, C2> for UnitNr {
    fn non_residue() -> Fp2<F, C2> {
        Fp2::new(F::zero(), F::one())
    }
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

impl<F: FieldCore + std::fmt::Debug, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> std::fmt::Debug
    for Fp4<F, C2, C4>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp4")
            .field("c0", &self.c0)
            .field("c1", &self.c1)
            .finish()
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

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Eq for Fp4<F, C2, C4> {}

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
impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> AddAssign for Fp4<F, C2, C4> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> SubAssign for Fp4<F, C2, C4> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Mul for Fp4<F, C2, C4> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let nr2 = C4::non_residue();
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        Self::new(
            v0 + (nr2 * v1),
            (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1,
        )
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
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let c0 = Fp2::<F, C2>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let c1 = Fp2::<F, C2>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::new(c0, c1);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> AdditiveGroup for Fp4<F, C2, C4> {
    const ZERO: Self = Self {
        c0: Fp2::ZERO,
        c1: Fp2::ZERO,
        _cfg: PhantomData,
    };
}

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> FieldCore for Fp4<F, C2, C4> {
    fn one() -> Self {
        Self::new(Fp2::one(), Fp2::zero())
    }

    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    fn square(&self) -> Self {
        let nr2 = C4::non_residue();
        let v0 = self.c0.square();
        let v1 = self.c1.square();
        Self::new(v0 + nr2 * v1, (self.c0 + self.c0) * self.c1)
    }

    fn inv(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inv()?;
        Some(Self::new(self.c0 * inv_n, (-self.c1) * inv_n))
    }

    const TWO_INV: Self = Self {
        c0: Fp2::TWO_INV,
        c1: Fp2::ZERO,
        _cfg: PhantomData,
    };
}

impl<F: FieldCore + FieldSampling + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> FieldSampling
    for Fp4<F, C2, C4>
{
    fn sample<R: RngCore>(rng: &mut R) -> Self {
        Self::new(Fp2::sample(rng), Fp2::sample(rng))
    }
}

impl<F: FieldCore + FromSmallInt + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> FromSmallInt
    for Fp4<F, C2, C4>
{
    fn from_u64(val: u64) -> Self {
        Self::new(Fp2::from_u64(val), Fp2::zero())
    }

    fn from_i64(val: i64) -> Self {
        Self::new(Fp2::from_i64(val), Fp2::zero())
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

// Convenience aliases for extension fields with NR = 2 (valid for all Hachi
// pseudo-Mersenne primes where p ≡ 5 mod 8).

/// Quadratic extension over any `F` with non-residue 2.
pub type Ext2<F> = Fp2<F, TwoNr>;

/// Quartic extension as tower `Ext2<F>[v]/(v^2 - u)`.
pub type Ext4<F> = Fp4<F, TwoNr, UnitNr>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::lift::ExtField;
    use crate::algebra::Fp64;
    use crate::{FieldCore, FieldSampling, FromSmallInt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;
    type E2 = Ext2<F>;
    type E4 = Ext4<F>;

    #[test]
    fn fp2_add_sub_identity() {
        let a = E2::new(F::from_u64(3), F::from_u64(5));
        let b = E2::new(F::from_u64(7), F::from_u64(11));
        let c = a + b;
        assert_eq!(c - b, a);
        assert_eq!(c - a, b);
    }

    #[test]
    fn fp2_mul_one() {
        let a = E2::new(F::from_u64(42), F::from_u64(13));
        assert_eq!(a * E2::one(), a);
        assert_eq!(E2::one() * a, a);
    }

    #[test]
    fn fp2_mul_commutativity() {
        let mut rng = StdRng::seed_from_u64(1234);
        let a = E2::sample(&mut rng);
        let b = E2::sample(&mut rng);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn fp2_karatsuba_matches_schoolbook() {
        let mut rng = StdRng::seed_from_u64(5678);
        for _ in 0..100 {
            let a = E2::sample(&mut rng);
            let b = E2::sample(&mut rng);
            let nr = <TwoNr as Fp2Config<F>>::non_residue();
            let expected = E2::new(
                (a.c0 * b.c0) + (nr * (a.c1 * b.c1)),
                (a.c0 * b.c1) + (a.c1 * b.c0),
            );
            assert_eq!(a * b, expected);
        }
    }

    #[test]
    fn fp2_square_matches_mul() {
        let mut rng = StdRng::seed_from_u64(9012);
        for _ in 0..100 {
            let a = E2::sample(&mut rng);
            assert_eq!(a.square(), a * a, "square mismatch for {a:?}");
        }
    }

    #[test]
    fn fp2_inv() {
        let mut rng = StdRng::seed_from_u64(3456);
        for _ in 0..50 {
            let a = E2::sample(&mut rng);
            if !a.is_zero() {
                let inv = a.inv().unwrap();
                assert_eq!(a * inv, E2::one());
            }
        }
    }

    #[test]
    fn fp4_mul_commutativity() {
        let mut rng = StdRng::seed_from_u64(7890);
        let a = E4::sample(&mut rng);
        let b = E4::sample(&mut rng);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn fp4_square_matches_mul() {
        let mut rng = StdRng::seed_from_u64(1111);
        for _ in 0..50 {
            let a = E4::sample(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn fp4_inv() {
        let mut rng = StdRng::seed_from_u64(2222);
        for _ in 0..50 {
            let a = E4::sample(&mut rng);
            if !a.is_zero() {
                let inv = a.inv().unwrap();
                assert_eq!(a * inv, E4::one());
            }
        }
    }

    #[test]
    fn from_small_int_fp2() {
        let a = E2::from_u64(42);
        assert_eq!(a, E2::new(F::from_u64(42), F::zero()));

        let b = E2::from_i64(-3);
        assert_eq!(b, E2::new(F::from_i64(-3), F::zero()));

        let c = E2::from_u8(7);
        assert_eq!(c, E2::from_u64(7));

        let d = E2::from_u32(100_000);
        assert_eq!(d, E2::from_u64(100_000));
    }

    #[test]
    fn from_small_int_fp4() {
        let a = E4::from_u64(42);
        assert_eq!(a, E4::new(E2::from_u64(42), E2::zero()));

        let b = E4::from_i64(-7);
        assert_eq!(b, E4::new(E2::from_i64(-7), E2::zero()));
    }

    #[test]
    fn ext_field_degree() {
        assert_eq!(<F as ExtField<F>>::EXT_DEGREE, 1);
        assert_eq!(<E2 as ExtField<F>>::EXT_DEGREE, 2);
        assert_eq!(<E4 as ExtField<F>>::EXT_DEGREE, 4);
    }

    #[test]
    fn ext_field_from_base_slice() {
        let c0 = F::from_u64(3);
        let c1 = F::from_u64(5);
        let e2 = E2::from_base_slice(&[c0, c1]);
        assert_eq!(e2, E2::new(c0, c1));

        let c2 = F::from_u64(7);
        let c3 = F::from_u64(11);
        let e4 = E4::from_base_slice(&[c0, c1, c2, c3]);
        assert_eq!(e4, E4::new(E2::new(c0, c1), E2::new(c2, c3)));
    }

    #[test]
    fn eq_impl() {
        let a = E2::new(F::from_u64(1), F::from_u64(2));
        let b = E2::new(F::from_u64(1), F::from_u64(2));
        let c = E2::new(F::from_u64(1), F::from_u64(3));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
