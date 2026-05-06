//! Quadratic and quartic extension fields.

use super::wide::{AccumPair, HasUnreducedOps};
use crate::{BalancedDigitLookup, FieldCore, HalvingField};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{FromPrimitiveInt, Invertible, RandomSampling, RingCore};

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
/// All Akita pseudo-Mersenne primes (`2^k - c` with `c ≡ 3 mod 8`)
/// satisfy this.
pub struct TwoNr;

impl<F: FieldCore + FromPrimitiveInt> Fp2Config<F> for TwoNr {
    fn non_residue() -> F {
        F::from_u64(2)
    }
}
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
#[repr(transparent)]
pub struct Fp2<F: FieldCore, C: Fp2Config<F>> {
    /// Coefficients `[c0, c1]` in basis `[1, u]`.
    pub coeffs: [F; 2],
    _cfg: PhantomData<fn() -> C>,
}

impl<F: FieldCore, C: Fp2Config<F>> Fp2<F, C> {
    /// Construct `c0 + c1 * u`.
    #[inline]
    pub fn new(c0: F, c1: F) -> Self {
        Self {
            coeffs: [c0, c1],
            _cfg: PhantomData,
        }
    }

    /// Degree-0 coefficient.
    #[inline]
    pub fn c0(&self) -> F {
        self.coeffs[0]
    }

    /// Degree-1 coefficient.
    #[inline]
    pub fn c1(&self) -> F {
        self.coeffs[1]
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new(F::one(), F::zero())
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero() && self.coeffs[1].is_zero()
    }

    /// Construct from a `u64` embedded in the base field.
    #[inline]
    pub fn from_u64(val: u64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(F::from_u64(val), F::zero())
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(F::from_i64(val), F::zero())
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
        Self::new(self.coeffs[0], -self.coeffs[1])
    }

    /// Return the norm in the base field: `c0^2 - NR * c1^2`.
    #[inline]
    pub fn norm(self) -> F {
        (self.coeffs[0] * self.coeffs[0]) - Self::mul_nr(self.coeffs[1] * self.coeffs[1])
    }
}

impl<F: FieldCore + std::fmt::Debug, C: Fp2Config<F>> std::fmt::Debug for Fp2<F, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp2").field("coeffs", &self.coeffs).finish()
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
        self.coeffs[0] == other.coeffs[0] && self.coeffs[1] == other.coeffs[1]
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Eq for Fp2<F, C> {}

impl<F: FieldCore, C: Fp2Config<F>> Add for Fp2<F, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Sub for Fp2<F, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Neg for Fp2<F, C> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.coeffs[0], -self.coeffs[1])
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
        let v0 = self.coeffs[0] * rhs.coeffs[0];
        let v1 = self.coeffs[1] * rhs.coeffs[1];
        Self::new(
            v0 + Self::mul_nr(v1),
            (self.coeffs[0] + self.coeffs[1]) * (rhs.coeffs[0] + rhs.coeffs[1]) - v0 - v1,
        )
    }
}
impl<F: FieldCore, C: Fp2Config<F>> MulAssign for Fp2<F, C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
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
        self.coeffs[0].check()?;
        self.coeffs[1].check()?;
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize, C: Fp2Config<F>> AkitaSerialize for Fp2<F, C> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs[0].serialize_with_mode(&mut writer, compress)?;
        self.coeffs[1].serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs[0].serialized_size(compress) + self.coeffs[1].serialized_size(compress)
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>, C: Fp2Config<F>> AkitaDeserialize
    for Fp2<F, C>
{
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

impl<F: FieldCore + Valid, C: Fp2Config<F>> RingCore for Fp2<F, C> {
    /// Specialized squaring: 2 base-field multiplications instead of 3.
    ///
    /// `(c0 + c1·u)^2 = (c0^2 + NR·c1^2) + (2·c0·c1)·u`
    fn square(&self) -> Self {
        let v0 = self.coeffs[0] * self.coeffs[0];
        let v1 = self.coeffs[1] * self.coeffs[1];
        Self::new(
            v0 + Self::mul_nr(v1),
            (self.coeffs[0] + self.coeffs[0]) * self.coeffs[1],
        )
    }
}

impl<F: FieldCore + Valid, C: Fp2Config<F>> Invertible for Fp2<F, C> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inverse()?;
        Some(Self::new(self.coeffs[0] * inv_n, (-self.coeffs[1]) * inv_n))
    }
}

impl<F: HalvingField + Valid, C: Fp2Config<F>> HalvingField for Fp2<F, C> {
    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs[0].half(), self.coeffs[1].half())
    }
}

impl<F: FieldCore + RandomSampling + Valid, C: Fp2Config<F>> RandomSampling for Fp2<F, C> {
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(F::random(rng), F::random(rng))
    }
}

impl<F: FieldCore + FromPrimitiveInt + Valid, C: Fp2Config<F>> FromPrimitiveInt for Fp2<F, C> {
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new(F::from_u128(val), F::zero())
    }

    fn from_i128(val: i128) -> Self {
        Self::new(F::from_i128(val), F::zero())
    }
}

impl<F: FieldCore + BalancedDigitLookup + Valid, C: Fp2Config<F>> BalancedDigitLookup
    for Fp2<F, C>
{
}

impl<F: HasUnreducedOps + Valid, C: Fp2Config<F>> HasUnreducedOps for Fp2<F, C> {
    type MulU64Accum = AccumPair<F::MulU64Accum>;
    type ProductAccum = AccumPair<F::ProductAccum>;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> AccumPair<F::MulU64Accum> {
        AccumPair(
            self.coeffs[0].mul_u64_unreduced(small),
            self.coeffs[1].mul_u64_unreduced(small),
        )
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> AccumPair<F::ProductAccum> {
        // Karatsuba: (c0 + c1·u)(d0 + d1·u) = (c0·d0 + NR·c1·d1) + (c0·d1 + c1·d0)·u
        let v0 = self.coeffs[0].mul_to_product_accum(other.coeffs[0]);
        let v1 = self.coeffs[1].mul_to_product_accum(other.coeffs[1]);
        let cross = (self.coeffs[0] + self.coeffs[1])
            .mul_to_product_accum(other.coeffs[0] + other.coeffs[1]);

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

/// Parameters for a tower-basis quartic extension over `Fp2<F, C2>`.
pub trait TowerBasisFp4Config<F: FieldCore, C2: Fp2Config<F>> {
    /// Non-residue `NR2` in `Fp2` such that `v^2 = NR2`.
    fn non_residue() -> Fp2<F, C2>;
}

/// Parameters for a power-basis quartic extension over base field `F`.
pub trait PowerBasisFp4Config<F: FieldCore> {
    /// Non-residue `W` such that `v^4 = W`.
    fn w() -> F;
}

impl<F, C> PowerBasisFp4Config<F> for C
where
    F: FieldCore,
    C: Fp2Config<F>,
{
    fn w() -> F {
        C::non_residue()
    }
}

/// `TowerBasisFp4Config` with non-residue `u ∈ Fp2` (the element `(0, 1)`).
///
/// This is the standard tower choice: `v^2 = u`, hence `v^4 = NR`.
pub struct UnitNr;

impl<F: FieldCore, C2: Fp2Config<F>> TowerBasisFp4Config<F, C2> for UnitNr {
    fn non_residue() -> Fp2<F, C2> {
        Fp2::new(F::zero(), F::one())
    }
}

/// Default quadratic extension used by Akita field tests and helpers.
pub type Ext2<F> = Fp2<F, TwoNr>;

/// Quartic extension element `b0 + b1 * v` over `Fp2`, where `v^2 = NR2`.
#[repr(transparent)]
pub struct TowerBasisFp4<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> {
    /// Coefficients `[b0, b1]` in tower basis `[1, v]` over `Fp2`.
    pub coeffs: [Fp2<F, C2>; 2],
    _cfg: PhantomData<fn() -> C4>,
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> TowerBasisFp4<F, C2, C4> {
    /// Construct `c0 + c1 * v`.
    #[inline]
    pub fn new(c0: Fp2<F, C2>, c1: Fp2<F, C2>) -> Self {
        Self {
            coeffs: [c0, c1],
            _cfg: PhantomData,
        }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new(Fp2::zero(), Fp2::zero())
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new(Fp2::one(), Fp2::zero())
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs[0].is_zero() && self.coeffs[1].is_zero()
    }

    /// Construct from a `u64` embedded in the base field.
    #[inline]
    pub fn from_u64(val: u64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(Fp2::from_u64(val), Fp2::zero())
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(Fp2::from_i64(val), Fp2::zero())
    }

    /// Return the norm in `Fp2`: `c0^2 - NR2 * c1^2`.
    #[inline]
    pub fn norm(self) -> Fp2<F, C2> {
        let nr2 = C4::non_residue();
        (self.coeffs[0] * self.coeffs[0]) - (nr2 * (self.coeffs[1] * self.coeffs[1]))
    }
}

impl<F: FieldCore + std::fmt::Debug, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>>
    std::fmt::Debug for TowerBasisFp4<F, C2, C4>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TowerBasisFp4")
            .field("c0", &self.coeffs[0])
            .field("c1", &self.coeffs[1])
            .finish()
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Clone
    for TowerBasisFp4<F, C2, C4>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Copy
    for TowerBasisFp4<F, C2, C4>
{
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Default
    for TowerBasisFp4<F, C2, C4>
{
    fn default() -> Self {
        Self::new(
            Fp2::new(F::zero(), F::zero()),
            Fp2::new(F::zero(), F::zero()),
        )
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> PartialEq
    for TowerBasisFp4<F, C2, C4>
{
    fn eq(&self, other: &Self) -> bool {
        self.coeffs[0] == other.coeffs[0] && self.coeffs[1] == other.coeffs[1]
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Eq
    for TowerBasisFp4<F, C2, C4>
{
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Add
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Sub
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Neg
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.coeffs[0], -self.coeffs[1])
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> AddAssign
    for TowerBasisFp4<F, C2, C4>
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> SubAssign
    for TowerBasisFp4<F, C2, C4>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Mul
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let nr2 = C4::non_residue();
        let v0 = self.coeffs[0] * rhs.coeffs[0];
        let v1 = self.coeffs[1] * rhs.coeffs[1];
        Self::new(
            v0 + (nr2 * v1),
            (self.coeffs[0] + self.coeffs[1]) * (rhs.coeffs[0] + rhs.coeffs[1]) - v0 - v1,
        )
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> MulAssign
    for TowerBasisFp4<F, C2, C4>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Add<&'a Self>
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Sub<&'a Self>
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Mul<&'a Self>
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Valid
    for TowerBasisFp4<F, C2, C4>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs[0].check()?;
        self.coeffs[1].check()?;
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> AkitaSerialize
    for TowerBasisFp4<F, C2, C4>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs[0].serialize_with_mode(&mut writer, compress)?;
        self.coeffs[1].serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs[0].serialized_size(compress) + self.coeffs[1].serialized_size(compress)
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        C2: Fp2Config<F>,
        C4: TowerBasisFp4Config<F, C2>,
    > AkitaDeserialize for TowerBasisFp4<F, C2, C4>
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

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> RingCore
    for TowerBasisFp4<F, C2, C4>
{
    fn square(&self) -> Self {
        let nr2 = C4::non_residue();
        let v0 = self.coeffs[0].square();
        let v1 = self.coeffs[1].square();
        Self::new(
            v0 + nr2 * v1,
            (self.coeffs[0] + self.coeffs[0]) * self.coeffs[1],
        )
    }
}

impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Invertible
    for TowerBasisFp4<F, C2, C4>
{
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inverse()?;
        Some(Self::new(self.coeffs[0] * inv_n, (-self.coeffs[1]) * inv_n))
    }
}

impl<F: HalvingField + Valid, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> HalvingField
    for TowerBasisFp4<F, C2, C4>
{
    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs[0].half(), self.coeffs[1].half())
    }
}

impl<F: FieldCore + RandomSampling + Valid, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>>
    RandomSampling for TowerBasisFp4<F, C2, C4>
{
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(Fp2::random(rng), Fp2::random(rng))
    }
}

impl<F: FieldCore + FromPrimitiveInt + Valid, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>>
    FromPrimitiveInt for TowerBasisFp4<F, C2, C4>
{
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new(Fp2::from_u128(val), Fp2::zero())
    }

    fn from_i128(val: i128) -> Self {
        Self::new(Fp2::from_i128(val), Fp2::zero())
    }
}

impl<
        F: FieldCore + BalancedDigitLookup + Valid,
        C2: Fp2Config<F>,
        C4: TowerBasisFp4Config<F, C2>,
    > BalancedDigitLookup for TowerBasisFp4<F, C2, C4>
{
}

/// Quartic extension element `a0 + a1*v + a2*v^2 + a3*v^3`, where `v^4 = W`.
#[repr(transparent)]
pub struct PowerBasisFp4<F: FieldCore, C: PowerBasisFp4Config<F>> {
    /// Coefficients `[a0, a1, a2, a3]` in basis `[1, v, v^2, v^3]`.
    pub coeffs: [F; 4],
    _cfg: PhantomData<fn() -> C>,
}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> PowerBasisFp4<F, C> {
    /// Construct from power-basis coefficients `[a0, a1, a2, a3]`.
    #[inline]
    pub fn new(coeffs: [F; 4]) -> Self {
        Self {
            coeffs,
            _cfg: PhantomData,
        }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new([F::zero(), F::zero(), F::zero(), F::zero()])
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new([F::one(), F::zero(), F::zero(), F::zero()])
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|coeff| coeff.is_zero())
    }

    /// Construct from a `u64` embedded in the base field.
    #[inline]
    pub fn from_u64(val: u64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new([F::from_u64(val), F::zero(), F::zero(), F::zero()])
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new([F::from_i64(val), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: FieldCore + std::fmt::Debug, C: PowerBasisFp4Config<F>> std::fmt::Debug
    for PowerBasisFp4<F, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PowerBasisFp4")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> Clone for PowerBasisFp4<F, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> Copy for PowerBasisFp4<F, C> {}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> Default for PowerBasisFp4<F, C> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> PartialEq for PowerBasisFp4<F, C> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> Eq for PowerBasisFp4<F, C> {}

impl<F: FieldCore, C: PowerBasisFp4Config<F>> Add for PowerBasisFp4<F, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> Sub for PowerBasisFp4<F, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> Neg for PowerBasisFp4<F, C> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(std::array::from_fn(|i| -self.coeffs[i]))
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> AddAssign for PowerBasisFp4<F, C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> SubAssign for PowerBasisFp4<F, C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> Mul for PowerBasisFp4<F, C> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let [a0, a1, a2, a3] = self.coeffs;
        let [b0, b1, b2, b3] = rhs.coeffs;
        let w = C::w();
        Self::new([
            a0 * b0 + w * (a1 * b3 + a2 * b2 + a3 * b1),
            a0 * b1 + a1 * b0 + w * (a2 * b3 + a3 * b2),
            a0 * b2 + a1 * b1 + a2 * b0 + w * (a3 * b3),
            a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0,
        ])
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> MulAssign for PowerBasisFp4<F, C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore, C: PowerBasisFp4Config<F>> Add<&'a Self> for PowerBasisFp4<F, C> {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, F: FieldCore, C: PowerBasisFp4Config<F>> Sub<&'a Self> for PowerBasisFp4<F, C> {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, F: FieldCore, C: PowerBasisFp4Config<F>> Mul<&'a Self> for PowerBasisFp4<F, C> {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid, C: PowerBasisFp4Config<F>> Valid for PowerBasisFp4<F, C> {
    fn check(&self) -> Result<(), SerializationError> {
        for coeff in self.coeffs {
            coeff.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize, C: PowerBasisFp4Config<F>> AkitaSerialize
    for PowerBasisFp4<F, C>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in self.coeffs {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs
            .iter()
            .map(|coeff| coeff.serialized_size(compress))
            .sum()
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>, C: PowerBasisFp4Config<F>>
    AkitaDeserialize for PowerBasisFp4<F, C>
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let coeffs = [
            F::deserialize_with_mode(&mut reader, compress, validate, &())?,
            F::deserialize_with_mode(&mut reader, compress, validate, &())?,
            F::deserialize_with_mode(&mut reader, compress, validate, &())?,
            F::deserialize_with_mode(&mut reader, compress, validate, &())?,
        ];
        let out = Self::new(coeffs);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid, C: PowerBasisFp4Config<F>> RingCore for PowerBasisFp4<F, C> {
    fn square(&self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let w = C::w();
        let two = F::one() + F::one();
        let a0a1 = a0 * a1;
        let a0a2 = a0 * a2;
        let a0a3 = a0 * a3;
        let a1a2 = a1 * a2;
        let a1a3 = a1 * a3;
        let a2a3 = a2 * a3;
        Self::new([
            a0.square() + w * (two * a1a3 + a2.square()),
            two * (a0a1 + w * a2a3),
            two * a0a2 + a1.square() + w * a3.square(),
            two * (a0a3 + a1a2),
        ])
    }
}

impl<F: FieldCore + Valid, C: PowerBasisFp4Config<F>> Invertible for PowerBasisFp4<F, C> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let [a0, a1, a2, a3] = self.coeffs;
        let w = C::w();
        let two = F::one() + F::one();

        let d0 = a0.square() + w * a2.square() - two * w * (a1 * a3);
        let d1 = two * (a0 * a2) - a1.square() - w * a3.square();
        let inv_norm = (d0.square() - w * d1.square()).inverse()?;
        let e0 = d0 * inv_norm;
        let e1 = -d1 * inv_norm;

        Some(Self::new([
            a0 * e0 + w * (a2 * e1),
            -(a1 * e0 + w * (a3 * e1)),
            a0 * e1 + a2 * e0,
            -(a1 * e1 + a3 * e0),
        ]))
    }
}

impl<F: HalvingField + Valid, C: PowerBasisFp4Config<F>> HalvingField for PowerBasisFp4<F, C> {
    #[inline]
    fn half(self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i].half()))
    }
}

impl<F: FieldCore + RandomSampling + Valid, C: PowerBasisFp4Config<F>> RandomSampling
    for PowerBasisFp4<F, C>
{
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new([
            F::random(rng),
            F::random(rng),
            F::random(rng),
            F::random(rng),
        ])
    }
}

impl<F: FieldCore + FromPrimitiveInt + Valid, C: PowerBasisFp4Config<F>> FromPrimitiveInt
    for PowerBasisFp4<F, C>
{
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new([F::from_u128(val), F::zero(), F::zero(), F::zero()])
    }

    fn from_i128(val: i128) -> Self {
        Self::new([F::from_i128(val), F::zero(), F::zero(), F::zero()])
    }
}

impl<F: FieldCore + BalancedDigitLookup + Valid, C: PowerBasisFp4Config<F>> BalancedDigitLookup
    for PowerBasisFp4<F, C>
{
}

impl<F, C> From<PowerBasisFp4<F, C>> for TowerBasisFp4<F, C, UnitNr>
where
    F: FieldCore,
    C: Fp2Config<F> + PowerBasisFp4Config<F>,
{
    fn from(x: PowerBasisFp4<F, C>) -> Self {
        let [a0, a1, a2, a3] = x.coeffs;
        Self::new(Fp2::new(a0, a2), Fp2::new(a1, a3))
    }
}

impl<F, C> From<TowerBasisFp4<F, C, UnitNr>> for PowerBasisFp4<F, C>
where
    F: FieldCore,
    C: Fp2Config<F> + PowerBasisFp4Config<F>,
{
    fn from(x: TowerBasisFp4<F, C, UnitNr>) -> Self {
        let [b0, b1] = x.coeffs;
        Self::new([b0.coeffs[0], b1.coeffs[0], b0.coeffs[1], b1.coeffs[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::lift::ExtField;
    use crate::Fp64;
    use crate::{FromPrimitiveInt, Invertible};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;
    type E2 = Ext2<F>;
    type E4 = TowerBasisFp4<F, TwoNr, UnitNr>;
    type P4 = PowerBasisFp4<F, TwoNr>;

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
        let a = E2::random(&mut rng);
        let b = E2::random(&mut rng);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn fp2_karatsuba_matches_schoolbook() {
        let mut rng = StdRng::seed_from_u64(5678);
        for _ in 0..100 {
            let a = E2::random(&mut rng);
            let b = E2::random(&mut rng);
            let nr = <TwoNr as Fp2Config<F>>::non_residue();
            let expected = E2::new(
                (a.coeffs[0] * b.coeffs[0]) + (nr * (a.coeffs[1] * b.coeffs[1])),
                (a.coeffs[0] * b.coeffs[1]) + (a.coeffs[1] * b.coeffs[0]),
            );
            assert_eq!(a * b, expected);
        }
    }

    #[test]
    fn fp2_square_matches_mul() {
        let mut rng = StdRng::seed_from_u64(9012);
        for _ in 0..100 {
            let a = E2::random(&mut rng);
            assert_eq!(a.square(), a * a, "square mismatch for {a:?}");
        }
    }

    #[test]
    fn fp2_inv() {
        let mut rng = StdRng::seed_from_u64(3456);
        for _ in 0..50 {
            let a = E2::random(&mut rng);
            if !a.is_zero() {
                let inv = a.inverse().unwrap();
                assert_eq!(a * inv, E2::one());
            }
        }
    }

    #[test]
    fn fp4_mul_commutativity() {
        let mut rng = StdRng::seed_from_u64(7890);
        let a = E4::random(&mut rng);
        let b = E4::random(&mut rng);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn fp4_square_matches_mul() {
        let mut rng = StdRng::seed_from_u64(1111);
        for _ in 0..50 {
            let a = E4::random(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn fp4_inv() {
        let mut rng = StdRng::seed_from_u64(2222);
        for _ in 0..50 {
            let a = E4::random(&mut rng);
            if !a.is_zero() {
                let inv = a.inverse().unwrap();
                assert_eq!(a * inv, E4::one());
            }
        }
    }

    #[test]
    fn power_basis_fp4_square_matches_mul() {
        let mut rng = StdRng::seed_from_u64(3333);
        for _ in 0..50 {
            let a = P4::random(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn power_basis_fp4_inv() {
        let mut rng = StdRng::seed_from_u64(4444);
        for _ in 0..50 {
            let a = P4::random(&mut rng);
            if !a.is_zero() {
                let inv = a.inverse().unwrap();
                assert_eq!(a * inv, P4::one());
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
        assert_eq!(e4, E4::new(E2::new(c0, c2), E2::new(c1, c3)));

        let p4 = P4::from_base_slice(&[c0, c1, c2, c3]);
        assert_eq!(p4, P4::new([c0, c1, c2, c3]));
    }

    #[test]
    fn tower_and_power_basis_fp4_multiplication_agree() {
        let x_p = P4::new([
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let y_p = P4::new([
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ]);
        let x_t: E4 = x_p.into();
        let y_t: E4 = y_p.into();

        let got: P4 = (x_t * y_t).into();
        assert_eq!(got, x_p * y_p);
    }

    #[test]
    fn power_basis_fp4_transcript_limb_order_is_univariate() {
        let x = P4::new([
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        assert_eq!(
            <P4 as ExtField<F>>::to_base_vec(&x),
            vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4)
            ]
        );
    }

    #[test]
    fn tower_basis_fp4_transcript_limb_order_is_univariate() {
        let x = E4::new(
            E2::new(F::from_u64(1), F::from_u64(3)),
            E2::new(F::from_u64(2), F::from_u64(4)),
        );
        assert_eq!(
            <E4 as ExtField<F>>::to_base_vec(&x),
            vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4)
            ]
        );
    }

    #[test]
    fn extension_fields_are_array_layouts() {
        assert_eq!(core::mem::size_of::<E2>(), core::mem::size_of::<[F; 2]>());
        assert_eq!(core::mem::align_of::<E2>(), core::mem::align_of::<[F; 2]>());
        assert_eq!(core::mem::size_of::<P4>(), core::mem::size_of::<[F; 4]>());
        assert_eq!(core::mem::align_of::<P4>(), core::mem::align_of::<[F; 4]>());
        assert_eq!(core::mem::size_of::<E4>(), core::mem::size_of::<[E2; 2]>());
        assert_eq!(
            core::mem::align_of::<E4>(),
            core::mem::align_of::<[E2; 2]>()
        );
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
