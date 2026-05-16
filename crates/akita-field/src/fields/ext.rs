//! Quadratic and quartic extension fields.

use super::wide::{AccumPair, HasUnreducedOps};
use super::{fp128::Fp128, fp32::Fp32, fp64::Fp64};
use crate::{BalancedDigitLookup, CanonicalField, FieldCore, HalvingField};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{FromPrimitiveInt, Invertible, RandomSampling, RingCore};
use rand_core::RngCore;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Arithmetic shape shared by scalar and packed extension coefficients.
pub trait ExtensionCoeff<F: FieldCore>:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
}

impl<F, A> ExtensionCoeff<F> for A
where
    F: FieldCore,
    A: Copy + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
{
}

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

    #[inline]
    fn mul_non_residue<A, B>(x: A, _from_base: B) -> A
    where
        A: ExtensionCoeff<F>,
        B: FnOnce(F) -> A,
    {
        x + x
    }
}

/// Parameters for an `Fp2` quadratic extension over base field `F`.
pub trait Fp2Config<F: FieldCore> {
    /// Whether the non-residue is -1.
    ///
    /// When `true`, multiplication by the non-residue is a free negation and
    /// the Karatsuba/squaring routines can avoid a base-field multiply.
    const IS_NEG_ONE: bool = false;

    /// Non-residue `NR` such that `u^2 = NR`.
    fn non_residue() -> F;

    /// Multiply a coefficient by the quadratic non-residue.
    #[inline]
    fn mul_non_residue<A, B>(x: A, from_base: B) -> A
    where
        A: ExtensionCoeff<F>,
        B: FnOnce(F) -> A,
    {
        if Self::IS_NEG_ONE {
            from_base(F::zero()) - x
        } else {
            from_base(Self::non_residue()) * x
        }
    }
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
        C::mul_non_residue(x, |base| base)
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
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Sub for Fp2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
        )
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Neg for Fp2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.coeffs[0], -self.coeffs[1])
    }
}
impl<F: FieldCore, C: Fp2Config<F>> AddAssign for Fp2<F, C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] + rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] + rhs.coeffs[1];
    }
}
impl<F: FieldCore, C: Fp2Config<F>> SubAssign for Fp2<F, C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] - rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] - rhs.coeffs[1];
    }
}
impl<F: FieldCore, C: Fp2Config<F>> Mul for Fp2<F, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let a0b0 = self.coeffs[0] * rhs.coeffs[0];
        let a1b1 = self.coeffs[1] * rhs.coeffs[1];
        let a0b1 = self.coeffs[0] * rhs.coeffs[1];
        let a1b0 = self.coeffs[1] * rhs.coeffs[0];
        Self::new(a0b0 + Self::mul_nr(a1b1), a0b1 + a1b0)
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
    #[inline(always)]
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
        let c00 = self.coeffs[0].mul_to_product_accum(other.coeffs[0]);
        let c11 = self.coeffs[1].mul_to_product_accum(other.coeffs[1]);
        let c01 = self.coeffs[0].mul_to_product_accum(other.coeffs[1]);
        let c10 = self.coeffs[1].mul_to_product_accum(other.coeffs[0]);

        let nr_c11 = if C::IS_NEG_ONE { -c11 } else { c11 + c11 };
        AccumPair(c00 + nr_c11, c01 + c10)
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

    /// Multiply an `Fp2` element by the tower non-residue.
    #[inline]
    fn mul_non_residue(x: Fp2<F, C2>) -> Fp2<F, C2> {
        Self::non_residue() * x
    }
}

/// Parameters for a power-basis quartic extension over base field `F`.
pub trait PowerBasisFp4Config<F: FieldCore> {
    /// Non-residue `W` such that `v^4 = W`.
    fn w() -> F;

    /// Multiply a coefficient by `W`.
    #[inline]
    fn mul_w<A, B>(x: A, from_base: B) -> A
    where
        A: ExtensionCoeff<F>,
        B: FnOnce(F) -> A,
    {
        from_base(Self::w()) * x
    }
}

impl<F, C> PowerBasisFp4Config<F> for C
where
    F: FieldCore,
    C: Fp2Config<F>,
{
    fn w() -> F {
        C::non_residue()
    }

    #[inline]
    fn mul_w<A, B>(x: A, from_base: B) -> A
    where
        A: ExtensionCoeff<F>,
        B: FnOnce(F) -> A,
    {
        C::mul_non_residue(x, from_base)
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

    #[inline]
    fn mul_non_residue(x: Fp2<F, C2>) -> Fp2<F, C2> {
        Fp2::new(C2::mul_non_residue(x.coeffs[1], |base| base), x.coeffs[0])
    }
}

/// Default quadratic extension used by Akita field tests and helpers.
pub type Ext2<F> = Fp2<F, TwoNr>;

/// Multiply power-basis quartic coefficient arrays over `F[v] / (v^4 - W)`.
#[inline]
pub(crate) fn power_basis_fp4_mul_coeffs<F, C, A, B>(a: [A; 4], b: [A; 4], from_base: B) -> [A; 4]
where
    F: FieldCore,
    C: PowerBasisFp4Config<F>,
    A: ExtensionCoeff<F>,
    B: Copy + Fn(F) -> A,
{
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    [
        a0 * b0 + C::mul_w(a1 * b3 + a2 * b2 + a3 * b1, from_base),
        a0 * b1 + a1 * b0 + C::mul_w(a2 * b3 + a3 * b2, from_base),
        a0 * b2 + a1 * b1 + a2 * b0 + C::mul_w(a3 * b3, from_base),
        a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0,
    ]
}

/// Multiply ring-subfield quartic coefficient arrays in `[1, e1, e2, e3]` basis.
#[inline]
pub(crate) fn ring_subfield_fp4_mul_coeffs<F, A>(a: [A; 4], b: [A; 4]) -> [A; 4]
where
    F: FieldCore,
    A: ExtensionCoeff<F>,
{
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    let tail0 = a1 * b1 + a2 * b2 + a3 * b3;
    [
        a0 * b0 + tail0 + tail0,
        a0 * b1 + a1 * b0 + a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2,
        a0 * b2 + a2 * b0 + a1 * b1 + a1 * b3 + a3 * b1 - a3 * b3,
        a0 * b3 + a3 * b0 + a1 * b2 + a2 * b1 - a2 * b3 - a3 * b2,
    ]
}

/// Backend hook for scalar power-basis quartic multiplication.
///
/// The default is the generic coefficient formula. Concrete base fields can
/// override this when their representation supports fusing product sums before
/// reduction.
pub trait PowerBasisFp4MulBackend<C>: FieldCore
where
    C: PowerBasisFp4Config<Self>,
{
    /// Multiply two power-basis coefficient arrays in `F[v] / (v^4 - W)`.
    #[inline(always)]
    fn power_basis_fp4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        power_basis_fp4_mul_coeffs::<Self, C, Self, _>(a, b, |base| base)
    }
}

impl<const P: u64, C> PowerBasisFp4MulBackend<C> for Fp64<P> where C: PowerBasisFp4Config<Self> {}
impl<const P: u128, C> PowerBasisFp4MulBackend<C> for Fp128<P> where C: PowerBasisFp4Config<Self> {}

impl<const P: u32, C> PowerBasisFp4MulBackend<C> for Fp32<P>
where
    C: PowerBasisFp4Config<Self>,
{
    #[inline(always)]
    fn power_basis_fp4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        if C::w().to_limbs() != 2 {
            return power_basis_fp4_mul_coeffs::<Self, C, Self, _>(a, b, |base| base);
        }

        #[inline(always)]
        fn product<const P: u32>(a: Fp32<P>, b: Fp32<P>) -> u128 {
            (a.to_limbs() as u128) * (b.to_limbs() as u128)
        }

        #[inline(always)]
        fn reduce<const P: u32>(x: u128) -> Fp32<P> {
            Fp32::<P>::from_canonical_u128_reduced(x)
        }

        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        [
            reduce(product(a0, b0) + 2 * (product(a1, b3) + product(a2, b2) + product(a3, b1))),
            reduce(product(a0, b1) + product(a1, b0) + 2 * (product(a2, b3) + product(a3, b2))),
            reduce(product(a0, b2) + product(a1, b1) + product(a2, b0) + 2 * product(a3, b3)),
            reduce(product(a0, b3) + product(a1, b2) + product(a2, b1) + product(a3, b0)),
        ]
    }
}

/// Backend hook for scalar ring-subfield quartic multiplication.
///
/// The default is the generic coefficient formula. Concrete base fields can
/// override this when their representation supports fusing product sums before
/// reduction.
pub trait RingSubfieldFp4MulBackend: FieldCore {
    /// Multiply two ring-subfield coefficient arrays in `[1, e1, e2, e3]` basis.
    #[inline(always)]
    fn ring_subfield_fp4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        ring_subfield_fp4_mul_coeffs::<Self, Self>(a, b)
    }
}

impl<const P: u64> RingSubfieldFp4MulBackend for Fp64<P> {}
impl<const P: u128> RingSubfieldFp4MulBackend for Fp128<P> {}

impl<const P: u32> RingSubfieldFp4MulBackend for Fp32<P> {
    #[inline(always)]
    fn ring_subfield_fp4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        #[inline(always)]
        fn product<const P: u32>(a: Fp32<P>, b: Fp32<P>) -> u128 {
            (a.to_limbs() as u128) * (b.to_limbs() as u128)
        }

        #[inline(always)]
        fn reduce<const P: u32>(x: u128) -> Fp32<P> {
            Fp32::<P>::from_canonical_u128_reduced(x)
        }

        let [a0, a1, a2, a3] = a;
        let [b0, b1, b2, b3] = b;
        let modulus_square = (P as u128) * (P as u128);
        [
            reduce(product(a0, b0) + 2 * (product(a1, b1) + product(a2, b2) + product(a3, b3))),
            reduce(
                product(a0, b1)
                    + product(a1, b0)
                    + product(a1, b2)
                    + product(a2, b1)
                    + product(a2, b3)
                    + product(a3, b2),
            ),
            reduce(
                product(a0, b2)
                    + product(a2, b0)
                    + product(a1, b1)
                    + product(a1, b3)
                    + product(a3, b1)
                    + modulus_square
                    - product(a3, b3),
            ),
            reduce(
                product(a0, b3)
                    + product(a3, b0)
                    + product(a1, b2)
                    + product(a2, b1)
                    + 2 * modulus_square
                    - product(a2, b3)
                    - product(a3, b2),
            ),
        ]
    }
}

#[inline(always)]
fn tower_basis_fp4_mul_coeffs<F, C2, C4>(a: [Fp2<F, C2>; 2], b: [Fp2<F, C2>; 2]) -> [Fp2<F, C2>; 2]
where
    F: FieldCore,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
{
    let v0 = a[0] * b[0];
    let v1 = a[1] * b[1];
    [
        v0 + C4::mul_non_residue(v1),
        (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1,
    ]
}

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
        (self.coeffs[0] * self.coeffs[0]) - C4::mul_non_residue(self.coeffs[1] * self.coeffs[1])
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
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let a0 = self.coeffs[0];
        let a1 = self.coeffs[1];
        let b0 = rhs.coeffs[0];
        let b1 = rhs.coeffs[1];
        Self::new(
            Fp2::new(a0.coeffs[0] + b0.coeffs[0], a0.coeffs[1] + b0.coeffs[1]),
            Fp2::new(a1.coeffs[0] + b1.coeffs[0], a1.coeffs[1] + b1.coeffs[1]),
        )
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Sub
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let a0 = self.coeffs[0];
        let a1 = self.coeffs[1];
        let b0 = rhs.coeffs[0];
        let b1 = rhs.coeffs[1];
        Self::new(
            Fp2::new(a0.coeffs[0] - b0.coeffs[0], a0.coeffs[1] - b0.coeffs[1]),
            Fp2::new(a1.coeffs[0] - b1.coeffs[0], a1.coeffs[1] - b1.coeffs[1]),
        )
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> Neg
    for TowerBasisFp4<F, C2, C4>
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let a0 = self.coeffs[0];
        let a1 = self.coeffs[1];
        Self::new(
            Fp2::new(-a0.coeffs[0], -a0.coeffs[1]),
            Fp2::new(-a1.coeffs[0], -a1.coeffs[1]),
        )
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> AddAssign
    for TowerBasisFp4<F, C2, C4>
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0].coeffs[0] = self.coeffs[0].coeffs[0] + rhs.coeffs[0].coeffs[0];
        self.coeffs[0].coeffs[1] = self.coeffs[0].coeffs[1] + rhs.coeffs[0].coeffs[1];
        self.coeffs[1].coeffs[0] = self.coeffs[1].coeffs[0] + rhs.coeffs[1].coeffs[0];
        self.coeffs[1].coeffs[1] = self.coeffs[1].coeffs[1] + rhs.coeffs[1].coeffs[1];
    }
}
impl<F: FieldCore, C2: Fp2Config<F>, C4: TowerBasisFp4Config<F, C2>> SubAssign
    for TowerBasisFp4<F, C2, C4>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0].coeffs[0] = self.coeffs[0].coeffs[0] - rhs.coeffs[0].coeffs[0];
        self.coeffs[0].coeffs[1] = self.coeffs[0].coeffs[1] - rhs.coeffs[0].coeffs[1];
        self.coeffs[1].coeffs[0] = self.coeffs[1].coeffs[0] - rhs.coeffs[1].coeffs[0];
        self.coeffs[1].coeffs[1] = self.coeffs[1].coeffs[1] - rhs.coeffs[1].coeffs[1];
    }
}
impl<F, C2, C4> Mul for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let nr = C4::non_residue();
        if nr.coeffs[0].is_zero() && nr.coeffs[1] == F::one() {
            let [c0, c1, c2, c3] = <F as PowerBasisFp4MulBackend<C2>>::power_basis_fp4_mul(
                [
                    self.coeffs[0].coeffs[0],
                    self.coeffs[1].coeffs[0],
                    self.coeffs[0].coeffs[1],
                    self.coeffs[1].coeffs[1],
                ],
                [
                    rhs.coeffs[0].coeffs[0],
                    rhs.coeffs[1].coeffs[0],
                    rhs.coeffs[0].coeffs[1],
                    rhs.coeffs[1].coeffs[1],
                ],
            );
            Self::new(Fp2::new(c0, c2), Fp2::new(c1, c3))
        } else {
            let [c0, c1] = tower_basis_fp4_mul_coeffs::<F, C2, C4>(self.coeffs, rhs.coeffs);
            Self::new(c0, c1)
        }
    }
}
impl<F, C2, C4> MulAssign for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
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
impl<'a, F, C2, C4> Mul<&'a Self> for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
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

impl<F, C2, C4> RingCore for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
{
    #[inline(always)]
    fn square(&self) -> Self {
        let v0 = self.coeffs[0].square();
        let v1 = self.coeffs[1].square();
        Self::new(
            v0 + C4::mul_non_residue(v1),
            (self.coeffs[0] + self.coeffs[0]) * self.coeffs[1],
        )
    }
}

impl<F, C2, C4> Invertible for TowerBasisFp4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
{
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inverse()?;
        Some(Self::new(self.coeffs[0] * inv_n, (-self.coeffs[1]) * inv_n))
    }
}

impl<F, C2, C4> HalvingField for TowerBasisFp4<F, C2, C4>
where
    F: HalvingField + Valid + PowerBasisFp4MulBackend<C2>,
    C2: Fp2Config<F>,
    C4: TowerBasisFp4Config<F, C2>,
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
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new([
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
            self.coeffs[2] + rhs.coeffs[2],
            self.coeffs[3] + rhs.coeffs[3],
        ])
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> Sub for PowerBasisFp4<F, C> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new([
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
            self.coeffs[2] - rhs.coeffs[2],
            self.coeffs[3] - rhs.coeffs[3],
        ])
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> Neg for PowerBasisFp4<F, C> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new([
            -self.coeffs[0],
            -self.coeffs[1],
            -self.coeffs[2],
            -self.coeffs[3],
        ])
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> AddAssign for PowerBasisFp4<F, C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] + rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] + rhs.coeffs[1];
        self.coeffs[2] = self.coeffs[2] + rhs.coeffs[2];
        self.coeffs[3] = self.coeffs[3] + rhs.coeffs[3];
    }
}
impl<F: FieldCore, C: PowerBasisFp4Config<F>> SubAssign for PowerBasisFp4<F, C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] - rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] - rhs.coeffs[1];
        self.coeffs[2] = self.coeffs[2] - rhs.coeffs[2];
        self.coeffs[3] = self.coeffs[3] - rhs.coeffs[3];
    }
}
impl<F: PowerBasisFp4MulBackend<C>, C: PowerBasisFp4Config<F>> Mul for PowerBasisFp4<F, C> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(F::power_basis_fp4_mul(self.coeffs, rhs.coeffs))
    }
}
impl<F: PowerBasisFp4MulBackend<C>, C: PowerBasisFp4Config<F>> MulAssign for PowerBasisFp4<F, C> {
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
impl<'a, F: PowerBasisFp4MulBackend<C>, C: PowerBasisFp4Config<F>> Mul<&'a Self>
    for PowerBasisFp4<F, C>
{
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

impl<F, C> RingCore for PowerBasisFp4<F, C>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C>,
    C: PowerBasisFp4Config<F>,
{
    #[inline(always)]
    fn square(&self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let two = F::one() + F::one();
        let a0a1 = a0 * a1;
        let a0a2 = a0 * a2;
        let a0a3 = a0 * a3;
        let a1a2 = a1 * a2;
        let a1a3 = a1 * a3;
        let a2a3 = a2 * a3;
        Self::new([
            a0.square() + C::mul_w(two * a1a3 + a2.square(), |base| base),
            two * (a0a1 + C::mul_w(a2a3, |base| base)),
            two * a0a2 + a1.square() + C::mul_w(a3.square(), |base| base),
            two * (a0a3 + a1a2),
        ])
    }
}

impl<F, C> Invertible for PowerBasisFp4<F, C>
where
    F: FieldCore + Valid + PowerBasisFp4MulBackend<C>,
    C: PowerBasisFp4Config<F>,
{
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let [a0, a1, a2, a3] = self.coeffs;
        let two = F::one() + F::one();

        let d0 = a0.square() + C::mul_w(a2.square(), |base| base)
            - C::mul_w(two * (a1 * a3), |base| base);
        let d1 = two * (a0 * a2) - a1.square() - C::mul_w(a3.square(), |base| base);
        let inv_norm = (d0.square() - C::mul_w(d1.square(), |base| base)).inverse()?;
        let e0 = d0 * inv_norm;
        let e1 = -d1 * inv_norm;

        Some(Self::new([
            a0 * e0 + C::mul_w(a2 * e1, |base| base),
            -(a1 * e0 + C::mul_w(a3 * e1, |base| base)),
            a0 * e1 + a2 * e0,
            -(a1 * e1 + a3 * e0),
        ]))
    }
}

impl<F, C> HalvingField for PowerBasisFp4<F, C>
where
    F: HalvingField + Valid + PowerBasisFp4MulBackend<C>,
    C: PowerBasisFp4Config<F>,
{
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

/// Quartic fixed-subfield element in the Akita cyclotomic basis.
///
/// Coordinates are `[c0, c1, c2, c3]` in basis `[1, e1, e2, e3]`, where
/// `e_j = zeta^(jm) + zeta^(-jm)` for `m = D / 8` inside a compatible
/// cyclotomic ring. The scalar arithmetic is independent of the concrete ring
/// dimension `D`.
#[repr(transparent)]
pub struct RingSubfieldFp4<F: FieldCore> {
    /// Coefficients in basis `[1, e1, e2, e3]`.
    pub coeffs: [F; 4],
}

impl<F: FieldCore> RingSubfieldFp4<F> {
    /// Construct from ring-subfield basis coefficients `[c0, c1, c2, c3]`.
    #[inline]
    pub fn new(coeffs: [F; 4]) -> Self {
        Self { coeffs }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new([F::zero(); 4])
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

    #[inline(always)]
    fn fp2_mul_by_e2_nr(lhs: (F, F), rhs: (F, F)) -> (F, F) {
        let (a0, a1) = lhs;
        let (b0, b1) = rhs;
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let c1 = (a0 + a1) * (b0 + b1) - v0 - v1;
        let c0 = v0 + v1 + v1;
        (c0, c1)
    }

    #[inline(always)]
    fn fp2_square_by_e2_nr(x: (F, F)) -> (F, F) {
        let (a0, a1) = x;
        let a0a1 = a0 * a1;
        (a0.square() + a1.square() + a1.square(), a0a1 + a0a1)
    }

    #[inline(always)]
    fn fp2_mul_by_e1_nr(x: (F, F)) -> (F, F) {
        let (x0, x1) = x;
        (x0 + x0 + x1 + x1, x0 + x1 + x1)
    }

    #[inline(always)]
    fn fp2_inverse_by_e2_nr(x: (F, F)) -> Option<(F, F)> {
        let (x0, x1) = x;
        let inv_norm = (x0.square() - (x1.square() + x1.square())).inverse()?;
        Some((x0 * inv_norm, -x1 * inv_norm))
    }
}

impl<F: FieldCore + std::fmt::Debug> std::fmt::Debug for RingSubfieldFp4<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingSubfieldFp4")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}

impl<F: FieldCore> Clone for RingSubfieldFp4<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore> Copy for RingSubfieldFp4<F> {}

impl<F: FieldCore> Default for RingSubfieldFp4<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldCore> PartialEq for RingSubfieldFp4<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<F: FieldCore> Eq for RingSubfieldFp4<F> {}

impl<F: FieldCore> Add for RingSubfieldFp4<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new([
            self.coeffs[0] + rhs.coeffs[0],
            self.coeffs[1] + rhs.coeffs[1],
            self.coeffs[2] + rhs.coeffs[2],
            self.coeffs[3] + rhs.coeffs[3],
        ])
    }
}

impl<F: FieldCore> Sub for RingSubfieldFp4<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new([
            self.coeffs[0] - rhs.coeffs[0],
            self.coeffs[1] - rhs.coeffs[1],
            self.coeffs[2] - rhs.coeffs[2],
            self.coeffs[3] - rhs.coeffs[3],
        ])
    }
}

impl<F: FieldCore> Neg for RingSubfieldFp4<F> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new([
            -self.coeffs[0],
            -self.coeffs[1],
            -self.coeffs[2],
            -self.coeffs[3],
        ])
    }
}

impl<F: FieldCore> AddAssign for RingSubfieldFp4<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] + rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] + rhs.coeffs[1];
        self.coeffs[2] = self.coeffs[2] + rhs.coeffs[2];
        self.coeffs[3] = self.coeffs[3] + rhs.coeffs[3];
    }
}

impl<F: FieldCore> SubAssign for RingSubfieldFp4<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0] = self.coeffs[0] - rhs.coeffs[0];
        self.coeffs[1] = self.coeffs[1] - rhs.coeffs[1];
        self.coeffs[2] = self.coeffs[2] - rhs.coeffs[2];
        self.coeffs[3] = self.coeffs[3] - rhs.coeffs[3];
    }
}

impl<F: RingSubfieldFp4MulBackend> Mul for RingSubfieldFp4<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(F::ring_subfield_fp4_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F: RingSubfieldFp4MulBackend> MulAssign for RingSubfieldFp4<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore> Add<&'a Self> for RingSubfieldFp4<F> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, F: FieldCore> Sub<&'a Self> for RingSubfieldFp4<F> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, F: RingSubfieldFp4MulBackend> Mul<&'a Self> for RingSubfieldFp4<F> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid> Valid for RingSubfieldFp4<F> {
    fn check(&self) -> Result<(), SerializationError> {
        for coeff in self.coeffs {
            coeff.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for RingSubfieldFp4<F> {
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

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for RingSubfieldFp4<F>
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

impl<F: FieldCore + Valid + RingSubfieldFp4MulBackend> RingCore for RingSubfieldFp4<F> {
    #[inline(always)]
    fn square(&self) -> Self {
        let [a0, a1, a2, a3] = self.coeffs;
        let a = (a0, a2);
        let b = (a1 - a3, a3);
        let aa = Self::fp2_square_by_e2_nr(a);
        let bb = Self::fp2_square_by_e2_nr(b);
        let ab = Self::fp2_mul_by_e2_nr(a, b);
        let constant = Self::fp2_mul_by_e1_nr(bb);
        let coeff_e1 = (ab.0 + ab.0, ab.1 + ab.1);
        Self::new([
            aa.0 + constant.0,
            coeff_e1.0 + coeff_e1.1,
            aa.1 + constant.1,
            coeff_e1.1,
        ])
    }
}

impl<F: FieldCore + Valid + RingSubfieldFp4MulBackend> Invertible for RingSubfieldFp4<F> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let [a0, a1, a2, a3] = self.coeffs;
        let a = (a0, a2);
        let b = (a1 - a3, a3);

        let aa = Self::fp2_square_by_e2_nr(a);
        let bb = Self::fp2_square_by_e2_nr(b);
        let norm = {
            let nr_bb = Self::fp2_mul_by_e1_nr(bb);
            (aa.0 - nr_bb.0, aa.1 - nr_bb.1)
        };
        let inv_norm = Self::fp2_inverse_by_e2_nr(norm)?;
        let constant = Self::fp2_mul_by_e2_nr(a, inv_norm);
        let e1_coeff = Self::fp2_mul_by_e2_nr((-b.0, -b.1), inv_norm);

        Some(Self::new([
            constant.0,
            e1_coeff.0 + e1_coeff.1,
            constant.1,
            e1_coeff.1,
        ]))
    }
}

impl<F: HalvingField + Valid + RingSubfieldFp4MulBackend> HalvingField for RingSubfieldFp4<F> {
    #[inline]
    fn half(self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i].half()))
    }
}

impl<F: FieldCore + RandomSampling + Valid> RandomSampling for RingSubfieldFp4<F> {
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new([
            F::random(rng),
            F::random(rng),
            F::random(rng),
            F::random(rng),
        ])
    }
}

impl<F: FieldCore + FromPrimitiveInt + Valid> FromPrimitiveInt for RingSubfieldFp4<F> {
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

impl<F: FieldCore + BalancedDigitLookup + Valid> BalancedDigitLookup for RingSubfieldFp4<F> {}

impl<F> HasUnreducedOps for RingSubfieldFp4<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + RingSubfieldFp4MulBackend,
{
    type MulU64Accum = Self;
    type ProductAccum = Self;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum {
        let small = F::from_u64(small);
        Self::new(self.coeffs.map(|coeff| coeff * small))
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Self::ProductAccum {
        self * other
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self {
        accum
    }

    #[inline]
    fn reduce_product_accum(accum: Self::ProductAccum) -> Self {
        accum
    }
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use crate::fields::lift::{
        canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
        ExtField, FrobeniusExtField,
    };
    use crate::Fp64;
    use crate::{FromPrimitiveInt, Invertible};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;
    type E2 = Ext2<F>;
    type E4 = TowerBasisFp4<F, TwoNr, UnitNr>;
    type P4 = PowerBasisFp4<F, TwoNr>;
    type R4 = RingSubfieldFp4<F>;

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
    fn ring_subfield_fp4_multiplication_table() {
        let two = F::from_u64(2);
        let e1 = R4::new([F::zero(), F::one(), F::zero(), F::zero()]);
        let e2 = R4::new([F::zero(), F::zero(), F::one(), F::zero()]);
        let e3 = R4::new([F::zero(), F::zero(), F::zero(), F::one()]);
        let two_const = R4::new([two, F::zero(), F::zero(), F::zero()]);

        assert_eq!(e1 * e1, two_const + e2);
        assert_eq!(e1 * e2, e1 + e3);
        assert_eq!(e1 * e3, e2);
        assert_eq!(e2 * e2, two_const);
        assert_eq!(e2 * e3, e1 - e3);
        assert_eq!(e3 * e3, two_const - e2);
    }

    #[test]
    fn ring_subfield_fp4_square_matches_mul() {
        let mut rng = StdRng::seed_from_u64(5555);
        for _ in 0..50 {
            let a = R4::random(&mut rng);
            assert_eq!(a.square(), a * a);
        }
    }

    #[test]
    fn ring_subfield_fp4_inv() {
        let mut rng = StdRng::seed_from_u64(6666);
        for _ in 0..50 {
            let a = R4::random(&mut rng);
            if !a.is_zero() {
                let inv = a.inverse().unwrap();
                assert_eq!(a * inv, R4::one());
            }
        }
    }

    #[test]
    fn frobenius_fp2_is_conjugation() {
        let x = E2::new(F::from_u64(13), F::from_u64(21));
        assert_eq!(<E2 as FrobeniusExtField<F>>::frobenius_pow(x, 0), x);
        assert_eq!(
            <E2 as FrobeniusExtField<F>>::frobenius_pow(x, 1),
            x.conjugate()
        );
        assert_eq!(<E2 as FrobeniusExtField<F>>::frobenius_pow(x, 2), x);
        assert_eq!(
            <E2 as FrobeniusExtField<F>>::frobenius_inv_pow(x, 1),
            x.conjugate()
        );
    }

    #[test]
    fn canonical_moore_thetas_solve_fp2() {
        validate_canonical_frobenius_thetas::<F, E2>(2).unwrap();
        let thetas = canonical_frobenius_thetas::<F, E2>(2).unwrap();
        let z = [
            E2::new(F::from_u64(3), F::from_u64(5)),
            E2::new(F::from_u64(7), F::from_u64(11)),
        ];
        let r = (0..2)
            .map(|row| {
                thetas
                    .iter()
                    .zip(z.iter())
                    .fold(E2::zero(), |acc, (&theta, &z_h)| {
                        acc + <E2 as FrobeniusExtField<F>>::frobenius_inv_pow(theta, row) * z_h
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            solve_frobenius_moore::<F, E2>(&thetas, &r).unwrap(),
            z.to_vec()
        );
    }

    #[test]
    fn canonical_ring_subfield_thetas_are_the_packing_basis() {
        let thetas = canonical_frobenius_thetas::<F, R4>(4).unwrap();
        assert_eq!(
            thetas[0],
            R4::new([F::one(), F::zero(), F::zero(), F::zero()])
        );
        assert_eq!(
            thetas[1],
            R4::new([F::zero(), F::one(), F::zero(), F::zero()])
        );
        assert_eq!(
            thetas[2],
            R4::new([F::zero(), F::zero(), F::one(), F::zero()])
        );
        assert_eq!(
            thetas[3],
            R4::new([F::zero(), F::zero(), F::zero(), F::one()])
        );
        validate_canonical_frobenius_thetas::<F, R4>(4).unwrap();
    }

    #[test]
    fn duplicate_moore_theta_rejects() {
        let theta = E2::one();
        let err = solve_frobenius_moore::<F, E2>(&[theta, theta], &[E2::one(), E2::one()])
            .expect_err("duplicate theta should be singular");
        assert!(format!("{err}").contains("singular"));
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
        assert_eq!(<R4 as ExtField<F>>::EXT_DEGREE, 4);
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

        let r4 = R4::from_base_slice(&[c0, c1, c2, c3]);
        assert_eq!(r4, R4::new([c0, c1, c2, c3]));
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
        assert_eq!(core::mem::size_of::<R4>(), core::mem::size_of::<[F; 4]>());
        assert_eq!(core::mem::align_of::<R4>(), core::mem::align_of::<[F; 4]>());
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
