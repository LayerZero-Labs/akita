use super::*;

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
        let v0 = self.coeffs[0] * rhs.coeffs[0];
        let v1 = self.coeffs[1] * rhs.coeffs[1];
        let cross = (self.coeffs[0] + self.coeffs[1]) * (rhs.coeffs[0] + rhs.coeffs[1]);
        Self::new(v0 + Self::mul_nr(v1), cross - v0 - v1)
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

/// Identity-stub `HasUnreducedOps` for `Fp2` variants without a dedicated
/// delayed-reduction accumulator. `ProductAccum = Self`, so every multiply
/// reduces immediately. Same pattern as `RingSubfieldFp4<Fp64/Fp128>` and
/// `RingSubfieldFp8<*>`.
macro_rules! impl_fp2_unreduced_identity {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty, C: Fp2Config<$base<$p>>> HasUnreducedOps for Fp2<$base<$p>, C> {
            type MulU64Accum = Self;
            type ProductAccum = Self;

            #[inline]
            fn mul_u64_unreduced(self, small: u64) -> Self {
                self * Self::from_u64(small)
            }
            #[inline]
            fn mul_to_product_accum(self, other: Self) -> Self {
                self * other
            }
            #[inline]
            fn reduce_mul_u64_accum(accum: Self) -> Self {
                accum
            }
            #[inline]
            fn reduce_product_accum(accum: Self) -> Self {
                accum
            }
        }
    };
}

impl_fp2_unreduced_identity!(Fp16<P: u32>);
impl_fp2_unreduced_identity!(Fp32<P: u32>);
impl_fp2_unreduced_identity!(Fp128<P: u128>);

macro_rules! impl_fp2_default_optimized_fold {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty, C: Fp2Config<$base<$p>>> HasOptimizedFold for Fp2<$base<$p>, C> {
            type FoldCtx = Self;
            #[inline]
            fn precompute_fold(r: Self) -> Self {
                r
            }
            #[inline]
            fn fold_one(r: &Self, even: Self, odd: Self) -> Self {
                even + *r * (odd - even)
            }
        }
    };
}

impl_fp2_default_optimized_fold!(Fp16<P: u32>);
impl_fp2_default_optimized_fold!(Fp32<P: u32>);
impl_fp2_default_optimized_fold!(Fp64<P: u64>);
impl_fp2_default_optimized_fold!(Fp128<P: u128>);

/// Widening `Fp2<Fp64<P>, C>` multiplication with delayed reduction.
///
/// Each base-field product is split into (lo64, hi64) halves and stored in
/// `Fp64ProductAccum` slots. For `IS_NEG_ONE` configs the `P^2` bias keeps
/// the constant coefficient non-negative.
#[inline(always)]
pub(crate) fn fp2_mul_to_accum_fp64<const P: u64, C: Fp2Config<Fp64<P>>>(
    a: [Fp64<P>; 2],
    b: [Fp64<P>; 2],
) -> Fp2Fp64ProductAccum {
    let p00: u128 = a[0].mul_wide(b[0]);
    let p11 = a[1].mul_wide(b[1]);
    let p01 = a[0].mul_wide(b[1]);
    let p10 = a[1].mul_wide(b[0]);

    let c0_wide = if C::IS_NEG_ONE {
        let modulus_sq = (P as u128) * (P as u128);
        p00.wrapping_add(modulus_sq).wrapping_sub(p11)
    } else {
        p00.wrapping_add(p11).wrapping_add(p11)
    };
    let c1_wide = p01.wrapping_add(p10);

    Fp2Fp64ProductAccum([
        c0_wide as u64 as u128,
        (c0_wide >> 64) as u64 as u128,
        c1_wide as u64 as u128,
        (c1_wide >> 64) as u64 as u128,
    ])
}

impl<const P: u64, C: Fp2Config<Fp64<P>>> HasUnreducedOps for Fp2<Fp64<P>, C> {
    type MulU64Accum = AccumPair<<Fp64<P> as HasUnreducedOps>::MulU64Accum>;
    type ProductAccum = Fp2Fp64ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum {
        AccumPair(
            self.coeffs[0].mul_u64_unreduced(small),
            self.coeffs[1].mul_u64_unreduced(small),
        )
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp2Fp64ProductAccum {
        fp2_mul_to_accum_fp64::<P, C>(self.coeffs, other.coeffs)
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self {
        Self::new(
            Fp64::<P>::reduce_mul_u64_accum(accum.0),
            Fp64::<P>::reduce_mul_u64_accum(accum.1),
        )
    }

    #[inline]
    fn reduce_product_accum(accum: Fp2Fp64ProductAccum) -> Self {
        let [c0, c1] = accum.reduce::<P>();
        Self::new(c0, c1)
    }
}

/// Default quadratic extension used by Akita field tests and helpers.
pub type Ext2<F> = Fp2<F, TwoNr>;
