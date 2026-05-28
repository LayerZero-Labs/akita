use super::*;

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
