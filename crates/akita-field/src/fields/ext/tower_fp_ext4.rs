use super::*;

/// Parameters for a tower-basis quartic extension over `FpExt2<F, C2>`.
pub trait TowerBasisFpExt4Config<F: FieldCore, C2: FpExt2Config<F>> {
    /// Non-residue `NR2` in `FpExt2` such that `v^2 = NR2`.
    fn non_residue() -> FpExt2<F, C2>;

    /// Multiply an `FpExt2` element by the tower non-residue.
    #[inline]
    fn mul_non_residue(x: FpExt2<F, C2>) -> FpExt2<F, C2> {
        Self::non_residue() * x
    }
}

/// `TowerBasisFpExt4Config` with non-residue `u ∈ FpExt2` (the element `(0, 1)`).
///
/// This is the standard tower choice: `v^2 = u`, hence `v^4 = NR`.
pub struct UnitNr;

impl<F: FieldCore, C2: FpExt2Config<F>> TowerBasisFpExt4Config<F, C2> for UnitNr {
    fn non_residue() -> FpExt2<F, C2> {
        FpExt2::new(F::zero(), F::one())
    }

    #[inline]
    fn mul_non_residue(x: FpExt2<F, C2>) -> FpExt2<F, C2> {
        FpExt2::new(C2::mul_non_residue(x.coeffs[1], |base| base), x.coeffs[0])
    }
}

#[inline(always)]
fn tower_basis_fp_ext4_mul_coeffs<F, C2, C4>(
    a: [FpExt2<F, C2>; 2],
    b: [FpExt2<F, C2>; 2],
) -> [FpExt2<F, C2>; 2]
where
    F: FieldCore,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    let v0 = a[0] * b[0];
    let v1 = a[1] * b[1];
    [
        v0 + C4::mul_non_residue(v1),
        (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1,
    ]
}

/// Quartic extension element `b0 + b1 * v` over `FpExt2`, where `v^2 = NR2`.
#[repr(transparent)]
pub struct TowerBasisFpExt4<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> {
    /// Coefficients `[b0, b1]` in tower basis `[1, v]` over `FpExt2`.
    pub coeffs: [FpExt2<F, C2>; 2],
    _cfg: PhantomData<fn() -> C4>,
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>>
    TowerBasisFpExt4<F, C2, C4>
{
    /// Construct `c0 + c1 * v`.
    #[inline]
    pub fn new(c0: FpExt2<F, C2>, c1: FpExt2<F, C2>) -> Self {
        Self {
            coeffs: [c0, c1],
            _cfg: PhantomData,
        }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new(FpExt2::zero(), FpExt2::zero())
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new(FpExt2::one(), FpExt2::zero())
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
        Self::new(FpExt2::from_u64(val), FpExt2::zero())
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(FpExt2::from_i64(val), FpExt2::zero())
    }

    /// Return the norm in `FpExt2`: `c0^2 - NR2 * c1^2`.
    #[inline]
    pub fn norm(self) -> FpExt2<F, C2> {
        (self.coeffs[0] * self.coeffs[0]) - C4::mul_non_residue(self.coeffs[1] * self.coeffs[1])
    }
}

impl<F: FieldCore + std::fmt::Debug, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>>
    std::fmt::Debug for TowerBasisFpExt4<F, C2, C4>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TowerBasisFpExt4")
            .field("c0", &self.coeffs[0])
            .field("c1", &self.coeffs[1])
            .finish()
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Clone
    for TowerBasisFpExt4<F, C2, C4>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Copy
    for TowerBasisFpExt4<F, C2, C4>
{
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Default
    for TowerBasisFpExt4<F, C2, C4>
{
    fn default() -> Self {
        Self::new(
            FpExt2::new(F::zero(), F::zero()),
            FpExt2::new(F::zero(), F::zero()),
        )
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> PartialEq
    for TowerBasisFpExt4<F, C2, C4>
{
    fn eq(&self, other: &Self) -> bool {
        self.coeffs[0] == other.coeffs[0] && self.coeffs[1] == other.coeffs[1]
    }
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Eq
    for TowerBasisFpExt4<F, C2, C4>
{
}

impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Add
    for TowerBasisFpExt4<F, C2, C4>
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let a0 = self.coeffs[0];
        let a1 = self.coeffs[1];
        let b0 = rhs.coeffs[0];
        let b1 = rhs.coeffs[1];
        Self::new(
            FpExt2::new(a0.coeffs[0] + b0.coeffs[0], a0.coeffs[1] + b0.coeffs[1]),
            FpExt2::new(a1.coeffs[0] + b1.coeffs[0], a1.coeffs[1] + b1.coeffs[1]),
        )
    }
}
impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Sub
    for TowerBasisFpExt4<F, C2, C4>
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let a0 = self.coeffs[0];
        let a1 = self.coeffs[1];
        let b0 = rhs.coeffs[0];
        let b1 = rhs.coeffs[1];
        Self::new(
            FpExt2::new(a0.coeffs[0] - b0.coeffs[0], a0.coeffs[1] - b0.coeffs[1]),
            FpExt2::new(a1.coeffs[0] - b1.coeffs[0], a1.coeffs[1] - b1.coeffs[1]),
        )
    }
}
impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Neg
    for TowerBasisFpExt4<F, C2, C4>
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let a0 = self.coeffs[0];
        let a1 = self.coeffs[1];
        Self::new(
            FpExt2::new(-a0.coeffs[0], -a0.coeffs[1]),
            FpExt2::new(-a1.coeffs[0], -a1.coeffs[1]),
        )
    }
}
impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> AddAssign
    for TowerBasisFpExt4<F, C2, C4>
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coeffs[0].coeffs[0] = self.coeffs[0].coeffs[0] + rhs.coeffs[0].coeffs[0];
        self.coeffs[0].coeffs[1] = self.coeffs[0].coeffs[1] + rhs.coeffs[0].coeffs[1];
        self.coeffs[1].coeffs[0] = self.coeffs[1].coeffs[0] + rhs.coeffs[1].coeffs[0];
        self.coeffs[1].coeffs[1] = self.coeffs[1].coeffs[1] + rhs.coeffs[1].coeffs[1];
    }
}
impl<F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> SubAssign
    for TowerBasisFpExt4<F, C2, C4>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coeffs[0].coeffs[0] = self.coeffs[0].coeffs[0] - rhs.coeffs[0].coeffs[0];
        self.coeffs[0].coeffs[1] = self.coeffs[0].coeffs[1] - rhs.coeffs[0].coeffs[1];
        self.coeffs[1].coeffs[0] = self.coeffs[1].coeffs[0] - rhs.coeffs[1].coeffs[0];
        self.coeffs[1].coeffs[1] = self.coeffs[1].coeffs[1] - rhs.coeffs[1].coeffs[1];
    }
}
impl<F, C2, C4> Mul for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let nr = C4::non_residue();
        if nr.coeffs[0].is_zero() && nr.coeffs[1] == F::one() {
            let [c0, c1, c2, c3] = <F as PowerBasisFpExt4MulBackend<C2>>::power_basis_fp_ext4_mul(
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
            Self::new(FpExt2::new(c0, c2), FpExt2::new(c1, c3))
        } else {
            let [c0, c1] = tower_basis_fp_ext4_mul_coeffs::<F, C2, C4>(self.coeffs, rhs.coeffs);
            Self::new(c0, c1)
        }
    }
}
impl<F, C2, C4> MulAssign for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Add<&'a Self>
    for TowerBasisFpExt4<F, C2, C4>
{
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}
impl<'a, F: FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Sub<&'a Self>
    for TowerBasisFpExt4<F, C2, C4>
{
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}
impl<'a, F, C2, C4> Mul<&'a Self> for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>> Valid
    for TowerBasisFpExt4<F, C2, C4>
{
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs[0].check()?;
        self.coeffs[1].check()?;
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>>
    AkitaSerialize for TowerBasisFpExt4<F, C2, C4>
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
        C2: FpExt2Config<F>,
        C4: TowerBasisFpExt4Config<F, C2>,
    > AkitaDeserialize for TowerBasisFpExt4<F, C2, C4>
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let c0 = FpExt2::<F, C2>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let c1 = FpExt2::<F, C2>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self::new(c0, c1);
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F, C2, C4> RingCore for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
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

impl<F, C2, C4> Invertible for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let inv_n = self.norm().inverse()?;
        Some(Self::new(self.coeffs[0] * inv_n, (-self.coeffs[1]) * inv_n))
    }
}

impl<F, C2, C4> HalvingField for TowerBasisFpExt4<F, C2, C4>
where
    F: HalvingField + Valid + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
    #[inline]
    fn half(self) -> Self {
        Self::new(self.coeffs[0].half(), self.coeffs[1].half())
    }
}

impl<
        F: FieldCore + RandomSampling + Valid,
        C2: FpExt2Config<F>,
        C4: TowerBasisFpExt4Config<F, C2>,
    > RandomSampling for TowerBasisFpExt4<F, C2, C4>
{
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(FpExt2::random(rng), FpExt2::random(rng))
    }
}

impl<
        F: FieldCore + FromPrimitiveInt + Valid,
        C2: FpExt2Config<F>,
        C4: TowerBasisFpExt4Config<F, C2>,
    > FromPrimitiveInt for TowerBasisFpExt4<F, C2, C4>
{
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new(FpExt2::from_u128(val), FpExt2::zero())
    }

    fn from_i128(val: i128) -> Self {
        Self::new(FpExt2::from_i128(val), FpExt2::zero())
    }
}

impl<
        F: FieldCore + BalancedDigitLookup + Valid,
        C2: FpExt2Config<F>,
        C4: TowerBasisFpExt4Config<F, C2>,
    > BalancedDigitLookup for TowerBasisFpExt4<F, C2, C4>
{
}

/// Identity-stub `HasUnreducedOps`: `ProductAccum = Self`, so every multiply
/// reduces immediately. Same pattern as the `FpExt2` / `RingSubfieldFpExt8` stubs.
impl<F, C2, C4> HasUnreducedOps for TowerBasisFpExt4<F, C2, C4>
where
    F: FieldCore + Valid + FromPrimitiveInt + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
    C4: TowerBasisFpExt4Config<F, C2>,
{
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

impl<F, C2> MulBaseUnreduced<F> for TowerBasisFpExt4<F, C2, UnitNr>
where
    F: FieldCore + Valid + FromPrimitiveInt + PowerBasisFpExt4MulBackend<C2>,
    C2: FpExt2Config<F>,
{
}
