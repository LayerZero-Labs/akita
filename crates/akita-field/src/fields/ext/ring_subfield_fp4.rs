use super::*;

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
