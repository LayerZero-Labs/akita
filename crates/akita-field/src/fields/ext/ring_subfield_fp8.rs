use super::*;

#[inline(always)]
fn ring_subfield_fp8_add_phi<F: FieldCore>(out: &mut [F; 8], idx: usize, value: F) {
    match idx {
        0 => out[0] += value + value,
        1..=7 => out[idx] += value,
        8 => {}
        9..=15 => out[16 - idx] -= value,
        _ => unreachable!("fp8 Chebyshev index out of range"),
    }
}

#[inline(always)]
fn ring_subfield_fp8_mul_coeffs<F: FieldCore>(a: [F; 8], b: [F; 8]) -> [F; 8] {
    let mut out = [F::zero(); 8];

    let diag = std::array::from_fn::<_, 8, _>(|i| a[i] * b[i]);
    out[0] += diag[0];

    for k in 1..8 {
        let mixed = (a[0] + a[k]) * (b[0] + b[k]) - diag[0] - diag[k];
        out[k] += mixed;
    }

    for (i, diag_i) in diag.iter().copied().enumerate().skip(1) {
        out[0] += diag_i + diag_i;
        ring_subfield_fp8_add_phi(&mut out, i + i, diag_i);
    }

    for i in 1..8 {
        for j in (i + 1)..8 {
            let mixed = (a[i] + a[j]) * (b[i] + b[j]) - diag[i] - diag[j];
            ring_subfield_fp8_add_phi(&mut out, i + j, mixed);
            ring_subfield_fp8_add_phi(&mut out, j - i, mixed);
        }
    }

    out
}

/// Backend hook for scalar ring-subfield degree-8 multiplication.
pub trait RingSubfieldFp8MulBackend: FieldCore {
    /// Multiply coefficient arrays in `[1, e1, ..., e7]` basis.
    #[inline(always)]
    fn ring_subfield_fp8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        ring_subfield_fp8_mul_coeffs::<Self>(a, b)
    }
}

impl<const P: u32> RingSubfieldFp8MulBackend for Fp16<P> {}
impl<const P: u32> RingSubfieldFp8MulBackend for Fp32<P> {}
impl<const P: u64> RingSubfieldFp8MulBackend for Fp64<P> {}
impl<const P: u128> RingSubfieldFp8MulBackend for Fp128<P> {}

/// Widening `RingSubfieldFp8<Fp16<P>>` multiplication that defers the
/// per-coefficient Solinas reduction, returning a
/// [`RingSubfieldFp8Fp16ProductAccum`] instead of canonical coefficients.
///
/// This expands `ring_subfield_fp8_mul_coeffs` term-for-term in basis
/// `[1, e1, ..., e7]` (using `e_i·e_j = e_{i+j} + e_{|i-j|}`, `e_0 = 2`,
/// `e_8 = 0`, `e_{8+t} = -e_{8-t}`), but keeps each base product
/// `a_i·b_j < P² < 2^32` in a `u64` slot. Where the ring formula subtracts `k`
/// base products, an offset `k · P²` is added first so the slot stays
/// non-negative; since `P² ≡ 0 (mod P)` the offsets do not change the reduced
/// value. The φ(X) ring reduction is fully fused into the formulas — only the
/// base-field modular reduction is deferred to
/// [`RingSubfieldFp8Fp16ProductAccum::reduce`].
#[inline(always)]
pub(crate) fn ring_subfield_fp8_mul_to_accum_fp16<const P: u32>(
    a: [Fp16<P>; 8],
    b: [Fp16<P>; 8],
) -> RingSubfieldFp8Fp16ProductAccum {
    let p = |i: usize, j: usize| -> u64 { a[i].mul_wide(b[j]) as u64 };
    let ms = (P as u64) * (P as u64);
    // Each slot stays `< 15·P² < 2^36`, so it is computed exactly in `u64`; it is
    // then widened to the accumulator's `u128` slots, whose batch-summation
    // headroom must cover the dense EOR round's `half`-sized accumulation.
    let slots: [u64; 8] = [
        // out[0] = p00 + 2·(p11 + p22 + ... + p77)
        p(0, 0) + 2 * (p(1, 1) + p(2, 2) + p(3, 3) + p(4, 4) + p(5, 5) + p(6, 6) + p(7, 7)),
        // out[1] = (p01+p10) + m12 + m23 + m34 + m45 + m56 + m67
        p(0, 1)
            + p(1, 0)
            + (p(1, 2) + p(2, 1))
            + (p(2, 3) + p(3, 2))
            + (p(3, 4) + p(4, 3))
            + (p(4, 5) + p(5, 4))
            + (p(5, 6) + p(6, 5))
            + (p(6, 7) + p(7, 6)),
        // out[2] = (p02+p20) + p11 + m13 + m24 + m35 + m46 + m57 + P² − p77
        p(0, 2)
            + p(2, 0)
            + p(1, 1)
            + (p(1, 3) + p(3, 1))
            + (p(2, 4) + p(4, 2))
            + (p(3, 5) + p(5, 3))
            + (p(4, 6) + p(6, 4))
            + (p(5, 7) + p(7, 5))
            + ms
            - p(7, 7),
        // out[3] = (p03+p30) + m12 + m14 + m25 + m36 + m47 + 2·P² − m67
        p(0, 3)
            + p(3, 0)
            + (p(1, 2) + p(2, 1))
            + (p(1, 4) + p(4, 1))
            + (p(2, 5) + p(5, 2))
            + (p(3, 6) + p(6, 3))
            + (p(4, 7) + p(7, 4))
            + 2 * ms
            - (p(6, 7) + p(7, 6)),
        // out[4] = (p04+p40) + p22 + m13 + m15 + m26 + m37 + 3·P² − p66 − m57
        p(0, 4)
            + p(4, 0)
            + p(2, 2)
            + (p(1, 3) + p(3, 1))
            + (p(1, 5) + p(5, 1))
            + (p(2, 6) + p(6, 2))
            + (p(3, 7) + p(7, 3))
            + 3 * ms
            - p(6, 6)
            - (p(5, 7) + p(7, 5)),
        // out[5] = (p05+p50) + m14 + m23 + m16 + m27 + 4·P² − m47 − m56
        p(0, 5)
            + p(5, 0)
            + (p(1, 4) + p(4, 1))
            + (p(2, 3) + p(3, 2))
            + (p(1, 6) + p(6, 1))
            + (p(2, 7) + p(7, 2))
            + 4 * ms
            - (p(4, 7) + p(7, 4))
            - (p(5, 6) + p(6, 5)),
        // out[6] = (p06+p60) + p33 + m15 + m24 + m17 + 5·P² − p55 − m37 − m46
        p(0, 6)
            + p(6, 0)
            + p(3, 3)
            + (p(1, 5) + p(5, 1))
            + (p(2, 4) + p(4, 2))
            + (p(1, 7) + p(7, 1))
            + 5 * ms
            - p(5, 5)
            - (p(3, 7) + p(7, 3))
            - (p(4, 6) + p(6, 4)),
        // out[7] = (p07+p70) + m16 + m25 + m34 + 6·P² − m27 − m36 − m45
        p(0, 7)
            + p(7, 0)
            + (p(1, 6) + p(6, 1))
            + (p(2, 5) + p(5, 2))
            + (p(3, 4) + p(4, 3))
            + 6 * ms
            - (p(2, 7) + p(7, 2))
            - (p(3, 6) + p(6, 3))
            - (p(4, 5) + p(5, 4)),
    ];
    RingSubfieldFp8Fp16ProductAccum(slots.map(u128::from))
}

/// Degree-8 ring subfield element in canonical basis `[1, e1, ..., e7]`.
#[repr(transparent)]
pub struct RingSubfieldFp8<F: FieldCore> {
    /// Coefficients in basis `[1, e1, ..., e7]`.
    pub coeffs: [F; 8],
}

impl<F: FieldCore> RingSubfieldFp8<F> {
    /// Construct from canonical ring-subfield basis coefficients.
    #[inline]
    pub fn new(coeffs: [F; 8]) -> Self {
        Self { coeffs }
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self::new([F::zero(); 8])
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::one()
            } else {
                F::zero()
            }
        }))
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
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_u64(val)
            } else {
                F::zero()
            }
        }))
    }

    /// Construct from an `i64` embedded in the base field.
    #[inline]
    pub fn from_i64(val: i64) -> Self
    where
        F: FromPrimitiveInt,
    {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_i64(val)
            } else {
                F::zero()
            }
        }))
    }
}

impl<F: FieldCore + std::fmt::Debug> std::fmt::Debug for RingSubfieldFp8<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingSubfieldFp8")
            .field("coeffs", &self.coeffs)
            .finish()
    }
}

impl<F: FieldCore> Clone for RingSubfieldFp8<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FieldCore> Copy for RingSubfieldFp8<F> {}

impl<F: FieldCore> Default for RingSubfieldFp8<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldCore> PartialEq for RingSubfieldFp8<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<F: FieldCore> Eq for RingSubfieldFp8<F> {}

impl<F: FieldCore> Add for RingSubfieldFp8<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<F: FieldCore> Sub for RingSubfieldFp8<F> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<F: FieldCore> Neg for RingSubfieldFp8<F> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(std::array::from_fn(|i| -self.coeffs[i]))
    }
}

impl<F: FieldCore> AddAssign for RingSubfieldFp8<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<F: FieldCore> SubAssign for RingSubfieldFp8<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<F: RingSubfieldFp8MulBackend> Mul for RingSubfieldFp8<F> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(F::ring_subfield_fp8_mul(self.coeffs, rhs.coeffs))
    }
}

impl<F: RingSubfieldFp8MulBackend> MulAssign for RingSubfieldFp8<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, F: FieldCore> Add<&'a Self> for RingSubfieldFp8<F> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, F: FieldCore> Sub<&'a Self> for RingSubfieldFp8<F> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, F: RingSubfieldFp8MulBackend> Mul<&'a Self> for RingSubfieldFp8<F> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<F: FieldCore + Valid> Valid for RingSubfieldFp8<F> {
    fn check(&self) -> Result<(), SerializationError> {
        for coeff in self.coeffs {
            coeff.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for RingSubfieldFp8<F> {
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
    for RingSubfieldFp8<F>
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

impl<F: FieldCore + Valid + RingSubfieldFp8MulBackend> RingCore for RingSubfieldFp8<F> {
    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }
}

impl<F: FieldCore + Valid + RingSubfieldFp8MulBackend> Invertible for RingSubfieldFp8<F> {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let mut aug = [[F::zero(); 9]; 8];
        for col in 0..8 {
            let mut basis = [F::zero(); 8];
            basis[col] = F::one();
            let product = *self * Self::new(basis);
            for (row, coeff) in product.coeffs.iter().copied().enumerate() {
                aug[row][col] = coeff;
            }
        }
        aug[0][8] = F::one();

        for col in 0..8 {
            let pivot = (col..8).find(|&row| !aug[row][col].is_zero())?;
            if pivot != col {
                aug.swap(col, pivot);
            }
            let inv = aug[col][col].inverse()?;
            for entry in &mut aug[col][col..=8] {
                *entry *= inv;
            }
            for row in 0..8 {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                if factor.is_zero() {
                    continue;
                }
                let pivot_row = aug[col];
                for (target, pivot) in aug[row][col..=8]
                    .iter_mut()
                    .zip(pivot_row[col..=8].iter().copied())
                {
                    *target -= factor * pivot;
                }
            }
        }

        Some(Self::new(std::array::from_fn(|i| aug[i][8])))
    }
}

impl<F: HalvingField + Valid + RingSubfieldFp8MulBackend> HalvingField for RingSubfieldFp8<F> {
    #[inline]
    fn half(self) -> Self {
        Self::new(std::array::from_fn(|i| self.coeffs[i].half()))
    }
}

impl<F: FieldCore + RandomSampling + Valid> RandomSampling for RingSubfieldFp8<F> {
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(std::array::from_fn(|_| F::random(rng)))
    }
}

impl<F: FieldCore + FromPrimitiveInt + Valid> FromPrimitiveInt for RingSubfieldFp8<F> {
    fn from_u64(val: u64) -> Self {
        Self::from_u64(val)
    }

    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    fn from_u128(val: u128) -> Self {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_u128(val)
            } else {
                F::zero()
            }
        }))
    }

    fn from_i128(val: i128) -> Self {
        Self::new(std::array::from_fn(|i| {
            if i == 0 {
                F::from_i128(val)
            } else {
                F::zero()
            }
        }))
    }
}

impl<F: FieldCore + BalancedDigitLookup + Valid> BalancedDigitLookup for RingSubfieldFp8<F> {}

impl<const P: u32> HasUnreducedOps for RingSubfieldFp8<Fp16<P>> {
    type MulU64Accum = Self;
    type ProductAccum = RingSubfieldFp8Fp16ProductAccum;

    // `ring_subfield_fp8_mul_to_accum_fp16` widens each Fp16 limb product
    // (< P² < 2^32) into a u64 slot with no `mod 2^64` wrap, so summing a small
    // batch and reducing once matches per-limb reduce-then-add exactly. Covered
    // by `ring_subfield_fp8_fp16_accum_summation`.
    const DELAYED_PRODUCT_SUM_IS_EXACT: bool = true;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum {
        let small = Fp16::<P>::from_u64(small);
        Self::new(self.coeffs.map(|coeff| coeff * small))
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Self::ProductAccum {
        ring_subfield_fp8_mul_to_accum_fp16(self.coeffs, other.coeffs)
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self {
        accum
    }

    #[inline]
    fn reduce_product_accum(accum: Self::ProductAccum) -> Self {
        Self::new(accum.reduce::<P>())
    }
}

impl<const P: u32> HasOptimizedFold for RingSubfieldFp8<Fp16<P>> {
    type FoldCtx = FoldMatrixFp16;

    #[inline]
    fn precompute_fold(r: Self) -> FoldMatrixFp16 {
        let [r0, r1, r2, r3, r4, r5, r6, r7] = r.coeffs;
        let two = Fp16::<P>::from_u64(2);
        // Column c is the canonical coordinates of `r · e_c` (e_0 = 1) in the
        // `[1, e1, ..., e7]` basis; row 0 of column c is the "1"-coefficient.
        // Equivalently, entry [t][c] is the coefficient of input limb `c` in
        // output coefficient `t` of the degree-8 ring multiply.
        let lt = |x: Fp16<P>| x.to_limbs() as u32;
        FoldMatrixFp16([
            [
                lt(r0),
                lt(two * r1),
                lt(two * r2),
                lt(two * r3),
                lt(two * r4),
                lt(two * r5),
                lt(two * r6),
                lt(two * r7),
            ],
            [
                lt(r1),
                lt(r0 + r2),
                lt(r1 + r3),
                lt(r2 + r4),
                lt(r3 + r5),
                lt(r4 + r6),
                lt(r5 + r7),
                lt(r6),
            ],
            [
                lt(r2),
                lt(r1 + r3),
                lt(r0 + r4),
                lt(r1 + r5),
                lt(r2 + r6),
                lt(r3 + r7),
                lt(r4),
                lt(r5 - r7),
            ],
            [
                lt(r3),
                lt(r2 + r4),
                lt(r1 + r5),
                lt(r0 + r6),
                lt(r1 + r7),
                lt(r2),
                lt(r3 - r7),
                lt(r4 - r6),
            ],
            [
                lt(r4),
                lt(r3 + r5),
                lt(r2 + r6),
                lt(r1 + r7),
                lt(r0),
                lt(r1 - r7),
                lt(r2 - r6),
                lt(r3 - r5),
            ],
            [
                lt(r5),
                lt(r4 + r6),
                lt(r3 + r7),
                lt(r2),
                lt(r1 - r7),
                lt(r0 - r6),
                lt(r1 - r5),
                lt(r2 - r4),
            ],
            [
                lt(r6),
                lt(r5 + r7),
                lt(r4),
                lt(r3 - r7),
                lt(r2 - r6),
                lt(r1 - r5),
                lt(r0 - r4),
                lt(r1 - r3),
            ],
            [
                lt(r7),
                lt(r6),
                lt(r5 - r7),
                lt(r4 - r6),
                lt(r3 - r5),
                lt(r2 - r4),
                lt(r1 - r3),
                lt(r0 - r2),
            ],
        ])
    }

    #[inline]
    fn fold_one(ctx: &FoldMatrixFp16, even: Self, odd: Self) -> Self {
        let m = &ctx.0;
        // d = odd − even, each limb canonical in [0, P) < 2^16.
        let d: [u32; 8] =
            std::array::from_fn(|j| (odd.coeffs[j] - even.coeffs[j]).to_limbs() as u32);
        // Each product < 2^16·2^16 = 2^32, sum of 8 < 2^35 — exact in u64.
        let folded: [Fp16<P>; 8] = std::array::from_fn(|row| {
            let acc: u64 = (m[row][0] as u64) * (d[0] as u64)
                + (m[row][1] as u64) * (d[1] as u64)
                + (m[row][2] as u64) * (d[2] as u64)
                + (m[row][3] as u64) * (d[3] as u64)
                + (m[row][4] as u64) * (d[4] as u64)
                + (m[row][5] as u64) * (d[5] as u64)
                + (m[row][6] as u64) * (d[6] as u64)
                + (m[row][7] as u64) * (d[7] as u64);
            Fp16::<P>::from_u64(acc) + even.coeffs[row]
        });
        RingSubfieldFp8::new(folded)
    }
}

macro_rules! impl_ring_subfield_fp8_unreduced_identity {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasUnreducedOps for RingSubfieldFp8<$base<$p>> {
            type MulU64Accum = Self;
            type ProductAccum = Self;

            #[inline]
            fn mul_u64_unreduced(self, small: u64) -> Self {
                let small = $base::<$p>::from_u64(small);
                Self::new(self.coeffs.map(|coeff| coeff * small))
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

impl_ring_subfield_fp8_unreduced_identity!(Fp32<P: u32>);
impl_ring_subfield_fp8_unreduced_identity!(Fp64<P: u64>);
impl_ring_subfield_fp8_unreduced_identity!(Fp128<P: u128>);

macro_rules! impl_ring_subfield_fp8_default_optimized_fold {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasOptimizedFold for RingSubfieldFp8<$base<$p>> {
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

impl_ring_subfield_fp8_default_optimized_fold!(Fp32<P: u32>);
impl_ring_subfield_fp8_default_optimized_fold!(Fp64<P: u64>);
impl_ring_subfield_fp8_default_optimized_fold!(Fp128<P: u128>);
