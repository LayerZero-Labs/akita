use super::*;

/// Chebyshev `φ` fold-back for a degree-8 accumulator, using caller-supplied
/// add/sub so the same routine serves scalar, `i64`, and SIMD lane types.
///
/// `φ(k)` maps a product onto the `[1, e1, ..., e7]` basis:
/// `k = 0 → 2·constant`, `1 ≤ k ≤ 7 → +e_k`, `k = 8 → 0`,
/// `9 ≤ k ≤ 15 → −e_{16−k}`.
#[inline(always)]
fn fp8_add_phi<V: Copy>(
    out: &mut [V; 8],
    idx: usize,
    value: V,
    add: &impl Fn(V, V) -> V,
    sub: &impl Fn(V, V) -> V,
) {
    match idx {
        0 => out[0] = add(out[0], add(value, value)),
        1..=7 => out[idx] = add(out[idx], value),
        8 => {}
        9..=15 => out[16 - idx] = sub(out[16 - idx], value),
        _ => unreachable!("fp8 Chebyshev index out of range"),
    }
}

/// Karatsuba schedule for `RingSubfieldFp8` multiplication in the Chebyshev
/// basis, generic over a lane type `V` and its add/sub/mul.
///
/// One schedule serves every backend: the scalar field default, the `i64`
/// Fp16 path, and the NEON / AVX2 / AVX-512 SIMD kernels. The schedule is
/// purely an additive combination of products, so callers that reduce per
/// operation (field or intrinsic ops) and callers that defer reduction to the
/// end (the `i64` Fp16 path) are both correct, provided the `i64` accumulator
/// does not overflow — which holds because every product is bounded by
/// `(2·P)² < 2^34` and at most a few dozen are summed per coefficient.
#[inline(always)]
pub(crate) fn ring_subfield_fp8_mul_schedule<V, A, S, M>(
    a: [V; 8],
    b: [V; 8],
    zero: V,
    add: A,
    sub: S,
    mul: M,
) -> [V; 8]
where
    V: Copy,
    A: Fn(V, V) -> V,
    S: Fn(V, V) -> V,
    M: Fn(V, V) -> V,
{
    let diag: [V; 8] = std::array::from_fn(|i| mul(a[i], b[i]));
    let mut out = [zero; 8];
    out[0] = diag[0];

    for k in 1..8 {
        let mixed = sub(sub(mul(add(a[0], a[k]), add(b[0], b[k])), diag[0]), diag[k]);
        out[k] = add(out[k], mixed);
    }

    for (i, &diag_i) in diag.iter().enumerate().skip(1) {
        out[0] = add(out[0], add(diag_i, diag_i));
        fp8_add_phi(&mut out, i + i, diag_i, &add, &sub);
    }

    for i in 1..8 {
        for j in (i + 1)..8 {
            let mixed = sub(sub(mul(add(a[i], a[j]), add(b[i], b[j])), diag[i]), diag[j]);
            fp8_add_phi(&mut out, i + j, mixed, &add, &sub);
            fp8_add_phi(&mut out, j - i, mixed, &add, &sub);
        }
    }

    out
}

/// Squaring schedule for `RingSubfieldFp8`, generic over a lane type `V`.
///
/// Uses `(a_i + a_j)² − a_i² − a_j² = 2·a_i·a_j` to compute `a_i·a_j` directly
/// and double, saving one add and two subs per cross-term versus the Karatsuba
/// form. Shares `fp8_add_phi` with [`ring_subfield_fp8_mul_schedule`].
#[inline(always)]
pub(crate) fn ring_subfield_fp8_square_schedule<V, A, S, M>(
    a: [V; 8],
    zero: V,
    add: A,
    sub: S,
    mul: M,
) -> [V; 8]
where
    V: Copy,
    A: Fn(V, V) -> V,
    S: Fn(V, V) -> V,
    M: Fn(V, V) -> V,
{
    let sq: [V; 8] = std::array::from_fn(|i| mul(a[i], a[i]));
    let mut out = [zero; 8];
    out[0] = sq[0];

    for k in 1..8 {
        let cross = mul(a[0], a[k]);
        out[k] = add(out[k], add(cross, cross));
    }

    for (i, &sq_i) in sq.iter().enumerate().skip(1) {
        out[0] = add(out[0], add(sq_i, sq_i));
        fp8_add_phi(&mut out, i + i, sq_i, &add, &sub);
    }

    for i in 1..8 {
        for j in (i + 1)..8 {
            let cross = mul(a[i], a[j]);
            let doubled = add(cross, cross);
            fp8_add_phi(&mut out, i + j, doubled, &add, &sub);
            fp8_add_phi(&mut out, j - i, doubled, &add, &sub);
        }
    }

    out
}

#[inline(always)]
fn ring_subfield_fp8_mul_coeffs<F: FieldCore>(a: [F; 8], b: [F; 8]) -> [F; 8] {
    ring_subfield_fp8_mul_schedule(a, b, F::zero(), |x, y| x + y, |x, y| x - y, |x, y| x * y)
}

/// Backend hook for scalar ring-subfield degree-8 multiplication.
pub trait RingSubfieldFp8MulBackend: FieldCore {
    /// Multiply coefficient arrays in `[1, e1, ..., e7]` basis.
    #[inline(always)]
    fn ring_subfield_fp8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        ring_subfield_fp8_mul_coeffs::<Self>(a, b)
    }
}

impl<const P: u32> RingSubfieldFp8MulBackend for Fp16<P> {
    /// `Fp16` widens each `u16` coefficient to `i64` and runs the shared
    /// Karatsuba schedule with raw integer ops, deferring a single
    /// `rem_euclid` reduction per output coefficient. This avoids the `u16`
    /// overflow that the field-op default would hit on Karatsuba subtractions.
    #[inline(always)]
    fn ring_subfield_fp8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        let al: [i64; 8] = std::array::from_fn(|i| a[i].to_limbs() as i64);
        let bl: [i64; 8] = std::array::from_fn(|i| b[i].to_limbs() as i64);
        let out =
            ring_subfield_fp8_mul_schedule(al, bl, 0i64, |x, y| x + y, |x, y| x - y, |x, y| x * y);
        std::array::from_fn(|i| Fp16::<P>::from_canonical_u16(out[i].rem_euclid(P as i64) as u16))
    }
}
impl<const P: u32> RingSubfieldFp8MulBackend for Fp32<P> {}
impl<const P: u64> RingSubfieldFp8MulBackend for Fp64<P> {}
impl<const P: u128> RingSubfieldFp8MulBackend for Fp128<P> {}

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

impl<F> HasUnreducedOps for RingSubfieldFp8<F>
where
    F: FieldCore + FromPrimitiveInt + Valid + RingSubfieldFp8MulBackend,
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
