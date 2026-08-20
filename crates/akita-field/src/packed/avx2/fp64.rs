use super::*;

/// Number of `Fp64` lanes in an AVX2 packed vector.
pub(crate) const FP64_WIDTH: usize = 4;

/// AVX2 packed arithmetic for `Fp64<P>`, processing 4 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp64Avx2<const P: u64>(pub [Fp64<P>; FP64_WIDTH]);

impl<const P: u64> PackedFp64Avx2<P> {
    #[inline(always)]
    fn to_vec(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }

    /// Add two lane-wise 128-bit values represented as `(hi, lo)`.
    #[inline(always)]
    unsafe fn add128_vec(
        a_hi: __m256i,
        a_lo: __m256i,
        b_hi: __m256i,
        b_lo: __m256i,
    ) -> (__m256i, __m256i) {
        let sign = _mm256_set1_epi64x(i64::MIN);
        let lo = _mm256_add_epi64(a_lo, b_lo);
        let a_lo_s = _mm256_xor_si256(a_lo, sign);
        let lo_s = _mm256_xor_si256(lo, sign);
        let carry = _mm256_cmpgt_epi64(a_lo_s, lo_s);
        let hi = _mm256_sub_epi64(_mm256_add_epi64(a_hi, b_hi), carry);
        (hi, lo)
    }

    #[inline]
    unsafe fn reduce128_vec(hi: __m256i, lo: __m256i) -> __m256i {
        if Fp64::<P>::BITS < 64 {
            if Fp64::<P>::FOLD_IN_U64 {
                Self::reduce128_sub_word_narrow(hi, lo)
            } else {
                Self::reduce128_sub_word_wide(hi, lo)
            }
        } else {
            Self::reduce128_word_sized(hi, lo)
        }
    }

    /// Reduction for sub-word fields whose two folds fit in one `u64` lane.
    #[inline]
    unsafe fn reduce128_sub_word_narrow(hi: __m256i, lo: __m256i) -> __m256i {
        let mask_k = _mm256_set1_epi64x(Fp64::<P>::MASK64 as i64);
        let c_vec = _mm256_set1_epi64x(Fp64::<P>::C as i64);
        let p_vec = _mm256_set1_epi64x(P as i64);
        let shift_k = _mm_set_epi64x(0, Fp64::<P>::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Fp64::<P>::BITS) as i64);

        let lo_k = _mm256_and_si256(lo, mask_k);
        let lo_upper = _mm256_srl_epi64(lo, shift_k);
        let hi_shifted = _mm256_sll_epi64(hi, shift_64mk);
        let hi_k = _mm256_or_si256(lo_upper, hi_shifted);

        let c_hi_lo = _mm256_mul_epu32(c_vec, hi_k);
        let hi_k_top = _mm256_srli_epi64::<32>(hi_k);
        let c_hi_top = _mm256_mul_epu32(c_vec, hi_k_top);
        let c_hi_top_shifted = _mm256_slli_epi64::<32>(c_hi_top);
        let c_hi = _mm256_add_epi64(c_hi_lo, c_hi_top_shifted);

        let fold1 = _mm256_add_epi64(lo_k, c_hi);

        let fold1_lo_k = _mm256_and_si256(fold1, mask_k);
        let fold1_hi = _mm256_srl_epi64(fold1, shift_k);
        let c_fold1_hi = _mm256_mul_epu32(c_vec, fold1_hi);
        let fold2 = _mm256_add_epi64(fold1_lo_k, c_fold1_hi);

        let reduced = _mm256_sub_epi64(fold2, p_vec);
        let sign = _mm256_set1_epi64x(i64::MIN);
        let fold2_s = _mm256_xor_si256(fold2, sign);
        let reduced_s = _mm256_xor_si256(reduced, sign);
        let fold2_lt = _mm256_cmpgt_epi64(reduced_s, fold2_s);
        _mm256_blendv_epi8(reduced, fold2, fold2_lt)
    }

    /// Reduction for sub-word fields where `C * (product >> BITS)` can
    /// overflow a `u64` lane. The product by `C` is split into exact low and
    /// high halves, then the carry is included in the second Solinas fold.
    #[inline]
    unsafe fn reduce128_sub_word_wide(hi: __m256i, lo: __m256i) -> __m256i {
        let mask_k = _mm256_set1_epi64x(Fp64::<P>::MASK64 as i64);
        let c_vec = _mm256_set1_epi64x(Fp64::<P>::C as i64);
        let p_vec = _mm256_set1_epi64x(P as i64);
        let sign = _mm256_set1_epi64x(i64::MIN);
        let shift_k = _mm_set_epi64x(0, Fp64::<P>::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Fp64::<P>::BITS) as i64);

        let lo_k = _mm256_and_si256(lo, mask_k);
        let lo_upper = _mm256_srl_epi64(lo, shift_k);
        let hi_shifted = _mm256_sll_epi64(hi, shift_64mk);
        let high = _mm256_or_si256(lo_upper, hi_shifted);
        let high_overflow = _mm256_srl_epi64(hi, shift_k);

        // C fits in u32 by the field invariant C(C + 1) < P. Splitting
        // `high` at 32 bits gives the exact 128-bit product C * high.
        let c_high_lo32 = _mm256_mul_epu32(c_vec, high);
        let high_hi32 = _mm256_srli_epi64::<32>(high);
        let c_high_hi32 = _mm256_mul_epu32(c_vec, high_hi32);
        let c_high_hi_shifted = _mm256_slli_epi64::<32>(c_high_hi32);
        let product_lo = _mm256_add_epi64(c_high_lo32, c_high_hi_shifted);
        let product_lo_s = _mm256_xor_si256(product_lo, sign);
        let c_high_lo32_s = _mm256_xor_si256(c_high_lo32, sign);
        let product_carry = _mm256_cmpgt_epi64(c_high_lo32_s, product_lo_s);
        let product_hi = _mm256_sub_epi64(_mm256_srli_epi64::<32>(c_high_hi32), product_carry);
        let product_hi = _mm256_add_epi64(product_hi, _mm256_mul_epu32(c_vec, high_overflow));

        let fold1_lo = _mm256_add_epi64(lo_k, product_lo);
        let fold1_lo_s = _mm256_xor_si256(fold1_lo, sign);
        let product_lo_s = _mm256_xor_si256(product_lo, sign);
        let fold1_carry = _mm256_cmpgt_epi64(product_lo_s, fold1_lo_s);
        let fold1_hi = _mm256_sub_epi64(product_hi, fold1_carry);

        let fold1_lo_k = _mm256_and_si256(fold1_lo, mask_k);
        let fold1_lo_upper = _mm256_srl_epi64(fold1_lo, shift_k);
        let fold1_hi_shifted = _mm256_sll_epi64(fold1_hi, shift_64mk);
        let fold1_high = _mm256_or_si256(fold1_lo_upper, fold1_hi_shifted);

        // After one fold, fold1_high is a small multiple of C, so this u32
        // multiply is exact for both base products and fused Ext2 coefficients.
        let c_fold1_high = _mm256_mul_epu32(c_vec, fold1_high);
        let fold2 = _mm256_add_epi64(fold1_lo_k, c_fold1_high);

        let reduced = _mm256_sub_epi64(fold2, p_vec);
        let fold2_s = _mm256_xor_si256(fold2, sign);
        let reduced_s = _mm256_xor_si256(reduced, sign);
        let fold2_lt = _mm256_cmpgt_epi64(reduced_s, fold2_s);
        _mm256_blendv_epi8(reduced, fold2, fold2_lt)
    }

    /// Reduction for BITS == 64. Uses XOR-with-SIGN_BIT trick for unsigned
    /// overflow detection.
    #[inline]
    unsafe fn reduce128_word_sized(hi: __m256i, lo: __m256i) -> __m256i {
        let c_vec = _mm256_set1_epi64x(Fp64::<P>::C as i64);
        let p_vec = _mm256_set1_epi64x(P as i64);
        let sign = _mm256_set1_epi64x(i64::MIN);
        let c_hi_lo = _mm256_mul_epu32(c_vec, hi);
        let hi_hi = _mm256_srli_epi64::<32>(hi);
        let c_hi_hi = _mm256_mul_epu32(c_vec, hi_hi);

        let c_hi_hi_lo32 = _mm256_slli_epi64::<32>(c_hi_hi);
        let c_hi_carry = _mm256_srli_epi64::<32>(c_hi_hi);

        let sum_lo = _mm256_add_epi64(c_hi_lo, c_hi_hi_lo32);
        let c_hi_lo_s = _mm256_xor_si256(c_hi_lo, sign);
        let sum_lo_s = _mm256_xor_si256(sum_lo, sign);
        let carry0 = _mm256_cmpgt_epi64(c_hi_lo_s, sum_lo_s);
        let overflow = _mm256_sub_epi64(c_hi_carry, carry0);

        let s = _mm256_add_epi64(lo, sum_lo);
        let lo_s = _mm256_xor_si256(lo, sign);
        let s_s = _mm256_xor_si256(s, sign);
        let carry1 = _mm256_cmpgt_epi64(lo_s, s_s);
        let total_overflow = _mm256_sub_epi64(overflow, carry1);

        let final_corr = _mm256_mul_epu32(c_vec, total_overflow);
        let result = _mm256_add_epi64(s, final_corr);
        let s2_s = _mm256_xor_si256(s, sign);
        let result_s = _mm256_xor_si256(result, sign);
        let carry_f = _mm256_cmpgt_epi64(s2_s, result_s);
        let corr_f = _mm256_and_si256(carry_f, c_vec);
        let result = _mm256_add_epi64(result, corr_f);

        let result_s2 = _mm256_xor_si256(result, sign);
        let p_s = _mm256_xor_si256(p_vec, sign);
        let lt_p = _mm256_cmpgt_epi64(p_s, result_s2);
        let sub_amt = _mm256_andnot_si256(lt_p, p_vec);
        _mm256_sub_epi64(result, sub_amt)
    }
}

impl<const P: u64> Default for PackedFp64Avx2<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp64(0); FP64_WIDTH])
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Avx2<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Avx2").field(&self.0).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Avx2<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u64> Eq for PackedFp64Avx2<P> {}

impl<const P: u64> Add for PackedFp64Avx2<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm256_set1_epi64x(P as i64);

            let result = if Fp64::<P>::BITS < 64 {
                // a + b < 2P < 2^64: no overflow.
                let s = _mm256_add_epi64(a, b);
                let r = _mm256_sub_epi64(s, p);
                // If subtraction wrapped, r is negative as i64. Otherwise r
                // is canonical and therefore below 2^63.
                let borrow = _mm256_cmpgt_epi64(_mm256_setzero_si256(), r);
                _mm256_blendv_epi8(r, s, borrow)
            } else {
                // a + b can overflow u64.
                let s = _mm256_add_epi64(a, b);
                let sign = _mm256_set1_epi64x(i64::MIN);
                let a_s = _mm256_xor_si256(a, sign);
                let s_s = _mm256_xor_si256(s, sign);
                let overflow = _mm256_cmpgt_epi64(a_s, s_s);
                let c = _mm256_set1_epi64x(Fp64::<P>::C as i64);
                let s_plus_c = _mm256_add_epi64(s, c);
                let s_minus_p = _mm256_sub_epi64(s, p);
                let p_s = _mm256_xor_si256(p, sign);
                let lt_p = _mm256_cmpgt_epi64(p_s, s_s);
                let no_of = _mm256_blendv_epi8(s_minus_p, s, lt_p);
                _mm256_blendv_epi8(no_of, s_plus_c, overflow)
            };

            Self::from_vec(result)
        }
    }
}

impl<const P: u64> Sub for PackedFp64Avx2<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm256_set1_epi64x(P as i64);
            let d = _mm256_sub_epi64(a, b);

            let result = if Fp64::<P>::BITS < 64 {
                // A wrapped difference is negative as i64, while both the
                // direct and corrected canonical differences are not.
                let underflow = _mm256_cmpgt_epi64(_mm256_setzero_si256(), d);
                _mm256_blendv_epi8(d, _mm256_add_epi64(d, p), underflow)
            } else {
                let sign = _mm256_set1_epi64x(i64::MIN);
                let a_s = _mm256_xor_si256(a, sign);
                let b_s = _mm256_xor_si256(b, sign);
                let underflow = _mm256_cmpgt_epi64(b_s, a_s);
                _mm256_add_epi64(d, _mm256_and_si256(underflow, p))
            };
            Self::from_vec(result)
        }
    }
}

impl<const P: u64> Mul for PackedFp64Avx2<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let (hi, lo) = mul64_64_256(self.to_vec(), rhs.to_vec());
            Self::from_vec(Self::reduce128_vec(hi, lo))
        }
    }
}

impl<const P: u64> PackedValue for PackedFp64Avx2<P> {
    type Value = Fp64<P>;
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0), f(1), f(2), f(3)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP64_WIDTH);
        self.0[lane]
    }
}

impl<const P: u64> AddAssign for PackedFp64Avx2<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Avx2<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Avx2<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Avx2<P> {
    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP64_WIDTH])
    }

    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        if C::NON_RESIDUE_KIND == FpExt2NonResidueKind::Two && fp64_ext2_two_avx_fusion_safe::<P>()
        {
            unsafe {
                let (p00_hi, p00_lo) = mul64_64_256(a0.to_vec(), b0.to_vec());
                let (p11_hi, p11_lo) = mul64_64_256(a1.to_vec(), b1.to_vec());
                let (p01_hi, p01_lo) = mul64_64_256(a0.to_vec(), b1.to_vec());
                let (p10_hi, p10_lo) = mul64_64_256(a1.to_vec(), b0.to_vec());
                let (z0_hi, z0_lo) = Self::add128_vec(p00_hi, p00_lo, p11_hi, p11_lo);
                let (z0_hi, z0_lo) = Self::add128_vec(z0_hi, z0_lo, p11_hi, p11_lo);
                let (z1_hi, z1_lo) = Self::add128_vec(p01_hi, p01_lo, p10_hi, p10_lo);
                return (
                    Self::from_vec(Self::reduce128_sub_word_wide(z0_hi, z0_lo)),
                    Self::from_vec(Self::reduce128_sub_word_wide(z1_hi, z1_lo)),
                );
            }
        }

        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }
}
