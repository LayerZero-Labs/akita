use super::*;

/// Number of `Fp64` lanes in an AVX-512 packed vector.
pub(crate) const FP64_WIDTH: usize = 8;

/// AVX-512 packed arithmetic for `Fp64<P>`, processing 8 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp64Avx512<const P: u64>(pub [Fp64<P>; FP64_WIDTH]);

impl<const P: u64> PackedFp64Avx512<P> {
    #[inline(always)]
    fn to_vec(self) -> __m512i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m512i) -> Self {
        unsafe { transmute(v) }
    }

    /// Add two lane-wise 128-bit values represented as `(hi, lo)`.
    #[inline(always)]
    unsafe fn add128_vec(
        a_hi: __m512i,
        a_lo: __m512i,
        b_hi: __m512i,
        b_lo: __m512i,
    ) -> (__m512i, __m512i) {
        let lo = _mm512_add_epi64(a_lo, b_lo);
        let carry = _mm512_cmplt_epu64_mask(lo, a_lo);
        let hi_sum = _mm512_add_epi64(a_hi, b_hi);
        let hi = _mm512_mask_add_epi64(hi_sum, carry, hi_sum, _mm512_set1_epi64(1));
        (hi, lo)
    }

    /// Vectorized 128-bit Solinas reduction for p = 2^BITS - C.
    /// Given (hi, lo) = 128-bit product, computes result ≡ (hi*2^64 + lo) mod p.
    #[inline]
    unsafe fn reduce128_vec(hi: __m512i, lo: __m512i) -> __m512i {
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
    unsafe fn reduce128_sub_word_narrow(hi: __m512i, lo: __m512i) -> __m512i {
        let mask_k = _mm512_set1_epi64(Fp64::<P>::MASK64 as i64);
        let c_vec = _mm512_set1_epi64(Fp64::<P>::C as i64);
        let p_vec = _mm512_set1_epi64(P as i64);
        let shift_k = _mm_set_epi64x(0, Fp64::<P>::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Fp64::<P>::BITS) as i64);

        let lo_k = _mm512_and_si512(lo, mask_k);
        let lo_upper = _mm512_srl_epi64(lo, shift_k);
        let hi_shifted = _mm512_sll_epi64(hi, shift_64mk);
        let hi_k = _mm512_or_si512(lo_upper, hi_shifted);

        // c * hi_k: hi_k may exceed 32 bits, split into lo32 and top
        let c_hi_lo = _mm512_mul_epu32(c_vec, hi_k);
        let hi_k_top = _mm512_srli_epi64::<32>(hi_k);
        let c_hi_top = _mm512_mul_epu32(c_vec, hi_k_top);
        let c_hi_top_shifted = _mm512_slli_epi64::<32>(c_hi_top);
        let c_hi = _mm512_add_epi64(c_hi_lo, c_hi_top_shifted);

        let fold1 = _mm512_add_epi64(lo_k, c_hi);

        let fold1_lo_k = _mm512_and_si512(fold1, mask_k);
        let fold1_hi = _mm512_srl_epi64(fold1, shift_k);
        let c_fold1_hi = _mm512_mul_epu32(c_vec, fold1_hi);
        let fold2 = _mm512_add_epi64(fold1_lo_k, c_fold1_hi);

        let reduced = _mm512_sub_epi64(fold2, p_vec);
        _mm512_min_epu64(fold2, reduced)
    }

    /// Reduction for sub-word fields where `C * (product >> BITS)` can
    /// overflow a `u64` lane. The product by `C` is split into exact low and
    /// high halves, then the carry is included in the second Solinas fold.
    #[inline]
    unsafe fn reduce128_sub_word_wide(hi: __m512i, lo: __m512i) -> __m512i {
        let mask_k = _mm512_set1_epi64(Fp64::<P>::MASK64 as i64);
        let c_vec = _mm512_set1_epi64(Fp64::<P>::C as i64);
        let p_vec = _mm512_set1_epi64(P as i64);
        let one = _mm512_set1_epi64(1);
        let shift_k = _mm_set_epi64x(0, Fp64::<P>::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Fp64::<P>::BITS) as i64);

        let lo_k = _mm512_and_si512(lo, mask_k);
        let lo_upper = _mm512_srl_epi64(lo, shift_k);
        let hi_shifted = _mm512_sll_epi64(hi, shift_64mk);
        let high = _mm512_or_si512(lo_upper, hi_shifted);
        let high_overflow = _mm512_srl_epi64(hi, shift_k);

        // C fits in u32 by the field invariant C(C + 1) < P. Splitting
        // `high` at 32 bits gives the exact 128-bit product C * high.
        let c_high_lo32 = _mm512_mul_epu32(c_vec, high);
        let high_hi32 = _mm512_srli_epi64::<32>(high);
        let c_high_hi32 = _mm512_mul_epu32(c_vec, high_hi32);
        let c_high_hi_shifted = _mm512_slli_epi64::<32>(c_high_hi32);
        let product_lo = _mm512_add_epi64(c_high_lo32, c_high_hi_shifted);
        let product_carry = _mm512_cmplt_epu64_mask(product_lo, c_high_lo32);
        let product_hi = _mm512_mask_add_epi64(
            _mm512_srli_epi64::<32>(c_high_hi32),
            product_carry,
            _mm512_srli_epi64::<32>(c_high_hi32),
            one,
        );
        let product_hi = _mm512_add_epi64(product_hi, _mm512_mul_epu32(c_vec, high_overflow));

        let fold1_lo = _mm512_add_epi64(lo_k, product_lo);
        let fold1_carry = _mm512_cmplt_epu64_mask(fold1_lo, product_lo);
        let fold1_hi = _mm512_mask_add_epi64(product_hi, fold1_carry, product_hi, one);

        let fold1_lo_k = _mm512_and_si512(fold1_lo, mask_k);
        let fold1_lo_upper = _mm512_srl_epi64(fold1_lo, shift_k);
        let fold1_hi_shifted = _mm512_sll_epi64(fold1_hi, shift_64mk);
        let fold1_high = _mm512_or_si512(fold1_lo_upper, fold1_hi_shifted);

        // After one fold, fold1_high is a small multiple of C, so this u32
        // multiply is exact for both base products and fused Ext2 coefficients.
        let c_fold1_high = _mm512_mul_epu32(c_vec, fold1_high);
        let fold2 = _mm512_add_epi64(fold1_lo_k, c_fold1_high);

        let ge_mask = _mm512_cmpge_epu64_mask(fold2, p_vec);
        _mm512_mask_sub_epi64(fold2, ge_mask, fold2, p_vec)
    }

    /// Reduction for BITS == 64 (e.g. p = 2^64 - 87). Tracks overflow from
    /// c*hi exceeding 64 bits, using native unsigned comparisons.
    #[inline]
    unsafe fn reduce128_word_sized(hi: __m512i, lo: __m512i) -> __m512i {
        let c_vec = _mm512_set1_epi64(Fp64::<P>::C as i64);
        let p_vec = _mm512_set1_epi64(P as i64);
        let one = _mm512_set1_epi64(1);

        // c * hi_lo32
        let c_hi_lo = _mm512_mul_epu32(c_vec, hi);
        // c * hi_hi32
        let hi_hi = _mm512_srli_epi64::<32>(hi);
        let c_hi_hi = _mm512_mul_epu32(c_vec, hi_hi);

        let c_hi_hi_lo32 = _mm512_slli_epi64::<32>(c_hi_hi);
        let c_hi_carry = _mm512_srli_epi64::<32>(c_hi_hi);

        // Lower 64 bits of c * hi
        let sum_lo = _mm512_add_epi64(c_hi_lo, c_hi_hi_lo32);
        let carry0 = _mm512_cmplt_epu64_mask(sum_lo, c_hi_lo);
        let overflow = _mm512_mask_add_epi64(c_hi_carry, carry0, c_hi_carry, one);

        // lo + sum_lo
        let s = _mm512_add_epi64(lo, sum_lo);
        let carry1 = _mm512_cmplt_epu64_mask(s, lo);
        let total_overflow = _mm512_mask_add_epi64(overflow, carry1, overflow, one);

        // Fold overflow: total_overflow * c (at most ~2^15)
        let final_corr = _mm512_mul_epu32(c_vec, total_overflow);
        let result = _mm512_add_epi64(s, final_corr);
        let carry_f = _mm512_cmplt_epu64_mask(result, s);
        let result = _mm512_mask_add_epi64(result, carry_f, result, c_vec);

        let ge_mask = _mm512_cmpge_epu64_mask(result, p_vec);
        _mm512_mask_sub_epi64(result, ge_mask, result, p_vec)
    }
}

impl<const P: u64> Default for PackedFp64Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp64(0); FP64_WIDTH])
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Avx512").field(&self.0).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u64> Eq for PackedFp64Avx512<P> {}

impl<const P: u64> Add for PackedFp64Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm512_set1_epi64(P as i64);

            let result = if Fp64::<P>::BITS < 64 {
                let s = _mm512_add_epi64(a, b);
                _mm512_min_epu64(s, _mm512_sub_epi64(s, p))
            } else {
                let s = _mm512_add_epi64(a, b);
                let overflow = _mm512_cmplt_epu64_mask(s, a);
                let c = _mm512_set1_epi64(Fp64::<P>::C as i64);
                let geq_p = _mm512_cmpge_epu64_mask(s, p);
                let no_of = _mm512_mask_sub_epi64(s, geq_p, s, p);
                let s_plus_c = _mm512_add_epi64(s, c);
                _mm512_mask_blend_epi64(overflow, no_of, s_plus_c)
            };

            Self::from_vec(result)
        }
    }
}

impl<const P: u64> Sub for PackedFp64Avx512<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm512_set1_epi64(P as i64);
            let d = _mm512_sub_epi64(a, b);
            let result = if Fp64::<P>::BITS < 64 {
                _mm512_min_epu64(d, _mm512_add_epi64(d, p))
            } else {
                let underflow = _mm512_cmplt_epu64_mask(a, b);
                _mm512_mask_add_epi64(d, underflow, d, p)
            };
            Self::from_vec(result)
        }
    }
}

impl<const P: u64> Mul for PackedFp64Avx512<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let (hi, lo) = mul64_64_512(self.to_vec(), rhs.to_vec());
            Self::from_vec(Self::reduce128_vec(hi, lo))
        }
    }
}

impl<const P: u64> PackedValue for PackedFp64Avx512<P> {
    type Value = Fp64<P>;
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP64_WIDTH);
        self.0[lane]
    }
}

impl<const P: u64> AddAssign for PackedFp64Avx512<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Avx512<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Avx512<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Avx512<P> {
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
                let (p00_hi, p00_lo) = mul64_64_512(a0.to_vec(), b0.to_vec());
                let (p11_hi, p11_lo) = mul64_64_512(a1.to_vec(), b1.to_vec());
                let (p01_hi, p01_lo) = mul64_64_512(a0.to_vec(), b1.to_vec());
                let (p10_hi, p10_lo) = mul64_64_512(a1.to_vec(), b0.to_vec());
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
