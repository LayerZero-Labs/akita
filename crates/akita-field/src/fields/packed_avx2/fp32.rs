use super::*;

/// Number of `Fp32` lanes in an AVX2 packed vector.
pub const FP32_WIDTH: usize = 8;

/// AVX2 packed arithmetic for `Fp32<P>`, processing 8 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp32Avx2<const P: u32>(pub [Fp32<P>; FP32_WIDTH]);

impl<const P: u32> PackedFp32Avx2<P> {
    const BITS: u32 = 32 - P.leading_zeros();

    const C: u32 = {
        let c = if Self::BITS == 32 {
            0u32.wrapping_sub(P)
        } else {
            (1u32 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            (c as u64) * (c as u64 + 1) < P as u64,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };

    const MASK_U64: u64 = if Self::BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << Self::BITS) - 1
    };

    /// Whether two Solinas folds suffice to bring the sum of four
    /// `(P-1)^2` products into `[0, 2*P)` for the final canonicalize step.
    /// Mirrors `PackedFp32Neon::TWO_FOLD_FOUR_PRODUCT_OK`. When `false`,
    /// `solinas_reduce` / `solinas_reduce_with_carry` must do a third fold
    /// before handing off to `pack_and_canonicalize`.
    const TWO_FOLD_FOUR_PRODUCT_OK: bool = {
        let c = Self::C as u64;
        4 * c * c + 3 * c <= (1u64 << Self::BITS)
    };

    #[inline(always)]
    fn to_vec(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }

    /// Multiply each `u64` lane by `C`. Building block of Solinas reduction;
    /// the `C == 1` fast path skips the multiply entirely for Mersenne-like
    /// primes. Mirrors `PackedFp32Neon::mul_c_u64`.
    ///
    /// AVX2 has no native 64×64-bit multiply, so we split `x` into two 32-bit
    /// halves, multiply each by `C` with `_mm256_mul_epu32` (32×32→64), then
    /// recombine: `x*C = x_lo*C + ((x_hi*C) << 32)` (mod 2^64). The previous
    /// implementation used a single `_mm256_mul_epu32(x, c_vec)` which only
    /// reads the *low 32 bits* of `x` and silently dropped bit 32+ — fine for
    /// `BITS == 32` (where the caller's `prod >> 32` always fits in 32 bits)
    /// but wrong for `BITS == 31` and `C != 1` where `prod >> 31` can occupy
    /// 33 bits.
    #[inline(always)]
    unsafe fn mul_c_u64(x: __m256i) -> __m256i {
        if Self::C == 1 {
            return x;
        }
        let c_vec = _mm256_set1_epi64x(Self::C as i64);
        let lo_part = _mm256_mul_epu32(x, c_vec);
        let hi_part = _mm256_mul_epu32(_mm256_srli_epi64::<32>(x), c_vec);
        _mm256_add_epi64(lo_part, _mm256_slli_epi64::<32>(hi_part))
    }

    /// Plonky3-style Mersenne31 multiply (P = 2^31 - 1). Specialized fold
    /// using `_mm256_srli_epi64::<31>` shifts. Used by the `Mul` impl when
    /// `Self::BITS == 31 && Self::C == 1`.
    #[inline(always)]
    unsafe fn mul_mersenne31_vec(a: __m256i, b: __m256i) -> __m256i {
        unsafe {
            let lhs_odd_dbl = _mm256_srli_epi64::<31>(a);
            let rhs_odd = movehdup_epi32(b);

            let prod_odd_dbl = _mm256_mul_epu32(rhs_odd, lhs_odd_dbl);
            let prod_evn = _mm256_mul_epu32(b, a);

            let prod_odd_lo_dirty = _mm256_slli_epi64::<31>(prod_odd_dbl);
            let prod_evn_hi = _mm256_srli_epi64::<31>(prod_evn);

            let prod_lo_dirty = _mm256_blend_epi32::<0b1010_1010>(prod_evn, prod_odd_lo_dirty);
            let prod_hi = _mm256_blend_epi32::<0b1010_1010>(prod_evn_hi, prod_odd_dbl);

            let p = _mm256_set1_epi32(P as i32);
            let prod_lo = _mm256_and_si256(prod_lo_dirty, p);
            let folded = _mm256_add_epi32(prod_lo, prod_hi);
            _mm256_min_epu32(folded, _mm256_sub_epi32(folded, p))
        }
    }

    /// Vector form of field add: 8-lane add + canonicalize to `[0, P)`.
    /// Mirrors `PackedFp32Neon::add_vec`.
    #[inline(always)]
    unsafe fn add_vec(a: __m256i, b: __m256i) -> __m256i {
        let p = _mm256_set1_epi32(P as i32);
        if Self::BITS <= 31 {
            let t = _mm256_add_epi32(a, b);
            let u = _mm256_sub_epi32(t, p);
            _mm256_min_epu32(t, u)
        } else {
            // BITS == 32: a + b may overflow u32. Detect via unsigned compare
            // (sign-bit-XOR trick), correct by adding C (since 2^32 ≡ C mod P),
            // then conditional subtract P.
            let c = _mm256_set1_epi32(Self::C as i32);
            let t = _mm256_add_epi32(a, b);
            let sign32 = _mm256_set1_epi32(i32::MIN);
            let overflow =
                _mm256_cmpgt_epi32(_mm256_xor_si256(a, sign32), _mm256_xor_si256(t, sign32));
            let t2 = _mm256_add_epi32(t, _mm256_and_si256(overflow, c));
            let r = _mm256_sub_epi32(t2, p);
            _mm256_min_epu32(t2, r)
        }
    }

    /// Vector form of field sub: 8-lane sub + canonicalize to `[0, P)`.
    /// Mirrors `PackedFp32Neon::sub_vec`.
    #[inline(always)]
    unsafe fn sub_vec(a: __m256i, b: __m256i) -> __m256i {
        let p = _mm256_set1_epi32(P as i32);
        if Self::BITS <= 31 {
            let t = _mm256_sub_epi32(a, b);
            let u = _mm256_add_epi32(t, p);
            _mm256_min_epu32(t, u)
        } else {
            // BITS == 32: t = a - b may underflow. If a < b, t wraps to
            // t + 2^32; we want t + P = t + 2^32 - C, i.e. subtract C.
            let t = _mm256_sub_epi32(a, b);
            let sign32 = _mm256_set1_epi32(i32::MIN);
            let underflow =
                _mm256_cmpgt_epi32(_mm256_xor_si256(b, sign32), _mm256_xor_si256(a, sign32));
            let c = _mm256_set1_epi32(Self::C as i32);
            _mm256_sub_epi32(t, _mm256_and_si256(underflow, c))
        }
    }

    /// Vector form of field mul: 8-lane Solinas multiply + canonicalize.
    /// Mirrors `PackedFp32Neon::mul_vec`.
    #[inline(always)]
    unsafe fn mul_vec(a: __m256i, b: __m256i) -> __m256i {
        let prod_evn = _mm256_mul_epu32(a, b);
        let a_odd = movehdup_epi32(a);
        let b_odd = movehdup_epi32(b);
        let prod_odd = _mm256_mul_epu32(a_odd, b_odd);
        Self::solinas_reduce(prod_evn, prod_odd)
    }

    /// `(sum + rhs, carry + overflow_bit)`: 4-lane `u64` accumulating add with
    /// per-lane carry tracking. Mirrors `PackedFp32Neon::add_u64_with_carry`.
    #[inline(always)]
    unsafe fn add_u64_with_carry(sum: __m256i, rhs: __m256i, carry: __m256i) -> (__m256i, __m256i) {
        let next = _mm256_add_epi64(sum, rhs);
        // unsigned compare next < sum: equivalent to "an overflow happened".
        let sign = _mm256_set1_epi64x(i64::MIN);
        let sum_s = _mm256_xor_si256(sum, sign);
        let next_s = _mm256_xor_si256(next, sign);
        let overflow_mask = _mm256_cmpgt_epi64(sum_s, next_s);
        let one = _mm256_set1_epi64x(1);
        let overflow_bit = _mm256_and_si256(overflow_mask, one);
        (next, _mm256_add_epi64(carry, overflow_bit))
    }

    /// Multiply each lane's low 32 bits by `Fp32::<P>::SHIFT64_MOD_P`. Used
    /// to fold accumulated overflow count back into the Solinas reduction.
    /// Mirrors `PackedFp32Neon::carry_correction`.
    #[inline(always)]
    unsafe fn carry_correction(carry: __m256i) -> __m256i {
        let shift = _mm256_set1_epi64x(Fp32::<P>::SHIFT64_MOD_P as i64);
        _mm256_mul_epu32(carry, shift)
    }

    /// 4-way fused multiply-accumulate with a single end-reduction.
    /// Computes `sum_i a[i] * b[i]` lane-wise and canonicalizes. The key
    /// fused operation for `RingSubfieldFp4` and power-basis Fp4 multiply.
    /// Mirrors `PackedFp32Neon::dot_product_4_vec`, including the
    /// `BITS <= 31` carry-free fast path: four `(2^31 - 1)^2` products sum
    /// to less than `2^64`, so partial sums never overflow a `u64` lane
    /// and we can drop the `add_u64_with_carry` chain plus the trailing
    /// `carry_correction`. The `if Self::BITS <= 31` is a const condition
    /// and dead-code-eliminated at compile time.
    #[inline(always)]
    unsafe fn dot_product_4_vec(a: [__m256i; 4], b: [__m256i; 4]) -> __m256i {
        let mut sum_evn = _mm256_mul_epu32(a[0], b[0]);
        let mut sum_odd = _mm256_mul_epu32(movehdup_epi32(a[0]), movehdup_epi32(b[0]));

        if Self::BITS <= 31 {
            for i in 1..4 {
                let prod_evn = _mm256_mul_epu32(a[i], b[i]);
                let prod_odd = _mm256_mul_epu32(movehdup_epi32(a[i]), movehdup_epi32(b[i]));
                sum_evn = _mm256_add_epi64(sum_evn, prod_evn);
                sum_odd = _mm256_add_epi64(sum_odd, prod_odd);
            }
            return Self::solinas_reduce(sum_evn, sum_odd);
        }

        let mut carry_evn = _mm256_setzero_si256();
        let mut carry_odd = _mm256_setzero_si256();

        for i in 1..4 {
            let prod_evn = _mm256_mul_epu32(a[i], b[i]);
            let prod_odd = _mm256_mul_epu32(movehdup_epi32(a[i]), movehdup_epi32(b[i]));
            let (s_evn, c_evn) = Self::add_u64_with_carry(sum_evn, prod_evn, carry_evn);
            let (s_odd, c_odd) = Self::add_u64_with_carry(sum_odd, prod_odd, carry_odd);
            sum_evn = s_evn;
            sum_odd = s_odd;
            carry_evn = c_evn;
            carry_odd = c_odd;
        }

        Self::solinas_reduce_with_carry(sum_evn, sum_odd, carry_evn, carry_odd)
    }

    /// Multiply by an `Fp2` non-residue (used by `fp2_mul`). Recognizes the
    /// `nr == -1` and `nr == 2` fast paths to avoid full multiplies.
    /// Mirrors `PackedFp32Neon::mul_nr_vec`.
    #[inline(always)]
    unsafe fn mul_nr_vec<C>(x: __m256i) -> __m256i
    where
        C: Fp2Config<Fp32<P>>,
    {
        if C::IS_NEG_ONE {
            Self::sub_vec(_mm256_setzero_si256(), x)
        } else if C::non_residue().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_non_residue(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    /// Multiply by the power-basis `w` (used by `power_basis_fp4_mul`).
    /// Recognizes the `w == 2` fast path. Mirrors `PackedFp32Neon::mul_w_vec`.
    #[inline(always)]
    unsafe fn mul_w_vec<C>(x: __m256i) -> __m256i
    where
        C: PowerBasisFp4Config<Fp32<P>>,
    {
        if C::w().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_w(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    /// Two-or-three-fold Solinas reduction of 4+4 `u64` products → 8 `u32`
    /// lanes. Inputs are the even-lane and odd-lane product vectors from
    /// `_mm256_mul_epu32`. Mirrors `PackedFp32Neon::solinas_reduce`.
    ///
    /// The `Self::BITS == 31` branches use immediate-shift
    /// `_mm256_srli_epi64::<31>` instead of the generic variable-shift
    /// `_mm256_srl_epi64(.., shift)`, mirroring the same specialisation
    /// the base-field `Mul` impl uses on Mersenne31, so extension-field
    /// operations on Mersenne31 get the same per-shift win.
    ///
    /// Two folds always suffice when `Self::TWO_FOLD_FOUR_PRODUCT_OK`. When
    /// it doesn't (large `C` such that `4*C^2 + 3*C > 2^BITS`), we run a
    /// third fold so `pack_and_canonicalize`'s single subtract-and-min step
    /// is enough to land in `[0, P)`.
    #[inline(always)]
    unsafe fn solinas_reduce(prod_evn: __m256i, prod_odd: __m256i) -> __m256i {
        let mask = _mm256_set1_epi64x(Self::MASK_U64 as i64);
        let shift = _mm_set_epi64x(0, Self::BITS as i64);

        // Fold 1
        let evn_lo = _mm256_and_si256(prod_evn, mask);
        let evn_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(prod_evn)
        } else {
            _mm256_srl_epi64(prod_evn, shift)
        };
        let evn_f1 = _mm256_add_epi64(evn_lo, Self::mul_c_u64(evn_hi));

        let odd_lo = _mm256_and_si256(prod_odd, mask);
        let odd_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(prod_odd)
        } else {
            _mm256_srl_epi64(prod_odd, shift)
        };
        let odd_f1 = _mm256_add_epi64(odd_lo, Self::mul_c_u64(odd_hi));

        // Fold 2
        let evn_f1_lo = _mm256_and_si256(evn_f1, mask);
        let evn_f1_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(evn_f1)
        } else {
            _mm256_srl_epi64(evn_f1, shift)
        };
        let evn_f2 = _mm256_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

        let odd_f1_lo = _mm256_and_si256(odd_f1, mask);
        let odd_f1_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(odd_f1)
        } else {
            _mm256_srl_epi64(odd_f1, shift)
        };
        let odd_f2 = _mm256_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

        // Optional third fold for large-C primes (e.g. Generic31Offset32787)
        // where two folds leave residue > 2*P.
        let (evn_final, odd_final) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
            (evn_f2, odd_f2)
        } else {
            let evn_f2_lo = _mm256_and_si256(evn_f2, mask);
            let evn_f2_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(evn_f2)
            } else {
                _mm256_srl_epi64(evn_f2, shift)
            };
            let odd_f2_lo = _mm256_and_si256(odd_f2, mask);
            let odd_f2_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(odd_f2)
            } else {
                _mm256_srl_epi64(odd_f2, shift)
            };
            (
                _mm256_add_epi64(evn_f2_lo, Self::mul_c_u64(evn_f2_hi)),
                _mm256_add_epi64(odd_f2_lo, Self::mul_c_u64(odd_f2_hi)),
            )
        };

        Self::pack_and_canonicalize(evn_final, odd_final)
    }

    /// Same as `solinas_reduce` but with extra carry counts to fold in. Used
    /// by `dot_product_4_vec` when accumulating multiple `u64` products.
    /// Mirrors `PackedFp32Neon::solinas_reduce_with_carry`.
    #[inline(always)]
    unsafe fn solinas_reduce_with_carry(
        prod_evn: __m256i,
        prod_odd: __m256i,
        carry_evn: __m256i,
        carry_odd: __m256i,
    ) -> __m256i {
        let mask = _mm256_set1_epi64x(Self::MASK_U64 as i64);
        let shift = _mm_set_epi64x(0, Self::BITS as i64);

        // Fold 1 with carry correction
        let evn_lo = _mm256_and_si256(prod_evn, mask);
        let evn_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(prod_evn)
        } else {
            _mm256_srl_epi64(prod_evn, shift)
        };
        let evn_f1 = _mm256_add_epi64(
            _mm256_add_epi64(evn_lo, Self::mul_c_u64(evn_hi)),
            Self::carry_correction(carry_evn),
        );

        let odd_lo = _mm256_and_si256(prod_odd, mask);
        let odd_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(prod_odd)
        } else {
            _mm256_srl_epi64(prod_odd, shift)
        };
        let odd_f1 = _mm256_add_epi64(
            _mm256_add_epi64(odd_lo, Self::mul_c_u64(odd_hi)),
            Self::carry_correction(carry_odd),
        );

        // Fold 2
        let evn_f1_lo = _mm256_and_si256(evn_f1, mask);
        let evn_f1_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(evn_f1)
        } else {
            _mm256_srl_epi64(evn_f1, shift)
        };
        let evn_f2 = _mm256_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

        let odd_f1_lo = _mm256_and_si256(odd_f1, mask);
        let odd_f1_hi = if Self::BITS == 31 {
            _mm256_srli_epi64::<31>(odd_f1)
        } else {
            _mm256_srl_epi64(odd_f1, shift)
        };
        let odd_f2 = _mm256_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

        // Optional third fold (see `solinas_reduce`).
        let (evn_final, odd_final) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
            (evn_f2, odd_f2)
        } else {
            let evn_f2_lo = _mm256_and_si256(evn_f2, mask);
            let evn_f2_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(evn_f2)
            } else {
                _mm256_srl_epi64(evn_f2, shift)
            };
            let odd_f2_lo = _mm256_and_si256(odd_f2, mask);
            let odd_f2_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(odd_f2)
            } else {
                _mm256_srl_epi64(odd_f2, shift)
            };
            (
                _mm256_add_epi64(evn_f2_lo, Self::mul_c_u64(evn_f2_hi)),
                _mm256_add_epi64(odd_f2_lo, Self::mul_c_u64(odd_f2_hi)),
            )
        };

        Self::pack_and_canonicalize(evn_final, odd_final)
    }

    /// Combine 4+4 `u64` lanes (in range `[0, 2P)`) into 8 `u32` lanes
    /// canonicalized to `[0, P)`. For `BITS < 32` the values fit in `u32`,
    /// so we can pack first and subtract `P` at `u32` width. For `BITS == 32`
    /// the worst case can exceed `u32::MAX`, so we conditionally subtract `P`
    /// at `u64` width first, then pack. Mirrors the post-fold tail of
    /// `PackedFp32Neon::solinas_reduce`.
    #[inline(always)]
    unsafe fn pack_and_canonicalize(evn_f2: __m256i, odd_f2: __m256i) -> __m256i {
        if Self::BITS < 32 {
            let odd_shifted = _mm256_slli_epi64::<32>(odd_f2);
            let combined = _mm256_blend_epi32::<0b10101010>(evn_f2, odd_shifted);
            let p = _mm256_set1_epi32(P as i32);
            let reduced = _mm256_sub_epi32(combined, p);
            _mm256_min_epu32(combined, reduced)
        } else {
            let p_u64 = _mm256_set1_epi64x(P as i64);
            let sign = _mm256_set1_epi64x(i64::MIN);
            let p_s = _mm256_xor_si256(p_u64, sign);

            let red_evn = _mm256_sub_epi64(evn_f2, p_u64);
            let evn_s = _mm256_xor_si256(evn_f2, sign);
            let keep_evn = _mm256_cmpgt_epi64(p_s, evn_s);
            let out_evn = _mm256_blendv_epi8(red_evn, evn_f2, keep_evn);

            let red_odd = _mm256_sub_epi64(odd_f2, p_u64);
            let odd_s = _mm256_xor_si256(odd_f2, sign);
            let keep_odd = _mm256_cmpgt_epi64(p_s, odd_s);
            let out_odd = _mm256_blendv_epi8(red_odd, odd_f2, keep_odd);

            let odd_shifted = _mm256_slli_epi64::<32>(out_odd);
            _mm256_blend_epi32::<0b10101010>(out_evn, odd_shifted)
        }
    }
}

impl<const P: u32> Default for PackedFp32Avx2<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp32(0); FP32_WIDTH])
    }
}

impl<const P: u32> fmt::Debug for PackedFp32Avx2<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp32Avx2").field(&self.0).finish()
    }
}

impl<const P: u32> PartialEq for PackedFp32Avx2<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u32> Eq for PackedFp32Avx2<P> {}

impl<const P: u32> Add for PackedFp32Avx2<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self::from_vec(Self::add_vec(self.to_vec(), rhs.to_vec())) }
    }
}

impl<const P: u32> Sub for PackedFp32Avx2<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self::from_vec(Self::sub_vec(self.to_vec(), rhs.to_vec())) }
    }
}

impl<const P: u32> Mul for PackedFp32Avx2<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();

            if Self::BITS == 31 && Self::C == 1 {
                return Self::from_vec(Self::mul_mersenne31_vec(a, b));
            }

            let prod_evn = _mm256_mul_epu32(a, b);
            let a_odd = movehdup_epi32(a);
            let b_odd = movehdup_epi32(b);
            let prod_odd = _mm256_mul_epu32(a_odd, b_odd);

            let mask = _mm256_set1_epi64x(Self::MASK_U64 as i64);
            let shift = _mm_set_epi64x(0, Self::BITS as i64);

            // Fold 1
            let evn_lo = _mm256_and_si256(prod_evn, mask);
            let evn_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(prod_evn)
            } else {
                _mm256_srl_epi64(prod_evn, shift)
            };
            let evn_f1 = _mm256_add_epi64(evn_lo, Self::mul_c_u64(evn_hi));

            let odd_lo = _mm256_and_si256(prod_odd, mask);
            let odd_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(prod_odd)
            } else {
                _mm256_srl_epi64(prod_odd, shift)
            };
            let odd_f1 = _mm256_add_epi64(odd_lo, Self::mul_c_u64(odd_hi));

            // Fold 2
            let evn_f1_lo = _mm256_and_si256(evn_f1, mask);
            let evn_f1_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(evn_f1)
            } else {
                _mm256_srl_epi64(evn_f1, shift)
            };
            let evn_f2 = _mm256_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

            let odd_f1_lo = _mm256_and_si256(odd_f1, mask);
            let odd_f1_hi = if Self::BITS == 31 {
                _mm256_srli_epi64::<31>(odd_f1)
            } else {
                _mm256_srl_epi64(odd_f1, shift)
            };
            let odd_f2 = _mm256_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

            // Recombine even/odd: shift odd results into high 32-bit positions, blend.
            let odd_shifted = _mm256_slli_epi64::<32>(odd_f2);
            let combined = _mm256_blend_epi32::<0b10101010>(evn_f2, odd_shifted);

            // Conditional subtract P
            let p_vec = _mm256_set1_epi32(P as i32);
            let reduced = _mm256_sub_epi32(combined, p_vec);
            Self::from_vec(_mm256_min_epu32(combined, reduced))
        }
    }
}

impl<const P: u32> PackedValue for PackedFp32Avx2<P> {
    type Value = Fp32<P>;
    const WIDTH: usize = FP32_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP32_WIDTH);
        self.0[lane]
    }
}

impl<const P: u32> AddAssign for PackedFp32Avx2<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp32Avx2<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp32Avx2<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> PackedField for PackedFp32Avx2<P> {
    type Scalar = Fp32<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP32_WIDTH])
    }

    #[inline(always)]
    fn fp2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: Fp2Config<Self::Scalar>,
    {
        unsafe {
            let a0 = a0.to_vec();
            let a1 = a1.to_vec();
            let b0 = b0.to_vec();
            let b1 = b1.to_vec();

            let v0 = Self::mul_vec(a0, b0);
            let v1 = Self::mul_vec(a1, b1);
            let cross = Self::mul_vec(Self::add_vec(a0, a1), Self::add_vec(b0, b1));

            (
                Self::from_vec(Self::add_vec(v0, Self::mul_nr_vec::<C>(v1))),
                Self::from_vec(Self::sub_vec(Self::sub_vec(cross, v0), v1)),
            )
        }
    }

    #[inline(always)]
    fn power_basis_fp4_mul<C>(a: [Self; 4], b: [Self; 4]) -> [Self; 4]
    where
        C: PowerBasisFp4Config<Self::Scalar>,
    {
        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let [b0, b1, b2, b3] = b.map(Self::to_vec);

            if C::w().0 == 2 {
                let two_b1 = Self::add_vec(b1, b1);
                let two_b2 = Self::add_vec(b2, b2);
                let two_b3 = Self::add_vec(b3, b3);
                return [
                    Self::from_vec(Self::dot_product_4_vec(
                        [a0, a1, a2, a3],
                        [b0, two_b3, two_b2, two_b1],
                    )),
                    Self::from_vec(Self::dot_product_4_vec(
                        [a0, a1, a2, a3],
                        [b1, b0, two_b3, two_b2],
                    )),
                    Self::from_vec(Self::dot_product_4_vec(
                        [a0, a1, a2, a3],
                        [b2, b1, b0, two_b3],
                    )),
                    Self::from_vec(Self::dot_product_4_vec([a0, a1, a2, a3], [b3, b2, b1, b0])),
                ];
            }

            let c0_tail = Self::add_vec(
                Self::add_vec(Self::mul_vec(a1, b3), Self::mul_vec(a2, b2)),
                Self::mul_vec(a3, b1),
            );
            let c1_tail = Self::add_vec(Self::mul_vec(a2, b3), Self::mul_vec(a3, b2));
            let c2_tail = Self::mul_vec(a3, b3);

            [
                Self::from_vec(Self::add_vec(
                    Self::mul_vec(a0, b0),
                    Self::mul_w_vec::<C>(c0_tail),
                )),
                Self::from_vec(Self::add_vec(
                    Self::add_vec(Self::mul_vec(a0, b1), Self::mul_vec(a1, b0)),
                    Self::mul_w_vec::<C>(c1_tail),
                )),
                Self::from_vec(Self::add_vec(
                    Self::add_vec(
                        Self::add_vec(Self::mul_vec(a0, b2), Self::mul_vec(a1, b1)),
                        Self::mul_vec(a2, b0),
                    ),
                    Self::mul_w_vec::<C>(c2_tail),
                )),
                Self::from_vec(Self::add_vec(
                    Self::add_vec(
                        Self::add_vec(Self::mul_vec(a0, b3), Self::mul_vec(a1, b2)),
                        Self::mul_vec(a2, b1),
                    ),
                    Self::mul_vec(a3, b0),
                )),
            ]
        }
    }

    #[inline(always)]
    fn ring_subfield_fp4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let [b0, b1, b2, b3] = b.map(Self::to_vec);
            let two_b1 = Self::add_vec(b1, b1);
            let two_b2 = Self::add_vec(b2, b2);
            let two_b3 = Self::add_vec(b3, b3);
            let b0_plus_b2 = Self::add_vec(b0, b2);
            let b1_plus_b3 = Self::add_vec(b1, b3);
            let b1_minus_b3 = Self::sub_vec(b1, b3);
            let b0_minus_b2 = Self::sub_vec(b0, b2);
            [
                Self::from_vec(Self::dot_product_4_vec(
                    [a0, a1, a2, a3],
                    [b0, two_b1, two_b2, two_b3],
                )),
                Self::from_vec(Self::dot_product_4_vec(
                    [a0, a1, a2, a3],
                    [b1, b0_plus_b2, b1_plus_b3, b2],
                )),
                Self::from_vec(Self::dot_product_4_vec(
                    [a0, a1, a2, a3],
                    [b2, b1_plus_b3, b0, b1_minus_b3],
                )),
                Self::from_vec(Self::dot_product_4_vec(
                    [a0, a1, a2, a3],
                    [b3, b2, b1_minus_b3, b0_minus_b2],
                )),
            ]
        }
    }

    #[inline(always)]
    fn ring_subfield_fp4_square(a: [Self; 4]) -> [Self; 4] {
        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let x0 = a0;
            let x1 = a2;
            let y0 = Self::sub_vec(a1, a3);
            let y1 = a3;

            let x0x1 = Self::mul_vec(x0, x1);
            let y0y1 = Self::mul_vec(y0, y1);
            let x1_square = Self::mul_vec(x1, x1);
            let y1_square = Self::mul_vec(y1, y1);
            let aa = (
                Self::add_vec(Self::mul_vec(x0, x0), Self::add_vec(x1_square, x1_square)),
                Self::add_vec(x0x1, x0x1),
            );
            let bb = (
                Self::add_vec(Self::mul_vec(y0, y0), Self::add_vec(y1_square, y1_square)),
                Self::add_vec(y0y1, y0y1),
            );

            let v0 = Self::mul_vec(x0, y0);
            let v1 = Self::mul_vec(x1, y1);
            let ab = (
                Self::add_vec(v0, Self::add_vec(v1, v1)),
                Self::sub_vec(
                    Self::sub_vec(
                        Self::mul_vec(Self::add_vec(x0, x1), Self::add_vec(y0, y1)),
                        v0,
                    ),
                    v1,
                ),
            );
            let constant = (
                Self::add_vec(Self::add_vec(bb.0, bb.0), Self::add_vec(bb.1, bb.1)),
                Self::add_vec(bb.0, Self::add_vec(bb.1, bb.1)),
            );
            let coeff_e1 = (Self::add_vec(ab.0, ab.0), Self::add_vec(ab.1, ab.1));

            [
                Self::from_vec(Self::add_vec(aa.0, constant.0)),
                Self::from_vec(Self::add_vec(coeff_e1.0, coeff_e1.1)),
                Self::from_vec(Self::add_vec(aa.1, constant.1)),
                Self::from_vec(coeff_e1.1),
            ]
        }
    }

    #[inline(always)]
    fn ring_subfield_fp4_inverse(a: [Self; 4]) -> Option<[Self; 4]>
    where
        Self::Scalar: Invertible,
    {
        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let zero = _mm256_setzero_si256();
            let x0 = a0;
            let x1 = a2;
            let y0 = Self::sub_vec(a1, a3);
            let y1 = a3;

            let x1_square = Self::mul_vec(x1, x1);
            let y1_square = Self::mul_vec(y1, y1);
            let aa0 = Self::add_vec(Self::mul_vec(x0, x0), Self::add_vec(x1_square, x1_square));
            let aa1 = {
                let x0x1 = Self::mul_vec(x0, x1);
                Self::add_vec(x0x1, x0x1)
            };
            let bb0 = Self::add_vec(Self::mul_vec(y0, y0), Self::add_vec(y1_square, y1_square));
            let bb1 = {
                let y0y1 = Self::mul_vec(y0, y1);
                Self::add_vec(y0y1, y0y1)
            };
            let nr_bb0 = Self::add_vec(Self::add_vec(bb0, bb0), Self::add_vec(bb1, bb1));
            let nr_bb1 = Self::add_vec(bb0, Self::add_vec(bb1, bb1));
            let norm0 = Self::sub_vec(aa0, nr_bb0);
            let norm1 = Self::sub_vec(aa1, nr_bb1);

            let inv_norm_base = {
                let norm1_square = Self::mul_vec(norm1, norm1);
                let norm_base = Self::sub_vec(
                    Self::mul_vec(norm0, norm0),
                    Self::add_vec(norm1_square, norm1_square),
                );
                Self::from_vec(norm_base).inverse()?.to_vec()
            };
            let inv_norm0 = Self::mul_vec(norm0, inv_norm_base);
            let inv_norm1 = Self::mul_vec(Self::sub_vec(zero, norm1), inv_norm_base);

            let v0 = Self::mul_vec(x0, inv_norm0);
            let v1 = Self::mul_vec(x1, inv_norm1);
            let constant0 = Self::add_vec(v0, Self::add_vec(v1, v1));
            let constant1 = Self::sub_vec(
                Self::sub_vec(
                    Self::mul_vec(Self::add_vec(x0, x1), Self::add_vec(inv_norm0, inv_norm1)),
                    v0,
                ),
                v1,
            );

            let neg_y0 = Self::sub_vec(zero, y0);
            let neg_y1 = Self::sub_vec(zero, y1);
            let w0 = Self::mul_vec(neg_y0, inv_norm0);
            let w1 = Self::mul_vec(neg_y1, inv_norm1);
            let e1_coeff0 = Self::add_vec(w0, Self::add_vec(w1, w1));
            let e1_coeff1 = Self::sub_vec(
                Self::sub_vec(
                    Self::mul_vec(
                        Self::add_vec(neg_y0, neg_y1),
                        Self::add_vec(inv_norm0, inv_norm1),
                    ),
                    w0,
                ),
                w1,
            );

            Some([
                Self::from_vec(constant0),
                Self::from_vec(Self::add_vec(e1_coeff0, e1_coeff1)),
                Self::from_vec(constant1),
                Self::from_vec(e1_coeff1),
            ])
        }
    }

    #[inline(always)]
    fn tower_basis_fp4_mul<C2, C4>(a: [Self; 4], b: [Self; 4]) -> [Self; 4]
    where
        C2: Fp2Config<Self::Scalar>,
        C4: TowerBasisFp4Config<Self::Scalar, C2>,
    {
        let nr = C4::non_residue();
        if nr.coeffs[0].is_zero() && nr.coeffs[1] == Self::Scalar::one() {
            return Self::power_basis_fp4_mul::<C2>(a, b);
        }

        unsafe {
            let [a0, a1, a2, a3] = a.map(Self::to_vec);
            let [b0, b1, b2, b3] = b.map(Self::to_vec);

            let (v0_0, v0_1) = Self::fp2_mul::<C2>(
                Self::from_vec(a0),
                Self::from_vec(a2),
                Self::from_vec(b0),
                Self::from_vec(b2),
            );
            let (v1_0, v1_1) = Self::fp2_mul::<C2>(
                Self::from_vec(a1),
                Self::from_vec(a3),
                Self::from_vec(b1),
                Self::from_vec(b3),
            );
            let (nr_v1_0, nr_v1_1) = Self::fp2_mul::<C2>(
                Self::broadcast(nr.coeffs[0]),
                Self::broadcast(nr.coeffs[1]),
                v1_0,
                v1_1,
            );
            let (cross_0, cross_1) = Self::fp2_mul::<C2>(
                Self::from_vec(Self::add_vec(a0, a1)),
                Self::from_vec(Self::add_vec(a2, a3)),
                Self::from_vec(Self::add_vec(b0, b1)),
                Self::from_vec(Self::add_vec(b2, b3)),
            );
            [
                v0_0 + nr_v1_0,
                cross_0 - v0_0 - v1_0,
                v0_1 + nr_v1_1,
                cross_1 - v0_1 - v1_1,
            ]
        }
    }
}
