//! AVX2 packed backends for Fp32, Fp64, Fp128.
//!
//! Techniques adapted from plonky2 (Goldilocks) and plonky3 (Mersenne-31).

use super::packed::{PackedField, PackedValue};
use crate::fields::ext::{Fp2Config, PowerBasisFp4Config, TowerBasisFp4Config};
use crate::fields::{Fp128, Fp32, Fp64};
use crate::Invertible;
use core::arch::x86_64::*;
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// Duplicate high 32 bits of each 64-bit lane into the low 32 bits.
/// Uses the float `movehdup` instruction which runs on port 5 (doesn't compete
/// with multiply on ports 0/1).
#[inline(always)]
unsafe fn movehdup_epi32(x: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)))
}

#[inline(always)]
unsafe fn moveldup_epi32(x: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(x)))
}

/// 64×64→128 schoolbook multiply using 32×32→64 partial products.
/// Returns (hi, lo) representing the 128-bit product.
#[inline]
unsafe fn mul64_64_256(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    let x_hi = movehdup_epi32(x);
    let y_hi = movehdup_epi32(y);

    let mul_ll = _mm256_mul_epu32(x, y);
    let mul_lh = _mm256_mul_epu32(x, y_hi);
    let mul_hl = _mm256_mul_epu32(x_hi, y);
    let mul_hh = _mm256_mul_epu32(x_hi, y_hi);

    let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
    let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);
    let mask32 = _mm256_set1_epi64x(0xFFFF_FFFF_i64);
    let t0_lo = _mm256_and_si256(t0, mask32);
    let t0_hi = _mm256_srli_epi64::<32>(t0);
    let t1 = _mm256_add_epi64(mul_lh, t0_lo);
    let t2 = _mm256_add_epi64(mul_hh, t0_hi);
    let t1_hi = _mm256_srli_epi64::<32>(t1);
    let res_hi = _mm256_add_epi64(t2, t1_hi);

    let t1_lo = moveldup_epi32(t1);
    let res_lo = _mm256_blend_epi32::<0b10101010>(mul_ll, t1_lo);

    (res_hi, res_lo)
}

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

    #[inline(always)]
    fn to_vec(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }

    /// Multiply each lane's low 32 bits by `C`. Building block of Solinas
    /// reduction. The `C == 1` fast path is from PR #99's fp31 work.
    /// Mirrors `PackedFp32Neon::mul_c_u64`.
    #[inline(always)]
    unsafe fn mul_c_u64(x: __m256i) -> __m256i {
        if Self::C == 1 {
            x
        } else {
            let c_vec = _mm256_set1_epi64x(Self::C as i64);
            _mm256_mul_epu32(x, c_vec)
        }
    }

    /// Plonky3-style Mersenne31 multiply (P = 2^31 - 1). Specialized fold
    /// using `_mm256_srli_epi64::<31>` shifts. Used by the `Mul` impl when
    /// `Self::BITS == 31 && Self::C == 1`. From PR #99.
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
    /// Mirrors `PackedFp32Neon::dot_product_4_vec`.
    #[inline(always)]
    unsafe fn dot_product_4_vec(a: [__m256i; 4], b: [__m256i; 4]) -> __m256i {
        let mut sum_evn = _mm256_mul_epu32(a[0], b[0]);
        let mut sum_odd = _mm256_mul_epu32(movehdup_epi32(a[0]), movehdup_epi32(b[0]));
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

    /// Two-fold Solinas reduction of 4+4 `u64` products → 8 `u32` lanes.
    /// Inputs are the even-lane and odd-lane product vectors from
    /// `_mm256_mul_epu32`. Mirrors `PackedFp32Neon::solinas_reduce`.
    ///
    /// The `Self::BITS == 31` branches mirror the immediate-shift
    /// specialisation PR #99 added in the base-field `Mul` impl, so
    /// extension-field operations on Mersenne31 get the same per-shift win.
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

        Self::pack_and_canonicalize(evn_f2, odd_f2)
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

        Self::pack_and_canonicalize(evn_f2, odd_f2)
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

/// Number of `Fp64` lanes in an AVX2 packed vector.
pub const FP64_WIDTH: usize = 4;

/// AVX2 packed arithmetic for `Fp64<P>`, processing 4 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp64Avx2<const P: u64>(pub [Fp64<P>; FP64_WIDTH]);

impl<const P: u64> PackedFp64Avx2<P> {
    const BITS: u32 = 64 - P.leading_zeros();

    const C_LO: u64 = {
        let c = if Self::BITS == 64 {
            0u64.wrapping_sub(P)
        } else {
            (1u64 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        c
    };

    const MASK64: u64 = if Self::BITS < 64 {
        (1u64 << Self::BITS) - 1
    } else {
        u64::MAX
    };

    #[inline(always)]
    fn to_vec(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }

    #[inline]
    unsafe fn reduce128_vec(hi: __m256i, lo: __m256i) -> __m256i {
        if Self::BITS < 64 {
            Self::reduce128_small_k(hi, lo)
        } else {
            Self::reduce128_full_k(hi, lo)
        }
    }

    /// Reduction for BITS < 64. All intermediates fit in u64 — no overflow.
    #[inline]
    unsafe fn reduce128_small_k(hi: __m256i, lo: __m256i) -> __m256i {
        let mask_k = _mm256_set1_epi64x(Self::MASK64 as i64);
        let c_vec = _mm256_set1_epi64x(Self::C_LO as i64);
        let p_vec = _mm256_set1_epi64x(P as i64);
        let shift_k = _mm_set_epi64x(0, Self::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Self::BITS) as i64);

        let lo_k = _mm256_and_si256(lo, mask_k);
        let lo_upper = _mm256_srl_epi64(lo, shift_k);
        let hi_shifted = _mm256_sll_epi64(hi, shift_64mk);
        let hi_k = _mm256_or_si256(lo_upper, hi_shifted);

        let c_hi_lo = _mm256_mul_epu32(c_vec, hi_k);
        let hi_k_top = _mm256_srli_epi64::<32>(hi_k);
        let c_hi_top = _mm256_mul_epu32(c_vec, hi_k_top);
        let c_hi_top_shifted = _mm256_slli_epi64::<32>(c_hi_top);
        let c_hi_full = _mm256_add_epi64(c_hi_lo, c_hi_top_shifted);

        let fold1 = _mm256_add_epi64(lo_k, c_hi_full);

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

    /// Reduction for BITS == 64. Uses XOR-with-SIGN_BIT trick for unsigned
    /// overflow detection.
    #[inline]
    unsafe fn reduce128_full_k(hi: __m256i, lo: __m256i) -> __m256i {
        let c_vec = _mm256_set1_epi64x(Self::C_LO as i64);
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

            let result = if Self::BITS <= 62 {
                // a + b < 2P < 2^63: no overflow.
                let s = _mm256_add_epi64(a, b);
                let r = _mm256_sub_epi64(s, p);
                // s < P? Use signed compare after shift trick.
                let sign = _mm256_set1_epi64x(i64::MIN);
                let s_s = _mm256_xor_si256(s, sign);
                let p_s = _mm256_xor_si256(p, sign);
                let borrow = _mm256_cmpgt_epi64(p_s, s_s);
                _mm256_blendv_epi8(r, s, borrow)
            } else {
                // a + b can overflow u64.
                let s = _mm256_add_epi64(a, b);
                let sign = _mm256_set1_epi64x(i64::MIN);
                let a_s = _mm256_xor_si256(a, sign);
                let s_s = _mm256_xor_si256(s, sign);
                let overflow = _mm256_cmpgt_epi64(a_s, s_s);
                let c = _mm256_set1_epi64x(Self::C_LO as i64);
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

            let sign = _mm256_set1_epi64x(i64::MIN);
            let a_s = _mm256_xor_si256(a, sign);
            let b_s = _mm256_xor_si256(b, sign);
            let underflow = _mm256_cmpgt_epi64(b_s, a_s);
            let corrected = _mm256_add_epi64(d, p);
            Self::from_vec(_mm256_blendv_epi8(d, corrected, underflow))
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

    /// Override of the default `fp2_mul` to force inlining of the underlying
    /// 4-lane `__m256i` add/mul. Mirrors `PackedFp64Neon::fp2_mul`.
    #[inline(always)]
    fn fp2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: Fp2Config<Self::Scalar>,
    {
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }
}

/// Number of `Fp128` lanes in an AVX2 packed vector.
pub const FP128_WIDTH: usize = 4;

/// AVX2 packed arithmetic for `Fp128<P>`, 4 lanes in SoA layout.
///
/// Stores 4 elements as separate `lo` and `hi` `u64` arrays, enabling
/// vectorized add/sub via `__m256i`.  Mul remains scalar per-lane.
#[derive(Clone, Copy)]
pub struct PackedFp128Avx2<const P: u128> {
    lo: [u64; FP128_WIDTH],
    hi: [u64; FP128_WIDTH],
}

impl<const P: u128> PackedFp128Avx2<P> {
    const P_LO: u64 = P as u64;
    const P_HI: u64 = (P >> 64) as u64;
}

impl<const P: u128> Default for PackedFp128Avx2<P> {
    #[inline]
    fn default() -> Self {
        Self {
            lo: [0; FP128_WIDTH],
            hi: [0; FP128_WIDTH],
        }
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Avx2<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems: Vec<_> = (0..FP128_WIDTH).map(|i| self.extract(i)).collect();
        f.debug_tuple("PackedFp128Avx2").field(&elems).finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Avx2<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

impl<const P: u128> Eq for PackedFp128Avx2<P> {}

impl<const P: u128> Add for PackedFp128Avx2<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm256_loadu_si256(self.lo.as_ptr().cast());
            let a_hi = _mm256_loadu_si256(self.hi.as_ptr().cast());
            let b_lo = _mm256_loadu_si256(rhs.lo.as_ptr().cast());
            let b_hi = _mm256_loadu_si256(rhs.hi.as_ptr().cast());
            let p_lo = _mm256_set1_epi64x(Self::P_LO as i64);
            let p_hi = _mm256_set1_epi64x(Self::P_HI as i64);
            let sign = _mm256_set1_epi64x(i64::MIN);
            let one = _mm256_set1_epi64x(1);

            // 128-bit add with unsigned compare emulation (XOR sign bit)
            let sum_lo = _mm256_add_epi64(a_lo, b_lo);
            let carry_lo =
                _mm256_cmpgt_epi64(_mm256_xor_si256(a_lo, sign), _mm256_xor_si256(sum_lo, sign));
            let carry_lo_bit = _mm256_and_si256(carry_lo, one);

            let hi_tmp = _mm256_add_epi64(a_hi, b_hi);
            let ov1 =
                _mm256_cmpgt_epi64(_mm256_xor_si256(a_hi, sign), _mm256_xor_si256(hi_tmp, sign));
            let sum_hi = _mm256_add_epi64(hi_tmp, carry_lo_bit);
            let ov2 = _mm256_cmpgt_epi64(
                _mm256_xor_si256(hi_tmp, sign),
                _mm256_xor_si256(sum_hi, sign),
            );
            let carry_128 = _mm256_or_si256(ov1, ov2);

            // 128-bit subtract P
            let red_lo = _mm256_sub_epi64(sum_lo, p_lo);
            let borrow_lo =
                _mm256_cmpgt_epi64(_mm256_xor_si256(p_lo, sign), _mm256_xor_si256(sum_lo, sign));
            let borrow_lo_bit = _mm256_and_si256(borrow_lo, one);

            let red_hi_tmp = _mm256_sub_epi64(sum_hi, p_hi);
            let bw1 =
                _mm256_cmpgt_epi64(_mm256_xor_si256(p_hi, sign), _mm256_xor_si256(sum_hi, sign));
            let red_hi = _mm256_sub_epi64(red_hi_tmp, borrow_lo_bit);
            let bw2 = _mm256_cmpgt_epi64(
                _mm256_xor_si256(borrow_lo_bit, sign),
                _mm256_xor_si256(red_hi_tmp, sign),
            );
            let borrow = _mm256_or_si256(bw1, bw2);

            // use_reduced = carry_128 | !borrow
            let not_borrow = _mm256_xor_si256(borrow, _mm256_set1_epi64x(-1));
            let use_reduced = _mm256_or_si256(carry_128, not_borrow);
            let out_lo = _mm256_blendv_epi8(sum_lo, red_lo, use_reduced);
            let out_hi = _mm256_blendv_epi8(sum_hi, red_hi, use_reduced);

            let mut result = Self::default();
            _mm256_storeu_si256(result.lo.as_mut_ptr().cast(), out_lo);
            _mm256_storeu_si256(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Sub for PackedFp128Avx2<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm256_loadu_si256(self.lo.as_ptr().cast());
            let a_hi = _mm256_loadu_si256(self.hi.as_ptr().cast());
            let b_lo = _mm256_loadu_si256(rhs.lo.as_ptr().cast());
            let b_hi = _mm256_loadu_si256(rhs.hi.as_ptr().cast());
            let p_lo = _mm256_set1_epi64x(Self::P_LO as i64);
            let p_hi = _mm256_set1_epi64x(Self::P_HI as i64);
            let sign = _mm256_set1_epi64x(i64::MIN);
            let one = _mm256_set1_epi64x(1);

            // 128-bit sub
            let diff_lo = _mm256_sub_epi64(a_lo, b_lo);
            let borrow_lo =
                _mm256_cmpgt_epi64(_mm256_xor_si256(b_lo, sign), _mm256_xor_si256(a_lo, sign));
            let borrow_lo_bit = _mm256_and_si256(borrow_lo, one);

            let hi_tmp = _mm256_sub_epi64(a_hi, b_hi);
            let bw1 =
                _mm256_cmpgt_epi64(_mm256_xor_si256(b_hi, sign), _mm256_xor_si256(a_hi, sign));
            let diff_hi = _mm256_sub_epi64(hi_tmp, borrow_lo_bit);
            let bw2 = _mm256_cmpgt_epi64(
                _mm256_xor_si256(borrow_lo_bit, sign),
                _mm256_xor_si256(hi_tmp, sign),
            );
            let borrow_128 = _mm256_or_si256(bw1, bw2);

            // Correction: add P back where underflow occurred
            let corr_lo = _mm256_add_epi64(diff_lo, p_lo);
            let carry_lo = _mm256_cmpgt_epi64(
                _mm256_xor_si256(diff_lo, sign),
                _mm256_xor_si256(corr_lo, sign),
            );
            let carry_lo_bit = _mm256_and_si256(carry_lo, one);
            let corr_hi = _mm256_add_epi64(diff_hi, p_hi);
            let corr_hi = _mm256_add_epi64(corr_hi, carry_lo_bit);

            let out_lo = _mm256_blendv_epi8(diff_lo, corr_lo, borrow_128);
            let out_hi = _mm256_blendv_epi8(diff_hi, corr_hi, borrow_128);

            let mut result = Self::default();
            _mm256_storeu_si256(result.lo.as_mut_ptr().cast(), out_lo);
            _mm256_storeu_si256(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Mul for PackedFp128Avx2<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::default();
        for i in 0..FP128_WIDTH {
            let a = Fp128::<P>([self.lo[i], self.hi[i]]);
            let b = Fp128::<P>([rhs.lo[i], rhs.hi[i]]);
            let r = a * b;
            out.lo[i] = r.0[0];
            out.hi[i] = r.0[1];
        }
        out
    }
}

impl<const P: u128> PackedValue for PackedFp128Avx2<P> {
    type Value = Fp128<P>;
    const WIDTH: usize = FP128_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        let mut lo = [0u64; FP128_WIDTH];
        let mut hi = [0u64; FP128_WIDTH];
        for i in 0..FP128_WIDTH {
            let v = f(i);
            lo[i] = v.0[0];
            hi[i] = v.0[1];
        }
        Self { lo, hi }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP128_WIDTH);
        Fp128([self.lo[lane], self.hi[lane]])
    }
}

impl<const P: u128> AddAssign for PackedFp128Avx2<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for PackedFp128Avx2<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for PackedFp128Avx2<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u128> PackedField for PackedFp128Avx2<P> {
    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self {
            lo: [value.0[0]; FP128_WIDTH],
            hi: [value.0[1]; FP128_WIDTH],
        }
    }
}
