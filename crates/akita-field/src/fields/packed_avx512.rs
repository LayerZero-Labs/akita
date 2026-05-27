//! AVX-512 packed backends for Fp16, Fp32, Fp64, Fp128.
//!
//! Requires AVX-512F + AVX-512DQ. Uses native unsigned comparisons and mask
//! registers for branchless conditionals.

use super::packed::{PackedField, PackedValue};
use crate::fields::ext::{Fp2Config, PowerBasisFp4Config, TowerBasisFp4Config};
use crate::fields::{Fp128, Fp16, Fp32, Fp64};
use crate::Invertible;
use core::arch::x86_64::*;
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[inline(always)]
unsafe fn movehdup_epi32_512(x: __m512i) -> __m512i {
    _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)))
}

#[inline(always)]
unsafe fn moveldup_epi32_512(x: __m512i) -> __m512i {
    _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(x)))
}

/// 64×64→128 schoolbook multiply using 32×32→64 partial products.
/// Returns (hi, lo) representing the 128-bit product.
/// Adapted from plonky3's Goldilocks AVX-512 backend.
#[inline]
unsafe fn mul64_64_512(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let x_hi = movehdup_epi32_512(x);
    let y_hi = movehdup_epi32_512(y);

    let mul_ll = _mm512_mul_epu32(x, y);
    let mul_lh = _mm512_mul_epu32(x, y_hi);
    let mul_hl = _mm512_mul_epu32(x_hi, y);
    let mul_hh = _mm512_mul_epu32(x_hi, y_hi);

    let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
    let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
    let mask32 = _mm512_set1_epi64(0xFFFF_FFFF_i64);
    let t0_lo = _mm512_and_si512(t0, mask32);
    let t0_hi = _mm512_srli_epi64::<32>(t0);
    let t1 = _mm512_add_epi64(mul_lh, t0_lo);
    let t2 = _mm512_add_epi64(mul_hh, t0_hi);
    let t1_hi = _mm512_srli_epi64::<32>(t1);
    let res_hi = _mm512_add_epi64(t2, t1_hi);

    let t1_lo = moveldup_epi32_512(t1);
    let res_lo = _mm512_mask_blend_epi32(0b0101_0101_0101_0101, t1_lo, mul_ll);

    (res_hi, res_lo)
}

/// Number of `Fp32` lanes in an AVX-512 packed vector.
pub const FP32_WIDTH: usize = 16;

/// AVX-512 packed arithmetic for `Fp32<P>`, processing 16 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp32Avx512<const P: u32>(pub [Fp32<P>; FP32_WIDTH]);

impl<const P: u32> PackedFp32Avx512<P> {
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
    fn to_vec(self) -> __m512i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m512i) -> Self {
        unsafe { transmute(v) }
    }

    /// Multiply each `u64` lane by `C`. Building block of Solinas reduction;
    /// the `C == 1` fast path skips the multiply entirely for Mersenne-like
    /// primes.
    ///
    /// Uses `_mm512_mullo_epi64` (AVX-512DQ, single `vpmullq`) for full
    /// 64-bit width. The previous implementation used `_mm512_mul_epu32`
    /// which only reads the *low 32 bits* of each lane and silently dropped
    /// bit 32+ of the input — fine for `BITS == 32` (where the caller's
    /// `prod >> 32` always fits in 32 bits) but wrong for `BITS == 31` and
    /// `C != 1` where `prod >> 31` can occupy 33 bits.
    #[inline(always)]
    unsafe fn mul_c_u64(x: __m512i) -> __m512i {
        if Self::C == 1 {
            x
        } else {
            let c_vec = _mm512_set1_epi64(Self::C as i64);
            _mm512_mullo_epi64(x, c_vec)
        }
    }

    /// Plonky3-style Mersenne31 multiply (P = 2^31 - 1). Specialized using
    /// `_mm512_srli_epi64::<31>` shifts and 16-lane mask blends. Used by
    /// the `Mul` impl when `Self::BITS == 31 && Self::C == 1`.
    #[inline(always)]
    unsafe fn mul_mersenne31_vec(a: __m512i, b: __m512i) -> __m512i {
        unsafe {
            const EVENS: __mmask16 = 0b0101_0101_0101_0101;
            const ODDS: __mmask16 = 0b1010_1010_1010_1010;

            let lhs_evn_dbl = _mm512_add_epi32(a, a);
            let rhs_odd = movehdup_epi32_512(b);
            let lhs_odd_dbl = _mm512_srli_epi64::<31>(a);

            let prod_odd_dbl = _mm512_mul_epu32(lhs_odd_dbl, rhs_odd);
            let prod_evn_dbl = _mm512_mul_epu32(lhs_evn_dbl, b);

            let prod_lo_dbl =
                _mm512_mask_blend_epi32(ODDS, prod_evn_dbl, moveldup_epi32_512(prod_odd_dbl));
            let prod_hi =
                _mm512_mask_blend_epi32(EVENS, prod_odd_dbl, movehdup_epi32_512(prod_evn_dbl));
            let prod_lo = _mm512_srli_epi32::<1>(prod_lo_dbl);

            let p = _mm512_set1_epi32(P as i32);
            let folded = _mm512_add_epi32(prod_lo, prod_hi);
            _mm512_min_epu32(folded, _mm512_sub_epi32(folded, p))
        }
    }

    /// Vector form of field add: 16-lane add + canonicalize to `[0, P)`.
    /// Mirrors `PackedFp32Avx2::add_vec` with native AVX-512 mask compares.
    #[inline(always)]
    unsafe fn add_vec(a: __m512i, b: __m512i) -> __m512i {
        let p = _mm512_set1_epi32(P as i32);
        if Self::BITS <= 31 {
            let t = _mm512_add_epi32(a, b);
            let u = _mm512_sub_epi32(t, p);
            _mm512_min_epu32(t, u)
        } else {
            let c = _mm512_set1_epi32(Self::C as i32);
            let t = _mm512_add_epi32(a, b);
            let overflow = _mm512_cmplt_epu32_mask(t, a);
            let t2 = _mm512_mask_add_epi32(t, overflow, t, c);
            let geq_p = _mm512_cmpge_epu32_mask(t2, p);
            _mm512_mask_sub_epi32(t2, geq_p, t2, p)
        }
    }

    /// Vector form of field sub: 16-lane sub + canonicalize to `[0, P)`.
    /// Mirrors `PackedFp32Avx2::sub_vec` with native AVX-512 mask compares.
    #[inline(always)]
    unsafe fn sub_vec(a: __m512i, b: __m512i) -> __m512i {
        let p = _mm512_set1_epi32(P as i32);
        if Self::BITS <= 31 {
            let t = _mm512_sub_epi32(a, b);
            let u = _mm512_add_epi32(t, p);
            _mm512_min_epu32(t, u)
        } else {
            let t = _mm512_sub_epi32(a, b);
            let underflow = _mm512_cmplt_epu32_mask(a, b);
            _mm512_mask_add_epi32(t, underflow, t, p)
        }
    }

    /// Vector form of field mul: 16-lane Solinas multiply + canonicalize.
    #[inline(always)]
    unsafe fn mul_vec(a: __m512i, b: __m512i) -> __m512i {
        let prod_evn = _mm512_mul_epu32(a, b);
        let a_odd = movehdup_epi32_512(a);
        let b_odd = movehdup_epi32_512(b);
        let prod_odd = _mm512_mul_epu32(a_odd, b_odd);
        Self::solinas_reduce(prod_evn, prod_odd)
    }

    /// `(sum + rhs, carry + overflow_bit)`: 8-lane `u64` accumulating add with
    /// per-lane carry tracking. AVX-512 uses native unsigned mask compare.
    #[inline(always)]
    unsafe fn add_u64_with_carry(sum: __m512i, rhs: __m512i, carry: __m512i) -> (__m512i, __m512i) {
        let next = _mm512_add_epi64(sum, rhs);
        let overflow = _mm512_cmplt_epu64_mask(next, sum);
        let one = _mm512_set1_epi64(1);
        let new_carry = _mm512_mask_add_epi64(carry, overflow, carry, one);
        (next, new_carry)
    }

    /// Multiply each lane's low 32 bits by `Fp32::<P>::SHIFT64_MOD_P`. Used
    /// to fold accumulated overflow count back into the Solinas reduction.
    #[inline(always)]
    unsafe fn carry_correction(carry: __m512i) -> __m512i {
        let shift = _mm512_set1_epi64(Fp32::<P>::SHIFT64_MOD_P as i64);
        _mm512_mul_epu32(carry, shift)
    }

    /// 4-way fused multiply-accumulate with a single end-reduction.
    /// Mirrors `PackedFp32Avx2::dot_product_4_vec` at 16 lanes, including
    /// the `BITS <= 31` carry-free fast path (four `(2^31 - 1)^2` products
    /// sum to less than `2^64`, so partial sums never overflow a `u64`
    /// lane and the `add_u64_with_carry` / `carry_correction` chain drops
    /// out). The `if Self::BITS <= 31` is a const condition and
    /// dead-code-eliminated at compile time.
    #[inline(always)]
    unsafe fn dot_product_4_vec(a: [__m512i; 4], b: [__m512i; 4]) -> __m512i {
        let mut sum_evn = _mm512_mul_epu32(a[0], b[0]);
        let mut sum_odd = _mm512_mul_epu32(movehdup_epi32_512(a[0]), movehdup_epi32_512(b[0]));

        if Self::BITS <= 31 {
            for i in 1..4 {
                let prod_evn = _mm512_mul_epu32(a[i], b[i]);
                let prod_odd = _mm512_mul_epu32(movehdup_epi32_512(a[i]), movehdup_epi32_512(b[i]));
                sum_evn = _mm512_add_epi64(sum_evn, prod_evn);
                sum_odd = _mm512_add_epi64(sum_odd, prod_odd);
            }
            return Self::solinas_reduce(sum_evn, sum_odd);
        }

        let mut carry_evn = _mm512_setzero_si512();
        let mut carry_odd = _mm512_setzero_si512();

        for i in 1..4 {
            let prod_evn = _mm512_mul_epu32(a[i], b[i]);
            let prod_odd = _mm512_mul_epu32(movehdup_epi32_512(a[i]), movehdup_epi32_512(b[i]));
            let (s_evn, c_evn) = Self::add_u64_with_carry(sum_evn, prod_evn, carry_evn);
            let (s_odd, c_odd) = Self::add_u64_with_carry(sum_odd, prod_odd, carry_odd);
            sum_evn = s_evn;
            sum_odd = s_odd;
            carry_evn = c_evn;
            carry_odd = c_odd;
        }

        Self::solinas_reduce_with_carry(sum_evn, sum_odd, carry_evn, carry_odd)
    }

    /// Multiply by an `Fp2` non-residue. Recognizes `nr == -1` and `nr == 2`
    /// fast paths.
    #[inline(always)]
    unsafe fn mul_nr_vec<C>(x: __m512i) -> __m512i
    where
        C: Fp2Config<Fp32<P>>,
    {
        if C::IS_NEG_ONE {
            Self::sub_vec(_mm512_setzero_si512(), x)
        } else if C::non_residue().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_non_residue(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    /// Multiply by the power-basis `w`. Recognizes `w == 2` fast path.
    #[inline(always)]
    unsafe fn mul_w_vec<C>(x: __m512i) -> __m512i
    where
        C: PowerBasisFp4Config<Fp32<P>>,
    {
        if C::w().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_w(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    /// Two-or-three-fold Solinas reduction of 8+8 `u64` products → 16 `u32`
    /// lanes.
    ///
    /// The `Self::BITS == 31` branches use immediate-shift
    /// `_mm512_srli_epi64::<31>` instead of the generic variable-shift
    /// `_mm512_srl_epi64(.., shift)`, mirroring the same specialisation
    /// the base-field `Mul` impl uses on Mersenne31, so extension-field
    /// operations on Mersenne31 get the same per-shift win.
    ///
    /// Two folds always suffice when `Self::TWO_FOLD_FOUR_PRODUCT_OK`. When
    /// it doesn't (large `C` such that `4*C^2 + 3*C > 2^BITS`), we run a
    /// third fold so `pack_and_canonicalize`'s single subtract-and-min step
    /// is enough to land in `[0, P)`. Mirrors `PackedFp32Neon::solinas_reduce`.
    #[inline(always)]
    unsafe fn solinas_reduce(prod_evn: __m512i, prod_odd: __m512i) -> __m512i {
        let mask = _mm512_set1_epi64(Self::MASK_U64 as i64);
        let shift = _mm_set_epi64x(0, Self::BITS as i64);

        // Fold 1
        let evn_lo = _mm512_and_si512(prod_evn, mask);
        let evn_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(prod_evn)
        } else {
            _mm512_srl_epi64(prod_evn, shift)
        };
        let evn_f1 = _mm512_add_epi64(evn_lo, Self::mul_c_u64(evn_hi));

        let odd_lo = _mm512_and_si512(prod_odd, mask);
        let odd_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(prod_odd)
        } else {
            _mm512_srl_epi64(prod_odd, shift)
        };
        let odd_f1 = _mm512_add_epi64(odd_lo, Self::mul_c_u64(odd_hi));

        // Fold 2
        let evn_f1_lo = _mm512_and_si512(evn_f1, mask);
        let evn_f1_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(evn_f1)
        } else {
            _mm512_srl_epi64(evn_f1, shift)
        };
        let evn_f2 = _mm512_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

        let odd_f1_lo = _mm512_and_si512(odd_f1, mask);
        let odd_f1_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(odd_f1)
        } else {
            _mm512_srl_epi64(odd_f1, shift)
        };
        let odd_f2 = _mm512_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

        // Optional third fold for large-C primes (e.g. Generic31Offset32787)
        // where two folds leave residue > 2*P.
        let (evn_final, odd_final) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
            (evn_f2, odd_f2)
        } else {
            let evn_f2_lo = _mm512_and_si512(evn_f2, mask);
            let evn_f2_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(evn_f2)
            } else {
                _mm512_srl_epi64(evn_f2, shift)
            };
            let odd_f2_lo = _mm512_and_si512(odd_f2, mask);
            let odd_f2_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(odd_f2)
            } else {
                _mm512_srl_epi64(odd_f2, shift)
            };
            (
                _mm512_add_epi64(evn_f2_lo, Self::mul_c_u64(evn_f2_hi)),
                _mm512_add_epi64(odd_f2_lo, Self::mul_c_u64(odd_f2_hi)),
            )
        };

        Self::pack_and_canonicalize(evn_final, odd_final)
    }

    /// Same as `solinas_reduce` but with extra carry counts to fold in.
    #[inline(always)]
    unsafe fn solinas_reduce_with_carry(
        prod_evn: __m512i,
        prod_odd: __m512i,
        carry_evn: __m512i,
        carry_odd: __m512i,
    ) -> __m512i {
        let mask = _mm512_set1_epi64(Self::MASK_U64 as i64);
        let shift = _mm_set_epi64x(0, Self::BITS as i64);

        // Fold 1 with carry correction
        let evn_lo = _mm512_and_si512(prod_evn, mask);
        let evn_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(prod_evn)
        } else {
            _mm512_srl_epi64(prod_evn, shift)
        };
        let evn_f1 = _mm512_add_epi64(
            _mm512_add_epi64(evn_lo, Self::mul_c_u64(evn_hi)),
            Self::carry_correction(carry_evn),
        );

        let odd_lo = _mm512_and_si512(prod_odd, mask);
        let odd_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(prod_odd)
        } else {
            _mm512_srl_epi64(prod_odd, shift)
        };
        let odd_f1 = _mm512_add_epi64(
            _mm512_add_epi64(odd_lo, Self::mul_c_u64(odd_hi)),
            Self::carry_correction(carry_odd),
        );

        // Fold 2
        let evn_f1_lo = _mm512_and_si512(evn_f1, mask);
        let evn_f1_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(evn_f1)
        } else {
            _mm512_srl_epi64(evn_f1, shift)
        };
        let evn_f2 = _mm512_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

        let odd_f1_lo = _mm512_and_si512(odd_f1, mask);
        let odd_f1_hi = if Self::BITS == 31 {
            _mm512_srli_epi64::<31>(odd_f1)
        } else {
            _mm512_srl_epi64(odd_f1, shift)
        };
        let odd_f2 = _mm512_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

        // Optional third fold (see `solinas_reduce`).
        let (evn_final, odd_final) = if Self::TWO_FOLD_FOUR_PRODUCT_OK {
            (evn_f2, odd_f2)
        } else {
            let evn_f2_lo = _mm512_and_si512(evn_f2, mask);
            let evn_f2_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(evn_f2)
            } else {
                _mm512_srl_epi64(evn_f2, shift)
            };
            let odd_f2_lo = _mm512_and_si512(odd_f2, mask);
            let odd_f2_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(odd_f2)
            } else {
                _mm512_srl_epi64(odd_f2, shift)
            };
            (
                _mm512_add_epi64(evn_f2_lo, Self::mul_c_u64(evn_f2_hi)),
                _mm512_add_epi64(odd_f2_lo, Self::mul_c_u64(odd_f2_hi)),
            )
        };

        Self::pack_and_canonicalize(evn_final, odd_final)
    }

    /// Combine 8+8 `u64` lanes into 16 `u32` lanes canonicalized to `[0, P)`.
    /// AVX-512 uses native unsigned compare masks and `_mm512_min_epu64`,
    /// simplifying both branches vs the AVX2 implementation.
    #[inline(always)]
    unsafe fn pack_and_canonicalize(evn_f2: __m512i, odd_f2: __m512i) -> __m512i {
        if Self::BITS < 32 {
            let odd_shifted = _mm512_slli_epi64::<32>(odd_f2);
            let combined = _mm512_mask_blend_epi32(0b1010_1010_1010_1010, evn_f2, odd_shifted);
            let p = _mm512_set1_epi32(P as i32);
            let reduced = _mm512_sub_epi32(combined, p);
            _mm512_min_epu32(combined, reduced)
        } else {
            let p_u64 = _mm512_set1_epi64(P as i64);

            let red_evn = _mm512_sub_epi64(evn_f2, p_u64);
            let out_evn = _mm512_min_epu64(evn_f2, red_evn);

            let red_odd = _mm512_sub_epi64(odd_f2, p_u64);
            let out_odd = _mm512_min_epu64(odd_f2, red_odd);

            let odd_shifted = _mm512_slli_epi64::<32>(out_odd);
            _mm512_mask_blend_epi32(0b1010_1010_1010_1010, out_evn, odd_shifted)
        }
    }
}

impl<const P: u32> Default for PackedFp32Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp32(0); FP32_WIDTH])
    }
}

impl<const P: u32> fmt::Debug for PackedFp32Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp32Avx512").field(&self.0).finish()
    }
}

impl<const P: u32> PartialEq for PackedFp32Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u32> Eq for PackedFp32Avx512<P> {}

impl<const P: u32> Add for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { Self::from_vec(Self::add_vec(self.to_vec(), rhs.to_vec())) }
    }
}

impl<const P: u32> Sub for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Self::from_vec(Self::sub_vec(self.to_vec(), rhs.to_vec())) }
    }
}

impl<const P: u32> Mul for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();

            if Self::BITS == 31 && Self::C == 1 {
                return Self::from_vec(Self::mul_mersenne31_vec(a, b));
            }

            let prod_evn = _mm512_mul_epu32(a, b);
            let a_odd = movehdup_epi32_512(a);
            let b_odd = movehdup_epi32_512(b);
            let prod_odd = _mm512_mul_epu32(a_odd, b_odd);

            let mask = _mm512_set1_epi64(Self::MASK_U64 as i64);
            let shift = _mm_set_epi64x(0, Self::BITS as i64);

            // Fold 1
            let evn_lo = _mm512_and_si512(prod_evn, mask);
            let evn_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(prod_evn)
            } else {
                _mm512_srl_epi64(prod_evn, shift)
            };
            let evn_f1 = _mm512_add_epi64(evn_lo, Self::mul_c_u64(evn_hi));

            let odd_lo = _mm512_and_si512(prod_odd, mask);
            let odd_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(prod_odd)
            } else {
                _mm512_srl_epi64(prod_odd, shift)
            };
            let odd_f1 = _mm512_add_epi64(odd_lo, Self::mul_c_u64(odd_hi));

            // Fold 2
            let evn_f1_lo = _mm512_and_si512(evn_f1, mask);
            let evn_f1_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(evn_f1)
            } else {
                _mm512_srl_epi64(evn_f1, shift)
            };
            let evn_f2 = _mm512_add_epi64(evn_f1_lo, Self::mul_c_u64(evn_f1_hi));

            let odd_f1_lo = _mm512_and_si512(odd_f1, mask);
            let odd_f1_hi = if Self::BITS == 31 {
                _mm512_srli_epi64::<31>(odd_f1)
            } else {
                _mm512_srl_epi64(odd_f1, shift)
            };
            let odd_f2 = _mm512_add_epi64(odd_f1_lo, Self::mul_c_u64(odd_f1_hi));

            // Recombine even/odd
            let odd_shifted = _mm512_slli_epi64::<32>(odd_f2);
            let combined = _mm512_mask_blend_epi32(0b1010101010101010, evn_f2, odd_shifted);

            // Conditional subtract P
            let p_vec = _mm512_set1_epi32(P as i32);
            let reduced = _mm512_sub_epi32(combined, p_vec);
            Self::from_vec(_mm512_min_epu32(combined, reduced))
        }
    }
}

impl<const P: u32> PackedValue for PackedFp32Avx512<P> {
    type Value = Fp32<P>;
    const WIDTH: usize = FP32_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([
            f(0),
            f(1),
            f(2),
            f(3),
            f(4),
            f(5),
            f(6),
            f(7),
            f(8),
            f(9),
            f(10),
            f(11),
            f(12),
            f(13),
            f(14),
            f(15),
        ])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP32_WIDTH);
        self.0[lane]
    }
}

impl<const P: u32> AddAssign for PackedFp32Avx512<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp32Avx512<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp32Avx512<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> PackedField for PackedFp32Avx512<P> {
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
            let zero = _mm512_setzero_si512();
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

/// Number of `Fp64` lanes in an AVX-512 packed vector.
pub const FP64_WIDTH: usize = 8;

/// AVX-512 packed arithmetic for `Fp64<P>`, processing 8 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp64Avx512<const P: u64>(pub [Fp64<P>; FP64_WIDTH]);

impl<const P: u64> PackedFp64Avx512<P> {
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
    fn to_vec(self) -> __m512i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m512i) -> Self {
        unsafe { transmute(v) }
    }

    /// Vectorized 128-bit Solinas reduction for p = 2^BITS - C.
    /// Given (hi, lo) = 128-bit product, computes result ≡ (hi*2^64 + lo) mod p.
    #[inline]
    unsafe fn reduce128_vec(hi: __m512i, lo: __m512i) -> __m512i {
        if Self::BITS < 64 {
            Self::reduce128_small_k(hi, lo)
        } else {
            Self::reduce128_full_k(hi, lo)
        }
    }

    /// Reduction for BITS < 64 (e.g. 40-bit prime). No overflow issues: all
    /// intermediates fit in u64.
    #[inline]
    unsafe fn reduce128_small_k(hi: __m512i, lo: __m512i) -> __m512i {
        let mask_k = _mm512_set1_epi64(Self::MASK64 as i64);
        let c_vec = _mm512_set1_epi64(Self::C_LO as i64);
        let p_vec = _mm512_set1_epi64(P as i64);
        let shift_k = _mm_set_epi64x(0, Self::BITS as i64);
        let shift_64mk = _mm_set_epi64x(0, (64 - Self::BITS) as i64);

        let lo_k = _mm512_and_si512(lo, mask_k);
        let lo_upper = _mm512_srl_epi64(lo, shift_k);
        let hi_shifted = _mm512_sll_epi64(hi, shift_64mk);
        let hi_k = _mm512_or_si512(lo_upper, hi_shifted);

        // c * hi_k: hi_k may exceed 32 bits, split into lo32 and top
        let c_hi_lo = _mm512_mul_epu32(c_vec, hi_k);
        let hi_k_top = _mm512_srli_epi64::<32>(hi_k);
        let c_hi_top = _mm512_mul_epu32(c_vec, hi_k_top);
        let c_hi_top_shifted = _mm512_slli_epi64::<32>(c_hi_top);
        let c_hi_full = _mm512_add_epi64(c_hi_lo, c_hi_top_shifted);

        let fold1 = _mm512_add_epi64(lo_k, c_hi_full);

        let fold1_lo_k = _mm512_and_si512(fold1, mask_k);
        let fold1_hi = _mm512_srl_epi64(fold1, shift_k);
        let c_fold1_hi = _mm512_mul_epu32(c_vec, fold1_hi);
        let fold2 = _mm512_add_epi64(fold1_lo_k, c_fold1_hi);

        let reduced = _mm512_sub_epi64(fold2, p_vec);
        _mm512_min_epu64(fold2, reduced)
    }

    /// Reduction for BITS == 64 (e.g. p = 2^64 - 87). Tracks overflow from
    /// c*hi exceeding 64 bits, using native unsigned comparisons.
    #[inline]
    unsafe fn reduce128_full_k(hi: __m512i, lo: __m512i) -> __m512i {
        let c_vec = _mm512_set1_epi64(Self::C_LO as i64);
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

            let result = if Self::BITS <= 62 {
                let s = _mm512_add_epi64(a, b);
                let geq_p = _mm512_cmpge_epu64_mask(s, p);
                _mm512_mask_sub_epi64(s, geq_p, s, p)
            } else {
                let s = _mm512_add_epi64(a, b);
                let overflow = _mm512_cmplt_epu64_mask(s, a);
                let c = _mm512_set1_epi64(Self::C_LO as i64);
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
            let underflow = _mm512_cmplt_epu64_mask(a, b);
            Self::from_vec(_mm512_mask_add_epi64(d, underflow, d, p))
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
}

/// Number of `Fp128` lanes in an AVX-512 packed vector.
pub const FP128_WIDTH: usize = 8;

/// AVX-512 packed arithmetic for `Fp128<P>`, 8 lanes in SoA layout.
///
/// Stores 8 elements as separate `lo` and `hi` `u64` arrays, enabling
/// vectorized add/sub via `__m512i`.  Mul remains scalar per-lane.
#[derive(Clone, Copy)]
pub struct PackedFp128Avx512<const P: u128> {
    lo: [u64; FP128_WIDTH],
    hi: [u64; FP128_WIDTH],
}

impl<const P: u128> PackedFp128Avx512<P> {
    const P_LO: u64 = P as u64;
    const P_HI: u64 = (P >> 64) as u64;
}

impl<const P: u128> Default for PackedFp128Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self {
            lo: [0; FP128_WIDTH],
            hi: [0; FP128_WIDTH],
        }
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems: Vec<_> = (0..FP128_WIDTH).map(|i| self.extract(i)).collect();
        f.debug_tuple("PackedFp128Avx512").field(&elems).finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

impl<const P: u128> Eq for PackedFp128Avx512<P> {}

impl<const P: u128> Add for PackedFp128Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm512_loadu_si512(self.lo.as_ptr().cast());
            let a_hi = _mm512_loadu_si512(self.hi.as_ptr().cast());
            let b_lo = _mm512_loadu_si512(rhs.lo.as_ptr().cast());
            let b_hi = _mm512_loadu_si512(rhs.hi.as_ptr().cast());
            let p_lo = _mm512_set1_epi64(Self::P_LO as i64);
            let p_hi = _mm512_set1_epi64(Self::P_HI as i64);
            let one = _mm512_set1_epi64(1);

            // 128-bit add: (sum_hi, sum_lo) = (a_hi, a_lo) + (b_hi, b_lo)
            let sum_lo = _mm512_add_epi64(a_lo, b_lo);
            let carry_lo = _mm512_cmplt_epu64_mask(sum_lo, a_lo);
            let hi_tmp = _mm512_add_epi64(a_hi, b_hi);
            let ov1 = _mm512_cmplt_epu64_mask(hi_tmp, a_hi);
            let sum_hi = _mm512_mask_add_epi64(hi_tmp, carry_lo, hi_tmp, one);
            let ov2 = _mm512_cmplt_epu64_mask(sum_hi, hi_tmp);
            let carry_128 = ov1 | ov2;

            // 128-bit subtract P: (red_hi, red_lo) = (sum_hi, sum_lo) - P
            let red_lo = _mm512_sub_epi64(sum_lo, p_lo);
            let borrow_lo = _mm512_cmplt_epu64_mask(sum_lo, p_lo);
            let red_hi_tmp = _mm512_sub_epi64(sum_hi, p_hi);
            let bw1 = _mm512_cmplt_epu64_mask(sum_hi, p_hi);
            let red_hi = _mm512_mask_sub_epi64(red_hi_tmp, borrow_lo, red_hi_tmp, one);
            let bw2 = _mm512_cmplt_epu64_mask(red_hi_tmp, _mm512_maskz_mov_epi64(borrow_lo, one));
            let borrow = bw1 | bw2;

            // Use reduced if: overflow happened OR subtraction didn't borrow
            let use_reduced = carry_128 | !borrow;
            let out_lo = _mm512_mask_blend_epi64(use_reduced, sum_lo, red_lo);
            let out_hi = _mm512_mask_blend_epi64(use_reduced, sum_hi, red_hi);

            let mut result = Self::default();
            _mm512_storeu_si512(result.lo.as_mut_ptr().cast(), out_lo);
            _mm512_storeu_si512(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Sub for PackedFp128Avx512<P> {
    type Output = Self;
    // `bw1 | bw2` below is correct 128-bit borrow wiring (mask OR), not an
    // arithmetic bug; suppress the lint locally rather than module-wide.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a_lo = _mm512_loadu_si512(self.lo.as_ptr().cast());
            let a_hi = _mm512_loadu_si512(self.hi.as_ptr().cast());
            let b_lo = _mm512_loadu_si512(rhs.lo.as_ptr().cast());
            let b_hi = _mm512_loadu_si512(rhs.hi.as_ptr().cast());
            let p_lo = _mm512_set1_epi64(Self::P_LO as i64);
            let p_hi = _mm512_set1_epi64(Self::P_HI as i64);
            let one = _mm512_set1_epi64(1);

            // 128-bit sub: (diff_hi, diff_lo) = (a_hi, a_lo) - (b_hi, b_lo)
            let diff_lo = _mm512_sub_epi64(a_lo, b_lo);
            let borrow_lo = _mm512_cmplt_epu64_mask(a_lo, b_lo);
            let hi_tmp = _mm512_sub_epi64(a_hi, b_hi);
            let bw1 = _mm512_cmplt_epu64_mask(a_hi, b_hi);
            let diff_hi = _mm512_mask_sub_epi64(hi_tmp, borrow_lo, hi_tmp, one);
            let bw2 = _mm512_cmplt_epu64_mask(hi_tmp, _mm512_maskz_mov_epi64(borrow_lo, one));
            let borrow_128 = bw1 | bw2;

            // Correction: add P back where underflow occurred
            let corr_lo = _mm512_add_epi64(diff_lo, p_lo);
            let carry_lo = _mm512_cmplt_epu64_mask(corr_lo, diff_lo);
            let corr_hi = _mm512_add_epi64(diff_hi, p_hi);
            let corr_hi = _mm512_mask_add_epi64(corr_hi, carry_lo, corr_hi, one);

            let out_lo = _mm512_mask_blend_epi64(borrow_128, diff_lo, corr_lo);
            let out_hi = _mm512_mask_blend_epi64(borrow_128, diff_hi, corr_hi);

            let mut result = Self::default();
            _mm512_storeu_si512(result.lo.as_mut_ptr().cast(), out_lo);
            _mm512_storeu_si512(result.hi.as_mut_ptr().cast(), out_hi);
            result
        }
    }
}

impl<const P: u128> Mul for PackedFp128Avx512<P> {
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

impl<const P: u128> PackedValue for PackedFp128Avx512<P> {
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

impl<const P: u128> AddAssign for PackedFp128Avx512<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for PackedFp128Avx512<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for PackedFp128Avx512<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u128> PackedField for PackedFp128Avx512<P> {
    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self {
            lo: [value.0[0]; FP128_WIDTH],
            hi: [value.0[1]; FP128_WIDTH],
        }
    }
}

// ---------------------------------------------------------------------------
// PackedFp16Avx512 — 16 lanes of Fp16 in __m512i (widened to u32)
// ---------------------------------------------------------------------------

/// Number of packed `Fp16` lanes in an AVX-512 vector.
///
/// Each u16 element is widened to u32 for arithmetic, so 16 lanes
/// fit in one `__m512i` (16 × 32 = 512 bits).
pub const FP16_WIDTH: usize = 16;

/// AVX-512 packed `Fp16` backend: 16 lanes, stored as `[u16; 16]`,
/// widened to u32 in `__m512i` for all arithmetic.
#[derive(Clone, Copy)]
pub struct PackedFp16Avx512<const P: u32> {
    vals: [u16; FP16_WIDTH],
}

impl<const P: u32> PackedFp16Avx512<P> {
    const BITS: u32 = 32 - P.leading_zeros();

    /// Widen 16 u16 values to 16 u32 values in `__m512i`.
    #[inline(always)]
    unsafe fn widen(x: &[u16; FP16_WIDTH]) -> __m512i {
        let lo = _mm256_loadu_si256(x.as_ptr().cast());
        _mm512_cvtepu16_epi32(lo)
    }

    /// Truncate 16 u32 values (in [0, P)) back to `[u16; 16]`.
    #[inline(always)]
    unsafe fn narrow(v: __m512i) -> [u16; FP16_WIDTH] {
        let lo = _mm512_cvtepi32_epi16(v);
        let mut out = [0u16; FP16_WIDTH];
        _mm256_storeu_si256(out.as_mut_ptr().cast(), lo);
        out
    }

    #[inline(always)]
    unsafe fn add_vec(a: __m512i, b: __m512i) -> __m512i {
        let p32 = _mm512_set1_epi32(P as i32);
        let t = _mm512_add_epi32(a, b);
        let u = _mm512_sub_epi32(t, p32);
        _mm512_min_epu32(t, u)
    }

    #[inline(always)]
    unsafe fn sub_vec(a: __m512i, b: __m512i) -> __m512i {
        let p32 = _mm512_set1_epi32(P as i32);
        let t = _mm512_sub_epi32(a, b);
        let u = _mm512_add_epi32(t, p32);
        _mm512_min_epu32(t, u)
    }

    #[inline(always)]
    unsafe fn mul_vec(a: __m512i, b: __m512i) -> __m512i {
        let prod = _mm512_mullo_epi32(a, b);
        Self::solinas_reduce(prod)
    }

    /// Three-fold Solinas reduction of 16 u32 products in `__m512i`.
    ///
    /// Three folds suffice for all valid `Fp16<P>` parameters
    /// (`BITS ≤ 16`, `C(C+1) < P`). Worst-case bound after fold 3:
    ///   fold1 ≤ (C+1)·2^BITS → fold2 ≤ 2^BITS + C² − 2C
    ///   fold3 ≤ C² − C − 1 < 2^BITS (since C < √P ≤ 2⁸).
    #[inline(always)]
    unsafe fn solinas_reduce(prod: __m512i) -> __m512i {
        let mask = _mm512_set1_epi32((1u32 << Self::BITS) as i32 - 1);
        let c = _mm512_set1_epi32(Fp16::<P>::C as i32);
        let shift = _mm_set_epi64x(0, Self::BITS as i64);

        let fold = |x: __m512i| -> __m512i {
            let lo = _mm512_and_si512(x, mask);
            let hi = _mm512_srl_epi32(x, shift);
            _mm512_add_epi32(lo, _mm512_mullo_epi32(hi, c))
        };

        let f1 = fold(prod);
        let f2 = fold(f1);
        let f3 = fold(f2);

        let p32 = _mm512_set1_epi32(P as i32);
        _mm512_min_epu32(f3, _mm512_sub_epi32(f3, p32))
    }
}

impl<const P: u32> PackedValue for PackedFp16Avx512<P> {
    type Value = Fp16<P>;
    const WIDTH: usize = FP16_WIDTH;

    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        let vals: [Fp16<P>; FP16_WIDTH] = std::array::from_fn(f);
        Self {
            vals: vals.map(|v| v.to_limbs()),
        }
    }

    fn extract(&self, lane: usize) -> Self::Value {
        Fp16::from_canonical_u16(self.vals[lane])
    }
}

impl<const P: u32> fmt::Debug for PackedFp16Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.vals.iter()).finish()
    }
}

impl<const P: u32> Default for PackedFp16Avx512<P> {
    fn default() -> Self {
        Self {
            vals: [0; FP16_WIDTH],
        }
    }
}

impl<const P: u32> PartialEq for PackedFp16Avx512<P> {
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u32> Eq for PackedFp16Avx512<P> {}

impl<const P: u32> Add for PackedFp16Avx512<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let a = Self::widen(&self.vals);
            let b = Self::widen(&rhs.vals);
            Self {
                vals: Self::narrow(Self::add_vec(a, b)),
            }
        }
    }
}

impl<const P: u32> Sub for PackedFp16Avx512<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a = Self::widen(&self.vals);
            let b = Self::widen(&rhs.vals);
            Self {
                vals: Self::narrow(Self::sub_vec(a, b)),
            }
        }
    }
}

impl<const P: u32> Mul for PackedFp16Avx512<P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a = Self::widen(&self.vals);
            let b = Self::widen(&rhs.vals);
            Self {
                vals: Self::narrow(Self::mul_vec(a, b)),
            }
        }
    }
}

impl<const P: u32> AddAssign for PackedFp16Avx512<P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp16Avx512<P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp16Avx512<P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Chebyshev φ fold-back for AVX-512 Fp16 `__m512i` accumulators (K=8).
#[inline(always)]
unsafe fn avx512_ring_subfield_fp8_add_phi_16<const P: u32>(
    out: &mut [__m512i; 8],
    idx: usize,
    value: __m512i,
) {
    match idx {
        0 => {
            out[0] =
                PackedFp16Avx512::<P>::add_vec(out[0], PackedFp16Avx512::<P>::add_vec(value, value))
        }
        1..=7 => out[idx] = PackedFp16Avx512::<P>::add_vec(out[idx], value),
        8 => {}
        9..=15 => out[16 - idx] = PackedFp16Avx512::<P>::sub_vec(out[16 - idx], value),
        _ => unreachable!(),
    }
}

impl<const P: u32> PackedField for PackedFp16Avx512<P> {
    type Scalar = Fp16<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self {
            vals: [value.to_limbs(); FP16_WIDTH],
        }
    }

    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }

    /// Chebyshev-basis Karatsuba multiplication for `RingSubfieldFp8` lanes.
    ///
    /// Widens all 8 coefficients to u32 once at entry, runs the full fp8
    /// algorithm in `__m512i` u32 arithmetic, and narrows back at exit.
    #[inline(always)]
    fn ring_subfield_fp8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        unsafe {
            let a: [__m512i; 8] = std::array::from_fn(|i| Self::widen(&a[i].vals));
            let b: [__m512i; 8] = std::array::from_fn(|i| Self::widen(&b[i].vals));
            let zero = _mm512_setzero_si512();
            let mut out = [zero; 8];

            let diag: [__m512i; 8] = std::array::from_fn(|i| Self::mul_vec(a[i], b[i]));
            out[0] = diag[0];

            for k in 1..8 {
                let mixed = Self::sub_vec(
                    Self::sub_vec(
                        Self::mul_vec(Self::add_vec(a[0], a[k]), Self::add_vec(b[0], b[k])),
                        diag[0],
                    ),
                    diag[k],
                );
                out[k] = Self::add_vec(out[k], mixed);
            }

            for (i, &diag_i) in diag.iter().enumerate().skip(1) {
                out[0] = Self::add_vec(out[0], Self::add_vec(diag_i, diag_i));
                avx512_ring_subfield_fp8_add_phi_16::<P>(&mut out, i + i, diag_i);
            }

            for i in 1..8usize {
                for j in (i + 1)..8usize {
                    let mixed = Self::sub_vec(
                        Self::sub_vec(
                            Self::mul_vec(Self::add_vec(a[i], a[j]), Self::add_vec(b[i], b[j])),
                            diag[i],
                        ),
                        diag[j],
                    );
                    avx512_ring_subfield_fp8_add_phi_16::<P>(&mut out, i + j, mixed);
                    avx512_ring_subfield_fp8_add_phi_16::<P>(&mut out, j - i, mixed);
                }
            }

            std::array::from_fn(|i| Self {
                vals: Self::narrow(out[i]),
            })
        }
    }

    #[inline(always)]
    fn ring_subfield_fp8_square(a: [Self; 8]) -> [Self; 8] {
        unsafe {
            let a: [__m512i; 8] = std::array::from_fn(|i| Self::widen(&a[i].vals));
            let zero = _mm512_setzero_si512();
            let mut out = [zero; 8];

            let sq: [__m512i; 8] = std::array::from_fn(|i| Self::mul_vec(a[i], a[i]));
            out[0] = sq[0];

            for k in 1..8 {
                let cross = Self::mul_vec(a[0], a[k]);
                out[k] = Self::add_vec(out[k], Self::add_vec(cross, cross));
            }

            for (i, &sq_i) in sq.iter().enumerate().skip(1) {
                out[0] = Self::add_vec(out[0], Self::add_vec(sq_i, sq_i));
                avx512_ring_subfield_fp8_add_phi_16::<P>(&mut out, i + i, sq_i);
            }

            for i in 1..8usize {
                for j in (i + 1)..8usize {
                    let cross = Self::mul_vec(a[i], a[j]);
                    let doubled = Self::add_vec(cross, cross);
                    avx512_ring_subfield_fp8_add_phi_16::<P>(&mut out, i + j, doubled);
                    avx512_ring_subfield_fp8_add_phi_16::<P>(&mut out, j - i, doubled);
                }
            }

            std::array::from_fn(|i| Self {
                vals: Self::narrow(out[i]),
            })
        }
    }
}
