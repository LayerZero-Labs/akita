//! AVX-512 packed backends for Fp32, Fp64, Fp128.
//!
//! Requires AVX-512F + AVX-512DQ. Uses native unsigned comparisons and mask
//! registers for branchless conditionals.

use super::packed::{PackedField, PackedValue};
use crate::algebra::fields::{Fp128, Fp32, Fp64};
use crate::FieldCore;
use core::arch::x86_64::*;
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, Mul, Sub};

// ===== Helpers =====

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

// ===== PackedFp32Avx512 (16-wide) =====

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

    #[inline(always)]
    fn to_vec(self) -> __m512i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    unsafe fn from_vec(v: __m512i) -> Self {
        unsafe { transmute(v) }
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
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm512_set1_epi32(P as i32);

            let result = if Self::BITS <= 31 {
                let t = _mm512_add_epi32(a, b);
                let u = _mm512_sub_epi32(t, p);
                _mm512_min_epu32(t, u)
            } else {
                let c = _mm512_set1_epi32(Self::C as i32);
                let t = _mm512_add_epi32(a, b);
                // Step 1: correct overflow (2^32 ≡ C mod P)
                let overflow = _mm512_cmplt_epu32_mask(t, a);
                let t2 = _mm512_mask_add_epi32(t, overflow, t, c);
                // Step 2: subtract P if t2 >= P
                let geq_p = _mm512_cmpge_epu32_mask(t2, p);
                _mm512_mask_sub_epi32(t2, geq_p, t2, p)
            };

            Self::from_vec(result)
        }
    }
}

impl<const P: u32> Sub for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();
            let p = _mm512_set1_epi32(P as i32);

            let result = if Self::BITS <= 31 {
                let t = _mm512_sub_epi32(a, b);
                let u = _mm512_add_epi32(t, p);
                _mm512_min_epu32(t, u)
            } else {
                let t = _mm512_sub_epi32(a, b);
                let underflow = _mm512_cmplt_epu32_mask(a, b);
                _mm512_mask_add_epi32(t, underflow, t, p)
            };

            Self::from_vec(result)
        }
    }
}

impl<const P: u32> Mul for PackedFp32Avx512<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a = self.to_vec();
            let b = rhs.to_vec();

            let prod_evn = _mm512_mul_epu32(a, b);
            let a_odd = movehdup_epi32_512(a);
            let b_odd = movehdup_epi32_512(b);
            let prod_odd = _mm512_mul_epu32(a_odd, b_odd);

            let mask = _mm512_set1_epi64(Self::MASK_U64 as i64);
            let c_vec = _mm512_set1_epi64(Self::C as i64);
            let shift = _mm_set_epi64x(0, Self::BITS as i64);

            // Fold 1
            let evn_lo = _mm512_and_si512(prod_evn, mask);
            let evn_hi = _mm512_srl_epi64(prod_evn, shift);
            let evn_f1 = _mm512_add_epi64(evn_lo, _mm512_mul_epu32(evn_hi, c_vec));

            let odd_lo = _mm512_and_si512(prod_odd, mask);
            let odd_hi = _mm512_srl_epi64(prod_odd, shift);
            let odd_f1 = _mm512_add_epi64(odd_lo, _mm512_mul_epu32(odd_hi, c_vec));

            // Fold 2
            let evn_f1_lo = _mm512_and_si512(evn_f1, mask);
            let evn_f1_hi = _mm512_srl_epi64(evn_f1, shift);
            let evn_f2 = _mm512_add_epi64(evn_f1_lo, _mm512_mul_epu32(evn_f1_hi, c_vec));

            let odd_f1_lo = _mm512_and_si512(odd_f1, mask);
            let odd_f1_hi = _mm512_srl_epi64(odd_f1, shift);
            let odd_f2 = _mm512_add_epi64(odd_f1_lo, _mm512_mul_epu32(odd_f1_hi, c_vec));

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

impl<const P: u32> PackedField for PackedFp32Avx512<P> {
    type Scalar = Fp32<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP32_WIDTH])
    }
}

// ===== PackedFp64Avx512 (8-wide) =====

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

impl<const P: u64> PackedField for PackedFp64Avx512<P> {
    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP64_WIDTH])
    }
}

// ===== PackedFp128Avx512 (4-wide) =====

/// Number of `Fp128` lanes in an AVX-512 packed vector.
pub const FP128_WIDTH: usize = 4;

/// AVX-512 packed arithmetic for `Fp128<P>`, processing 4 lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedFp128Avx512<const P: u128>(pub [Fp128<P>; FP128_WIDTH]);

impl<const P: u128> PackedFp128Avx512<P> {
    const C: u128 = {
        let c = 0u128.wrapping_sub(P);
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(c < (1u128 << 64), "P must be 2^128 - c with c < 2^64");
        assert!(
            c * (c + 1) < P,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };
    const C_LO: u64 = Self::C as u64;
}

impl<const P: u128> Default for PackedFp128Avx512<P> {
    #[inline]
    fn default() -> Self {
        Self([Fp128::zero(); FP128_WIDTH])
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Avx512<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp128Avx512").field(&self.0).finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Avx512<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u128> Eq for PackedFp128Avx512<P> {}

impl<const P: u128> Add for PackedFp128Avx512<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl<const P: u128> Sub for PackedFp128Avx512<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl<const P: u128> Mul for PackedFp128Avx512<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
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
        Self([f(0), f(1), f(2), f(3)])
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP128_WIDTH);
        self.0[lane]
    }
}

impl<const P: u128> PackedField for PackedFp128Avx512<P> {
    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self([value; FP128_WIDTH])
    }
}
