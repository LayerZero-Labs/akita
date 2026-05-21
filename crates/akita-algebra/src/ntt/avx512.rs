//! x86_64 AVX-512 SIMD kernels for NTT butterfly, Montgomery multiply, and
//! pointwise operations.
//!
//! Requires AVX-512F + AVX-512DQ + AVX-512BW. The BW extension is needed for
//! the 16-bit `_mm512_mullo_epi16`/`_mm512_mulhi_epi16` used by the i16 path.
//!
//! Provides vectorized i32 (16 lanes, for Q64/Q128) and i16 (32 lanes, for
//! Q32) paths, mirroring [`super::neon`] and [`super::avx2`].
//!
//! Dispatch is controlled by [`super::use_simd_ntt`] (defined once at the
//! `ntt` module level and shared across all SIMD backends).

// AVX-512 intrinsics in stdarch became stable in Rust 1.89; this file is only
// compiled with AVX-512 enabled, which requires either nightly (for older
// rustc) or 1.89+. The workspace MSRV of 1.88 does not apply here.
#![allow(clippy::incompatible_msrv)]

use std::arch::x86_64::*;

use super::butterfly::NttTwiddles;
use super::prime::{MontCoeff, NttPrime};

// ============================================================================
// i32 path (R = 2^32, for primes < 2^30)
// ============================================================================

/// 16-wide Montgomery multiply for i32 primes.
#[inline(always)]
unsafe fn mont_mul_16x_i32(a: __m512i, b: __m512i, p: __m512i, pinv: __m512i) -> __m512i {
    let c_evn = _mm512_mul_epi32(a, b);
    let a_odd = _mm512_srli_epi64::<32>(a);
    let b_odd = _mm512_srli_epi64::<32>(b);
    let c_odd = _mm512_mul_epi32(a_odd, b_odd);

    let t_evn = _mm512_mullo_epi32(c_evn, pinv);
    let t_odd = _mm512_mullo_epi32(c_odd, pinv);

    let tp_evn = _mm512_mul_epi32(t_evn, p);
    let tp_odd = _mm512_mul_epi32(t_odd, p);

    let r_evn = _mm512_srli_epi64::<32>(_mm512_sub_epi64(c_evn, tp_evn));
    let r_odd = _mm512_srli_epi64::<32>(_mm512_sub_epi64(c_odd, tp_odd));

    let r_odd_shifted = _mm512_slli_epi64::<32>(r_odd);
    _mm512_mask_blend_epi32(0b1010_1010_1010_1010, r_evn, r_odd_shifted)
}

/// 16-wide range reduction for i32: maps `(-2p, 2p)` → `(-p, p)`. AVX-512
/// uses native signed mask compares for branchless conditionals.
#[inline(always)]
unsafe fn reduce_range_16x_i32(a: __m512i, p: __m512i) -> __m512i {
    let ge_mask = _mm512_cmpge_epi32_mask(a, p);
    let after_sub = _mm512_mask_sub_epi32(a, ge_mask, a, p);
    let zero = _mm512_setzero_si512();
    let lt_mask = _mm512_cmplt_epi32_mask(after_sub, zero);
    _mm512_mask_add_epi32(after_sub, lt_mask, after_sub, p)
}

pub(crate) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm512_set1_epi32(prime.p);
    let pinv_v = _mm512_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i32;
        let mut i = 0;
        while i + 16 <= D {
            let ai = _mm512_loadu_si512(a_ptr.add(i).cast());
            let psi = _mm512_loadu_si512(psi_ptr.add(i).cast());
            _mm512_storeu_si512(a_ptr.add(i).cast(), mont_mul_16x_i32(ai, psi, p_v, pinv_v));
            i += 16;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.psi_pows[i]);
            i += 1;
        }
    }

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm512_add_epi32(u, v);
                    let diff = _mm512_sub_epi32(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i32(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_16x_i32(diff, w, p_v, pinv_v),
                    );
                    j += 16;
                }
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    reduce_range_in_place_i32(a, p_v);
}

pub(crate) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm512_set1_epi32(prime.p);
    let pinv_v = _mm512_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v_raw = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_16x_i32(v_raw, w, p_v, pinv_v);
                    let sum = _mm512_add_epi32(u, v);
                    let diff = _mm512_sub_epi32(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i32(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_16x_i32(diff, p_v),
                    );
                    j += 16;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }

    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i32;
        let mut i = 0;
        while i + 16 <= D {
            let ai = _mm512_loadu_si512(a_ptr.add(i).cast());
            let f = _mm512_loadu_si512(fused_ptr.add(i).cast());
            _mm512_storeu_si512(a_ptr.add(i).cast(), mont_mul_16x_i32(ai, f, p_v, pinv_v));
            i += 16;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.d_inv_psi_inv[i]);
            i += 1;
        }
    }
}

pub(crate) unsafe fn forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm512_set1_epi32(prime.p);
    let pinv_v = _mm512_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm512_add_epi32(u, v);
                    let diff = _mm512_sub_epi32(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i32(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_16x_i32(diff, w, p_v, pinv_v),
                    );
                    j += 16;
                }
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }
    reduce_range_in_place_i32(a, p_v);
}

pub(crate) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm512_set1_epi32(prime.p);
    let pinv_v = _mm512_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v_raw = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_16x_i32(v_raw, w, p_v, pinv_v);
                    let sum = _mm512_add_epi32(u, v);
                    let diff = _mm512_sub_epi32(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i32(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_16x_i32(diff, p_v),
                    );
                    j += 16;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }

    {
        let d_inv = tw.d_inv;
        let d_inv_v = _mm512_set1_epi32(d_inv.raw());
        let mut i = 0;
        while i + 16 <= D {
            let ai = _mm512_loadu_si512(a_ptr.add(i).cast());
            _mm512_storeu_si512(
                a_ptr.add(i).cast(),
                mont_mul_16x_i32(ai, d_inv_v, p_v, pinv_v),
            );
            i += 16;
        }
        while i < D {
            a[i] = prime.mul(a[i], d_inv);
            i += 1;
        }
    }
}

/// 16-wide pointwise multiply-accumulate for a single CRT limb (i32).
///
/// # Safety
///
/// `acc`, `lhs`, `rhs` must be valid for `d` elements; `acc` must be uniquely
/// borrowed.
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let pinv_v = _mm512_set1_epi32(pinv);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 16 <= d {
        let a = _mm512_loadu_si512(acc.add(i).cast());
        let l = _mm512_loadu_si512(lhs.add(i).cast());
        let r = _mm512_loadu_si512(rhs.add(i).cast());
        let prod = mont_mul_16x_i32(l, r, p_v, pinv_v);
        let sum = _mm512_add_epi32(a, prod);
        _mm512_storeu_si512(acc.add(i).cast(), reduce_range_16x_i32(sum, p_v));
        i += 16;
    }
    while i < d {
        let prod = prime.mul(
            MontCoeff::from_raw(*lhs.add(i)),
            MontCoeff::from_raw(*rhs.add(i)),
        );
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

/// 16-wide add-and-reduce for a single CRT limb (i32).
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements and must not alias.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_v = _mm512_set1_epi32(p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 16 <= d {
        let a = _mm512_loadu_si512(acc.add(i).cast());
        let b = _mm512_loadu_si512(other.add(i).cast());
        _mm512_storeu_si512(
            acc.add(i).cast(),
            reduce_range_16x_i32(_mm512_add_epi32(a, b), p_v),
        );
        i += 16;
    }
    while i < d {
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

unsafe fn reduce_range_in_place_i32<const D: usize>(a: &mut [MontCoeff<i32>; D], p_v: __m512i) {
    let ptr = a.as_mut_ptr() as *mut i32;
    let mut i = 0;
    while i + 16 <= D {
        let val = _mm512_loadu_si512(ptr.add(i).cast());
        _mm512_storeu_si512(ptr.add(i).cast(), reduce_range_16x_i32(val, p_v));
        i += 16;
    }
}

// ============================================================================
// i16 path (R = 2^16, for primes < 2^14) — requires AVX-512BW
// ============================================================================

/// 32-wide Montgomery multiply for i16 primes. Same algebra as AVX2's
/// `mont_mul_16x_i16` but at twice the lane count.
#[inline(always)]
unsafe fn mont_mul_32x_i16(a: __m512i, b: __m512i, p: __m512i, pinv: __m512i) -> __m512i {
    let c_lo = _mm512_mullo_epi16(a, b);
    let c_hi = _mm512_mulhi_epi16(a, b);
    let t = _mm512_mullo_epi16(c_lo, pinv);
    let tp_hi = _mm512_mulhi_epi16(t, p);
    _mm512_sub_epi16(c_hi, tp_hi)
}

/// 32-wide range reduction for i16: maps `(-2p, 2p)` → `(-p, p)`.
#[inline(always)]
unsafe fn reduce_range_32x_i16(a: __m512i, p: __m512i) -> __m512i {
    let ge_mask = _mm512_cmpge_epi16_mask(a, p);
    let after_sub = _mm512_mask_sub_epi16(a, ge_mask, a, p);
    let zero = _mm512_setzero_si512();
    let lt_mask = _mm512_cmplt_epi16_mask(after_sub, zero);
    _mm512_mask_add_epi16(after_sub, lt_mask, after_sub, p)
}

pub(crate) unsafe fn forward_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm512_set1_epi16(prime.p);
    let pinv_v = _mm512_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i16;
        let mut i = 0;
        while i + 32 <= D {
            let ai = _mm512_loadu_si512(a_ptr.add(i).cast());
            let psi = _mm512_loadu_si512(psi_ptr.add(i).cast());
            _mm512_storeu_si512(a_ptr.add(i).cast(), mont_mul_32x_i16(ai, psi, p_v, pinv_v));
            i += 32;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.psi_pows[i]);
            i += 1;
        }
    }

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 32 {
                let mut j = 0;
                while j < len {
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm512_add_epi16(u, v);
                    let diff = _mm512_sub_epi16(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_32x_i16(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_32x_i16(diff, w, p_v, pinv_v),
                    );
                    j += 32;
                }
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }

    reduce_range_in_place_i16(a, p_v);
}

pub(crate) unsafe fn inverse_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm512_set1_epi16(prime.p);
    let pinv_v = _mm512_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 32 {
                let mut j = 0;
                while j < len {
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v_raw = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_32x_i16(v_raw, w, p_v, pinv_v);
                    let sum = _mm512_add_epi16(u, v);
                    let diff = _mm512_sub_epi16(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_32x_i16(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_32x_i16(diff, p_v),
                    );
                    j += 32;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }

    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i16;
        let mut i = 0;
        while i + 32 <= D {
            let ai = _mm512_loadu_si512(a_ptr.add(i).cast());
            let f = _mm512_loadu_si512(fused_ptr.add(i).cast());
            _mm512_storeu_si512(a_ptr.add(i).cast(), mont_mul_32x_i16(ai, f, p_v, pinv_v));
            i += 32;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.d_inv_psi_inv[i]);
            i += 1;
        }
    }
}

pub(crate) unsafe fn forward_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm512_set1_epi16(prime.p);
    let pinv_v = _mm512_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 32 {
                let mut j = 0;
                while j < len {
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm512_add_epi16(u, v);
                    let diff = _mm512_sub_epi16(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_32x_i16(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_32x_i16(diff, w, p_v, pinv_v),
                    );
                    j += 32;
                }
            } else {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
            }
            start += 2 * len;
        }
        len /= 2;
    }
    reduce_range_in_place_i16(a, p_v);
}

pub(crate) unsafe fn inverse_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm512_set1_epi16(prime.p);
    let pinv_v = _mm512_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 32 {
                let mut j = 0;
                while j < len {
                    let w = _mm512_loadu_si512(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm512_loadu_si512(a_ptr.add(start + j).cast());
                    let v_raw = _mm512_loadu_si512(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_32x_i16(v_raw, w, p_v, pinv_v);
                    let sum = _mm512_add_epi16(u, v);
                    let diff = _mm512_sub_epi16(u, v);
                    _mm512_storeu_si512(
                        a_ptr.add(start + j).cast(),
                        reduce_range_32x_i16(sum, p_v),
                    );
                    _mm512_storeu_si512(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_32x_i16(diff, p_v),
                    );
                    j += 32;
                }
            } else {
                for j in 0..len {
                    let w = tw.inv_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = prime.mul(a[start + j + len], w);
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                }
            }
            start += 2 * len;
        }
        len *= 2;
    }

    {
        let d_inv = tw.d_inv;
        let d_inv_v = _mm512_set1_epi16(d_inv.raw());
        let mut i = 0;
        while i + 32 <= D {
            let ai = _mm512_loadu_si512(a_ptr.add(i).cast());
            _mm512_storeu_si512(
                a_ptr.add(i).cast(),
                mont_mul_32x_i16(ai, d_inv_v, p_v, pinv_v),
            );
            i += 32;
        }
        while i < D {
            a[i] = prime.mul(a[i], d_inv);
            i += 1;
        }
    }
}

/// 32-wide pointwise multiply-accumulate for a single CRT limb (i16).
///
/// # Safety
///
/// `acc`, `lhs`, `rhs` must be valid for `d` elements; `acc` must be uniquely
/// borrowed.
pub(crate) unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    let p_v = _mm512_set1_epi16(p);
    let pinv_v = _mm512_set1_epi16(pinv);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 32 <= d {
        let a = _mm512_loadu_si512(acc.add(i).cast());
        let l = _mm512_loadu_si512(lhs.add(i).cast());
        let r = _mm512_loadu_si512(rhs.add(i).cast());
        let prod = mont_mul_32x_i16(l, r, p_v, pinv_v);
        let sum = _mm512_add_epi16(a, prod);
        _mm512_storeu_si512(acc.add(i).cast(), reduce_range_32x_i16(sum, p_v));
        i += 32;
    }
    while i < d {
        let prod = prime.mul(
            MontCoeff::from_raw(*lhs.add(i)),
            MontCoeff::from_raw(*rhs.add(i)),
        );
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(prod.raw()));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

/// 32-wide add-and-reduce for a single CRT limb (i16).
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements and must not alias.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_v = _mm512_set1_epi16(p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 32 <= d {
        let a = _mm512_loadu_si512(acc.add(i).cast());
        let b = _mm512_loadu_si512(other.add(i).cast());
        _mm512_storeu_si512(
            acc.add(i).cast(),
            reduce_range_32x_i16(_mm512_add_epi16(a, b), p_v),
        );
        i += 32;
    }
    while i < d {
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

unsafe fn reduce_range_in_place_i16<const D: usize>(a: &mut [MontCoeff<i16>; D], p_v: __m512i) {
    let ptr = a.as_mut_ptr() as *mut i16;
    let mut i = 0;
    while i + 32 <= D {
        let val = _mm512_loadu_si512(ptr.add(i).cast());
        _mm512_storeu_si512(ptr.add(i).cast(), reduce_range_32x_i16(val, p_v));
        i += 32;
    }
}
