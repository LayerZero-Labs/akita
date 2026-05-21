//! x86_64 AVX2 SIMD kernels for NTT butterfly, Montgomery multiply, and
//! pointwise operations.
//!
//! Provides vectorized i32 (for Q64/Q128) and i16 (for Q32) paths,
//! mirroring [`super::neon`].
//!
//! Dispatch is controlled by [`super::use_simd_ntt`] (defined once at the
//! `ntt` module level and shared across all SIMD backends).

use std::arch::x86_64::*;

use super::butterfly::NttTwiddles;
use super::prime::{MontCoeff, NttPrime};

// ============================================================================
// i32 path (R = 2^32, for primes < 2^30)
// ============================================================================

/// 8-wide Montgomery multiply for i32 primes.
///
/// Uses two 4-wide signed widening multiplications (one for even i32 lanes,
/// one for odd i32 lanes via right-shift), then reduces and recombines.
///
/// Inputs / outputs are in `(-p, p)` centered Montgomery representation.
#[inline(always)]
unsafe fn mont_mul_8x_i32(a: __m256i, b: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
    // Even lanes: low 32 bits of each 64-bit lane (lanes 0, 2, 4, 6 of i32).
    let c_evn = _mm256_mul_epi32(a, b);
    // Odd lanes: bring the high 32 bits of each 64-bit lane down to the low
    // 32 (lanes 1, 3, 5, 7 become the new low). Right-shift fills with zeros
    // in the high 32 of each 64-bit lane, but `mul_epi32` reads only the low
    // 32 bits and sign-extends them, so the operation is semantically signed.
    let a_odd = _mm256_srli_epi64::<32>(a);
    let b_odd = _mm256_srli_epi64::<32>(b);
    let c_odd = _mm256_mul_epi32(a_odd, b_odd);

    // t = (c mod 2^32) * pinv mod 2^32; only the low 32 bits matter.
    let t_evn = _mm256_mullo_epi32(c_evn, pinv);
    let t_odd = _mm256_mullo_epi32(c_odd, pinv);

    // tp = signed widening mul of (t mod 2^32) * (p mod 2^32) — 64-bit result.
    let tp_evn = _mm256_mul_epi32(t_evn, p);
    let tp_odd = _mm256_mul_epi32(t_odd, p);

    // (c - tp) is divisible by 2^32 by Montgomery construction.
    // After unsigned right shift by 32 the canonical i32 result lands in the
    // low 32 bits of each 64-bit lane (high 32 are zero); we'll discard those
    // in the blend below.
    let r_evn = _mm256_srli_epi64::<32>(_mm256_sub_epi64(c_evn, tp_evn));
    let r_odd = _mm256_srli_epi64::<32>(_mm256_sub_epi64(c_odd, tp_odd));

    // Combine: r_evn results sit in i32 lanes 0, 2, 4, 6; shift r_odd left 32
    // to land in i32 lanes 1, 3, 5, 7, then blend.
    let r_odd_shifted = _mm256_slli_epi64::<32>(r_odd);
    _mm256_blend_epi32::<0b1010_1010>(r_evn, r_odd_shifted)
}

/// 8-wide range reduction for i32: maps `(-2p, 2p)` → `(-p, p)`.
///
/// Two-step branchless: subtract `p` if `a >= p` (positive overflow), then
/// add `p` if the result went negative (negative wrap).
#[inline(always)]
unsafe fn reduce_range_8x_i32(a: __m256i, p: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    // a >= p (signed): emulated as !(p > a).
    let p_gt_a = _mm256_cmpgt_epi32(p, a);
    let all_ones = _mm256_set1_epi32(-1);
    let ge_mask = _mm256_xor_si256(p_gt_a, all_ones);
    let after_sub = _mm256_sub_epi32(a, _mm256_and_si256(p, ge_mask));
    // after_sub < 0 (signed) → add p.
    let lt_mask = _mm256_cmpgt_epi32(zero, after_sub);
    _mm256_add_epi32(after_sub, _mm256_and_si256(p, lt_mask))
}

/// AVX2-accelerated forward negacyclic NTT for i32 primes.
///
/// Processes 8 butterfly pairs per iteration when `len >= 8`;
/// falls back to scalar for the final stages.
pub(crate) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm256_set1_epi32(prime.p);
    let pinv_v = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    // Pre-twist by psi^i.
    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i32;
        let mut i = 0;
        while i + 8 <= D {
            let ai = _mm256_loadu_si256(a_ptr.add(i).cast());
            let psi = _mm256_loadu_si256(psi_ptr.add(i).cast());
            _mm256_storeu_si256(a_ptr.add(i).cast(), mont_mul_8x_i32(ai, psi, p_v, pinv_v));
            i += 8;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.psi_pows[i]);
            i += 1;
        }
    }

    // DIF butterfly stages.
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0;
                while j < len {
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());

                    let sum = _mm256_add_epi32(u, v);
                    let diff = _mm256_sub_epi32(u, v);

                    _mm256_storeu_si256(a_ptr.add(start + j).cast(), reduce_range_8x_i32(sum, p_v));
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_8x_i32(diff, w, p_v, pinv_v),
                    );
                    j += 8;
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

/// AVX2-accelerated inverse negacyclic NTT for i32 primes.
pub(crate) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm256_set1_epi32(prime.p);
    let pinv_v = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0;
                while j < len {
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v_raw = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_8x_i32(v_raw, w, p_v, pinv_v);

                    let sum = _mm256_add_epi32(u, v);
                    let diff = _mm256_sub_epi32(u, v);

                    _mm256_storeu_si256(a_ptr.add(start + j).cast(), reduce_range_8x_i32(sum, p_v));
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_8x_i32(diff, p_v),
                    );
                    j += 8;
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

    // Fused D^{-1} * psi^{-i} untwist.
    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i32;
        let mut i = 0;
        while i + 8 <= D {
            let ai = _mm256_loadu_si256(a_ptr.add(i).cast());
            let f = _mm256_loadu_si256(fused_ptr.add(i).cast());
            _mm256_storeu_si256(a_ptr.add(i).cast(), mont_mul_8x_i32(ai, f, p_v, pinv_v));
            i += 8;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.d_inv_psi_inv[i]);
            i += 1;
        }
    }
}

/// AVX2-accelerated forward cyclic NTT for i32 (no negacyclic twist).
pub(crate) unsafe fn forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm256_set1_epi32(prime.p);
    let pinv_v = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0;
                while j < len {
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm256_add_epi32(u, v);
                    let diff = _mm256_sub_epi32(u, v);
                    _mm256_storeu_si256(a_ptr.add(start + j).cast(), reduce_range_8x_i32(sum, p_v));
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_8x_i32(diff, w, p_v, pinv_v),
                    );
                    j += 8;
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

/// AVX2-accelerated inverse cyclic NTT for i32 (no negacyclic untwist).
pub(crate) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_v = _mm256_set1_epi32(prime.p);
    let pinv_v = _mm256_set1_epi32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 8 {
                let mut j = 0;
                while j < len {
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v_raw = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_8x_i32(v_raw, w, p_v, pinv_v);
                    let sum = _mm256_add_epi32(u, v);
                    let diff = _mm256_sub_epi32(u, v);
                    _mm256_storeu_si256(a_ptr.add(start + j).cast(), reduce_range_8x_i32(sum, p_v));
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_8x_i32(diff, p_v),
                    );
                    j += 8;
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

    // D^{-1} scaling.
    {
        let d_inv = tw.d_inv;
        let d_inv_v = _mm256_set1_epi32(d_inv.raw());
        let mut i = 0;
        while i + 8 <= D {
            let ai = _mm256_loadu_si256(a_ptr.add(i).cast());
            _mm256_storeu_si256(
                a_ptr.add(i).cast(),
                mont_mul_8x_i32(ai, d_inv_v, p_v, pinv_v),
            );
            i += 8;
        }
        while i < D {
            a[i] = prime.mul(a[i], d_inv);
            i += 1;
        }
    }
}

/// 8-wide pointwise multiply-accumulate for a single CRT limb (i32).
///
/// `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))` for `i in 0..d`.
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
    let p_v = _mm256_set1_epi32(p);
    let pinv_v = _mm256_set1_epi32(pinv);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 8 <= d {
        let a = _mm256_loadu_si256(acc.add(i).cast());
        let l = _mm256_loadu_si256(lhs.add(i).cast());
        let r = _mm256_loadu_si256(rhs.add(i).cast());
        let prod = mont_mul_8x_i32(l, r, p_v, pinv_v);
        let sum = _mm256_add_epi32(a, prod);
        _mm256_storeu_si256(acc.add(i).cast(), reduce_range_8x_i32(sum, p_v));
        i += 8;
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

/// 8-wide add-and-reduce for a single CRT limb (i32).
///
/// `acc[i] = reduce_range(acc[i] + other[i])` for `i in 0..d`.
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements and must not alias in a
/// way that violates Rust's mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_v = _mm256_set1_epi32(p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 8 <= d {
        let a = _mm256_loadu_si256(acc.add(i).cast());
        let b = _mm256_loadu_si256(other.add(i).cast());
        _mm256_storeu_si256(
            acc.add(i).cast(),
            reduce_range_8x_i32(_mm256_add_epi32(a, b), p_v),
        );
        i += 8;
    }
    while i < d {
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

/// In-place `reduce_range` over a full i32 array.
unsafe fn reduce_range_in_place_i32<const D: usize>(a: &mut [MontCoeff<i32>; D], p_v: __m256i) {
    let ptr = a.as_mut_ptr() as *mut i32;
    let mut i = 0;
    while i + 8 <= D {
        let val = _mm256_loadu_si256(ptr.add(i).cast());
        _mm256_storeu_si256(ptr.add(i).cast(), reduce_range_8x_i32(val, p_v));
        i += 8;
    }
}

// ============================================================================
// i16 path (R = 2^16, for primes < 2^14)
// ============================================================================

/// 16-wide Montgomery multiply for i16 primes.
///
/// Uses `mullo_epi16` for `c.lo16` and `t`, and `mulhi_epi16` (signed) for
/// `c.hi16` and `tp.hi16`. By the Montgomery invariant `c.lo16 == tp.lo16`,
/// so `(c - tp) >> 16 == c.hi16 - tp.hi16` exactly with no borrow needed.
/// Four instructions per 16-lane multiply.
#[inline(always)]
unsafe fn mont_mul_16x_i16(a: __m256i, b: __m256i, p: __m256i, pinv: __m256i) -> __m256i {
    let c_lo = _mm256_mullo_epi16(a, b);
    let c_hi = _mm256_mulhi_epi16(a, b);
    let t = _mm256_mullo_epi16(c_lo, pinv);
    let tp_hi = _mm256_mulhi_epi16(t, p);
    _mm256_sub_epi16(c_hi, tp_hi)
}

/// 16-wide range reduction for i16: maps `(-2p, 2p)` → `(-p, p)`.
#[inline(always)]
unsafe fn reduce_range_16x_i16(a: __m256i, p: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let p_gt_a = _mm256_cmpgt_epi16(p, a);
    let all_ones = _mm256_set1_epi16(-1);
    let ge_mask = _mm256_xor_si256(p_gt_a, all_ones);
    let after_sub = _mm256_sub_epi16(a, _mm256_and_si256(p, ge_mask));
    let lt_mask = _mm256_cmpgt_epi16(zero, after_sub);
    _mm256_add_epi16(after_sub, _mm256_and_si256(p, lt_mask))
}

/// AVX2-accelerated forward negacyclic NTT for i16 primes.
pub(crate) unsafe fn forward_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm256_set1_epi16(prime.p);
    let pinv_v = _mm256_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    // Pre-twist by psi^i.
    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i16;
        let mut i = 0;
        while i + 16 <= D {
            let ai = _mm256_loadu_si256(a_ptr.add(i).cast());
            let psi = _mm256_loadu_si256(psi_ptr.add(i).cast());
            _mm256_storeu_si256(a_ptr.add(i).cast(), mont_mul_16x_i16(ai, psi, p_v, pinv_v));
            i += 16;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.psi_pows[i]);
            i += 1;
        }
    }

    // DIF butterfly stages.
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm256_add_epi16(u, v);
                    let diff = _mm256_sub_epi16(u, v);
                    _mm256_storeu_si256(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i16(sum, p_v),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_16x_i16(diff, w, p_v, pinv_v),
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

    reduce_range_in_place_i16(a, p_v);
}

/// AVX2-accelerated inverse negacyclic NTT for i16 primes.
pub(crate) unsafe fn inverse_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm256_set1_epi16(prime.p);
    let pinv_v = _mm256_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v_raw = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_16x_i16(v_raw, w, p_v, pinv_v);
                    let sum = _mm256_add_epi16(u, v);
                    let diff = _mm256_sub_epi16(u, v);
                    _mm256_storeu_si256(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i16(sum, p_v),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_16x_i16(diff, p_v),
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

    // Fused D^{-1} * psi^{-i} untwist.
    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i16;
        let mut i = 0;
        while i + 16 <= D {
            let ai = _mm256_loadu_si256(a_ptr.add(i).cast());
            let f = _mm256_loadu_si256(fused_ptr.add(i).cast());
            _mm256_storeu_si256(a_ptr.add(i).cast(), mont_mul_16x_i16(ai, f, p_v, pinv_v));
            i += 16;
        }
        while i < D {
            a[i] = prime.mul(a[i], tw.d_inv_psi_inv[i]);
            i += 1;
        }
    }
}

/// AVX2-accelerated forward cyclic NTT for i16.
pub(crate) unsafe fn forward_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm256_set1_epi16(prime.p);
    let pinv_v = _mm256_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let sum = _mm256_add_epi16(u, v);
                    let diff = _mm256_sub_epi16(u, v);
                    _mm256_storeu_si256(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i16(sum, p_v),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        mont_mul_16x_i16(diff, w, p_v, pinv_v),
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
    reduce_range_in_place_i16(a, p_v);
}

/// AVX2-accelerated inverse cyclic NTT for i16.
pub(crate) unsafe fn inverse_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_v = _mm256_set1_epi16(prime.p);
    let pinv_v = _mm256_set1_epi16(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 16 {
                let mut j = 0;
                while j < len {
                    let w = _mm256_loadu_si256(tw_ptr.add(twiddle_base + j).cast());
                    let u = _mm256_loadu_si256(a_ptr.add(start + j).cast());
                    let v_raw = _mm256_loadu_si256(a_ptr.add(start + j + len).cast());
                    let v = mont_mul_16x_i16(v_raw, w, p_v, pinv_v);
                    let sum = _mm256_add_epi16(u, v);
                    let diff = _mm256_sub_epi16(u, v);
                    _mm256_storeu_si256(
                        a_ptr.add(start + j).cast(),
                        reduce_range_16x_i16(sum, p_v),
                    );
                    _mm256_storeu_si256(
                        a_ptr.add(start + j + len).cast(),
                        reduce_range_16x_i16(diff, p_v),
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

    // D^{-1} scaling.
    {
        let d_inv = tw.d_inv;
        let d_inv_v = _mm256_set1_epi16(d_inv.raw());
        let mut i = 0;
        while i + 16 <= D {
            let ai = _mm256_loadu_si256(a_ptr.add(i).cast());
            _mm256_storeu_si256(
                a_ptr.add(i).cast(),
                mont_mul_16x_i16(ai, d_inv_v, p_v, pinv_v),
            );
            i += 16;
        }
        while i < D {
            a[i] = prime.mul(a[i], d_inv);
            i += 1;
        }
    }
}

/// 16-wide pointwise multiply-accumulate for a single CRT limb (i16).
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
    let p_v = _mm256_set1_epi16(p);
    let pinv_v = _mm256_set1_epi16(pinv);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 16 <= d {
        let a = _mm256_loadu_si256(acc.add(i).cast());
        let l = _mm256_loadu_si256(lhs.add(i).cast());
        let r = _mm256_loadu_si256(rhs.add(i).cast());
        let prod = mont_mul_16x_i16(l, r, p_v, pinv_v);
        let sum = _mm256_add_epi16(a, prod);
        _mm256_storeu_si256(acc.add(i).cast(), reduce_range_16x_i16(sum, p_v));
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

/// 16-wide add-and-reduce for a single CRT limb (i16).
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements and must not alias in a
/// way that violates Rust's mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_v = _mm256_set1_epi16(p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 16 <= d {
        let a = _mm256_loadu_si256(acc.add(i).cast());
        let b = _mm256_loadu_si256(other.add(i).cast());
        _mm256_storeu_si256(
            acc.add(i).cast(),
            reduce_range_16x_i16(_mm256_add_epi16(a, b), p_v),
        );
        i += 16;
    }
    while i < d {
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

/// In-place `reduce_range` over a full i16 array.
unsafe fn reduce_range_in_place_i16<const D: usize>(a: &mut [MontCoeff<i16>; D], p_v: __m256i) {
    let ptr = a.as_mut_ptr() as *mut i16;
    let mut i = 0;
    while i + 16 <= D {
        let val = _mm256_loadu_si256(ptr.add(i).cast());
        _mm256_storeu_si256(ptr.add(i).cast(), reduce_range_16x_i16(val, p_v));
        i += 16;
    }
}
