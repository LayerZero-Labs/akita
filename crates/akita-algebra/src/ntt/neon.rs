//! AArch64 NEON SIMD kernels for NTT butterfly, Montgomery multiply,
//! and pointwise operations.
//!
//! Provides vectorized i32 (for Q64/Q128) and i16 (for Q32) paths.
//! Dispatch is controlled by [`use_neon_ntt`]: set `AKITA_SCALAR_NTT=1`
//! to force the scalar fallback for A/B performance comparison.

use std::arch::aarch64::*;
use std::sync::OnceLock;

use super::butterfly::NttTwiddles;
use super::prime::{MontCoeff, NttPrime};

/// Whether the NEON NTT path is active. Cached on first call.
/// Set `AKITA_SCALAR_NTT=1` to force scalar fallback.
/// Returns whether NEON NTT kernels are enabled at runtime.
pub fn use_neon_ntt() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AKITA_SCALAR_NTT").map_or(true, |v| v != "1"))
}

/// 4-wide Montgomery multiply for i32 primes.
///
/// Uses two 2-wide `vmull_s32` chains (since i32×i32→i64 only fills 2 lanes
/// of a 128-bit register) and combines the results.
#[inline(always)]
unsafe fn mont_mul_4x_i32(a: int32x4_t, b: int32x4_t, p: int32x2_t, pinv: int32x2_t) -> int32x4_t {
    let a_lo = vget_low_s32(a);
    let a_hi = vget_high_s32(a);
    let b_lo = vget_low_s32(b);
    let b_hi = vget_high_s32(b);

    // Low pair
    let c_lo = vmull_s32(a_lo, b_lo);
    let t_lo = vmul_s32(vmovn_s64(c_lo), pinv);
    let tp_lo = vmull_s32(t_lo, p);
    let r_lo = vmovn_s64(vshrq_n_s64::<32>(vsubq_s64(c_lo, tp_lo)));

    // High pair
    let c_hi = vmull_s32(a_hi, b_hi);
    let t_hi = vmul_s32(vmovn_s64(c_hi), pinv);
    let tp_hi = vmull_s32(t_hi, p);
    let r_hi = vmovn_s64(vshrq_n_s64::<32>(vsubq_s64(c_hi, tp_hi)));

    vcombine_s32(r_lo, r_hi)
}

/// 4-wide range reduction for i32: maps `(-2p, 2p)` → `(-p, p)`.
///
/// Uses comparison-first approach to avoid the i64 widening that the
/// scalar `csubp`/`caddp` path requires (since `a - p` can overflow i32).
#[inline(always)]
unsafe fn reduce_range_4x_i32(a: int32x4_t, p: int32x4_t) -> int32x4_t {
    let zero = vdupq_n_s32(0);

    // csubp: subtract p where a >= p
    let ge_mask = vcgeq_s32(a, p);
    let after_sub = vsubq_s32(
        a,
        vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(p), ge_mask)),
    );

    // caddp: add p where result < 0
    let lt_mask = vcltq_s32(after_sub, zero);
    vaddq_s32(
        after_sub,
        vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(p), lt_mask)),
    )
}

/// NEON-accelerated forward negacyclic NTT for i32 primes.
///
/// Processes 4 butterfly pairs per iteration when `len >= 4`;
/// falls back to scalar for the final 2 stages (`len = 2, 1`).
pub(crate) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_d = vdup_n_s32(prime.p);
    let pinv_d = vdup_n_s32(prime.pinv);
    let p_q = vdupq_n_s32(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    // Pre-twist by psi^i
    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i32;
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1q_s32(a_ptr.add(i));
            let psi = vld1q_s32(psi_ptr.add(i));
            vst1q_s32(a_ptr.add(i), mont_mul_4x_i32(ai, psi, p_d, pinv_d));
            i += 4;
        }
    }

    // DIF butterfly stages
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let u = vld1q_s32(a_ptr.add(start + j));
                    let v = vld1q_s32(a_ptr.add(start + j + len));
                    let w = vld1q_s32(tw_ptr.add(twiddle_base + j));

                    let sum = vaddq_s32(u, v);
                    let diff = vsubq_s32(u, v);

                    vst1q_s32(a_ptr.add(start + j), reduce_range_4x_i32(sum, p_q));
                    vst1q_s32(
                        a_ptr.add(start + j + len),
                        mont_mul_4x_i32(diff, w, p_d, pinv_d),
                    );
                    j += 4;
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

    // Final reduce_range pass
    reduce_range_in_place_i32(a, p_q);
}

/// NEON-accelerated inverse negacyclic NTT for i32 primes.
pub(crate) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_d = vdup_n_s32(prime.p);
    let pinv_d = vdup_n_s32(prime.pinv);
    let p_q = vdupq_n_s32(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    // DIT butterfly stages
    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let w = vld1q_s32(tw_ptr.add(twiddle_base + j));
                    let u = vld1q_s32(a_ptr.add(start + j));
                    let v_raw = vld1q_s32(a_ptr.add(start + j + len));
                    let v = mont_mul_4x_i32(v_raw, w, p_d, pinv_d);

                    let sum = vaddq_s32(u, v);
                    let diff = vsubq_s32(u, v);

                    vst1q_s32(a_ptr.add(start + j), reduce_range_4x_i32(sum, p_q));
                    vst1q_s32(a_ptr.add(start + j + len), reduce_range_4x_i32(diff, p_q));
                    j += 4;
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

    // Fused D^{-1} * psi^{-i} untwist
    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i32;
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1q_s32(a_ptr.add(i));
            let f = vld1q_s32(fused_ptr.add(i));
            vst1q_s32(a_ptr.add(i), mont_mul_4x_i32(ai, f, p_d, pinv_d));
            i += 4;
        }
    }
}

/// NEON-accelerated forward cyclic NTT for i32 (no negacyclic twist).
pub(crate) unsafe fn forward_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_d = vdup_n_s32(prime.p);
    let pinv_d = vdup_n_s32(prime.pinv);
    let p_q = vdupq_n_s32(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let u = vld1q_s32(a_ptr.add(start + j));
                    let v = vld1q_s32(a_ptr.add(start + j + len));
                    let w = vld1q_s32(tw_ptr.add(twiddle_base + j));
                    let sum = vaddq_s32(u, v);
                    let diff = vsubq_s32(u, v);
                    vst1q_s32(a_ptr.add(start + j), reduce_range_4x_i32(sum, p_q));
                    vst1q_s32(
                        a_ptr.add(start + j + len),
                        mont_mul_4x_i32(diff, w, p_d, pinv_d),
                    );
                    j += 4;
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
    reduce_range_in_place_i32(a, p_q);
}

/// NEON-accelerated inverse cyclic NTT for i32 (no negacyclic untwist).
pub(crate) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_d = vdup_n_s32(prime.p);
    let pinv_d = vdup_n_s32(prime.pinv);
    let p_q = vdupq_n_s32(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let w = vld1q_s32(tw_ptr.add(twiddle_base + j));
                    let u = vld1q_s32(a_ptr.add(start + j));
                    let v_raw = vld1q_s32(a_ptr.add(start + j + len));
                    let v = mont_mul_4x_i32(v_raw, w, p_d, pinv_d);
                    let sum = vaddq_s32(u, v);
                    let diff = vsubq_s32(u, v);
                    vst1q_s32(a_ptr.add(start + j), reduce_range_4x_i32(sum, p_q));
                    vst1q_s32(a_ptr.add(start + j + len), reduce_range_4x_i32(diff, p_q));
                    j += 4;
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

    // D^{-1} scaling
    {
        let d_inv = tw.d_inv;
        let d_inv_q = vdupq_n_s32(d_inv.raw());
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1q_s32(a_ptr.add(i));
            vst1q_s32(a_ptr.add(i), mont_mul_4x_i32(ai, d_inv_q, p_d, pinv_d));
            i += 4;
        }
    }
}

/// 4-wide pointwise multiply-accumulate for a single CRT limb (i32).
///
/// `acc[i] = reduce_range(acc[i] + mont_mul(lhs[i], rhs[i]))` for `i in 0..d`.
pub(crate) unsafe fn pointwise_mul_acc_i32(
    acc: *mut i32,
    lhs: *const i32,
    rhs: *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_d = vdup_n_s32(p);
    let pinv_d = vdup_n_s32(pinv);
    let p_q = vdupq_n_s32(p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 4 <= d {
        let a = vld1q_s32(acc.add(i));
        let l = vld1q_s32(lhs.add(i));
        let r = vld1q_s32(rhs.add(i));
        let prod = mont_mul_4x_i32(l, r, p_d, pinv_d);
        let sum = vaddq_s32(a, prod);
        vst1q_s32(acc.add(i), reduce_range_4x_i32(sum, p_q));
        i += 4;
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

/// 4-wide add-and-reduce for a single CRT limb (i32).
///
/// `acc[i] = reduce_range(acc[i] + other[i])` for `i in 0..d`.
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements, properly aligned for
/// NEON loads/stores, and must not alias in a way that violates Rust's
/// mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i32(acc: *mut i32, other: *const i32, d: usize, p: i32) {
    let p_q = vdupq_n_s32(p);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 4 <= d {
        let a = vld1q_s32(acc.add(i));
        let b = vld1q_s32(other.add(i));
        vst1q_s32(acc.add(i), reduce_range_4x_i32(vaddq_s32(a, b), p_q));
        i += 4;
    }
    while i < d {
        let sum = MontCoeff::from_raw((*acc.add(i)).wrapping_add(*other.add(i)));
        *acc.add(i) = prime.reduce_range(sum).raw();
        i += 1;
    }
}

/// In-place reduce_range over a full array.
unsafe fn reduce_range_in_place_i32<const D: usize>(a: &mut [MontCoeff<i32>; D], p_q: int32x4_t) {
    let ptr = a.as_mut_ptr() as *mut i32;
    let mut i = 0;
    while i + 4 <= D {
        let val = vld1q_s32(ptr.add(i));
        vst1q_s32(ptr.add(i), reduce_range_4x_i32(val, p_q));
        i += 4;
    }
}

/// 4-wide Montgomery multiply for i16 primes.
///
/// Natural 4-wide: `vmull_s16` produces `int32x4_t`.
#[inline(always)]
unsafe fn mont_mul_4x_i16(a: int16x4_t, b: int16x4_t, p: int16x4_t, pinv: int16x4_t) -> int16x4_t {
    let c = vmull_s16(a, b);
    let t = vmul_s16(vmovn_s32(c), pinv);
    let tp = vmull_s16(t, p);
    vmovn_s32(vshrq_n_s32::<16>(vsubq_s32(c, tp)))
}

/// 8-wide Montgomery multiply for i16 primes (two 4-wide chains).
#[inline(always)]
unsafe fn mont_mul_8x_i16(a: int16x8_t, b: int16x8_t, p: int16x4_t, pinv: int16x4_t) -> int16x8_t {
    let r_lo = mont_mul_4x_i16(vget_low_s16(a), vget_low_s16(b), p, pinv);
    let r_hi = mont_mul_4x_i16(vget_high_s16(a), vget_high_s16(b), p, pinv);
    vcombine_s16(r_lo, r_hi)
}

/// 8-wide range reduction for i16: `(-2p, 2p)` → `(-p, p)`.
///
/// Same comparison-first approach as i32 but on `int16x8_t`.
#[inline(always)]
unsafe fn reduce_range_8x_i16(a: int16x8_t, p: int16x8_t) -> int16x8_t {
    let zero = vdupq_n_s16(0);
    let ge_mask = vcgeq_s16(a, p);
    let after_sub = vsubq_s16(
        a,
        vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(p), ge_mask)),
    );
    let lt_mask = vcltq_s16(after_sub, zero);
    vaddq_s16(
        after_sub,
        vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(p), lt_mask)),
    )
}

/// NEON-accelerated forward negacyclic NTT for i16 primes.
///
/// Processes 4 butterflies per iteration when `len >= 4`;
/// scalar fallback for `len < 4`.
pub(crate) unsafe fn forward_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    // Pre-twist by psi^i
    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i16;
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1_s16(a_ptr.add(i));
            let psi = vld1_s16(psi_ptr.add(i));
            vst1_s16(a_ptr.add(i), mont_mul_4x_i16(ai, psi, p_d, pinv_d));
            i += 4;
        }
    }

    // DIF butterfly stages
    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v = vld1_s16(a_ptr.add(start + j + len));
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);

                    // reduce_range on 4-wide i16 (use 8-wide by padding)
                    let sum_q = vcombine_s16(sum, vdup_n_s16(0));
                    let sum_reduced = vget_low_s16(reduce_range_8x_i16(sum_q, p_q));

                    let diff_mul_w = mont_mul_4x_i16(diff, w, p_d, pinv_d);
                    vst1_s16(a_ptr.add(start + j), sum_reduced);
                    vst1_s16(a_ptr.add(start + j + len), diff_mul_w);
                    j += 4;
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

    // Final reduce_range pass
    reduce_range_in_place_i16(a, p_q);
}

/// NEON-accelerated inverse negacyclic NTT for i16 primes.
pub(crate) unsafe fn inverse_ntt_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v_raw = vld1_s16(a_ptr.add(start + j + len));
                    let v = mont_mul_4x_i16(v_raw, w, p_d, pinv_d);
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);
                    let reduced = reduce_range_8x_i16(vcombine_s16(sum, diff), p_q);
                    vst1_s16(a_ptr.add(start + j), vget_low_s16(reduced));
                    vst1_s16(a_ptr.add(start + j + len), vget_high_s16(reduced));
                    j += 4;
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

    // Fused D^{-1} * psi^{-i} untwist
    {
        let fused_ptr = tw.d_inv_psi_inv.as_ptr() as *const i16;
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1_s16(a_ptr.add(i));
            let f = vld1_s16(fused_ptr.add(i));
            vst1_s16(a_ptr.add(i), mont_mul_4x_i16(ai, f, p_d, pinv_d));
            i += 4;
        }
    }
}

/// NEON-accelerated forward cyclic NTT for i16.
pub(crate) unsafe fn forward_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v = vld1_s16(a_ptr.add(start + j + len));
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);
                    let sum_q = vcombine_s16(sum, vdup_n_s16(0));
                    vst1_s16(
                        a_ptr.add(start + j),
                        vget_low_s16(reduce_range_8x_i16(sum_q, p_q)),
                    );
                    vst1_s16(
                        a_ptr.add(start + j + len),
                        mont_mul_4x_i16(diff, w, p_d, pinv_d),
                    );
                    j += 4;
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
    reduce_range_in_place_i16(a, p_q);
}

/// NEON-accelerated inverse cyclic NTT for i16.
pub(crate) unsafe fn inverse_ntt_cyclic_i16<const D: usize>(
    a: &mut [MontCoeff<i16>; D],
    prime: NttPrime<i16>,
    tw: &NttTwiddles<i16, D>,
) {
    let p_d = vdup_n_s16(prime.p);
    let pinv_d = vdup_n_s16(prime.pinv);
    let p_q = vdupq_n_s16(prime.p);
    let a_ptr = a.as_mut_ptr() as *mut i16;

    let mut len = 1usize;
    while len < D {
        let twiddle_base = len - 1;
        let tw_ptr = tw.inv_twiddles.as_ptr() as *const i16;
        let mut start = 0usize;
        while start < D {
            if len >= 4 {
                let mut j = 0;
                while j < len {
                    let w = vld1_s16(tw_ptr.add(twiddle_base + j));
                    let u = vld1_s16(a_ptr.add(start + j));
                    let v_raw = vld1_s16(a_ptr.add(start + j + len));
                    let v = mont_mul_4x_i16(v_raw, w, p_d, pinv_d);
                    let sum = vadd_s16(u, v);
                    let diff = vsub_s16(u, v);
                    let reduced = reduce_range_8x_i16(vcombine_s16(sum, diff), p_q);
                    vst1_s16(a_ptr.add(start + j), vget_low_s16(reduced));
                    vst1_s16(a_ptr.add(start + j + len), vget_high_s16(reduced));
                    j += 4;
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

    // D^{-1} scaling
    {
        let d_inv = tw.d_inv;
        let d_inv_d = vdup_n_s16(d_inv.raw());
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1_s16(a_ptr.add(i));
            vst1_s16(a_ptr.add(i), mont_mul_4x_i16(ai, d_inv_d, p_d, pinv_d));
            i += 4;
        }
    }
}

/// 8-wide pointwise multiply-accumulate for a single CRT limb (i16).
pub(crate) unsafe fn pointwise_mul_acc_i16(
    acc: *mut i16,
    lhs: *const i16,
    rhs: *const i16,
    d: usize,
    p: i16,
    pinv: i16,
) {
    let p_d = vdup_n_s16(p);
    let pinv_d = vdup_n_s16(pinv);
    let p_q = vdupq_n_s16(p);
    let mut i = 0;
    while i + 8 <= d {
        let a = vld1q_s16(acc.add(i));
        let l = vld1q_s16(lhs.add(i));
        let r = vld1q_s16(rhs.add(i));
        let prod = mont_mul_8x_i16(l, r, p_d, pinv_d);
        let sum = vaddq_s16(a, prod);
        vst1q_s16(acc.add(i), reduce_range_8x_i16(sum, p_q));
        i += 8;
    }
    while i + 4 <= d {
        let a = vld1_s16(acc.add(i));
        let l = vld1_s16(lhs.add(i));
        let r = vld1_s16(rhs.add(i));
        let prod = mont_mul_4x_i16(l, r, p_d, pinv_d);
        let sum = vadd_s16(a, prod);
        let sum_q = vcombine_s16(sum, vdup_n_s16(0));
        vst1_s16(acc.add(i), vget_low_s16(reduce_range_8x_i16(sum_q, p_q)));
        i += 4;
    }
}

/// 8-wide add-and-reduce for a single CRT limb (i16).
///
/// # Safety
///
/// `acc` and `other` must be valid for `d` elements, properly aligned for
/// NEON loads/stores, and must not alias in a way that violates Rust's
/// mutable-reference rules.
#[cfg(feature = "parallel")]
pub unsafe fn add_reduce_i16(acc: *mut i16, other: *const i16, d: usize, p: i16) {
    let p_q = vdupq_n_s16(p);
    let mut i = 0;
    while i + 8 <= d {
        let a = vld1q_s16(acc.add(i));
        let b = vld1q_s16(other.add(i));
        vst1q_s16(acc.add(i), reduce_range_8x_i16(vaddq_s16(a, b), p_q));
        i += 8;
    }
    while i + 4 <= d {
        let a = vld1_s16(acc.add(i));
        let b = vld1_s16(other.add(i));
        let sum_q = vcombine_s16(vadd_s16(a, b), vdup_n_s16(0));
        vst1_s16(acc.add(i), vget_low_s16(reduce_range_8x_i16(sum_q, p_q)));
        i += 4;
    }
}

/// In-place reduce_range over a full i16 array.
unsafe fn reduce_range_in_place_i16<const D: usize>(a: &mut [MontCoeff<i16>; D], p_q: int16x8_t) {
    let ptr = a.as_mut_ptr() as *mut i16;
    let mut i = 0;
    while i + 8 <= D {
        let val = vld1q_s16(ptr.add(i));
        vst1q_s16(ptr.add(i), reduce_range_8x_i16(val, p_q));
        i += 8;
    }
    while i + 4 <= D {
        let val = vld1_s16(ptr.add(i));
        let padded = vcombine_s16(val, vdup_n_s16(0));
        vst1_s16(ptr.add(i), vget_low_s16(reduce_range_8x_i16(padded, p_q)));
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::super::butterfly::{
        forward_ntt as scalar_forward_ntt, forward_ntt_cyclic as scalar_forward_ntt_cyclic,
        inverse_ntt as scalar_inverse_ntt, inverse_ntt_cyclic as scalar_inverse_ntt_cyclic,
        NttTwiddles,
    };
    use super::super::prime::{MontCoeff, NttPrime};
    use super::*;

    fn random_mont_array_i32<const D: usize>(
        prime: NttPrime<i32>,
        seed: u64,
    ) -> [MontCoeff<i32>; D] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as i64 % prime.p as i64) as i32;
            prime.from_canonical(val)
        })
    }

    fn random_mont_array_i16<const D: usize>(
        prime: NttPrime<i16>,
        seed: u64,
    ) -> [MontCoeff<i16>; D] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) as i64 % prime.p as i64) as i16;
            prime.from_canonical(val)
        })
    }

    const TEST_PRIME_I32: i32 = 1073707009;
    const TEST_PRIME_I16: i16 = 13697;

    #[test]
    fn neon_forward_ntt_i32_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        let tw = NttTwiddles::<i32, 512>::compute(prime);
        let input = random_mont_array_i32::<512>(prime, 0xCAFE);

        let mut neon_result = input;
        unsafe { forward_ntt_i32(&mut neon_result, prime, &tw) };

        let mut scalar_result = input;
        scalar_forward_ntt(&mut scalar_result, prime, &tw);

        for i in 0..512 {
            let n = prime.to_canonical(neon_result[i]);
            let s = prime.to_canonical(scalar_result[i]);
            assert_eq!(n, s, "mismatch at index {i}: neon={n}, scalar={s}");
        }
    }

    #[test]
    fn neon_inverse_ntt_i32_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        let tw = NttTwiddles::<i32, 512>::compute(prime);
        let input = random_mont_array_i32::<512>(prime, 0xBEEF);

        let mut neon_result = input;
        unsafe { inverse_ntt_i32(&mut neon_result, prime, &tw) };

        let mut scalar_result = input;
        scalar_inverse_ntt(&mut scalar_result, prime, &tw);

        for i in 0..512 {
            let n = prime.to_canonical(neon_result[i]);
            let s = prime.to_canonical(scalar_result[i]);
            assert_eq!(n, s, "mismatch at index {i}: neon={n}, scalar={s}");
        }
    }

    #[test]
    fn neon_forward_inverse_roundtrip_i32() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        let tw = NttTwiddles::<i32, 512>::compute(prime);
        let input = random_mont_array_i32::<512>(prime, 0xDEAD);
        let canonical_input: Vec<i32> = input.iter().map(|c| prime.to_canonical(*c)).collect();

        let mut a = input;
        unsafe {
            forward_ntt_i32(&mut a, prime, &tw);
            inverse_ntt_i32(&mut a, prime, &tw);
        }

        for i in 0..512 {
            let result = prime.to_canonical(a[i]);
            assert_eq!(
                result, canonical_input[i],
                "roundtrip mismatch at index {i}"
            );
        }
    }

    #[test]
    fn neon_cyclic_ntt_i32_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        let tw = NttTwiddles::<i32, 512>::compute(prime);
        let input = random_mont_array_i32::<512>(prime, 0xFACE);

        let mut neon_fwd = input;
        unsafe { forward_ntt_cyclic_i32(&mut neon_fwd, prime, &tw) };

        let mut scalar_fwd = input;
        scalar_forward_ntt_cyclic(&mut scalar_fwd, prime, &tw);

        for i in 0..512 {
            let n = prime.to_canonical(neon_fwd[i]);
            let s = prime.to_canonical(scalar_fwd[i]);
            assert_eq!(n, s, "forward cyclic mismatch at {i}: neon={n}, scalar={s}");
        }

        let mut neon_inv = neon_fwd;
        unsafe { inverse_ntt_cyclic_i32(&mut neon_inv, prime, &tw) };

        let mut scalar_inv = scalar_fwd;
        scalar_inverse_ntt_cyclic(&mut scalar_inv, prime, &tw);

        for i in 0..512 {
            let n = prime.to_canonical(neon_inv[i]);
            let s = prime.to_canonical(scalar_inv[i]);
            assert_eq!(n, s, "inverse cyclic mismatch at {i}: neon={n}, scalar={s}");
        }
    }

    #[test]
    fn neon_pointwise_mul_acc_i32_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        const D: usize = 512;
        let acc_init = random_mont_array_i32::<D>(prime, 0x1111);
        let lhs = random_mont_array_i32::<D>(prime, 0x2222);
        let rhs = random_mont_array_i32::<D>(prime, 0x3333);

        let mut neon_acc = acc_init;
        unsafe {
            pointwise_mul_acc_i32(
                neon_acc.as_mut_ptr() as *mut i32,
                lhs.as_ptr() as *const i32,
                rhs.as_ptr() as *const i32,
                D,
                prime.p,
                prime.pinv,
            );
        }

        let mut scalar_acc = acc_init;
        for i in 0..D {
            let prod = prime.mul(lhs[i], rhs[i]);
            let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
            scalar_acc[i] = prime.reduce_range(sum);
        }

        for i in 0..D {
            let n = prime.to_canonical(neon_acc[i]);
            let s = prime.to_canonical(scalar_acc[i]);
            assert_eq!(n, s, "pointwise mul acc mismatch at {i}");
        }
    }

    #[test]
    fn neon_pointwise_mul_acc_i32_handles_scalar_tail() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        const D: usize = 6;
        let acc_init = random_mont_array_i32::<D>(prime, 0x4444);
        let lhs = random_mont_array_i32::<D>(prime, 0x5555);
        let rhs = random_mont_array_i32::<D>(prime, 0x6666);

        let mut neon_acc = acc_init;
        unsafe {
            pointwise_mul_acc_i32(
                neon_acc.as_mut_ptr() as *mut i32,
                lhs.as_ptr() as *const i32,
                rhs.as_ptr() as *const i32,
                D,
                prime.p,
                prime.pinv,
            );
        }

        let mut scalar_acc = acc_init;
        for i in 0..D {
            let prod = prime.mul(lhs[i], rhs[i]);
            let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
            scalar_acc[i] = prime.reduce_range(sum);
        }

        assert_eq!(neon_acc, scalar_acc);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn neon_add_reduce_i32_handles_scalar_tail() {
        let prime = NttPrime::compute(TEST_PRIME_I32);
        const D: usize = 6;
        let acc_init = random_mont_array_i32::<D>(prime, 0x7777);
        let other = random_mont_array_i32::<D>(prime, 0x8888);

        let mut neon_acc = acc_init;
        unsafe {
            add_reduce_i32(
                neon_acc.as_mut_ptr() as *mut i32,
                other.as_ptr() as *const i32,
                D,
                prime.p,
            );
        }

        let mut scalar_acc = acc_init;
        for i in 0..D {
            let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(other[i].raw()));
            scalar_acc[i] = prime.reduce_range(sum);
        }

        assert_eq!(neon_acc, scalar_acc);
    }

    #[test]
    fn neon_forward_ntt_i16_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I16);
        let tw = NttTwiddles::<i16, 64>::compute(prime);
        let input = random_mont_array_i16::<64>(prime, 0xABCD);

        let mut neon_result = input;
        unsafe { forward_ntt_i16(&mut neon_result, prime, &tw) };

        let mut scalar_result = input;
        scalar_forward_ntt(&mut scalar_result, prime, &tw);

        for i in 0..64 {
            let n = prime.to_canonical(neon_result[i]);
            let s = prime.to_canonical(scalar_result[i]);
            assert_eq!(n, s, "i16 forward mismatch at {i}: neon={n}, scalar={s}");
        }
    }

    #[test]
    fn neon_inverse_ntt_i16_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I16);
        let tw = NttTwiddles::<i16, 64>::compute(prime);
        let input = random_mont_array_i16::<64>(prime, 0xFEED);

        let mut neon_result = input;
        unsafe { inverse_ntt_i16(&mut neon_result, prime, &tw) };

        let mut scalar_result = input;
        scalar_inverse_ntt(&mut scalar_result, prime, &tw);

        for i in 0..64 {
            let n = prime.to_canonical(neon_result[i]);
            let s = prime.to_canonical(scalar_result[i]);
            assert_eq!(n, s, "i16 inverse mismatch at {i}: neon={n}, scalar={s}");
        }
    }

    #[test]
    fn neon_forward_inverse_roundtrip_i16() {
        let prime = NttPrime::compute(TEST_PRIME_I16);
        let tw = NttTwiddles::<i16, 64>::compute(prime);
        let input = random_mont_array_i16::<64>(prime, 0x7777);
        let canonical_input: Vec<i16> = input.iter().map(|c| prime.to_canonical(*c)).collect();

        let mut a = input;
        unsafe {
            forward_ntt_i16(&mut a, prime, &tw);
            inverse_ntt_i16(&mut a, prime, &tw);
        }

        for i in 0..64 {
            let result = prime.to_canonical(a[i]);
            assert_eq!(result, canonical_input[i], "i16 roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn neon_cyclic_i16_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I16);
        let tw = NttTwiddles::<i16, 64>::compute(prime);
        let input = random_mont_array_i16::<64>(prime, 0x9999);

        let mut neon_fwd = input;
        unsafe { forward_ntt_cyclic_i16(&mut neon_fwd, prime, &tw) };

        let mut scalar_fwd = input;
        scalar_forward_ntt_cyclic(&mut scalar_fwd, prime, &tw);

        for i in 0..64 {
            let n = prime.to_canonical(neon_fwd[i]);
            let s = prime.to_canonical(scalar_fwd[i]);
            assert_eq!(n, s, "i16 fwd cyclic mismatch at {i}");
        }

        let mut neon_inv = neon_fwd;
        unsafe { inverse_ntt_cyclic_i16(&mut neon_inv, prime, &tw) };

        let mut scalar_inv = scalar_fwd;
        scalar_inverse_ntt_cyclic(&mut scalar_inv, prime, &tw);

        for i in 0..64 {
            let n = prime.to_canonical(neon_inv[i]);
            let s = prime.to_canonical(scalar_inv[i]);
            assert_eq!(n, s, "i16 inv cyclic mismatch at {i}");
        }
    }

    #[test]
    fn neon_pointwise_mul_acc_i16_matches_scalar() {
        let prime = NttPrime::compute(TEST_PRIME_I16);
        const D: usize = 64;
        let acc_init = random_mont_array_i16::<D>(prime, 0xAAAA);
        let lhs = random_mont_array_i16::<D>(prime, 0xBBBB);
        let rhs = random_mont_array_i16::<D>(prime, 0xCCCC);

        let mut neon_acc = acc_init;
        unsafe {
            pointwise_mul_acc_i16(
                neon_acc.as_mut_ptr() as *mut i16,
                lhs.as_ptr() as *const i16,
                rhs.as_ptr() as *const i16,
                D,
                prime.p,
                prime.pinv,
            );
        }

        let mut scalar_acc = acc_init;
        for i in 0..D {
            let prod = prime.mul(lhs[i], rhs[i]);
            let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
            scalar_acc[i] = prime.reduce_range(sum);
        }

        for i in 0..D {
            let n = prime.to_canonical(neon_acc[i]);
            let s = prime.to_canonical(scalar_acc[i]);
            assert_eq!(n, s, "i16 pointwise mul acc mismatch at {i}");
        }
    }
}
