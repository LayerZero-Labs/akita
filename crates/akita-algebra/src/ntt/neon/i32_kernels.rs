use std::arch::aarch64::*;

use crate::ntt::batched_four_point_eligible;
use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::prime::{MontCoeff, NttPrime};

/// True 4-wide signed Montgomery multiply for i32 primes.
///
/// Computes `a * b * R^{-1} mod p` (R = 2^32) for all four lanes at once,
/// matching the scalar [`NttPrime::mont_mul_raw`] convention exactly
/// (`pinv = p^{-1} mod 2^32`, signed). Result lies in `(-p, p)`.
///
/// Uses the high-multiply formulation from Becker–Hwang–Kannwischer–Yang
/// ("Neon NTT"): `vqdmulhq_s32` yields `(2·a·b) >> 32` with a single 4-lane
/// multiply, so the reduction needs two `vqdmulhq_s32`, two `vmulq_s32`, and one
/// halving subtract instead of two 2-lane `vmull_s32` widening chains.
/// Every NTT prime here is `< 2^30`, so neither `2·a·b` nor `2·m·p` saturates
/// an `int32x4_t` after the `>> 32`.
#[inline(always)]
unsafe fn mont_mul_4x_i32(
    a: int32x4_t,
    b: int32x4_t,
    p_q: int32x4_t,
    pinv_q: int32x4_t,
) -> int32x4_t {
    // top = (2·a·b) >> 32, the doubled high word of the full product.
    let top = vqdmulhq_s32(a, b);
    // lo = (a·b) mod 2^32 = low word of the product.
    let lo = vmulq_s32(a, b);
    // m = lo · pinv mod 2^32 (signed), the Montgomery quotient digit.
    let m = vmulq_s32(lo, pinv_q);
    // sub = (2·m·p) >> 32, the doubled high word of m·p.
    let sub = vqdmulhq_s32(m, p_q);
    // (top - sub) >> 1 = (a·b - m·p) >> 32, exact since the low 32 bits cancel.
    vhsubq_s32(top, sub)
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

/// 4-wide `caddp`: add `p` to lanes that are negative, mapping `(-p, p)` → `[0, p)`.
///
/// Equivalent to [`reduce_range_4x_i32`] when the input is already in `(-p, p)`,
/// but skips the (always-false) `csubp` comparison.
#[inline(always)]
unsafe fn caddp_4x_i32(a: int32x4_t, p: int32x4_t) -> int32x4_t {
    let lt_mask = vcltq_s32(a, vdupq_n_s32(0));
    vaddq_s32(
        a,
        vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(p), lt_mask)),
    )
}

/// Vectorized final two DIF stages (`len = 2`, then `len = 1`) for forward i32 NTTs.
///
/// These stages have butterfly half-lengths below the 4-wide window, so the
/// naive stage loop drops to a scalar path that runs `~3·D/4` scalar Montgomery
/// multiplies per transform. Since each size-4 sub-DFT touches a contiguous run
/// of four coefficients, a stride-4 `vld4q` deinterleave lands the four members
/// of four independent sub-DFTs across the lanes of `r0..r3` (`r_e[lane]` holds
/// element `e` of sub-DFT `lane`). The two stages then run fully 4-wide with the
/// per-`j` twiddles broadcast across all sub-DFTs, and `vst4q` re-interleaves.
///
/// Requires `D` divisible by 16; callers fall back to the scalar tail otherwise.
#[inline(always)]
unsafe fn forward_dif_tail_i32<const D: usize>(
    a_ptr: *mut i32,
    fwd_twiddles: *const i32,
    p_q: int32x4_t,
    pinv_q: int32x4_t,
) {
    // tw0 is the (broadcast) len=1 twiddle; tw1/tw2 are the two len=2 twiddles.
    let tw0 = vdupq_n_s32(*fwd_twiddles);
    let tw1 = vdupq_n_s32(*fwd_twiddles.add(1));
    let tw2 = vdupq_n_s32(*fwd_twiddles.add(2));

    let mut base = 0usize;
    while base < D {
        let q = vld4q_s32(a_ptr.add(base));
        let (r0, r1, r2, r3) = (q.0, q.1, q.2, q.3);

        // len = 2: butterflies (e0,e2) with tw1, (e1,e3) with tw2.
        let s0 = reduce_range_4x_i32(vaddq_s32(r0, r2), p_q);
        let d0 = mont_mul_4x_i32(vsubq_s32(r0, r2), tw1, p_q, pinv_q);
        let s1 = reduce_range_4x_i32(vaddq_s32(r1, r3), p_q);
        let d1 = mont_mul_4x_i32(vsubq_s32(r1, r3), tw2, p_q, pinv_q);

        // len = 1: butterflies (e0,e1) and (e2,e3), both with tw0. All four
        // results land in (-p, p); the final `caddp` normalizes them to [0, p),
        // which folds the transform's closing reduce_range pass into this stage.
        let o0 = caddp_4x_i32(reduce_range_4x_i32(vaddq_s32(s0, s1), p_q), p_q);
        let o1 = caddp_4x_i32(mont_mul_4x_i32(vsubq_s32(s0, s1), tw0, p_q, pinv_q), p_q);
        let o2 = caddp_4x_i32(reduce_range_4x_i32(vaddq_s32(d0, d1), p_q), p_q);
        let o3 = caddp_4x_i32(mont_mul_4x_i32(vsubq_s32(d0, d1), tw0, p_q, pinv_q), p_q);

        vst4q_s32(a_ptr.add(base), int32x4x4_t(o0, o1, o2, o3));
        base += 16;
    }
}

/// Vectorized first two DIT stages (`len = 1`, then `len = 2`) for inverse i32 NTTs.
///
/// A stride-4 deinterleave places the same position from four independent
/// size-4 sub-transforms in each vector. This avoids `D` scalar Montgomery
/// multiplies at the inverse head and hands the transform back to the ordinary
/// four-wide stage loop at `len = 4`.
///
/// Requires `D` divisible by 16; callers retain the scalar stages otherwise.
#[inline(always)]
unsafe fn inverse_dit_head_i32<const D: usize>(
    a_ptr: *mut i32,
    inv_twiddles: *const i32,
    p_q: int32x4_t,
    pinv_q: int32x4_t,
) {
    let tw0 = vdupq_n_s32(*inv_twiddles);
    let tw1 = vdupq_n_s32(*inv_twiddles.add(1));
    let tw2 = vdupq_n_s32(*inv_twiddles.add(2));

    let mut base = 0usize;
    while base < D {
        let q = vld4q_s32(a_ptr.add(base));
        let (r0, r1, r2, r3) = (q.0, q.1, q.2, q.3);

        // len = 1: adjacent pairs use the same first-stage twiddle.
        let v1 = mont_mul_4x_i32(r1, tw0, p_q, pinv_q);
        let v3 = mont_mul_4x_i32(r3, tw0, p_q, pinv_q);
        let s0 = reduce_range_4x_i32(vaddq_s32(r0, v1), p_q);
        let d0 = reduce_range_4x_i32(vsubq_s32(r0, v1), p_q);
        let s1 = reduce_range_4x_i32(vaddq_s32(r2, v3), p_q);
        let d1 = reduce_range_4x_i32(vsubq_s32(r2, v3), p_q);

        // len = 2: positions (0,2) use tw1 and (1,3) use tw2.
        let v2 = mont_mul_4x_i32(s1, tw1, p_q, pinv_q);
        let v3 = mont_mul_4x_i32(d1, tw2, p_q, pinv_q);
        let o0 = reduce_range_4x_i32(vaddq_s32(s0, v2), p_q);
        let o2 = reduce_range_4x_i32(vsubq_s32(s0, v2), p_q);
        let o1 = reduce_range_4x_i32(vaddq_s32(d0, v3), p_q);
        let o3 = reduce_range_4x_i32(vsubq_s32(d0, v3), p_q);

        vst4q_s32(a_ptr.add(base), int32x4x4_t(o0, o1, o2, o3));
        base += 16;
    }
}

/// NEON-accelerated forward negacyclic NTT for i32 primes.
///
/// Processes 4 butterfly pairs per iteration while `len >= 4`, then runs the
/// final two stages (`len = 2, 1`) through the vectorized [`forward_dif_tail_i32`]
/// when `D` is a multiple of 16 (scalar fallback otherwise).
pub(crate) unsafe fn forward_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_q = vdupq_n_s32(prime.p);
    let pinv_q = vdupq_n_s32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    // Pre-twist by psi^i
    {
        let psi_ptr = tw.psi_pows.as_ptr() as *const i32;
        let mut i = 0;
        while i + 4 <= D {
            let ai = vld1q_s32(a_ptr.add(i));
            let psi = vld1q_s32(psi_ptr.add(i));
            vst1q_s32(a_ptr.add(i), mont_mul_4x_i32(ai, psi, p_q, pinv_q));
            i += 4;
        }
    }

    // DIF butterfly stages (4-wide while half-length permits).
    let mut len = D / 2;
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
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
                    mont_mul_4x_i32(diff, w, p_q, pinv_q),
                );
                j += 4;
            }
            start += 2 * len;
        }
        len /= 2;
    }

    // Final two stages (len = 2, 1). The vectorized tail already normalizes its
    // outputs to [0, p), so the closing reduce_range pass is only needed on the
    // scalar fallback (D not a multiple of 16).
    if batched_four_point_eligible::<D>(4) {
        forward_dif_tail_i32::<D>(a_ptr, tw.fwd_twiddles.as_ptr() as *const i32, p_q, pinv_q);
    } else {
        while len > 0 {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
                start += 2 * len;
            }
            len /= 2;
        }
        reduce_range_in_place_i32(a, p_q);
    }
}

/// NEON-accelerated inverse negacyclic NTT for i32 primes.
pub(crate) unsafe fn inverse_ntt_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_q = vdupq_n_s32(prime.p);
    let pinv_q = vdupq_n_s32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    // DIT butterfly stages. The first two stages are deinterleaved across four
    // independent size-4 sub-transforms when the degree permits it.
    let mut len = if batched_four_point_eligible::<D>(4) {
        inverse_dit_head_i32::<D>(a_ptr, tw.inv_twiddles.as_ptr() as *const i32, p_q, pinv_q);
        4
    } else {
        1
    };
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
                    let v = mont_mul_4x_i32(v_raw, w, p_q, pinv_q);

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
            vst1q_s32(a_ptr.add(i), mont_mul_4x_i32(ai, f, p_q, pinv_q));
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
    let p_q = vdupq_n_s32(prime.p);
    let pinv_q = vdupq_n_s32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = D / 2;
    while len >= 4 {
        let twiddle_base = len - 1;
        let tw_ptr = tw.fwd_twiddles.as_ptr() as *const i32;
        let mut start = 0usize;
        while start < D {
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
                    mont_mul_4x_i32(diff, w, p_q, pinv_q),
                );
                j += 4;
            }
            start += 2 * len;
        }
        len /= 2;
    }

    if batched_four_point_eligible::<D>(4) {
        forward_dif_tail_i32::<D>(a_ptr, tw.fwd_twiddles.as_ptr() as *const i32, p_q, pinv_q);
    } else {
        while len > 0 {
            let twiddle_base = len - 1;
            let mut start = 0usize;
            while start < D {
                for j in 0..len {
                    let w = tw.fwd_twiddles[twiddle_base + j];
                    let u = a[start + j];
                    let v = a[start + j + len];
                    let sum = u.raw().wrapping_add(v.raw());
                    let diff = u.raw().wrapping_sub(v.raw());
                    a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                    a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                }
                start += 2 * len;
            }
            len /= 2;
        }
        reduce_range_in_place_i32(a, p_q);
    }
}

/// NEON-accelerated inverse cyclic NTT for i32 (no negacyclic untwist).
pub(crate) unsafe fn inverse_ntt_cyclic_i32<const D: usize>(
    a: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    tw: &NttTwiddles<i32, D>,
) {
    let p_q = vdupq_n_s32(prime.p);
    let pinv_q = vdupq_n_s32(prime.pinv);
    let a_ptr = a.as_mut_ptr() as *mut i32;

    let mut len = if batched_four_point_eligible::<D>(4) {
        inverse_dit_head_i32::<D>(a_ptr, tw.inv_twiddles.as_ptr() as *const i32, p_q, pinv_q);
        4
    } else {
        1
    };
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
                    let v = mont_mul_4x_i32(v_raw, w, p_q, pinv_q);
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
            vst1q_s32(a_ptr.add(i), mont_mul_4x_i32(ai, d_inv_q, p_q, pinv_q));
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
    let p_q = vdupq_n_s32(p);
    let pinv_q = vdupq_n_s32(pinv);
    let prime = NttPrime::compute(p);
    let mut i = 0;
    while i + 4 <= d {
        let a = vld1q_s32(acc.add(i));
        let l = vld1q_s32(lhs.add(i));
        let r = vld1q_s32(rhs.add(i));
        let prod = mont_mul_4x_i32(l, r, p_q, pinv_q);
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

/// Convert signed i16 coefficients directly into an i32 Montgomery limb.
///
/// The protocol's i32 CRT primes are all larger than the complete i16 range,
/// so sign extension already gives the centered residue consumed by
/// `from_canonical`. Widening eight coefficients at a time avoids both the
/// temporary `[i32; D]` and a scalar table lookup for every output coefficient.
///
/// # Safety
///
/// `dst` and `src` must be valid for `d` elements and must not overlap. The
/// modulus must be larger than every absolute source coefficient.
pub(crate) unsafe fn centered_i16_to_mont_i32(
    dst: *mut i32,
    src: *const i16,
    d: usize,
    p: i32,
    pinv: i32,
    montsq: i32,
) {
    let p_q = vdupq_n_s32(p);
    let pinv_q = vdupq_n_s32(pinv);
    let montsq_q = vdupq_n_s32(montsq);
    let mut i = 0usize;
    while i + 8 <= d {
        let coefficients = vld1q_s16(src.add(i));
        let low = vmovl_s16(vget_low_s16(coefficients));
        let high = vmovl_high_s16(coefficients);
        vst1q_s32(dst.add(i), mont_mul_4x_i32(low, montsq_q, p_q, pinv_q));
        vst1q_s32(dst.add(i + 4), mont_mul_4x_i32(high, montsq_q, p_q, pinv_q));
        i += 8;
    }

    let prime = NttPrime::compute(p);
    while i < d {
        *dst.add(i) = prime.from_canonical(i32::from(*src.add(i))).raw();
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
