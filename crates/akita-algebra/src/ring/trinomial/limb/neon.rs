//! Eight independent, centered Montgomery lanes on AArch64.
//!
//! Unlike a lazy signed-lane kernel, both factors are always in
//! `[-floor(p/2), floor(p/2)]`, even for p > 2^14. The high-half instruction
//! saturates only for (-32768)*(-32768); neither the product nor its
//! correction (whose other factor is p < 32768) can hit that case.

use std::arch::aarch64::*;

use super::{TrinomialLimbDomain, DEGREE, LANES};

#[inline(always)]
unsafe fn center(value: int16x8_t, p: int16x8_t, half: int16x8_t) -> int16x8_t {
    // SAFETY: NEON register operations have no memory preconditions. Inputs
    // lie in (-p,p), so signed additions/subtractions cannot overflow.
    unsafe {
        let value = vsubq_s16(
            value,
            vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(value, half)), p),
        );
        vaddq_s16(
            value,
            vandq_s16(vreinterpretq_s16_u16(vcltq_s16(value, vnegq_s16(half))), p),
        )
    }
}

#[inline(always)]
unsafe fn mul(
    a: int16x8_t,
    b: int16x8_t,
    p: int16x8_t,
    pinv: int16x8_t,
    half: int16x8_t,
) -> int16x8_t {
    // SAFETY: centered operands have magnitude <= 9720. The correction may
    // have magnitude 32768, but its other factor is <= 19441. Thus doubled
    // products never saturate; shifting gives exact signed high halves.
    unsafe {
        let low = vmulq_s16(a, b);
        let high = vshrq_n_s16::<1>(vqdmulhq_s16(a, b));
        let correction = vmulq_s16(low, pinv);
        let correction_high = vshrq_n_s16::<1>(vqdmulhq_s16(correction, p));
        center(vsubq_s16(high, correction_high), p, half)
    }
}

pub(super) fn transform(domain: &TrinomialLimbDomain, values: &mut [i16; DEGREE], inverse: bool) {
    // SAFETY: AArch64 guarantees NEON. Every split has start + 3*stride
    // <= 81; each lane row is eight values. Pointers address disjoint rows
    // within the mutable 648-element array; twiddle rows each have 8 values.
    unsafe {
        let p = vdupq_n_s16(domain.prime() as i16);
        let half = vdupq_n_s16((domain.prime() / 2) as i16);
        let pinv = vdupq_n_s16(domain.arithmetic.pinv);
        let omega = vdupq_n_s16(if inverse {
            domain.inverse_omega
        } else {
            domain.omega
        });
        let ptr = values.as_mut_ptr();
        for index in 0..domain.splits.len() {
            let split = &domain.splits[if inverse {
                domain.splits.len() - 1 - index
            } else {
                index
            }];
            let r = vld1q_s16(if inverse {
                split.ri.as_ptr()
            } else {
                split.r.as_ptr()
            });
            let r2 = vld1q_s16(if inverse {
                split.ri2.as_ptr()
            } else {
                split.r2.as_ptr()
            });
            let step = split.stride * LANES;
            for t in 0..split.stride {
                let offset = (split.start + t) * LANES;
                let a = vld1q_s16(ptr.add(offset));
                let mut b = vld1q_s16(ptr.add(offset + step));
                let mut c = vld1q_s16(ptr.add(offset + 2 * step));
                if !inverse {
                    b = mul(b, r, p, pinv, half);
                    c = mul(c, r2, p, pinv, half);
                }
                let difference = mul(center(vsubq_s16(b, c), p, half), omega, p, pinv, half);
                let o0 = center(vaddq_s16(center(vaddq_s16(a, b), p, half), c), p, half);
                let mut o1 = center(
                    vaddq_s16(center(vsubq_s16(a, c), p, half), difference),
                    p,
                    half,
                );
                let mut o2 = center(
                    vsubq_s16(center(vsubq_s16(a, b), p, half), difference),
                    p,
                    half,
                );
                if inverse {
                    o1 = mul(o1, r, p, pinv, half);
                    o2 = mul(o2, r2, p, pinv, half);
                }
                vst1q_s16(ptr.add(offset), o0);
                vst1q_s16(ptr.add(offset + step), o1);
                vst1q_s16(ptr.add(offset + 2 * step), o2);
            }
        }
    }
}

pub(super) fn accumulate(sums: &mut [i32; DEGREE], lhs: &[i16; DEGREE], rhs: &[i16; DEGREE]) {
    // SAFETY: AArch64 guarantees NEON. 648 is divisible by 8; all loads and
    // stores address complete rows of their respective arrays. The enforced
    // fifteen-product checkpoint proves every signed 32-bit addition fits.
    unsafe {
        for offset in (0..DEGREE).step_by(LANES) {
            let a = vld1q_s16(lhs.as_ptr().add(offset));
            let b = vld1q_s16(rhs.as_ptr().add(offset));
            let ptr = sums.as_mut_ptr().add(offset);
            let low = vmlal_s16(vld1q_s32(ptr), vget_low_s16(a), vget_low_s16(b));
            let high = vmlal_high_s16(vld1q_s32(ptr.add(4)), a, b);
            vst1q_s32(ptr, low);
            vst1q_s32(ptr.add(4), high);
        }
    }
}
