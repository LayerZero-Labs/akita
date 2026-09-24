//! Direct negacyclic split tree, with binary nibble startup.

use std::arch::aarch64::*;

use super::binary::nibble_four;
use crate::ntt::{MontCoeff, NttPrime};

/// Reduce |x| < 2p to |y| < p with centered approximate division.
///
/// For v=round(2^31/p), qhat=round(x*v/2^31) differs from x/p by
/// at most 1/2+|x|/2^32 < 1 because p<2^30. Thus y=x-qhat*p
/// lies strictly between -p and p and has the same residue as x.
#[inline(always)]
pub(super) unsafe fn centered_reduce(
    x: int32x4_t,
    p: int32x4_t,
    reciprocal: int32x4_t,
) -> int32x4_t {
    vmlsq_s32(x, vqrdmulhq_s32(x, reciprocal), p)
}

/// Montgomery multiply by a prepared twiddle: companion=w*pinv (mod 2^32).
#[inline(always)]
unsafe fn twiddle_mul(x: int32x4_t, w: int32x4_t, companion: int32x4_t, p: int32x4_t) -> int32x4_t {
    let top = vqdmulhq_s32(x, w);
    let m = vmulq_s32(x, companion);
    vhsubq_s32(top, vqdmulhq_s32(m, p))
}

#[inline(always)]
unsafe fn canonical(x: int32x4_t, p: int32x4_t) -> int32x4_t {
    vaddq_s32(x, vandq_s32(vshrq_n_s32::<31>(x), p))
}

/// D is a power of two >=16; digits are validated binary; tables have D entries.
pub(super) unsafe fn forward_split4<const D: usize>(
    out: &mut [MontCoeff<i32>; D],
    digits: &[i8; D],
    prime: NttPrime<i32>,
    compact: &[i32],
    roots: &[i32],
    companions: &[i32],
) {
    let dst = out.as_mut_ptr().cast::<i32>();
    let byte_offsets = vcreate_u8(0x03020100_03020100);
    let offsets = vcombine_u8(byte_offsets, byte_offsets);
    for j in (0..D / 4).step_by(4) {
        let indices = nibble_four(digits, j, D / 4);
        let lookup = vaddq_u8(
            vreinterpretq_u8_u32(vmulq_n_u32(indices, 0x04040404)),
            offsets,
        );
        for segment in 0..4 {
            let ptr = compact.as_ptr().add(segment * 16).cast::<u8>();
            let table = uint8x16x4_t(
                vld1q_u8(ptr),
                vld1q_u8(ptr.add(16)),
                vld1q_u8(ptr.add(32)),
                vld1q_u8(ptr.add(48)),
            );
            let value = vreinterpretq_s32_u8(vqtbl4q_u8(table, lookup));
            vst1q_s32(dst.add(segment * (D / 4) + j), value);
        }
    }
    continue_split(out, prime, roots, companions, 4);
}

/// D is a power of two >=32; two centered nibble contributions sum to (-p,p).
pub(super) unsafe fn forward_split8<const D: usize>(
    out: &mut [MontCoeff<i32>; D],
    digits: &[i8; D],
    prime: NttPrime<i32>,
    table: &[i32],
    roots: &[i32],
    companions: &[i32],
) {
    let dst = out.as_mut_ptr().cast::<i32>();
    let byte_offsets = vcreate_u8(0x03020100_03020100);
    let offsets = vcombine_u8(byte_offsets, byte_offsets);
    for j in (0..D / 8).step_by(4) {
        let lo = nibble_four(digits, j, D / 8);
        let hi = nibble_four(digits, j + D / 2, D / 8);
        let lo = vaddq_u8(vreinterpretq_u8_u32(vmulq_n_u32(lo, 0x04040404)), offsets);
        let hi = vaddq_u8(vreinterpretq_u8_u32(vmulq_n_u32(hi, 0x04040404)), offsets);
        for segment in 0..8 {
            let ptr = table.as_ptr().add(segment * 32).cast::<u8>();
            let low_table = uint8x16x4_t(
                vld1q_u8(ptr),
                vld1q_u8(ptr.add(16)),
                vld1q_u8(ptr.add(32)),
                vld1q_u8(ptr.add(48)),
            );
            let high_table = uint8x16x4_t(
                vld1q_u8(ptr.add(64)),
                vld1q_u8(ptr.add(80)),
                vld1q_u8(ptr.add(96)),
                vld1q_u8(ptr.add(112)),
            );
            let low_value = vreinterpretq_s32_u8(vqtbl4q_u8(low_table, lo));
            let high_value = vreinterpretq_s32_u8(vqtbl4q_u8(high_table, hi));
            vst1q_s32(
                dst.add(segment * (D / 8) + j),
                vaddq_s32(low_value, high_value),
            );
        }
    }
    continue_split(out, prime, roots, companions, 8);
}

unsafe fn continue_split<const D: usize>(
    out: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    roots: &[i32],
    companions: &[i32],
    mut groups: usize,
) {
    let dst = out.as_mut_ptr().cast::<i32>();
    let p = vdupq_n_s32(prime.p);
    let v = ((1u64 << 31) + prime.p as u64 / 2) / prime.p as u64;
    let reciprocal = vdupq_n_s32(v as i32);
    let mut len = D / (2 * groups);
    while len >= 4 {
        for group in 0..groups {
            let w = vdupq_n_s32(roots[groups + group]);
            let companion = vdupq_n_s32(companions[groups + group]);
            let start = group * 2 * len;
            for j in (0..len).step_by(4) {
                let a = vld1q_s32(dst.add(start + j));
                let b = twiddle_mul(vld1q_s32(dst.add(start + j + len)), w, companion, p);
                vst1q_s32(
                    dst.add(start + j),
                    centered_reduce(vaddq_s32(a, b), p, reciprocal),
                );
                vst1q_s32(
                    dst.add(start + j + len),
                    centered_reduce(vsubq_s32(a, b), p, reciprocal),
                );
            }
        }
        groups *= 2;
        len /= 2;
    }
    // Four independent degree-four subtrees across NEON lanes. Their group
    // twiddles differ, so the len=1 twiddles are loaded deinterleaved.
    for base in (0..D).step_by(16) {
        let int32x4x4_t(a0, a1, a2, a3) = vld4q_s32(dst.add(base));
        let group = base / 4;
        let w = vld1q_s32(roots.as_ptr().add(D / 4 + group));
        let wp = vld1q_s32(companions.as_ptr().add(D / 4 + group));
        let c = twiddle_mul(a2, w, wp, p);
        let d = twiddle_mul(a3, w, wp, p);
        let b0 = centered_reduce(vaddq_s32(a0, c), p, reciprocal);
        let b2 = centered_reduce(vsubq_s32(a0, c), p, reciprocal);
        let b1 = centered_reduce(vaddq_s32(a1, d), p, reciprocal);
        let b3 = centered_reduce(vsubq_s32(a1, d), p, reciprocal);
        let int32x4x2_t(w0, w1) = vld2q_s32(roots.as_ptr().add(D / 2 + 2 * group));
        let int32x4x2_t(wp0, wp1) = vld2q_s32(companions.as_ptr().add(D / 2 + 2 * group));
        let c0 = twiddle_mul(b1, w0, wp0, p);
        let c1 = twiddle_mul(b3, w1, wp1, p);
        let o0 = canonical(centered_reduce(vaddq_s32(b0, c0), p, reciprocal), p);
        let o1 = canonical(centered_reduce(vsubq_s32(b0, c0), p, reciprocal), p);
        let o2 = canonical(centered_reduce(vaddq_s32(b2, c1), p, reciprocal), p);
        let o3 = canonical(centered_reduce(vsubq_s32(b2, c1), p, reciprocal), p);
        vst4q_s32(dst.add(base), int32x4x4_t(o0, o1, o2, o3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_barrett_bound_and_residue() {
        for prime in crate::ntt::tables::Q64_PRIMES {
            let p = i64::from(prime.p);
            let reciprocal = ((1i64 << 31) + p / 2) / p;
            let mut input = vec![
                -2 * p + 1,
                -p - 1,
                -p,
                -p + 1,
                -1,
                0,
                1,
                p - 1,
                p,
                p + 1,
                2 * p - 1,
                p / 2,
                -p / 2,
            ];
            let mut state = 0x43ee13d71u64;
            for _ in 0..65536 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                input.push((state % (4 * p - 1) as u64) as i64 - 2 * p + 1);
            }
            for &x in &input {
                let mut result = [0i32; 4];
                unsafe {
                    vst1q_s32(
                        result.as_mut_ptr(),
                        centered_reduce(
                            vdupq_n_s32(x as i32),
                            vdupq_n_s32(prime.p),
                            vdupq_n_s32(reciprocal as i32),
                        ),
                    );
                }
                for y in result {
                    assert!(i64::from(y).abs() < p, "x={x}, y={y}, p={p}");
                    assert_eq!(i64::from(y).rem_euclid(p), x.rem_euclid(p));
                }
            }
        }
    }
}
