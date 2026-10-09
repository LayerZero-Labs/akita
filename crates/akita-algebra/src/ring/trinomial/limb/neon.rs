//! Four signed 32-bit lanes; exact rounded-quotient twiddle multiplication.

use super::arithmetic::{Split, Twiddle};
use super::{TrinomialLimbDomain, DEGREE, LANES};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn mul(a: int32x4_t, w: int32x4_t, q: int32x4_t, p: int32x4_t) -> int32x4_t {
    // SAFETY: Register operations only. Centered twiddle quotients have
    // magnitude <=2^30, excluding the SQRDMULH saturating MIN*MIN case.
    // Wrapping low products recover the exact bounded signed remainder.
    // Assembly touches only registers. Early output constraints keep both
    // outputs distinct from every input: a must survive the first MUL.
    // The three instructions access no memory, stack, or condition flags.
    unsafe {
        let result;
        std::arch::asm!(
            "mul {result:v}.4s, {a:v}.4s, {w:v}.4s",
            "sqrdmulh {correction:v}.4s, {a:v}.4s, {q:v}.4s",
            "mls {result:v}.4s, {correction:v}.4s, {p:v}.4s",
            result = out(vreg) result,
            correction = out(vreg) _,
            a = in(vreg) a,
            w = in(vreg) w,
            q = in(vreg) q,
            p = in(vreg) p,
            options(pure, nomem, nostack, preserves_flags),
        );
        result
    }
}

#[inline(always)]
unsafe fn reduce_one(a: int32x4_t, quotient: int32x4_t, p: int32x4_t) -> int32x4_t {
    // SAFETY: Multiplication by the residue one uses the same exact rounded
    // quotient formula. The lazy B4 bound ensures the result lies in (-p,p).
    unsafe { vmlsq_s32(a, vqrdmulhq_s32(a, quotient), p) }
}

#[inline(always)]
unsafe fn center(a: int32x4_t, p: int32x4_t, half: int32x4_t) -> int32x4_t {
    // SAFETY: Input magnitude is strictly below p, so at most one correction
    // is needed; additions and subtractions fit signed 32-bit lanes.
    unsafe {
        let a = vsubq_s32(a, vandq_s32(vreinterpretq_s32_u32(vcgtq_s32(a, half)), p));
        vaddq_s32(
            a,
            vandq_s32(vreinterpretq_s32_u32(vcltq_s32(a, vnegq_s32(half))), p),
        )
    }
}

#[inline(always)]
unsafe fn weights(w: &[Twiddle; LANES], lane: usize) -> (int32x4_t, int32x4_t) {
    // SAFETY: lane is 0 or 4. Twiddle is repr(C), containing two i32 fields;
    // the size assertion proves no padding. The deinterleaving load reads
    // exactly four initialized value/quotient pairs inside the eight-element
    // array, producing one vector of each field.
    unsafe {
        let pair = vld2q_s32(w.as_ptr().add(lane).cast::<i32>());
        (pair.0, pair.1)
    }
}

struct ButterflyRegisters {
    omega: (int32x4_t, int32x4_t),
    one: (int32x4_t, int32x4_t),
    p: int32x4_t,
    half: int32x4_t,
    inverse: bool,
}

#[inline(always)]
unsafe fn row(
    ptr: *mut i32,
    offset: usize,
    step: usize,
    r: (int32x4_t, int32x4_t),
    r2: (int32x4_t, int32x4_t),
    registers: &ButterflyRegisters,
    normalize: bool,
) {
    // SAFETY: Caller supplies disjoint four-lane rows within the 648-value
    // array. Forward B4<5.67p<2^31 bounds all additions/subtractions;
    // inverse inputs are centered and its intermediates are <2p.
    unsafe {
        let ButterflyRegisters {
            omega,
            one,
            p,
            half,
            inverse,
        } = *registers;
        let a = vld1q_s32(ptr.add(offset));
        let mut b = vld1q_s32(ptr.add(offset + step));
        let mut c = vld1q_s32(ptr.add(offset + 2 * step));
        if !inverse {
            b = mul(b, r.0, r.1, p);
            c = mul(c, r2.0, r2.1, p);
        }
        let d = mul(vsubq_s32(b, c), omega.0, omega.1, p);
        let mut o0 = vaddq_s32(vaddq_s32(a, b), c);
        let mut o1 = vaddq_s32(vsubq_s32(a, c), d);
        let mut o2 = vsubq_s32(vsubq_s32(a, b), d);
        if inverse {
            o0 = center(reduce_one(o0, one.1, p), p, half);
            o1 = center(mul(o1, r.0, r.1, p), p, half);
            o2 = center(mul(o2, r2.0, r2.1, p), p, half);
        }
        if normalize && !inverse {
            o0 = reduce_one(o0, one.1, p);
            o1 = reduce_one(o1, one.1, p);
            o2 = reduce_one(o2, one.1, p);
        }
        vst1q_s32(ptr.add(offset), o0);
        vst1q_s32(ptr.add(offset + step), o1);
        vst1q_s32(ptr.add(offset + 2 * step), o2);
    }
}

pub(super) fn transform(
    domain: &TrinomialLimbDomain,
    values: &mut [i32; DEGREE],
    inverse: bool,
    skip_first: bool,
) {
    // SAFETY: NEON is statically available on this target. Each split covers disjoint eight-lane
    // rows inside the 648-value array, processed in two complete four-lane
    // halves. Twiddle loads are bounded by their eight-element arrays.
    unsafe {
        let p = vdupq_n_s32(domain.prime() as i32);
        let half = vdupq_n_s32((domain.prime() / 2) as i32);
        let w = if inverse {
            domain.inverse_omega
        } else {
            domain.omega
        };
        let omega = (vdupq_n_s32(w.value), vdupq_n_s32(w.quotient));
        let one = (
            vdupq_n_s32(domain.one.value),
            vdupq_n_s32(domain.one.quotient),
        );
        let registers = ButterflyRegisters {
            omega,
            one,
            p,
            half,
            inverse,
        };
        let ptr = values.as_mut_ptr();
        for index in usize::from(skip_first)..domain.splits.len() {
            let split: &Split = &domain.splits[if inverse {
                domain.splits.len() - 1 - index
            } else {
                index
            }];
            for lane in [0, 4] {
                let r = weights(if inverse { &split.ri } else { &split.r }, lane);
                let r2 = weights(if inverse { &split.ri2 } else { &split.r2 }, lane);
                let step = split.stride * LANES;
                if split.stride == 1 {
                    row(
                        ptr,
                        split.start * LANES + lane,
                        step,
                        r,
                        r2,
                        &registers,
                        true,
                    );
                } else {
                    for t in 0..split.stride {
                        row(
                            ptr,
                            (split.start + t) * LANES + lane,
                            step,
                            r,
                            r2,
                            &registers,
                            false,
                        );
                    }
                }
            }
        }
    }
}

pub(super) fn accumulate(sums: &mut [i64; DEGREE], lhs: &[i32; DEGREE], rhs: &[i32; DEGREE]) {
    // SAFETY: Complete four-lane loads are inside the input arrays; each
    // output pair is inside sums. The 128-product checkpoint proves that
    // signed 64-bit additions cannot overflow.
    unsafe {
        for offset in (0..DEGREE).step_by(4) {
            let a = vld1q_s32(lhs.as_ptr().add(offset));
            let b = vld1q_s32(rhs.as_ptr().add(offset));
            let ptr = sums.as_mut_ptr().add(offset);
            vst1q_s64(
                ptr,
                vmlal_s32(vld1q_s64(ptr), vget_low_s32(a), vget_low_s32(b)),
            );
            vst1q_s64(ptr.add(2), vmlal_high_s32(vld1q_s64(ptr.add(2)), a, b));
        }
    }
}

#[cfg(test)]
fn mul_lanes(arithmetic: super::arithmetic::Arithmetic, input: [i32; 4], w: Twiddle) -> [i32; 4] {
    let mut output = [0; 4];
    // SAFETY: NEON is statically available on this target; both arrays contain four initialized
    // lanes. The load/store cover their complete arrays and twiddles obey
    // the same centered quotient contract as the production transform.
    unsafe {
        let result = mul(
            vld1q_s32(input.as_ptr()),
            vdupq_n_s32(w.value),
            vdupq_n_s32(w.quotient),
            vdupq_n_s32(arithmetic.prime as i32),
        );
        vst1q_s32(output.as_mut_ptr(), result);
    }
    output
}

#[cfg(test)]
fn rounded_high_lanes(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    let mut output = [0; 4];
    // SAFETY: NEON is statically available on this target; the complete four-lane loads and
    // store stay within the initialized input and output arrays.
    unsafe {
        vst1q_s32(
            output.as_mut_ptr(),
            vqrdmulhq_s32(vld1q_s32(a.as_ptr()), vld1q_s32(b.as_ptr())),
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::super::arithmetic::{self, Arithmetic};
    use super::*;

    #[test]
    fn neon_constant_products_match_i128_at_lane_and_lazy_extremes() {
        for p in TrinomialLimbDomain::ADMITTED_PRIMES {
            let arithmetic = Arithmetic { prime: p };
            let bounds = arithmetic::forward_bounds(p);
            let mut operands = vec![i32::MIN, i32::MAX, -1, 0, 1];
            for bound in bounds {
                operands.extend([bound as i32, -(bound as i32)]);
            }
            let half = (p / 2) as i32;
            for weight in [-half, -1, 0, 1, half] {
                let twiddle = Twiddle::new(arithmetic, weight);
                let expected_quotient = (i128::from(weight) * (1i128 << 31) + i128::from(p / 2))
                    .div_euclid(i128::from(p));
                assert_eq!(i128::from(twiddle.quotient), expected_quotient);
                for offset in (0..operands.len()).step_by(4) {
                    let lanes =
                        std::array::from_fn(|lane| operands[(offset + lane) % operands.len()]);
                    let result = mul_lanes(arithmetic, lanes, twiddle);
                    for (a, actual) in lanes.into_iter().zip(result) {
                        let quotient = (i128::from(a) * expected_quotient + (1i128 << 30)) >> 31;
                        let expected =
                            i128::from(a) * i128::from(weight) - quotient * i128::from(p);
                        assert_eq!(i128::from(actual), expected);
                        assert_eq!(actual, arithmetic.mul(a, twiddle));
                        assert!(
                            i64::from(actual).abs()
                                <= arithmetic::multiply_bound(p, i64::from(a).abs())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn neon_rounded_high_matches_i128_including_saturation() {
        let extrema = [i32::MIN, i32::MAX, -(1 << 30), 1 << 30, -1, 0, 1];
        for a in extrema {
            for offset in 0..extrema.len() {
                let b = std::array::from_fn(|lane| extrema[(offset + lane) % extrema.len()]);
                let result = rounded_high_lanes([a; 4], b);
                for (b, actual) in b.into_iter().zip(result) {
                    let expected = ((i128::from(a) * i128::from(b) + (1i128 << 30)) >> 31)
                        .clamp(i128::from(i32::MIN), i128::from(i32::MAX));
                    assert_eq!(i128::from(actual), expected);
                    assert_eq!(actual, arithmetic::sqrdmulh(a, b));
                }
            }
        }
    }
}
