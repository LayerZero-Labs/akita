//! AArch64 NEON packed backends for Fp32, Fp64, Fp128.

use super::packed::{PackedField, PackedValue};
use crate::fields::ext::{Fp2Config, PowerBasisFp4Config, TowerBasisFp4Config};
use crate::fields::{Fp128, Fp32, Fp64};
use crate::Invertible;
use core::arch::aarch64::{
    uint32x2_t, uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, vandq_u32, vandq_u64, vbslq_u64,
    vcltq_u32, vcltq_u64, vcombine_u32, vdup_n_u32, vdupq_n_s64, vdupq_n_u32, vdupq_n_u64,
    vget_low_u32, vminq_u32, vmovn_u64, vmull_high_u32, vmull_u32, vorrq_u64, vshlq_u64, vsubq_u32,
    vsubq_u64,
};
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// Number of packed `Fp128` lanes in this backend.
pub const WIDTH: usize = 2;

/// True SoA layout for two packed `Fp128` lanes.
///
/// `lo = [lane0.lo, lane1.lo]`
/// `hi = [lane0.hi, lane1.hi]`
#[derive(Clone, Copy)]
pub struct PackedFp128Neon<const P: u128> {
    lo: [u64; 2],
    hi: [u64; 2],
}

#[inline(always)]
fn to_vec(x: [u64; 2]) -> uint64x2_t {
    unsafe { transmute::<[u64; 2], uint64x2_t>(x) }
}

#[inline(always)]
fn from_vec(v: uint64x2_t) -> [u64; 2] {
    unsafe { transmute::<uint64x2_t, [u64; 2]>(v) }
}

#[inline(always)]
fn mask_to_bit(mask: uint64x2_t) -> uint64x2_t {
    unsafe { vandq_u64(mask, vdupq_n_u64(1)) }
}

#[inline(always)]
const fn modulus_lo<const P: u128>() -> u64 {
    P as u64
}

#[inline(always)]
const fn modulus_hi<const P: u128>() -> u64 {
    (P >> 64) as u64
}

use super::util::{is_pow2_u64, log2_pow2_u64};

impl<const P: u128> PackedFp128Neon<P> {
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
    const C_SHIFT_KIND: i8 = {
        let c = Self::C_LO;
        if c > 1 && is_pow2_u64(c - 1) {
            1
        } else if c == u64::MAX || is_pow2_u64(c + 1) {
            -1
        } else {
            0
        }
    };
    const C_SHIFT: u32 = {
        let c = Self::C_LO;
        if Self::C_SHIFT_KIND == 1 {
            log2_pow2_u64(c - 1)
        } else if Self::C_SHIFT_KIND == -1 {
            if c == u64::MAX {
                64
            } else {
                log2_pow2_u64(c + 1)
            }
        } else {
            0
        }
    };

    #[inline(always)]
    fn mul_wide_u64(a: u64, b: u64) -> (u64, u64) {
        let prod = (a as u128) * (b as u128);
        (prod as u64, (prod >> 64) as u64)
    }

    #[inline(always)]
    fn mul_c_wide(x: u64) -> (u64, u64) {
        if Self::C_SHIFT_KIND == 1 {
            let v = ((x as u128) << Self::C_SHIFT) + x as u128;
            (v as u64, (v >> 64) as u64)
        } else if Self::C_SHIFT_KIND == -1 {
            let v = ((x as u128) << Self::C_SHIFT) - x as u128;
            (v as u64, (v >> 64) as u64)
        } else {
            Self::mul_wide_u64(Self::C_LO, x)
        }
    }

    #[inline(always)]
    fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> (u64, u64) {
        let (ct2_lo, ct2_hi) = Self::mul_c_wide(t2);

        let (s0, carry0) = t0.overflowing_add(ct2_lo);
        let (s1a, carry1a) = t1.overflowing_add(ct2_hi);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        if overflow | carry3 {
            (r0, r1)
        } else {
            (s0, s1)
        }
    }

    #[inline(always)]
    fn mul_raw_lane(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
        let (p00_lo, p00_hi) = Self::mul_wide_u64(a0, b0);
        let (p01_lo, p01_hi) = Self::mul_wide_u64(a0, b1);
        let (p10_lo, p10_hi) = Self::mul_wide_u64(a1, b0);
        let (p11_lo, p11_hi) = Self::mul_wide_u64(a1, b1);

        let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = p01_hi as u128 + p10_hi as u128 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        let (cr2_lo, cr2_hi) = Self::mul_c_wide(r2);
        let (cr3_lo, cr3_hi) = Self::mul_c_wide(r3);

        let t0_sum = r0 as u128 + cr2_lo as u128;
        let t0 = t0_sum as u64;
        let carryf = (t0_sum >> 64) as u64;

        let t1_sum = r1 as u128 + cr2_hi as u128 + cr3_lo as u128 + carryf as u128;
        let t1 = t1_sum as u64;

        let t2_sum = cr3_hi as u128 + (t1_sum >> 64);
        let t2 = t2_sum as u64;
        debug_assert_eq!(t2_sum >> 64, 0);

        Self::fold2_canonicalize(t0, t1, t2)
    }
}

impl<const P: u128> Default for PackedFp128Neon<P> {
    #[inline]
    fn default() -> Self {
        Self::broadcast(Fp128::zero())
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp128Neon")
            .field(&[self.extract(0), self.extract(1)])
            .finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.extract(0) == other.extract(0) && self.extract(1) == other.extract(1)
    }
}

impl<const P: u128> Eq for PackedFp128Neon<P> {}

impl<const P: u128> PackedValue for PackedFp128Neon<P> {
    type Value = Fp128<P>;
    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        let x0 = f(0);
        let x1 = f(1);
        Self {
            lo: [x0.0[0], x1.0[0]],
            hi: [x0.0[1], x1.0[1]],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < WIDTH);
        Fp128([self.lo[lane], self.hi[lane]])
    }
}

impl<const P: u128> Add for PackedFp128Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lo_a = to_vec(self.lo);
        let hi_a = to_vec(self.hi);
        let lo_b = to_vec(rhs.lo);
        let hi_b = to_vec(rhs.hi);

        let (out_lo, out_hi) = unsafe {
            let c_vec = vdupq_n_u64(Self::C_LO);

            // s = a + b (128-bit, two lanes).
            // Carry propagation uses raw comparison masks with sub: subtracting
            // a lane of all-1s is equivalent to adding 1 in wrapping arithmetic.
            let sum_lo = vaddq_u64(lo_a, lo_b);
            let carry_lo = vcltq_u64(sum_lo, lo_a);

            let hi_tmp = vaddq_u64(hi_a, hi_b);
            let carry_hi1 = vcltq_u64(hi_tmp, hi_a);
            let sum_hi = vsubq_u64(hi_tmp, carry_lo);
            let carry_hi2 = vcltq_u64(sum_hi, hi_tmp);
            let overflow = vorrq_u64(carry_hi1, carry_hi2);

            // t = s + C.  Since p = 2^128 - C, this is s - p (mod 2^128).
            // If s + C >= 2^128 then s >= p, so the reduced value t is correct.
            let t_lo = vaddq_u64(sum_lo, c_vec);
            let carry_c = vcltq_u64(t_lo, sum_lo);
            let t_hi = vsubq_u64(sum_hi, carry_c);
            let carry_t = vcltq_u64(t_hi, sum_hi);

            let use_reduced = vorrq_u64(overflow, carry_t);
            let out_lo = vbslq_u64(use_reduced, t_lo, sum_lo);
            let out_hi = vbslq_u64(use_reduced, t_hi, sum_hi);
            (out_lo, out_hi)
        };

        Self {
            lo: from_vec(out_lo),
            hi: from_vec(out_hi),
        }
    }
}

impl<const P: u128> Sub for PackedFp128Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lo_a = to_vec(self.lo);
        let hi_a = to_vec(self.hi);
        let lo_b = to_vec(rhs.lo);
        let hi_b = to_vec(rhs.hi);

        let (out_lo, out_hi) = unsafe {
            let p_lo = vdupq_n_u64(modulus_lo::<P>());
            let p_hi = vdupq_n_u64(modulus_hi::<P>());

            let diff_lo = vsubq_u64(lo_a, lo_b);
            let borrow_lo = mask_to_bit(vcltq_u64(lo_a, lo_b));

            let diff_hi_tmp = vsubq_u64(hi_a, hi_b);
            let borrow_hi1 = vcltq_u64(hi_a, hi_b);
            let diff_hi = vsubq_u64(diff_hi_tmp, borrow_lo);
            let borrow_hi2 = vcltq_u64(diff_hi_tmp, borrow_lo);
            let borrow_128 = vorrq_u64(borrow_hi1, borrow_hi2);

            let corr_lo = vaddq_u64(diff_lo, p_lo);
            let carry_lo = mask_to_bit(vcltq_u64(corr_lo, diff_lo));

            let corr_hi_tmp = vaddq_u64(diff_hi, p_hi);
            let corr_hi = vaddq_u64(corr_hi_tmp, carry_lo);

            let out_lo = vbslq_u64(borrow_128, corr_lo, diff_lo);
            let out_hi = vbslq_u64(borrow_128, corr_hi, diff_hi);
            (out_lo, out_hi)
        };

        Self {
            lo: from_vec(out_lo),
            hi: from_vec(out_hi),
        }
    }
}

impl<const P: u128> Mul for PackedFp128Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (o0_lo, o0_hi) = Self::mul_raw_lane(self.lo[0], self.hi[0], rhs.lo[0], rhs.hi[0]);
        let (o1_lo, o1_hi) = Self::mul_raw_lane(self.lo[1], self.hi[1], rhs.lo[1], rhs.hi[1]);

        Self {
            lo: [o0_lo, o1_lo],
            hi: [o0_hi, o1_hi],
        }
    }
}

impl<const P: u128> AddAssign for PackedFp128Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for PackedFp128Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for PackedFp128Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u128> PackedField for PackedFp128Neon<P> {
    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::from_fn(|_| value)
    }
}

/// Number of packed `Fp32` lanes.
pub const FP32_WIDTH: usize = 4;

/// NEON packed `Fp32` backend: 4 lanes in `uint32x4_t`.
#[derive(Clone, Copy)]
pub struct PackedFp32Neon<const P: u32> {
    vals: [u32; 4],
}

#[inline(always)]
fn to_vec32(x: [u32; 4]) -> uint32x4_t {
    unsafe { transmute::<[u32; 4], uint32x4_t>(x) }
}

#[inline(always)]
fn from_vec32(v: uint32x4_t) -> [u32; 4] {
    unsafe { transmute::<uint32x4_t, [u32; 4]>(v) }
}

impl<const P: u32> PackedFp32Neon<P> {
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

    const SHIFT64_MOD_P: u32 = {
        let c = Self::C as u128;
        let bits = Self::BITS;
        let mask = if bits == 32 {
            u32::MAX as u128
        } else {
            (1u128 << bits) - 1
        };
        let mut v = 1u128 << 64;
        while v >> bits != 0 {
            v = (v & mask) + c * (v >> bits);
        }
        let f = v as u64;
        let reduced = f.wrapping_sub(P as u64);
        let borrow = reduced >> 63;
        reduced.wrapping_add(borrow.wrapping_neg() & (P as u64)) as u32
    };

    #[inline(always)]
    fn to_vec(self) -> uint32x4_t {
        to_vec32(self.vals)
    }

    #[inline(always)]
    fn from_vec(v: uint32x4_t) -> Self {
        Self {
            vals: from_vec32(v),
        }
    }

    #[inline(always)]
    fn add_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p = vdupq_n_u32(P);
            if Self::BITS <= 31 {
                let t = vaddq_u32(a, b);
                vminq_u32(t, vsubq_u32(t, p))
            } else {
                let c = vdupq_n_u32(Self::C);
                let t = vaddq_u32(a, b);
                let overflow = vcltq_u32(t, a);
                let folded = vaddq_u32(t, vandq_u32(overflow, c));
                vminq_u32(folded, vsubq_u32(folded, p))
            }
        }
    }

    #[inline(always)]
    fn sub_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p = vdupq_n_u32(P);
            if Self::BITS <= 31 {
                let t = vsubq_u32(a, b);
                vminq_u32(t, vaddq_u32(t, p))
            } else {
                let t = vsubq_u32(a, b);
                let underflow = vcltq_u32(a, b);
                vsubq_u32(t, vandq_u32(underflow, vdupq_n_u32(Self::C)))
            }
        }
    }

    #[inline(always)]
    fn mul_vec(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let prod_lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
            let prod_hi = vmull_high_u32(a, b);
            Self::solinas_reduce(prod_lo, prod_hi)
        }
    }

    #[inline(always)]
    fn add_u64_with_carry(
        sum: uint64x2_t,
        rhs: uint64x2_t,
        carry: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            let next = vaddq_u64(sum, rhs);
            let overflow = vcltq_u64(next, sum);
            (next, vaddq_u64(carry, mask_to_bit(overflow)))
        }
    }

    #[inline(always)]
    fn carry_correction(carry: uint64x2_t) -> uint64x2_t {
        unsafe { vmull_u32(vmovn_u64(carry), vdup_n_u32(Self::SHIFT64_MOD_P)) }
    }

    #[inline(always)]
    fn dot_product_4_vec(a: [uint32x4_t; 4], b: [uint32x4_t; 4]) -> uint32x4_t {
        unsafe {
            let mut sum_lo = vmull_u32(vget_low_u32(a[0]), vget_low_u32(b[0]));
            let mut sum_hi = vmull_high_u32(a[0], b[0]);
            let mut carry_lo = vdupq_n_u64(0);
            let mut carry_hi = vdupq_n_u64(0);

            let prod_lo_1 = vmull_u32(vget_low_u32(a[1]), vget_low_u32(b[1]));
            let prod_hi_1 = vmull_high_u32(a[1], b[1]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_1, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_1, carry_hi);

            let prod_lo_2 = vmull_u32(vget_low_u32(a[2]), vget_low_u32(b[2]));
            let prod_hi_2 = vmull_high_u32(a[2], b[2]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_2, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_2, carry_hi);

            let prod_lo_3 = vmull_u32(vget_low_u32(a[3]), vget_low_u32(b[3]));
            let prod_hi_3 = vmull_high_u32(a[3], b[3]);
            (sum_lo, carry_lo) = Self::add_u64_with_carry(sum_lo, prod_lo_3, carry_lo);
            (sum_hi, carry_hi) = Self::add_u64_with_carry(sum_hi, prod_hi_3, carry_hi);

            Self::solinas_reduce_with_carry(sum_lo, sum_hi, carry_lo, carry_hi)
        }
    }

    #[inline(always)]
    fn mul_nr_vec<C>(x: uint32x4_t) -> uint32x4_t
    where
        C: Fp2Config<Fp32<P>>,
    {
        if C::IS_NEG_ONE {
            Self::sub_vec(unsafe { vdupq_n_u32(0) }, x)
        } else if C::non_residue().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_non_residue(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    #[inline(always)]
    fn mul_w_vec<C>(x: uint32x4_t) -> uint32x4_t
    where
        C: PowerBasisFp4Config<Fp32<P>>,
    {
        if C::w().0 == 2 {
            Self::add_vec(x, x)
        } else {
            C::mul_w(Self::from_vec(x), Self::broadcast).to_vec()
        }
    }

    #[inline(always)]
    fn mul_c_u64(hi: uint64x2_t, c: uint32x2_t) -> uint64x2_t {
        unsafe {
            let hi_narrow = vmovn_u64(hi);
            vmull_u32(hi_narrow, c)
        }
    }

    #[inline(always)]
    fn solinas_reduce(prod_lo: uint64x2_t, prod_hi: uint64x2_t) -> uint32x4_t {
        unsafe {
            let mask = vdupq_n_u64(Self::MASK_U64);
            let neg_bits = vdupq_n_s64(-(Self::BITS as i64));
            let c = vdup_n_u32(Self::C);

            let f1_lo = vaddq_u64(
                vandq_u64(prod_lo, mask),
                Self::mul_c_u64(vshlq_u64(prod_lo, neg_bits), c),
            );
            let f1_hi = vaddq_u64(
                vandq_u64(prod_hi, mask),
                Self::mul_c_u64(vshlq_u64(prod_hi, neg_bits), c),
            );

            let f2_lo = vaddq_u64(
                vandq_u64(f1_lo, mask),
                Self::mul_c_u64(vshlq_u64(f1_lo, neg_bits), c),
            );
            let f2_hi = vaddq_u64(
                vandq_u64(f1_hi, mask),
                Self::mul_c_u64(vshlq_u64(f1_hi, neg_bits), c),
            );

            if Self::BITS < 32 {
                let result = vcombine_u32(vmovn_u64(f2_lo), vmovn_u64(f2_hi));
                let p = vdupq_n_u32(P);
                vminq_u32(result, vsubq_u32(result, p))
            } else {
                let p_u64 = vdupq_n_u64(P as u64);

                let red_lo = vsubq_u64(f2_lo, p_u64);
                let keep_lo = vcltq_u64(f2_lo, p_u64);
                let out_lo = vbslq_u64(keep_lo, f2_lo, red_lo);

                let red_hi = vsubq_u64(f2_hi, p_u64);
                let keep_hi = vcltq_u64(f2_hi, p_u64);
                let out_hi = vbslq_u64(keep_hi, f2_hi, red_hi);

                vcombine_u32(vmovn_u64(out_lo), vmovn_u64(out_hi))
            }
        }
    }

    #[inline(always)]
    fn solinas_reduce_with_carry(
        prod_lo: uint64x2_t,
        prod_hi: uint64x2_t,
        carry_lo: uint64x2_t,
        carry_hi: uint64x2_t,
    ) -> uint32x4_t {
        unsafe {
            let mask = vdupq_n_u64(Self::MASK_U64);
            let neg_bits = vdupq_n_s64(-(Self::BITS as i64));
            let c = vdup_n_u32(Self::C);

            let f1_lo = vaddq_u64(
                vaddq_u64(
                    vandq_u64(prod_lo, mask),
                    Self::mul_c_u64(vshlq_u64(prod_lo, neg_bits), c),
                ),
                Self::carry_correction(carry_lo),
            );
            let f1_hi = vaddq_u64(
                vaddq_u64(
                    vandq_u64(prod_hi, mask),
                    Self::mul_c_u64(vshlq_u64(prod_hi, neg_bits), c),
                ),
                Self::carry_correction(carry_hi),
            );

            let f2_lo = vaddq_u64(
                vandq_u64(f1_lo, mask),
                Self::mul_c_u64(vshlq_u64(f1_lo, neg_bits), c),
            );
            let f2_hi = vaddq_u64(
                vandq_u64(f1_hi, mask),
                Self::mul_c_u64(vshlq_u64(f1_hi, neg_bits), c),
            );

            if Self::BITS < 32 {
                let result = vcombine_u32(vmovn_u64(f2_lo), vmovn_u64(f2_hi));
                let p = vdupq_n_u32(P);
                vminq_u32(result, vsubq_u32(result, p))
            } else {
                let p_u64 = vdupq_n_u64(P as u64);

                let red_lo = vsubq_u64(f2_lo, p_u64);
                let keep_lo = vcltq_u64(f2_lo, p_u64);
                let out_lo = vbslq_u64(keep_lo, f2_lo, red_lo);

                let red_hi = vsubq_u64(f2_hi, p_u64);
                let keep_hi = vcltq_u64(f2_hi, p_u64);
                let out_hi = vbslq_u64(keep_hi, f2_hi, red_hi);

                vcombine_u32(vmovn_u64(out_lo), vmovn_u64(out_hi))
            }
        }
    }
}

impl<const P: u32> Default for PackedFp32Neon<P> {
    #[inline]
    fn default() -> Self {
        Self { vals: [0; 4] }
    }
}

impl<const P: u32> fmt::Debug for PackedFp32Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp32Neon").field(&self.vals).finish()
    }
}

impl<const P: u32> PartialEq for PackedFp32Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u32> Eq for PackedFp32Neon<P> {}

impl<const P: u32> Add for PackedFp32Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vec(Self::add_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> Sub for PackedFp32Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vec(Self::sub_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> Mul for PackedFp32Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vec(Self::mul_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> PackedValue for PackedFp32Neon<P> {
    type Value = Fp32<P>;
    const WIDTH: usize = FP32_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self {
            vals: [f(0).0, f(1).0, f(2).0, f(3).0],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP32_WIDTH);
        Fp32(self.vals[lane])
    }
}

impl<const P: u32> AddAssign for PackedFp32Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp32Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp32Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> PackedField for PackedFp32Neon<P> {
    type Scalar = Fp32<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self { vals: [value.0; 4] }
    }

    #[inline(always)]
    fn fp2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: Fp2Config<Self::Scalar>,
    {
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

    #[inline(always)]
    fn power_basis_fp4_mul<C>(a: [Self; 4], b: [Self; 4]) -> [Self; 4]
    where
        C: PowerBasisFp4Config<Self::Scalar>,
    {
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

    #[inline(always)]
    fn ring_subfield_fp4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
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

    #[inline(always)]
    fn ring_subfield_fp4_square(a: [Self; 4]) -> [Self; 4] {
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

    #[inline(always)]
    fn ring_subfield_fp4_inverse(a: [Self; 4]) -> Option<[Self; 4]>
    where
        Self::Scalar: Invertible,
    {
        let [a0, a1, a2, a3] = a.map(Self::to_vec);
        let zero = unsafe { vdupq_n_u32(0) };
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

/// Number of packed `Fp64` lanes.
pub const FP64_WIDTH: usize = 2;

/// NEON packed `Fp64` backend: 2 lanes in `uint64x2_t`.
#[derive(Clone, Copy)]
pub struct PackedFp64Neon<const P: u64> {
    vals: [u64; 2],
}

impl<const P: u64> PackedFp64Neon<P> {
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

    const MASK_U128: u128 = if Self::BITS == 64 {
        u64::MAX as u128
    } else {
        (1u128 << Self::BITS) - 1
    };

    const FOLD_IN_U64: bool =
        Self::BITS < 64 && (Self::C_LO as u128) < (1u128 << (64 - Self::BITS));

    #[inline(always)]
    fn mul_c_narrow(x: u64) -> u64 {
        Self::C_LO.wrapping_mul(x)
    }

    #[inline(always)]
    fn reduce_product(x: u128) -> u64 {
        if Self::FOLD_IN_U64 {
            let lo = x as u64;
            let hi = (x >> 64) as u64;
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let f1 = (lo & Self::MASK64).wrapping_add(Self::mul_c_narrow(high));
            let f2 = (f1 & Self::MASK64).wrapping_add(Self::mul_c_narrow(f1 >> Self::BITS));
            let reduced = f2.wrapping_sub(P);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            let f1 =
                (x & Self::MASK_U128) + (Self::C_LO as u128) * ((x >> Self::BITS) as u64 as u128);
            let f2 =
                (f1 & Self::MASK_U128) + (Self::C_LO as u128) * ((f1 >> Self::BITS) as u64 as u128);
            let reduced = f2.wrapping_sub(P as u128);
            let borrow = reduced >> 127;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
        }
    }
}

impl<const P: u64> Default for PackedFp64Neon<P> {
    #[inline]
    fn default() -> Self {
        Self { vals: [0; 2] }
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Neon").field(&self.vals).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u64> Eq for PackedFp64Neon<P> {}

impl<const P: u64> Add for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let a = to_vec(self.vals);
        let b = to_vec(rhs.vals);
        let result = unsafe {
            let p = vdupq_n_u64(P);
            if Self::BITS == 64 {
                let s = vaddq_u64(a, b);
                let overflow = vcltq_u64(s, a);
                let folded = vaddq_u64(s, vandq_u64(overflow, vdupq_n_u64(Self::C_LO)));
                let reduced = vsubq_u64(folded, p);
                let borrow = vcltq_u64(folded, p);
                vbslq_u64(borrow, folded, reduced)
            } else if Self::BITS <= 62 {
                let s = vaddq_u64(a, b);
                let r = vsubq_u64(s, p);
                let borrow = vcltq_u64(s, p);
                vbslq_u64(borrow, s, r)
            } else {
                let s = vaddq_u64(a, b);
                let overflow = vcltq_u64(s, a);
                let c = vdupq_n_u64(Self::C_LO);
                let s_plus_c = vaddq_u64(s, c);
                let s_minus_p = vsubq_u64(s, p);
                let borrow = vcltq_u64(s, p);
                let no_of = vbslq_u64(borrow, s, s_minus_p);
                vbslq_u64(overflow, s_plus_c, no_of)
            }
        };
        Self {
            vals: from_vec(result),
        }
    }
}

impl<const P: u64> Sub for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let a = to_vec(self.vals);
        let b = to_vec(rhs.vals);
        let result = unsafe {
            let d = vsubq_u64(a, b);
            let underflow = vcltq_u64(a, b);
            if Self::BITS == 64 {
                vsubq_u64(d, vandq_u64(underflow, vdupq_n_u64(Self::C_LO)))
            } else {
                vbslq_u64(underflow, vaddq_u64(d, vdupq_n_u64(P)), d)
            }
        };
        Self {
            vals: from_vec(result),
        }
    }
}

impl<const P: u64> Mul for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let x0 = (self.vals[0] as u128) * (rhs.vals[0] as u128);
        let x1 = (self.vals[1] as u128) * (rhs.vals[1] as u128);
        let r0 = Self::reduce_product(x0);
        let r1 = Self::reduce_product(x1);
        Self { vals: [r0, r1] }
    }
}

impl<const P: u64> PackedValue for PackedFp64Neon<P> {
    type Value = Fp64<P>;
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self {
            vals: [f(0).0, f(1).0],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Value {
        debug_assert!(lane < FP64_WIDTH);
        Fp64(self.vals[lane])
    }
}

impl<const P: u64> AddAssign for PackedFp64Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Neon<P> {
    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self { vals: [value.0; 2] }
    }

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
