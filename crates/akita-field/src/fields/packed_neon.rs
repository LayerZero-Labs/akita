//! AArch64 NEON packed backends for Fp16, Fp32, Fp64, Fp128.

use super::packed::{PackedField, PackedValue};
use crate::fields::ext::{
    ring_subfield_fp8_mul_schedule, ring_subfield_fp8_square_schedule, Fp2Config,
    PowerBasisFp4Config, TowerBasisFp4Config,
};
use crate::fields::{Fp128, Fp16, Fp32, Fp64};
use crate::Invertible;
use core::arch::aarch64::{
    uint16x8_t, uint32x2_t, uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, vandq_u32, vandq_u64,
    vbslq_u64, vcltq_u32, vcltq_u64, vcombine_u32, vdup_n_u32, vdupq_n_s32, vdupq_n_s64,
    vdupq_n_u32, vdupq_n_u64, vget_low_u16, vget_low_u32, vminq_u32, vmlsq_u32, vmovn_u64,
    vmull_high_u16, vmull_high_u32, vmull_u16, vmull_u32, vmulq_u32, vorrq_u64, vqdmulhq_s32,
    vreinterpretq_s32_u32, vreinterpretq_u32_s32, vshlq_n_u64, vshlq_u64, vshrq_n_u64, vsubq_u32,
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

mod fp32;
pub use fp32::*;

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

// ---------------------------------------------------------------------------
// PackedFp16Neon — 8 lanes of Fp16 in uint16x8_t
// ---------------------------------------------------------------------------

/// Number of packed `Fp16` lanes.
pub const FP16_WIDTH: usize = 8;

/// NEON packed `Fp16` backend: 8 lanes in `uint16x8_t`.
#[derive(Clone, Copy)]
pub struct PackedFp16Neon<const P: u32> {
    vals: [u16; 8],
}

#[inline(always)]
fn to_vec16(x: [u16; 8]) -> uint16x8_t {
    unsafe { transmute::<[u16; 8], uint16x8_t>(x) }
}

#[inline(always)]
fn from_vec16(v: uint16x8_t) -> [u16; 8] {
    unsafe { transmute::<uint16x8_t, [u16; 8]>(v) }
}

impl<const P: u32> PackedFp16Neon<P> {
    const BITS: u32 = 32 - P.leading_zeros();

    #[inline(always)]
    fn to_vec(self) -> uint16x8_t {
        to_vec16(self.vals)
    }

    #[inline(always)]
    fn from_vec(v: uint16x8_t) -> Self {
        Self {
            vals: from_vec16(v),
        }
    }

    #[inline(always)]
    fn widen_lo(x: uint16x8_t) -> uint32x4_t {
        unsafe { core::arch::aarch64::vmovl_u16(vget_low_u16(x)) }
    }

    #[inline(always)]
    fn widen_hi(x: uint16x8_t) -> uint32x4_t {
        unsafe { core::arch::aarch64::vmovl_high_u16(x) }
    }

    #[inline(always)]
    fn narrow_u32(lo: uint32x4_t, hi: uint32x4_t) -> uint16x8_t {
        unsafe {
            use core::arch::aarch64::{vcombine_u16, vmovn_u32};
            vcombine_u16(vmovn_u32(lo), vmovn_u32(hi))
        }
    }

    #[inline(always)]
    fn add_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p32 = vdupq_n_u32(P);
            let sum = vaddq_u32(a, b);
            vminq_u32(sum, vsubq_u32(sum, p32))
        }
    }

    #[inline(always)]
    fn sub_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe {
            let p32 = vdupq_n_u32(P);
            let diff = vaddq_u32(vsubq_u32(a, b), p32);
            vminq_u32(diff, vsubq_u32(diff, p32))
        }
    }

    #[inline(always)]
    fn mul_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { Self::solinas_reduce_u32(vmulq_u32(a, b)) }
    }

    #[inline(always)]
    fn add_vec(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        Self::narrow_u32(
            Self::add_u32(Self::widen_lo(a), Self::widen_lo(b)),
            Self::add_u32(Self::widen_hi(a), Self::widen_hi(b)),
        )
    }

    #[inline(always)]
    fn sub_vec(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        Self::narrow_u32(
            Self::sub_u32(Self::widen_lo(a), Self::widen_lo(b)),
            Self::sub_u32(Self::widen_hi(a), Self::widen_hi(b)),
        )
    }

    /// Multiply 8 lanes: widens to 2×uint32x4_t, Solinas-reduces back to u16.
    #[inline(always)]
    fn mul_vec(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        unsafe {
            let prod_lo = vmull_u16(vget_low_u16(a), vget_low_u16(b));
            let prod_hi = vmull_high_u16(a, b);
            Self::solinas_reduce_16(prod_lo, prod_hi)
        }
    }

    /// Reduce two `uint32x4_t` (holding 8 × u32 products) back to `uint16x8_t`.
    ///
    /// Three Solinas folds suffice for all valid `Fp16<P>` parameters
    /// (BITS ≤ 16, C(C+1) < P). Worst-case bound after fold 3:
    ///   fold1 ≤ (C+1)·2^BITS  →  fold2 ≤ 2^BITS + C² - 2C
    ///   fold3 ≤ C² - C - 1  <  2^BITS     (since C < √P ≤ 2⁸)
    #[inline(always)]
    unsafe fn solinas_reduce_16(prod_lo: uint32x4_t, prod_hi: uint32x4_t) -> uint16x8_t {
        use core::arch::aarch64::{vcombine_u16, vmovn_u32};

        let red_lo = Self::solinas_reduce_u32(prod_lo);
        let red_hi = Self::solinas_reduce_u32(prod_hi);
        vcombine_u16(vmovn_u32(red_lo), vmovn_u32(red_hi))
    }

    #[inline(always)]
    unsafe fn solinas_reduce_u32(prod: uint32x4_t) -> uint32x4_t {
        use core::arch::aarch64::vshlq_u32;

        let mask = vdupq_n_u32((1u32 << Self::BITS) - 1);
        let neg_bits = vdupq_n_s32(-(Self::BITS as i32));

        let fold = |x: uint32x4_t| -> uint32x4_t {
            let lo = vandq_u32(x, mask);
            let hi = vshlq_u32(x, neg_bits);
            vaddq_u32(lo, vmulq_u32(hi, vdupq_n_u32(Fp16::<P>::C)))
        };

        let f1 = fold(prod);
        let f2 = fold(f1);
        let f3 = fold(f2);
        let p32 = vdupq_n_u32(P);
        vminq_u32(f3, vsubq_u32(f3, p32))
    }

    /// Run the shared fp8 multiply schedule on one widened `uint32x4_t` half.
    #[inline(always)]
    fn ring_subfield_fp8_mul_u32(a: [uint32x4_t; 8], b: [uint32x4_t; 8]) -> [uint32x4_t; 8] {
        let zero = unsafe { vdupq_n_u32(0) };
        ring_subfield_fp8_mul_schedule(a, b, zero, Self::add_u32, Self::sub_u32, Self::mul_u32)
    }

    /// Run the shared fp8 square schedule on one widened `uint32x4_t` half.
    #[inline(always)]
    fn ring_subfield_fp8_square_u32(a: [uint32x4_t; 8]) -> [uint32x4_t; 8] {
        let zero = unsafe { vdupq_n_u32(0) };
        ring_subfield_fp8_square_schedule(a, zero, Self::add_u32, Self::sub_u32, Self::mul_u32)
    }
}

impl<const P: u32> PackedValue for PackedFp16Neon<P> {
    type Value = Fp16<P>;
    const WIDTH: usize = FP16_WIDTH;

    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        let vals: [Fp16<P>; 8] = std::array::from_fn(f);
        Self {
            vals: vals.map(|v| v.to_limbs()),
        }
    }

    fn extract(&self, lane: usize) -> Self::Value {
        Fp16::from_canonical_u16(self.vals[lane])
    }
}

impl<const P: u32> fmt::Debug for PackedFp16Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.vals.iter()).finish()
    }
}

impl<const P: u32> Default for PackedFp16Neon<P> {
    fn default() -> Self {
        Self { vals: [0; 8] }
    }
}

impl<const P: u32> PartialEq for PackedFp16Neon<P> {
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u32> Eq for PackedFp16Neon<P> {}

impl<const P: u32> Add for PackedFp16Neon<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_vec(Self::add_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> Sub for PackedFp16Neon<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vec(Self::sub_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> Mul for PackedFp16Neon<P> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vec(Self::mul_vec(self.to_vec(), rhs.to_vec()))
    }
}

impl<const P: u32> AddAssign for PackedFp16Neon<P> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> SubAssign for PackedFp16Neon<P> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> MulAssign for PackedFp16Neon<P> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> PackedField for PackedFp16Neon<P> {
    type Scalar = Fp16<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self {
            vals: [value.to_limbs(); 8],
        }
    }

    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn ring_subfield_fp8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        let a_lo = std::array::from_fn(|i| Self::widen_lo(a[i].to_vec()));
        let a_hi = std::array::from_fn(|i| Self::widen_hi(a[i].to_vec()));
        let b_lo = std::array::from_fn(|i| Self::widen_lo(b[i].to_vec()));
        let b_hi = std::array::from_fn(|i| Self::widen_hi(b[i].to_vec()));
        let out_lo = Self::ring_subfield_fp8_mul_u32(a_lo, b_lo);
        let out_hi = Self::ring_subfield_fp8_mul_u32(a_hi, b_hi);
        std::array::from_fn(|i| Self::from_vec(Self::narrow_u32(out_lo[i], out_hi[i])))
    }

    #[inline(always)]
    fn ring_subfield_fp8_square(a: [Self; 8]) -> [Self; 8] {
        let a_lo = std::array::from_fn(|i| Self::widen_lo(a[i].to_vec()));
        let a_hi = std::array::from_fn(|i| Self::widen_hi(a[i].to_vec()));
        let out_lo = Self::ring_subfield_fp8_square_u32(a_lo);
        let out_hi = Self::ring_subfield_fp8_square_u32(a_hi);
        std::array::from_fn(|i| Self::from_vec(Self::narrow_u32(out_lo[i], out_hi[i])))
    }
}
