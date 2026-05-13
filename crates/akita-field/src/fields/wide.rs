//! Wide unreduced field accumulators for carry-free signed addition.
//!
//! Each type splits a canonical field element into 16-bit limbs stored in
//! `i32` slots.  Addition and negation are element-wise i32 ops — no carry
//! propagation, no modular reduction.  Reduction back to canonical form
//! happens once after accumulation via [`reduce`](Fp128x8i32::reduce).
//!
//! The i32 overflow budget is `i32::MAX / u16::MAX ≈ 32,769` signed
//! additions before any limb can overflow.

use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use crate::{AdditiveGroup, CanonicalField, FieldCore};

use super::fp128::Fp128;
use super::fp32::Fp32;
use super::fp64::Fp64;

/// Wide unreduced accumulator for `Fp32`: 2 × i32 limbs (16-bit data each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp32x2i32(pub [i32; 2]);

impl Fp32x2i32 {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 2]);

    /// Returns the zero accumulator.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }
}

impl<const P: u32> From<Fp32<P>> for Fp32x2i32 {
    #[inline]
    fn from(x: Fp32<P>) -> Self {
        let v = x.0;
        Self([(v & 0xFFFF) as i32, (v >> 16) as i32])
    }
}

impl Fp32x2i32 {
    /// Multiply every limb by a small signed scalar.
    ///
    /// Safe when `|small| * max_limb_magnitude` fits in i32. After `From`,
    /// limbs are in `[0, 0xFFFF]`, so `|small| ≤ 32_767` is safe for a single
    /// product.  For accumulation of `k` scaled values, require
    /// `k * |small| * 0xFFFF < i32::MAX`, i.e. roughly `k * |small| < 32_768`.
    #[inline]
    pub fn scale_i32(self, small: i32) -> Self {
        Self([self.0[0] * small, self.0[1] * small])
    }

    /// Reduce back to canonical `Fp32<P>`.
    ///
    /// Carry-propagates the i32 limbs into a signed value, normalizes to
    /// `[0, p)`, and returns the canonical field element.
    #[inline]
    pub fn reduce<const P: u32>(self) -> Fp32<P> {
        let [l0, l1] = self.0;
        // Carry-propagate: value = l0 + l1 * 2^16
        let wide = l0 as i64 + (l1 as i64) * (1i64 << 16);
        // Normalize to [0, p)
        let p = P as i64;
        let normalized = ((wide % p) + p) % p;
        Fp32::from_canonical_u32(normalized as u32)
    }
}

impl Add for Fp32x2i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl AddAssign for Fp32x2i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }
}

impl Sub for Fp32x2i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl SubAssign for Fp32x2i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
    }
}

impl Neg for Fp32x2i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

/// Wide unreduced accumulator for `Fp64`: 4 × i32 limbs (16-bit data each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp64x4i32(pub [i32; 4]);

impl Fp64x4i32 {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 4]);

    /// Returns the zero accumulator.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }
}

impl<const P: u64> From<Fp64<P>> for Fp64x4i32 {
    #[inline]
    fn from(x: Fp64<P>) -> Self {
        let v = x.0;
        Self([
            (v & 0xFFFF) as i32,
            ((v >> 16) & 0xFFFF) as i32,
            ((v >> 32) & 0xFFFF) as i32,
            ((v >> 48) & 0xFFFF) as i32,
        ])
    }
}

impl Fp64x4i32 {
    /// Multiply every limb by a small signed scalar. See [`Fp32x2i32::scale_i32`].
    #[inline]
    pub fn scale_i32(self, small: i32) -> Self {
        Self([
            self.0[0] * small,
            self.0[1] * small,
            self.0[2] * small,
            self.0[3] * small,
        ])
    }

    /// Reduce back to canonical `Fp64<P>`.
    #[inline]
    pub fn reduce<const P: u64>(self) -> Fp64<P> {
        let [l0, l1, l2, l3] = self.0;
        // Carry-propagate: value = l0 + l1*2^16 + l2*2^32 + l3*2^48
        let wide = l0 as i128
            + (l1 as i128) * (1i128 << 16)
            + (l2 as i128) * (1i128 << 32)
            + (l3 as i128) * (1i128 << 48);
        let p = P as i128;
        let normalized = ((wide % p) + p) % p;
        Fp64::<P>::from_canonical_u64(normalized as u64)
    }
}

#[cfg(target_arch = "aarch64")]
impl Add for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = vld1q_s32(self.0.as_ptr());
            let b = vld1q_s32(rhs.0.as_ptr());
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), vaddq_s32(a, b));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl AddAssign for Fp64x4i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Sub for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = vld1q_s32(self.0.as_ptr());
            let b = vld1q_s32(rhs.0.as_ptr());
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), vsubq_s32(a, b));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl SubAssign for Fp64x4i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Neg for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = vld1q_s32(self.0.as_ptr());
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), vnegq_s32(a));
            Self(out)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Add for Fp64x4i32 {
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

#[cfg(not(target_arch = "aarch64"))]
impl AddAssign for Fp64x4i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Sub for Fp64x4i32 {
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

#[cfg(not(target_arch = "aarch64"))]
impl SubAssign for Fp64x4i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
        self.0[3] -= rhs.0[3];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Neg for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

/// Wide unreduced accumulator for `Fp128`: 8 × i32 limbs (16-bit data each).
///
/// On AVX2, one element fits a single 256-bit YMM register.  On NEON, it
/// spans two 128-bit Q registers.  All arithmetic is carry-free element-wise
/// i32 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp128x8i32(pub [i32; 8]);

impl Fp128x8i32 {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 8]);

    /// Returns the zero accumulator.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128x8i32 {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        let lo = x.0[0];
        let hi = x.0[1];
        Self([
            (lo & 0xFFFF) as i32,
            ((lo >> 16) & 0xFFFF) as i32,
            ((lo >> 32) & 0xFFFF) as i32,
            ((lo >> 48) & 0xFFFF) as i32,
            (hi & 0xFFFF) as i32,
            ((hi >> 16) & 0xFFFF) as i32,
            ((hi >> 32) & 0xFFFF) as i32,
            ((hi >> 48) & 0xFFFF) as i32,
        ])
    }
}

impl Fp128x8i32 {
    /// Multiply every limb by a small signed scalar. See [`Fp32x2i32::scale_i32`].
    #[inline]
    pub fn scale_i32(self, small: i32) -> Self {
        Self([
            self.0[0] * small,
            self.0[1] * small,
            self.0[2] * small,
            self.0[3] * small,
            self.0[4] * small,
            self.0[5] * small,
            self.0[6] * small,
            self.0[7] * small,
        ])
    }

    /// Reduce back to canonical `Fp128<P>`.
    ///
    /// Carry-propagates the 8 × i32 limbs into unsigned u64 limbs, then
    /// applies Solinas reduction.
    #[inline]
    pub fn reduce<const P: u128>(self) -> Fp128<P> {
        let limbs = self.0;

        // Carry-propagate from low to high, accumulating into i64 slots.
        // Each i32 limb can be in [-32769*65535, 32769*65535] ≈ ±2^31.
        // After propagation, each 16-bit "digit" is in [0, 65535] and we
        // may have a signed residual in the top that overflows 128 bits.
        let mut carry: i64 = 0;
        let mut digits = [0u16; 8];
        for i in 0..8 {
            let v = limbs[i] as i64 + carry;
            // Arithmetic right-shift to propagate sign correctly
            digits[i] = (v & 0xFFFF) as u16;
            carry = v >> 16;
        }

        // Reassemble into u64 limbs
        let lo = digits[0] as u64
            | (digits[1] as u64) << 16
            | (digits[2] as u64) << 32
            | (digits[3] as u64) << 48;
        let hi = digits[4] as u64
            | (digits[5] as u64) << 16
            | (digits[6] as u64) << 32
            | (digits[7] as u64) << 48;

        // p = 2^128 - c, so 2^128 ≡ c (mod p).
        // value = lo + hi*2^64 + carry*2^128 ≡ lo + hi*2^64 + carry*c (mod p).
        let c = Fp128::<P>::C_LO;
        if carry == 0 {
            Fp128::<P>::from_canonical_u128_reduced(lo as u128 | (hi as u128) << 64)
        } else if carry > 0 {
            Fp128::<P>::solinas_reduce(&[lo, hi, carry as u64])
        } else {
            // carry < 0: value = base - |carry|*c.
            let neg_carry = (-carry) as u64;
            let sub = neg_carry as u128 * c as u128;
            let base = lo as u128 | (hi as u128) << 64;
            if base >= sub {
                Fp128::<P>::from_canonical_u128_reduced(base - sub)
            } else {
                let diff = sub - base;
                Fp128::<P>::from_canonical_u128_reduced(P - diff)
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Add for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s32(self.0.as_ptr());
            let a1 = vld1q_s32(self.0.as_ptr().add(4));
            let b0 = vld1q_s32(rhs.0.as_ptr());
            let b1 = vld1q_s32(rhs.0.as_ptr().add(4));
            let mut out = [0i32; 8];
            vst1q_s32(out.as_mut_ptr(), vaddq_s32(a0, b0));
            vst1q_s32(out.as_mut_ptr().add(4), vaddq_s32(a1, b1));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl AddAssign for Fp128x8i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Sub for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s32(self.0.as_ptr());
            let a1 = vld1q_s32(self.0.as_ptr().add(4));
            let b0 = vld1q_s32(rhs.0.as_ptr());
            let b1 = vld1q_s32(rhs.0.as_ptr().add(4));
            let mut out = [0i32; 8];
            vst1q_s32(out.as_mut_ptr(), vsubq_s32(a0, b0));
            vst1q_s32(out.as_mut_ptr().add(4), vsubq_s32(a1, b1));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl SubAssign for Fp128x8i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Neg for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s32(self.0.as_ptr());
            let a1 = vld1q_s32(self.0.as_ptr().add(4));
            let mut out = [0i32; 8];
            vst1q_s32(out.as_mut_ptr(), vnegq_s32(a0));
            vst1q_s32(out.as_mut_ptr().add(4), vnegq_s32(a1));
            Self(out)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Add for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6],
            self.0[7] + rhs.0[7],
        ])
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl AddAssign for Fp128x8i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
        self.0[4] += rhs.0[4];
        self.0[5] += rhs.0[5];
        self.0[6] += rhs.0[6];
        self.0[7] += rhs.0[7];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Sub for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
            self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6],
            self.0[7] - rhs.0[7],
        ])
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl SubAssign for Fp128x8i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
        self.0[3] -= rhs.0[3];
        self.0[4] -= rhs.0[4];
        self.0[5] -= rhs.0[5];
        self.0[6] -= rhs.0[6];
        self.0[7] -= rhs.0[7];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Neg for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            -self.0[0], -self.0[1], -self.0[2], -self.0[3], -self.0[4], -self.0[5], -self.0[6],
            -self.0[7],
        ])
    }
}

/// Accumulator for `Fp32 × u64` and `Fp32 × Fp32` products.
///
/// Products are split into two 64-bit limbs stored as u128 slots. The second
/// limb is zero for `Fp32 × Fp32` products.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp32ProductAccum(pub [u128; 2]);

impl Fp32ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 2]);

    /// Reduce accumulated products to a canonical `Fp32<P>`.
    #[inline]
    pub fn reduce<const P: u32>(self) -> Fp32<P> {
        let [s0, s1] = self.0;
        let a = Fp32::<P>::from_canonical_u128_reduced(s0);
        let b = Fp32::<P>::from_canonical_u128_reduced(s1);
        let shift = Fp32::<P>::from_canonical_u128_reduced(1u128 << 64);
        a + b * shift
    }
}

impl<const P: u32> From<Fp32<P>> for Fp32ProductAccum {
    #[inline]
    fn from(x: Fp32<P>) -> Self {
        Self([x.to_limbs() as u128, 0])
    }
}

impl Add for Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}
impl AddAssign for Fp32ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }
}
impl Sub for Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
        ])
    }
}
impl SubAssign for Fp32ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
    }
}
impl Neg for Fp32ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([self.0[0].wrapping_neg(), self.0[1].wrapping_neg()])
    }
}

/// Accumulator for `Fp64 × u64` products (also used for `Fp64 × Fp64`).
///
/// Each product is ≤ 128 bits, split into two u64 halves stored as u128 slots.
/// Headroom: 2^64 additions per slot before overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp64ProductAccum(pub [u128; 2]);

impl Fp64ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 2]);

    /// Reduce accumulated products to a canonical `Fp64<P>`.
    #[inline]
    pub fn reduce<const P: u64>(self) -> Fp64<P> {
        let [s0, s1] = self.0;
        // s0 = Σ lo_i, s1 = Σ hi_i; value = s0 + s1 * 2^64
        let a = Fp64::<P>::solinas_reduce(s0);
        let b = Fp64::<P>::solinas_reduce(s1);
        let shift = Fp64::<P>::solinas_reduce(1u128 << 64);
        let b_shifted = Fp64::<P>::solinas_reduce(b.mul_wide_u64(shift.to_limbs()));
        a + b_shifted
    }
}

impl<const P: u64> From<Fp64<P>> for Fp64ProductAccum {
    #[inline]
    fn from(x: Fp64<P>) -> Self {
        Self([x.to_limbs() as u128, 0])
    }
}

impl Add for Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}
impl AddAssign for Fp64ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }
}
impl Sub for Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
        ])
    }
}
impl SubAssign for Fp64ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
    }
}
impl Neg for Fp64ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([self.0[0].wrapping_neg(), self.0[1].wrapping_neg()])
    }
}

/// Accumulator for `Fp128 × u64` products.
///
/// Each `mul_wide_u64` produces 3 u64 limbs; stored as `[u128; 3]`.
/// Headroom: 2^64 additions per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128MulU64Accum(pub [u128; 3]);

impl Fp128MulU64Accum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 3]);

    /// Reduce to canonical `Fp128<P>`.
    #[inline]
    pub fn reduce<const P: u128>(self) -> Fp128<P> {
        let [s0, s1, s2] = self.0;
        let c0 = s0 >> 64;
        let r0 = s0 as u64;
        let t1 = s1 + c0;
        let r1 = t1 as u64;
        let c1 = t1 >> 64;
        let t2 = s2 + c1;
        let r2 = t2 as u64;
        let r3 = (t2 >> 64) as u64;
        Fp128::<P>::solinas_reduce(&[r0, r1, r2, r3])
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128MulU64Accum {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        let [lo, hi] = x.to_limbs();
        Self([lo as u128, hi as u128, 0])
    }
}

impl Add for Fp128MulU64Accum {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}
impl AddAssign for Fp128MulU64Accum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}
impl Sub for Fp128MulU64Accum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
        ])
    }
}
impl SubAssign for Fp128MulU64Accum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
    }
}
impl Neg for Fp128MulU64Accum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
        ])
    }
}

/// Accumulator for `Fp128 × Fp128` products.
///
/// Each `mul_wide` produces 4 u64 limbs; stored as `[u128; 4]`.
/// Headroom: 2^64 additions per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128ProductAccum(pub [u128; 4]);

impl Fp128ProductAccum {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 4]);

    /// Reduce to canonical `Fp128<P>`.
    #[inline]
    pub fn reduce<const P: u128>(self) -> Fp128<P> {
        let [s0, s1, s2, s3] = self.0;
        let c0 = s0 >> 64;
        let r0 = s0 as u64;
        let t1 = s1 + c0;
        let r1 = t1 as u64;
        let c1 = t1 >> 64;
        let t2 = s2 + c1;
        let r2 = t2 as u64;
        let c2 = t2 >> 64;
        let t3 = s3 + c2;
        let r3 = t3 as u64;
        let r4 = (t3 >> 64) as u64;
        Fp128::<P>::solinas_reduce(&[r0, r1, r2, r3, r4])
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128ProductAccum {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        let [lo, hi] = x.to_limbs();
        Self([lo as u128, hi as u128, 0, 0])
    }
}

impl Add for Fp128ProductAccum {
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
impl AddAssign for Fp128ProductAccum {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
    }
}
impl Sub for Fp128ProductAccum {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
        ])
    }
}
impl SubAssign for Fp128ProductAccum {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] = self.0[0].wrapping_sub(rhs.0[0]);
        self.0[1] = self.0[1].wrapping_sub(rhs.0[1]);
        self.0[2] = self.0[2].wrapping_sub(rhs.0[2]);
        self.0[3] = self.0[3].wrapping_sub(rhs.0[3]);
    }
}
impl Neg for Fp128ProductAccum {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

/// Pair accumulator for extension fields.
///
/// Wraps two base-field accumulators `(c0, c1)` component-wise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumPair<A>(pub A, pub A);

impl<A: AdditiveGroup> Add for AccumPair<A> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl<A: AdditiveGroup> AddAssign for AccumPair<A> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
impl<A: AdditiveGroup> Sub for AccumPair<A> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}
impl<A: AdditiveGroup> SubAssign for AccumPair<A> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}
impl<A: AdditiveGroup> Neg for AccumPair<A> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
}

/// Reduce a wide unreduced accumulator back to a canonical field element.
pub trait ReduceTo<F> {
    /// Carry-propagate and reduce to a canonical field element.
    fn reduce(self) -> F;
}

impl<const P: u32> ReduceTo<Fp32<P>> for Fp32x2i32 {
    #[inline]
    fn reduce(self) -> Fp32<P> {
        Fp32x2i32::reduce::<P>(self)
    }
}

impl<const P: u64> ReduceTo<Fp64<P>> for Fp64x4i32 {
    #[inline]
    fn reduce(self) -> Fp64<P> {
        Fp64x4i32::reduce::<P>(self)
    }
}

impl<const P: u128> ReduceTo<Fp128<P>> for Fp128x8i32 {
    #[inline]
    fn reduce(self) -> Fp128<P> {
        Fp128x8i32::reduce::<P>(self)
    }
}

/// Multi-level unreduced multiplication hierarchy.
///
/// Provides `field × u64` and `field × field` widening multiplies that return
/// accumulator types supporting carry-free addition. Reduction back to a
/// canonical field element happens once after accumulation.
pub trait HasUnreducedOps: FieldCore {
    /// Accumulator for `self × u64` products (narrower than full product).
    type MulU64Accum: AdditiveGroup;
    /// Accumulator for `self × self` products.
    type ProductAccum: AdditiveGroup;

    /// Widening `self × small` with no reduction.
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum;
    /// Widening `self × other` with no reduction.
    fn mul_to_product_accum(self, other: Self) -> Self::ProductAccum;

    /// Reduce a narrow-mul accumulator to a canonical field element.
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self;
    /// Reduce a full-product accumulator to a canonical field element.
    fn reduce_product_accum(accum: Self::ProductAccum) -> Self;
}

impl<const P: u64> HasUnreducedOps for Fp64<P> {
    type MulU64Accum = Fp64ProductAccum;
    type ProductAccum = Fp64ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp64ProductAccum {
        let wide = self.mul_wide_u64(small);
        Fp64ProductAccum([wide & u64::MAX as u128, wide >> 64])
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp64ProductAccum {
        let wide = self.mul_wide(other);
        Fp64ProductAccum([wide & u64::MAX as u128, wide >> 64])
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Fp64ProductAccum) -> Self {
        accum.reduce::<P>()
    }

    #[inline]
    fn reduce_product_accum(accum: Fp64ProductAccum) -> Self {
        accum.reduce::<P>()
    }
}

impl<const P: u32> HasUnreducedOps for Fp32<P> {
    type MulU64Accum = Fp32ProductAccum;
    type ProductAccum = Fp32ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp32ProductAccum {
        let wide = (self.to_limbs() as u128) * (small as u128);
        Fp32ProductAccum([wide & u64::MAX as u128, wide >> 64])
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp32ProductAccum {
        Fp32ProductAccum([self.mul_wide(other) as u128, 0])
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Fp32ProductAccum) -> Self {
        accum.reduce::<P>()
    }

    #[inline]
    fn reduce_product_accum(accum: Fp32ProductAccum) -> Self {
        accum.reduce::<P>()
    }
}

impl<const P: u128> HasUnreducedOps for Fp128<P> {
    type MulU64Accum = Fp128MulU64Accum;
    type ProductAccum = Fp128ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp128MulU64Accum {
        let [lo, mid, hi] = self.mul_wide_u64(small);
        Fp128MulU64Accum([lo as u128, mid as u128, hi as u128])
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp128ProductAccum {
        let [r0, r1, r2, r3] = self.mul_wide(other);
        Fp128ProductAccum([r0 as u128, r1 as u128, r2 as u128, r3 as u128])
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Fp128MulU64Accum) -> Self {
        accum.reduce::<P>()
    }

    #[inline]
    fn reduce_product_accum(accum: Fp128ProductAccum) -> Self {
        accum.reduce::<P>()
    }
}

/// Element-wise scaling of a wide accumulator by a small signed integer.
pub trait ScaleI32 {
    /// Scale each element by `small`.
    fn scale_i32(self, small: i32) -> Self;
}

impl ScaleI32 for Fp32x2i32 {
    #[inline]
    fn scale_i32(self, small: i32) -> Self {
        self.scale_i32(small)
    }
}

impl ScaleI32 for Fp64x4i32 {
    #[inline]
    fn scale_i32(self, small: i32) -> Self {
        self.scale_i32(small)
    }
}

impl ScaleI32 for Fp128x8i32 {
    #[inline]
    fn scale_i32(self, small: i32) -> Self {
        self.scale_i32(small)
    }
}

/// Associates a field type with its wide unreduced accumulator.
pub trait HasWide: FieldCore {
    /// The wide accumulator type.
    type Wide: AdditiveGroup + From<Self> + ReduceTo<Self> + ScaleI32;

    /// Convert `self` to wide form and scale every limb by `small`.
    ///
    /// Equivalent to `Self::Wide::from(self).scale_i32(small)` but avoids
    /// the trait-method ambiguity at call sites.
    #[inline]
    fn mul_small_to_wide(self, small: i32) -> Self::Wide {
        Self::Wide::from(self).scale_i32(small)
    }
}

impl<const P: u32> HasWide for Fp32<P> {
    type Wide = Fp32x2i32;
}

impl<const P: u64> HasWide for Fp64<P> {
    type Wide = Fp64x4i32;
}

impl<const P: u128> HasWide for Fp128<P> {
    type Wide = Fp128x8i32;
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use crate::fields::{Prime128Offset275, Prime24Offset3, Prime40Offset195};
    use crate::RandomSampling;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_core::RngCore;

    type F128 = Prime128Offset275;
    type F32 = Prime24Offset3;
    type F64 = Prime40Offset195;

    const P128: u128 = 0xfffffffffffffffffffffffffffffeed;
    const P32: u32 = (1 << 24) - 3;
    const P64: u64 = (1 << 40) - 195;

    #[test]
    fn fp128_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xdead_1234);
        for _ in 0..1000 {
            let a: F128 = RandomSampling::random(&mut rng);
            let wide = Fp128x8i32::from(a);
            let back = wide.reduce::<P128>();
            assert_eq!(a, back, "roundtrip failed for {a:?}");
        }
    }

    #[test]
    fn fp128_accumulate_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0xbeef_cafe_4321);
        let n = 1000;
        let vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let scalar_sum = vals.iter().fold(F128::zero(), |acc, &x| acc + x);

        let wide_sum = vals
            .iter()
            .fold(Fp128x8i32::zero(), |acc, &x| acc + Fp128x8i32::from(x));
        let reduced = wide_sum.reduce::<P128>();

        assert_eq!(scalar_sum, reduced);
    }

    #[test]
    fn fp128_add_sub_neg_match_scalar() {
        let mut rng = StdRng::seed_from_u64(0x1122_3344_5566);
        for _ in 0..500 {
            let a: F128 = RandomSampling::random(&mut rng);
            let b: F128 = RandomSampling::random(&mut rng);

            let wa = Fp128x8i32::from(a);
            let wb = Fp128x8i32::from(b);

            assert_eq!((wa + wb).reduce::<P128>(), a + b);
            assert_eq!((wa - wb).reduce::<P128>(), a - b);
            assert_eq!((-wa).reduce::<P128>(), -a);
        }
    }

    #[test]
    fn fp128_mixed_add_sub_stress() {
        let mut rng = StdRng::seed_from_u64(0xaaaa_bbbb_cccc);
        let n = 500;
        let vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let mut scalar = F128::zero();
        let mut wide = Fp128x8i32::zero();
        for (i, &v) in vals.iter().enumerate() {
            let wv = Fp128x8i32::from(v);
            if i % 3 == 0 {
                scalar -= v;
                wide -= wv;
            } else {
                scalar += v;
                wide += wv;
            }
        }
        assert_eq!(wide.reduce::<P128>(), scalar);
    }

    #[test]
    fn fp32_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0x3232_3232);
        for _ in 0..1000 {
            let a: F32 = RandomSampling::random(&mut rng);
            let wide = Fp32x2i32::from(a);
            let back = wide.reduce::<P32>();
            assert_eq!(a, back);
        }
    }

    #[test]
    fn fp32_accumulate_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x3232_abcd);
        let n = 1000;
        let vals: Vec<F32> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let scalar_sum = vals.iter().fold(F32::zero(), |acc, &x| acc + x);
        let wide_sum = vals
            .iter()
            .fold(Fp32x2i32::zero(), |acc, &x| acc + Fp32x2i32::from(x));
        assert_eq!(wide_sum.reduce::<P32>(), scalar_sum);
    }

    #[test]
    fn fp64_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0x6464_6464);
        for _ in 0..1000 {
            let a: F64 = RandomSampling::random(&mut rng);
            let wide = Fp64x4i32::from(a);
            let back = wide.reduce::<P64>();
            assert_eq!(a, back);
        }
    }

    #[test]
    fn fp64_accumulate_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x6464_beef);
        let n = 1000;
        let vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let scalar_sum = vals.iter().fold(F64::zero(), |acc, &x| acc + x);
        let wide_sum = vals
            .iter()
            .fold(Fp64x4i32::zero(), |acc, &x| acc + Fp64x4i32::from(x));
        assert_eq!(wide_sum.reduce::<P64>(), scalar_sum);
    }

    #[test]
    fn fp64_product_accum_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x6464_4444);
        let n = 500;
        let a_vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
        let b_vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let scalar_sum: F64 = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(F64::zero(), |acc, (&a, &b)| acc + a * b);

        let accum_sum = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(Fp64ProductAccum::ZERO, |acc, (&a, &b)| {
                acc + a.mul_to_product_accum(b)
            });
        assert_eq!(F64::reduce_product_accum(accum_sum), scalar_sum);
    }

    #[test]
    fn fp64_mul_u64_accum_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x6464_5555);
        let n = 500;
        let a_vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
        let b_vals: Vec<u64> = (0..n).map(|_| rng.next_u64() >> 32).collect();

        let scalar_sum: F64 = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(F64::zero(), |acc, (&a, &b)| acc + a * F64::from_u64(b));

        let accum_sum = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(Fp64ProductAccum::ZERO, |acc, (&a, &b)| {
                acc + a.mul_u64_unreduced(b)
            });
        assert_eq!(F64::reduce_mul_u64_accum(accum_sum), scalar_sum);
    }

    #[test]
    fn fp128_product_accum_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x0128_6666);
        let n = 500;
        let a_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
        let b_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let scalar_sum: F128 = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(F128::zero(), |acc, (&a, &b)| acc + a * b);

        let accum_sum = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(Fp128ProductAccum::ZERO, |acc, (&a, &b)| {
                acc + a.mul_to_product_accum(b)
            });
        assert_eq!(F128::reduce_product_accum(accum_sum), scalar_sum);
    }

    #[test]
    fn fp128_mul_u64_accum_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x0128_7777);
        let n = 500;
        let a_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
        let b_vals: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();

        let scalar_sum: F128 = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(F128::zero(), |acc, (&a, &b)| acc + a * F128::from_u64(b));

        let accum_sum = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(Fp128MulU64Accum::ZERO, |acc, (&a, &b)| {
                acc + a.mul_u64_unreduced(b)
            });
        assert_eq!(F128::reduce_mul_u64_accum(accum_sum), scalar_sum);
    }

    #[test]
    fn fp128_product_accum_sub_neg() {
        let mut rng = StdRng::seed_from_u64(0x0128_8888);
        let n = 500;
        let a_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
        let b_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

        let mut scalar_sum = F128::zero();
        let mut accum_pos = Fp128ProductAccum::ZERO;
        let mut accum_neg = Fp128ProductAccum::ZERO;
        for (i, (&a, &b)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
            let prod = a.mul_to_product_accum(b);
            if i % 2 == 0 {
                scalar_sum += a * b;
                accum_pos += prod;
            } else {
                scalar_sum -= a * b;
                accum_neg += prod;
            }
        }
        let result = F128::reduce_product_accum(accum_pos) - F128::reduce_product_accum(accum_neg);
        assert_eq!(result, scalar_sum);
    }
}
