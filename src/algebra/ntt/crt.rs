//! Limb-based helpers for `q = 2^k - c` style moduli.
//!
//! Big-`q` coefficients are stored in radix `2^14` limbs.
//! This module provides a portable scalar representation with the same limb layout.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

/// Limb radix bit-width (`2^14`).
pub const RADIX_BITS: u32 = 14;
const RADIX: i32 = 1 << RADIX_BITS;
const RADIX_MASK: i32 = RADIX - 1;

/// Fixed-width radix-`2^14` integer.
///
/// Limbs are little-endian: `limbs[0]` is least significant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimbQ<const L: usize> {
    /// Little-endian limbs.
    pub limbs: [u16; L],
}

impl<const L: usize> Default for LimbQ<L> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const L: usize> LimbQ<L> {
    /// Zero value.
    #[inline]
    pub const fn zero() -> Self {
        Self { limbs: [0; L] }
    }

    /// Construct directly from limbs.
    #[inline]
    pub const fn from_limbs(limbs: [u16; L]) -> Self {
        Self { limbs }
    }

    /// Conditional subtraction: if `self >= modulus`, return `self - modulus` (branchless).
    #[inline]
    pub fn csub_mod(self, modulus: Self) -> Self {
        // Compute self - modulus, tracking the final borrow.
        let mut diff = [0u16; L];
        let mut borrow = 0i32;
        for (i, df) in diff.iter_mut().enumerate() {
            let d = self.limbs[i] as i32 - modulus.limbs[i] as i32 + borrow;
            borrow = d >> 31;
            if i + 1 < L {
                *df = (d - borrow * RADIX) as u16;
            } else {
                *df = d as u16;
            }
        }
        // borrow = -1 if self < modulus (underflowed), 0 otherwise.
        // Branchless select: mask = 0xFFFF if underflow (keep self), 0 if not (keep diff).
        let mask = borrow as u16;
        let mut result = [0u16; L];
        for (i, r) in result.iter_mut().enumerate() {
            *r = (self.limbs[i] & mask) | (diff[i] & !mask);
        }
        Self { limbs: result }
    }
}

// ---- Standard conversions ----

impl<const L: usize> From<u128> for LimbQ<L> {
    fn from(mut x: u128) -> Self {
        let mut out = [0u16; L];
        for (i, limb) in out.iter_mut().enumerate() {
            if i + 1 < L {
                *limb = (x & (RADIX_MASK as u128)) as u16;
                x >>= RADIX_BITS;
            } else {
                *limb = x as u16;
            }
        }
        Self { limbs: out }
    }
}

impl<const L: usize> TryFrom<LimbQ<L>> for u128 {
    type Error = &'static str;

    fn try_from(limb: LimbQ<L>) -> Result<Self, Self::Error> {
        if (L as u32) * RADIX_BITS > 128 {
            return Err("LimbQ too wide for u128");
        }
        let mut acc = 0u128;
        for i in (0..L).rev() {
            acc <<= RADIX_BITS;
            acc |= limb.limbs[i] as u128;
        }
        Ok(acc)
    }
}

// ---- Ordering ----

impl<const L: usize> PartialOrd for LimbQ<L> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const L: usize> Ord for LimbQ<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..L).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

// ---- Arithmetic ----

impl<const L: usize> Add for LimbQ<L> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut out = [0u16; L];
        let mut carry = 0i32;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let s = self.limbs[i] as i32 + rhs.limbs[i] as i32 + carry;
            if i + 1 < L {
                carry = s >> RADIX_BITS;
                *out_limb = (s & RADIX_MASK) as u16;
            } else {
                *out_limb = s as u16;
            }
        }
        Self { limbs: out }
    }
}

impl<const L: usize> Sub for LimbQ<L> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut out = [0u16; L];
        let mut borrow = 0i32;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let d = self.limbs[i] as i32 - rhs.limbs[i] as i32 + borrow;
            if i + 1 < L {
                borrow = d >> 31;
                *out_limb = (d - borrow * RADIX) as u16;
            } else {
                *out_limb = d as u16;
            }
        }
        Self { limbs: out }
    }
}

// ---- Display ----

impl<const L: usize> fmt::Display for LimbQ<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(val) = u128::try_from(*self) {
            write!(f, "{val}")
        } else {
            write!(f, "LimbQ{:?}", self.limbs)
        }
    }
}

// ---- CRT data ----

/// CRT/q constants for a given parameter set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QData<const K: usize, const L: usize> {
    /// The modulus `q` in radix-`2^14` limbs.
    pub q: LimbQ<L>,
    /// `-P mod q` in limbs (where `P = prod p_i`).
    pub pmq: LimbQ<L>,
    /// `P/p_i mod q` in limbs.
    pub xvec: [LimbQ<L>; K],
    /// `k` in `q = 2^k - c`.
    pub logq: u32,
    /// `c` in `q = 2^k - c`.
    pub qoff: u16,
}

impl<const K: usize, const L: usize> QData<K, L> {
    /// `q` as `u128` when representable.
    #[inline]
    pub fn q_u128(self) -> Option<u128> {
        u128::try_from(self.q).ok()
    }
}
