//! Limb-based helpers for `q = 2^k - c` style moduli.
//!
//! Big-`q` coefficients are stored in radix `2^14` limbs.
//! This module provides a portable scalar representation with the same limb layout.

/// Limb radix bit-width used by Labrador (`2^14`).
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

    /// Build from an unsigned integer.
    #[inline]
    pub fn from_u128(mut x: u128) -> Self {
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

    /// Convert to `u128` when the limb width fits.
    #[inline]
    pub fn to_u128(self) -> Option<u128> {
        if (L as u32) * RADIX_BITS > 128 {
            return None;
        }
        let mut acc = 0u128;
        for i in (0..L).rev() {
            acc <<= RADIX_BITS;
            acc |= self.limbs[i] as u128;
        }
        Some(acc)
    }

    /// Lexicographic comparison over limbs (most-significant first).
    #[inline]
    pub fn less_than(&self, other: &Self) -> bool {
        for i in (0..L).rev() {
            if self.limbs[i] < other.limbs[i] {
                return true;
            }
            if self.limbs[i] > other.limbs[i] {
                return false;
            }
        }
        false
    }

    /// Limb-wise addition with radix carry propagation.
    #[inline]
    pub fn add_limbs(self, rhs: Self) -> Self {
        let mut out = [0u16; L];
        let mut carry = 0i32;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let mut s = self.limbs[i] as i32 + rhs.limbs[i] as i32 + carry;
            if i + 1 < L {
                carry = s >> RADIX_BITS;
                s &= RADIX_MASK;
            }
            *out_limb = s as u16;
        }
        Self { limbs: out }
    }

    /// Limb-wise subtraction with radix borrow propagation.
    #[inline]
    pub fn sub_limbs(self, rhs: Self) -> Self {
        let mut out = [0u16; L];
        let mut borrow = 0i32;
        for (i, out_limb) in out.iter_mut().enumerate() {
            let mut d = self.limbs[i] as i32 - rhs.limbs[i] as i32 + borrow;
            if i + 1 < L {
                if d < 0 {
                    d += RADIX;
                    borrow = -1;
                } else {
                    borrow = 0;
                }
            }
            *out_limb = d as u16;
        }
        Self { limbs: out }
    }

    /// Conditional subtraction: if `self >= modulus`, return `self - modulus`.
    #[inline]
    pub fn csub_mod(self, modulus: Self) -> Self {
        if !self.less_than(&modulus) {
            self.sub_limbs(modulus)
        } else {
            self
        }
    }
}

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
        self.q.to_u128()
    }
}
