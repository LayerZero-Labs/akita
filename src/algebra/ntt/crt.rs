//! CRT helpers: Garner reconstruction and limb-based modular arithmetic.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

use super::prime::{NttPrime, PrimeWidth};

/// Limb radix bit-width (`2^14`).
pub const RADIX_BITS: u32 = 14;
const RADIX: i32 = 1 << RADIX_BITS;
const RADIX_MASK: i32 = RADIX - 1;

/// Precomputed Garner inverse table for CRT reconstruction.
///
/// `gamma[i][j]` = `p_j^{-1} mod p_i` for `j < i`. Upper triangle and
/// diagonal entries are zero (unused).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GarnerData<W: PrimeWidth, const K: usize> {
    /// `gamma[i][j]` = `p_j^{-1} mod p_i` for `j < i`.
    pub gamma: [[W; K]; K],
}

impl<W: PrimeWidth, const K: usize> GarnerData<W, K> {
    /// Compute Garner constants from a set of NTT primes.
    pub fn compute(primes: &[NttPrime<W>; K]) -> Self {
        let mut gamma = [[W::default(); K]; K];
        for i in 1..K {
            let pi = primes[i].p.to_i64();
            #[allow(clippy::needless_range_loop)]
            for j in 0..i {
                let pj = primes[j].p.to_i64();
                let inv = mod_inverse_i64(pj, pi);
                gamma[i][j] = W::from_i64(inv);
            }
        }
        Self { gamma }
    }
}

/// Modular inverse via extended GCD, operating in `i64`.
fn mod_inverse_i64(a: i64, modulus: i64) -> i64 {
    let (mut t, mut new_t) = (0i64, 1i64);
    let (mut r, mut new_r) = (modulus, ((a % modulus) + modulus) % modulus);
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    assert_eq!(r, 1, "modular inverse does not exist");
    ((t % modulus) + modulus) % modulus
}

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
        let mask = borrow as u16;
        let mut result = [0u16; L];
        for (i, r) in result.iter_mut().enumerate() {
            *r = (self.limbs[i] & mask) | (diff[i] & !mask);
        }
        Self { limbs: result }
    }
}

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

impl<const L: usize> fmt::Display for LimbQ<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(val) = u128::try_from(*self) {
            write!(f, "{val}")
        } else {
            write!(f, "LimbQ{:?}", self.limbs)
        }
    }
}
