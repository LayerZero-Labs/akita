//! CRT helpers: Garner reconstruction and limb-based modular arithmetic.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

use super::prime::{NttPrime, PrimeWidth};
use akita_error::AkitaError;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmallNat {
    limbs: Vec<u32>,
}

impl SmallNat {
    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    fn mul_u128(&mut self, rhs: u128) {
        if rhs == 0 {
            self.limbs = vec![0];
            return;
        }
        let mut rhs_limbs = Vec::new();
        let mut value = rhs;
        while value != 0 {
            rhs_limbs.push(value as u32);
            value >>= 32;
        }
        let mut out = vec![0u32; self.limbs.len() + rhs_limbs.len()];
        for (i, &lhs) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &rhs) in rhs_limbs.iter().enumerate() {
                let index = i + j;
                let accum = u128::from(out[index]) + u128::from(lhs) * u128::from(rhs) + carry;
                out[index] = accum as u32;
                carry = accum >> 32;
            }
            let mut index = i + rhs_limbs.len();
            while carry != 0 {
                if index == out.len() {
                    out.push(0);
                }
                let accum = u128::from(out[index]) + carry;
                out[index] = accum as u32;
                carry = accum >> 32;
                index += 1;
            }
        }
        while out.len() > 1 && out.last() == Some(&0) {
            out.pop();
        }
        self.limbs = out;
    }
}

impl Ord for SmallNat {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => self.limbs.iter().rev().cmp(other.limbs.iter().rev()),
            ordering => ordering,
        }
    }
}

impl PartialOrd for SmallNat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact product capacity of a CRT residue profile.
///
/// The product is independent of the residue representation and execution
/// kernels. It can therefore compare homogeneous i16/i32 profiles, mixed
/// profiles, and wider SIMD-specific profiles through one exact bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrtCapacity {
    product: SmallNat,
}

impl CrtCapacity {
    /// Build a capacity from canonical prime moduli.
    pub fn from_prime_moduli(primes: impl IntoIterator<Item = u128>) -> Self {
        let mut product = SmallNat::one();
        for prime in primes {
            product.mul_u128(prime);
        }
        Self { product }
    }

    /// Extend this capacity with one additional canonical prime modulus.
    #[must_use]
    pub fn with_prime_modulus(mut self, prime: u128) -> Self {
        self.product.mul_u128(prime);
        self
    }

    /// Whether this CRT product can reconstruct the requested accumulation.
    ///
    /// The strict exactness condition is
    /// `2 * width * D * floor(q / 2) * rhs_abs_bound < product(primes)`.
    pub fn supports<F: crate::Field + crate::CanonicalEncoding, const D: usize>(
        &self,
        width: usize,
        rhs_abs_bound: u64,
    ) -> bool {
        self.supports_modulus(
            width,
            D,
            (-F::one())
                .to_u128_checked()
                .expect("Akita field element must fit in u128")
                + 1,
            rhs_abs_bound,
        )
    }

    /// Whether this CRT product can reconstruct an accumulation for an
    /// explicitly identified field modulus and runtime ring dimension.
    pub fn supports_modulus(
        &self,
        width: usize,
        ring_dimension: usize,
        modulus: u128,
        rhs_abs_bound: u64,
    ) -> bool {
        let mut required = SmallNat::one();
        required.mul_u128(2);
        required.mul_u128(width as u128);
        required.mul_u128(ring_dimension as u128);
        required.mul_u128(modulus / 2);
        required.mul_u128(u128::from(rhs_abs_bound));
        required < self.product
    }

    /// Conservative maximum matrix width supported at one coefficient bound.
    pub fn max_safe_width<F: crate::Field + crate::CanonicalEncoding, const D: usize>(
        &self,
        rhs_abs_bound: u64,
    ) -> Option<usize> {
        self.max_safe_width_for_modulus(
            D,
            (-F::one())
                .to_u128_checked()
                .expect("Akita field element must fit in u128")
                + 1,
            rhs_abs_bound,
        )
    }

    /// Conservative maximum matrix width for an explicitly identified field
    /// modulus and runtime ring dimension.
    pub fn max_safe_width_for_modulus(
        &self,
        ring_dimension: usize,
        modulus: u128,
        rhs_abs_bound: u64,
    ) -> Option<usize> {
        if rhs_abs_bound == 0 {
            return Some(usize::MAX);
        }
        if modulus <= 1
            || ring_dimension == 0
            || !self.supports_modulus(1, ring_dimension, modulus, rhs_abs_bound)
        {
            return None;
        }
        let mut low = 1usize;
        let mut high = 2usize;
        while self.supports_modulus(high, ring_dimension, modulus, rhs_abs_bound) {
            low = high;
            let Some(next) = high.checked_mul(2) else {
                if self.supports_modulus(usize::MAX, ring_dimension, modulus, rhs_abs_bound) {
                    return Some(usize::MAX);
                }
                high = usize::MAX;
                break;
            };
            high = next;
        }
        while low + 1 < high {
            let mid = low + (high - low) / 2;
            if self.supports_modulus(mid, ring_dimension, modulus, rhs_abs_bound) {
                low = mid;
            } else {
                high = mid;
            }
        }
        Some(low)
    }
}

/// Limb radix bit-width (`2^14`).
pub const RADIX_BITS: u32 = 14;
const RADIX: i32 = 1 << RADIX_BITS;
const RADIX_MASK: i32 = RADIX - 1;

/// Precomputed Garner inverse table for CRT reconstruction.
///
/// `gamma[i][j]` = `p_j^{-1} mod p_i` for `j < i`. Upper triangle and
/// diagonal entries are zero (unused).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GarnerData<const K: usize> {
    /// `gamma[i][j]` = `p_j^{-1} mod p_i` for `j < i`.
    pub gamma: [[u64; K]; K],
}

impl<const K: usize> GarnerData<K> {
    /// Compute Garner constants from a set of NTT primes.
    pub fn compute<W: PrimeWidth>(primes: &[NttPrime<W>; K]) -> Self {
        Self::try_from_moduli(primes.map(|prime| prime.p.to_i64() as u64))
            .expect("CRT primes must be pairwise coprime")
    }

    pub(crate) fn try_from_moduli(moduli: [u64; K]) -> Result<Self, AkitaError> {
        let mut gamma = [[0; K]; K];
        for i in 1..K {
            let pi = moduli[i];
            #[allow(clippy::needless_range_loop)]
            for j in 0..i {
                gamma[i][j] = modular_inverse(moduli[j] % pi, pi)?;
            }
        }
        Ok(Self { gamma })
    }

    pub(crate) fn centered_mixed_radix(&self, residues: [i128; K], moduli: [u64; K]) -> [i128; K] {
        let mut digits = [0; K];
        if K == 0 {
            return digits;
        }
        let first_modulus = i128::from(moduli[0]);
        digits[0] = center_mod(residues[0], first_modulus);
        for index in 1..K {
            let modulus = i128::from(moduli[index]);
            let mut digit = residues[index].rem_euclid(modulus);
            for (prior, prior_digit) in digits.iter().enumerate().take(index) {
                digit = (digit - prior_digit).rem_euclid(modulus);
                digit = (digit * i128::from(self.gamma[index][prior])).rem_euclid(modulus);
            }
            digits[index] = center_mod(digit, modulus);
        }
        digits
    }
}

fn center_mod(value: i128, modulus: i128) -> i128 {
    let value = value.rem_euclid(modulus);
    if value > modulus / 2 {
        value - modulus
    } else {
        value
    }
}

pub(crate) fn modular_inverse(value: u64, modulus: u64) -> Result<u64, AkitaError> {
    let (mut old_remainder, mut remainder) = (i128::from(modulus), i128::from(value));
    let (mut old_coefficient, mut coefficient) = (0i128, 1i128);
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
        (old_coefficient, coefficient) = (coefficient, old_coefficient - quotient * coefficient);
    }
    if old_remainder != 1 {
        return Err(AkitaError::InvalidSetup(
            "CRT primes are not pairwise coprime".into(),
        ));
    }
    Ok(old_coefficient.rem_euclid(i128::from(modulus)) as u64)
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

#[cfg(test)]
mod capacity_tests {
    use super::CrtCapacity;
    use crate::ntt::tables::{I16_TAIL_PRIME, Q32_PRIMES};
    use crate::PrimeWidth;
    use jolt_field::Prime32Offset99;

    #[test]
    fn q32_capacity_matches_existing_exact_widths() {
        let base =
            CrtCapacity::from_prime_moduli(Q32_PRIMES.iter().map(|prime| prime.p.to_i64() as u128));
        assert_eq!(
            base.max_safe_width::<Prime32Offset99, 128>(32_768),
            Some(63)
        );
        assert_eq!(
            base.clone()
                .with_prime_modulus(I16_TAIL_PRIME.p as u128)
                .max_safe_width::<Prime32Offset99, 128>(32_768),
            Some(786_406)
        );
    }

    #[test]
    fn mixed_wide_and_small_capacity_is_representation_independent() {
        let mixed =
            CrtCapacity::from_prime_moduli([1_125_899_906_826_241u128, I16_TAIL_PRIME.p as u128]);
        assert_eq!(
            mixed.max_safe_width::<Prime32Offset99, 128>(32_768),
            Some(768)
        );
        assert!(mixed.supports::<Prime32Offset99, 128>(128, 32_768));
    }
}
