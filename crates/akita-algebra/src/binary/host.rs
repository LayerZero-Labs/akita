//! Small portable binary fields used by the field-switch reference path.

use std::ops::{Add, AddAssign, Mul};

use super::product::portable_clmul;

/// `F_2[x]/(x^128 + x^7 + x^2 + x + 1)` in low-bit polynomial coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BinaryField128([u64; 2]);

impl BinaryField128 {
    /// Additive identity.
    pub const ZERO: Self = Self([0; 2]);
    /// Multiplicative identity.
    pub const ONE: Self = Self([1, 0]);

    /// Construct from the low-degree-first coefficient words.
    pub const fn from_words(words: [u64; 2]) -> Self {
        Self(words)
    }

    /// Return the low-degree-first coefficient words.
    pub const fn to_words(self) -> [u64; 2] {
        self.0
    }

    /// Multiply by the polynomial basis element `x`.
    pub(super) const fn mul_x(self) -> Self {
        let [lo, hi] = self.0;
        let carry = hi >> 63;
        Self([(lo << 1) ^ carry.wrapping_mul(0x87), (hi << 1) | (lo >> 63)])
    }
}

impl Add for BinaryField128 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] ^ rhs.0[0], self.0[1] ^ rhs.0[1]])
    }
}

impl AddAssign for BinaryField128 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for BinaryField128 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let [a0, a1] = self.0;
        let [b0, b1] = rhs.0;
        let d0 = portable_clmul(a0, b0);
        let d1 = portable_clmul(a1, b1);
        let cross = portable_clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
        reduce128([
            d0 as u64,
            (d0 >> 64) as u64 ^ cross as u64,
            d1 as u64 ^ (cross >> 64) as u64,
            (d1 >> 64) as u64,
        ])
    }
}

/// Reduce a degree-at-most-254 polynomial using `x^128 = x^7 + x^2 + x + 1`.
#[inline]
fn reduce128(product: [u64; 4]) -> BinaryField128 {
    let [p0, p1, h0, h1] = product;
    let t0 = h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
    let t1 = h1 ^ (h1 << 1) ^ (h0 >> 63) ^ (h1 << 2) ^ (h0 >> 62) ^ (h1 << 7) ^ (h0 >> 57);
    let overflow = (h1 >> 63) ^ (h1 >> 62) ^ (h1 >> 57);
    let folded = overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7);
    BinaryField128([p0 ^ t0 ^ folded, p1 ^ t1])
}

/// `K[y]/(y^3 + y + 1)`, where `K = F_2[x]/(x^64 + x^4 + x^3 + x + 1)`.
///
/// The three words are the low-to-high coefficients of `1, y, y^2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BinaryField192([u64; 3]);

impl BinaryField192 {
    /// Additive identity.
    pub const ZERO: Self = Self([0; 3]);
    /// Multiplicative identity.
    pub const ONE: Self = Self([1, 0, 0]);

    /// Construct from the low-degree-first extension coefficients.
    pub const fn from_words(words: [u64; 3]) -> Self {
        Self(words)
    }

    /// Return the low-degree-first extension coefficients.
    pub const fn to_words(self) -> [u64; 3] {
        self.0
    }

    /// Multiply each `K` coefficient by its polynomial basis element `x`.
    pub(super) const fn mul_x(self) -> Self {
        Self([k_mul_x(self.0[0]), k_mul_x(self.0[1]), k_mul_x(self.0[2])])
    }

    /// Multiply by the cubic-extension basis element `y`.
    pub(super) const fn mul_y(self) -> Self {
        let [c0, c1, c2] = self.0;
        Self([c2, c0 ^ c2, c1])
    }
}

impl Add for BinaryField192 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] ^ rhs.0[0],
            self.0[1] ^ rhs.0[1],
            self.0[2] ^ rhs.0[2],
        ])
    }
}

impl AddAssign for BinaryField192 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul for BinaryField192 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let [a0, a1, a2] = self.0;
        let [b0, b1, b2] = rhs.0;
        let d0 = portable_clmul(a0, b0);
        let d1 = portable_clmul(a1, b1);
        let d2 = portable_clmul(a2, b2);
        let c01 = portable_clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
        let c02 = portable_clmul(a0 ^ a2, b0 ^ b2) ^ d0 ^ d2;
        let c12 = portable_clmul(a1 ^ a2, b1 ^ b2) ^ d1 ^ d2;

        // Reduce the y^3 and y^4 terms with y^3 = y + 1 and y^4 = y^2 + y.
        Self([
            reduce64(d0 ^ c12),
            reduce64(c01 ^ c12 ^ d2),
            reduce64(d1 ^ c02 ^ d2),
        ])
    }
}

/// Reduce over `K` using `x^64 = x^4 + x^3 + x + 1`.
#[inline]
fn reduce64(product: u128) -> u64 {
    let low = product as u64;
    let high = (product >> 64) as u64;
    let first = u128::from(high)
        ^ (u128::from(high) << 1)
        ^ (u128::from(high) << 3)
        ^ (u128::from(high) << 4);
    let overflow = (first >> 64) as u64;
    let second = overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
    low ^ first as u64 ^ second
}

#[inline]
const fn k_mul_x(value: u64) -> u64 {
    (value << 1) ^ (value >> 63).wrapping_mul(0x1b)
}

#[cfg(test)]
mod tests;
