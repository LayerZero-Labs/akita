//! Binary scalar arithmetic for `F_2[X]/(X^162 + X^81 + 1)`.
//!
//! This is a separate characteristic-two type, not an odd-prime extension.
//! Coefficients are stored low-degree first in three little-endian words.
//! Multiplication selects carryless CPU instructions when available; every
//! platform also has a portable implementation. No protocol traits are added.

mod packed;
mod product;

pub use packed::PackedBinary162;

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// An element of the degree-162 binary scalar field.
///
/// The modulus is irreducible because 2 has order 162 modulo 243.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BinaryField162([u64; 3]);

impl BinaryField162 {
    /// Additive identity.
    pub const ZERO: Self = Self([0; 3]);
    /// Multiplicative identity.
    pub const ONE: Self = Self([1, 0, 0]);
    /// Number of polynomial coefficients.
    pub const DEGREE: usize = 162;
    const TOP_MASK: u64 = (1 << 34) - 1;

    /// Construct from coefficients, rejecting any bit above degree 161.
    pub const fn from_words(words: [u64; 3]) -> Option<Self> {
        if words[2] & !Self::TOP_MASK == 0 {
            Some(Self(words))
        } else {
            None
        }
    }

    /// Return the low-degree-first coefficient words.
    pub const fn to_words(self) -> [u64; 3] {
        self.0
    }

    /// Decode exactly 21 little-endian bytes, rejecting unused high bits.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; 21] = bytes.try_into().ok()?;
        if bytes[20] & !3 != 0 {
            return None;
        }
        let mut words = [0u64; 3];
        for (i, byte) in bytes.iter().enumerate() {
            words[i / 8] |= u64::from(*byte) << (8 * (i % 8));
        }
        Some(Self(words))
    }

    /// Encode in the unique 21-byte little-endian representation.
    pub fn to_bytes(self) -> [u8; 21] {
        std::array::from_fn(|i| (self.0[i / 8] >> (8 * (i % 8))) as u8)
    }

    /// Square this element.
    ///
    /// Hardware backends use three carryless products because cross terms
    /// vanish in characteristic two. Other targets use portable bit spreading.
    pub fn square(self) -> Self {
        product::kernels().square(self)
    }

    /// Return the sum of pairwise products, or `None` when the lengths differ.
    ///
    /// The implementation accumulates unreduced products and performs one
    /// final reduction. Runtime feature detection occurs before the inner loop.
    pub fn dot_product(lhs: &[Self], rhs: &[Self]) -> Option<Self> {
        (lhs.len() == rhs.len()).then(|| product::kernels().dot_product(lhs, rhs))
    }

    /// Multiplicative inverse, or `None` for zero.
    ///
    /// An addition chain for 161 builds `a^(2^161-1)` and squares it.
    /// This takes nine multiplications rather than 160.
    pub fn inverse(self) -> Option<Self> {
        if self == Self::ZERO {
            return None;
        }
        Some(product::kernels().inverse(self))
    }

    #[cfg(test)]
    fn square_n(mut self, count: usize) -> Self {
        for _ in 0..count {
            self = self.square();
        }
        self
    }
}

impl Add for BinaryField162 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }
}

impl Sub for BinaryField162 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Mul for BinaryField162 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        product::kernels().multiply(self, rhs)
    }
}

impl AddAssign for BinaryField162 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for BinaryField162 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for BinaryField162 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests;
