//! Binary scalar arithmetic for `F_2[X]/(X^162 + X^81 + 1)`.
//!
//! This is a separate characteristic-two type, not an odd-prime extension.
//! Coefficients are stored low-degree first in three little-endian words.
//! Multiplication selects carryless CPU instructions when available; every
//! platform also has a portable implementation. No protocol traits are added.

mod product;

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

    /// Square by interleaving zero bits, without a multiplication kernel.
    pub fn square(self) -> Self {
        let mut product = [0; 6];
        for (i, word) in self.0.into_iter().enumerate() {
            product[2 * i] = spread(word as u32);
            product[2 * i + 1] = spread((word >> 32) as u32);
        }
        Self::reduce(product)
    }

    /// Multiplicative inverse, or `None` for zero.
    ///
    /// An addition chain for 161 builds `a^(2^161-1)` and squares it.
    /// This takes nine multiplications rather than 160.
    pub fn inverse(self) -> Option<Self> {
        if self == Self::ZERO {
            return None;
        }
        let mut power = self;
        let mut power32 = self;
        for width in [1, 2, 4, 8, 16, 32, 64] {
            power = power.square_n(width) * power;
            if width == 16 {
                power32 = power;
            }
        }
        power = power.square_n(32) * power32;
        power = power.square() * self;
        Some(power.square())
    }

    fn square_n(mut self, count: usize) -> Self {
        for _ in 0..count {
            self = self.square();
        }
        self
    }

    fn reduce(p: [u64; 6]) -> Self {
        // Write P=L+X^162 H, then use X^162=X^81+1 twice.
        // H has degree <=160, so the second high part has degree <=79.
        let h = [
            (p[2] >> 34) | (p[3] << 30),
            (p[3] >> 34) | (p[4] << 30),
            (p[4] >> 34) | (p[5] << 30),
        ];
        let j = [
            h[0] ^ (h[1] >> 17) ^ (h[2] << 47),
            h[1] ^ (h[2] >> 17),
            h[2],
        ];
        Self([
            p[0] ^ j[0],
            p[1] ^ j[1] ^ (j[0] << 17),
            (p[2] ^ j[2] ^ (j[0] >> 47) ^ (j[1] << 17)) & Self::TOP_MASK,
        ])
    }
}

fn spread(value: u32) -> u64 {
    let mut x = u64::from(value);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555
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
        Self::reduce(product::multiply(self.0, rhs.0))
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
