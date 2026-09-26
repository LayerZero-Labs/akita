//! Small binary fields used by the field-switch reference path.

use std::ops::{Add, AddAssign, Mul};
use std::sync::OnceLock;

use super::product::portable_clmul;

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x86;

type Mul128 = fn(BinaryField128, BinaryField128) -> BinaryField128;
type Mul192 = fn(BinaryField192, BinaryField192) -> BinaryField192;
type Equality128 = fn(&[BinaryField128], &mut [BinaryField128]);
type Equality192 = fn(&[BinaryField192], &mut [BinaryField192]);

struct Kernels {
    multiply128: Mul128,
    multiply192: Mul192,
    equality128: Equality128,
    equality192: Equality192,
}

fn kernels() -> &'static Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    KERNELS.get_or_init(detect)
}

fn detect() -> Kernels {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes")
        && std::arch::is_aarch64_feature_detected!("pmull")
    {
        return Kernels {
            multiply128: |a, b| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm::multiply128(a, b) }
            },
            multiply192: |a, b| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm::multiply192(a, b) }
            },
            equality128: |point, output| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm::equality128(point, output) }
            },
            equality192: |point, output| {
                // SAFETY: this closure is installed only after detecting PMULL.
                unsafe { arm::equality192(point, output) }
            },
        };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        let mut equality128: Equality128 = |point, output| {
            // SAFETY: this closure is installed only after detecting PCLMUL.
            unsafe { x86::equality128(point, output) }
        };
        let mut equality192: Equality192 = |point, output| {
            // SAFETY: this closure is installed only after detecting PCLMUL.
            unsafe { x86::equality192(point, output) }
        };
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            equality128 = |point, output| {
                // SAFETY: this closure is installed only after detecting both features.
                unsafe { x86::equality128_vec4(point, output) }
            };
            equality192 = |point, output| {
                // SAFETY: this closure is installed only after detecting both features.
                unsafe { x86::equality192_vec4(point, output) }
            };
        }
        return Kernels {
            multiply128: |a, b| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86::multiply128(a, b) }
            },
            multiply192: |a, b| {
                // SAFETY: this closure is installed only after detecting PCLMUL.
                unsafe { x86::multiply192(a, b) }
            },
            equality128,
            equality192,
        };
    }
    Kernels {
        multiply128: portable_multiply128,
        multiply192: portable_multiply192,
        equality128: portable_equality128,
        equality192: portable_equality192,
    }
}

/// `F_2[x]/(x^128 + x^7 + x^2 + x + 1)` in low-bit polynomial coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
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

    /// Expand the multilinear equality weights for a pre-sized output table.
    pub(super) fn equality_weights(point: &[Self], output: &mut [Self]) {
        (kernels().equality128)(point, output);
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
        (kernels().multiply128)(self, rhs)
    }
}

#[inline]
pub(super) fn portable_multiply128(a: BinaryField128, b: BinaryField128) -> BinaryField128 {
    let [a0, a1] = a.0;
    let [b0, b1] = b.0;
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

/// Reduce a degree-at-most-254 polynomial using `x^128 = x^7 + x^2 + x + 1`.
#[inline]
pub(super) fn reduce128(product: [u64; 4]) -> BinaryField128 {
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
#[repr(transparent)]
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

    /// Expand the multilinear equality weights for a pre-sized output table.
    pub(super) fn equality_weights(point: &[Self], output: &mut [Self]) {
        (kernels().equality192)(point, output);
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
        (kernels().multiply192)(self, rhs)
    }
}

#[inline]
pub(super) fn portable_multiply192(a: BinaryField192, b: BinaryField192) -> BinaryField192 {
    let [a0, a1, a2] = a.0;
    let [b0, b1, b2] = b.0;
    let d0 = portable_clmul(a0, b0);
    let d1 = portable_clmul(a1, b1);
    let d2 = portable_clmul(a2, b2);
    let c01 = portable_clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    let c02 = portable_clmul(a0 ^ a2, b0 ^ b2) ^ d0 ^ d2;
    let c12 = portable_clmul(a1 ^ a2, b1 ^ b2) ^ d1 ^ d2;

    // Reduce the y^3 and y^4 terms with y^3 = y + 1 and y^4 = y^2 + y.
    BinaryField192([
        reduce64(d0 ^ c12),
        reduce64(c01 ^ c12 ^ d2),
        reduce64(d1 ^ c02 ^ d2),
    ])
}

/// Reduce over `K` using `x^64 = x^4 + x^3 + x + 1`.
#[inline]
pub(super) fn reduce64(product: u128) -> u64 {
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

fn portable_equality128(point: &[BinaryField128], output: &mut [BinaryField128]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField128::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        for j in 0..width {
            let hi = portable_multiply128(low[j], r);
            high[j] = hi;
            low[j] += hi;
        }
    }
}

fn portable_equality192(point: &[BinaryField192], output: &mut [BinaryField192]) {
    if output.is_empty() {
        return;
    }
    output[0] = BinaryField192::ONE;
    for (axis, &r) in point.iter().enumerate() {
        let width = 1 << axis;
        let (low, high) = output.split_at_mut(width);
        for j in 0..width {
            let hi = portable_multiply192(low[j], r);
            high[j] = hi;
            low[j] += hi;
        }
    }
}

#[cfg(test)]
mod tests;
