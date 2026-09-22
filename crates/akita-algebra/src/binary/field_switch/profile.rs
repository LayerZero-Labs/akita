use std::ops::{Add, AddAssign, BitXorAssign, Mul};

use super::super::{BinaryField128, BinaryField192};

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::BinaryField128 {}
    impl Sealed for super::BinaryField192 {}
}

/// One of the two supported host-field/binary-source coordinate contracts.
///
/// Sealed to F128/F128 and F64/F192. This is a coordinate contract for the
/// switch, not an implementation of the odd-characteristic field interface.
pub trait SwitchField:
    sealed::Sealed + Copy + Default + Eq + Add<Output = Self> + AddAssign + Mul<Output = Self>
{
    /// Source coefficients: polynomial bits in `u128` (F128) or `u64` (F64).
    type Source: Copy + Default + Eq + BitXorAssign + Into<u128> + TryFrom<u128>;
    /// Number of live binary basis coordinates.
    const ROWS: usize;
    /// Number of F162 batching coordinates, including canonical row padding.
    const BATCH_BITS: usize;
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Low-coordinate-first binary words; unused words are zero.
    fn coordinates(self) -> [u64; 3];
    /// Embed the source field into the host field.
    fn embed_source(value: Self::Source) -> Self;
    /// Expand equality weights into a caller-provided table of size 2^point.len().
    fn equality_weights(point: &[Self], output: &mut [Self]);
    /// Products with the ordered host basis; unused entries are zero.
    ///
    /// The fixed array bounds verifier scratch independently of source size.
    fn basis_products(self) -> [Self; 192];
}

impl SwitchField for BinaryField128 {
    type Source = u128;
    const ROWS: usize = 128;
    const BATCH_BITS: usize = 7;
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    fn coordinates(self) -> [u64; 3] {
        let [lo, hi] = self.to_words();
        [lo, hi, 0]
    }

    fn embed_source(value: u128) -> Self {
        Self::from_words([value as u64, (value >> 64) as u64])
    }

    fn equality_weights(point: &[Self], output: &mut [Self]) {
        Self::equality_weights(point, output);
    }

    fn basis_products(self) -> [Self; 192] {
        let mut result = [Self::ZERO; 192];
        let mut value = self;
        for slot in &mut result[..128] {
            *slot = value;
            value = value.mul_x();
        }
        result
    }
}

impl SwitchField for BinaryField192 {
    type Source = u64;
    const ROWS: usize = 192;
    const BATCH_BITS: usize = 8;
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    fn coordinates(self) -> [u64; 3] {
        self.to_words()
    }

    fn embed_source(value: u64) -> Self {
        Self::from_words([value, 0, 0])
    }

    fn equality_weights(point: &[Self], output: &mut [Self]) {
        Self::equality_weights(point, output);
    }

    fn basis_products(self) -> [Self; 192] {
        let mut result = [Self::ZERO; 192];
        let mut base = self;
        for block in result.chunks_exact_mut(64) {
            let mut value = base;
            for slot in block {
                *slot = value;
                value = value.mul_x();
            }
            base = base.mul_y();
        }
        result
    }
}
