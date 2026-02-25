//! Minimal `u256` helper used by `Fp128` reduction code.

/// Unsigned 256-bit integer represented as `(hi, lo)` 128-bit halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct U256 {
    /// High 128 bits.
    pub hi: u128,
    /// Low 128 bits.
    pub lo: u128,
}

impl U256 {
    /// Construct from `(hi, lo)` halves.
    #[inline]
    pub const fn new(hi: u128, lo: u128) -> Self {
        Self { hi, lo }
    }

    /// Full-width `u128 * u128 -> u256`.
    #[inline]
    pub fn mul_u128(a: u128, b: u128) -> Self {
        const MASK64: u128 = (1u128 << 64) - 1;

        let a0 = a & MASK64;
        let a1 = a >> 64;
        let b0 = b & MASK64;
        let b1 = b >> 64;

        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;

        let (mid, mid_overflow) = p01.overflowing_add(p10);
        let mid_lo_shift = mid << 64;
        let mid_hi = mid >> 64;

        let (lo, carry_lo) = p00.overflowing_add(mid_lo_shift);

        let mut hi = p11
            .wrapping_add(mid_hi)
            .wrapping_add((mid_overflow as u128) << 64);
        if carry_lo {
            hi = hi.wrapping_add(1);
        }

        Self { hi, lo }
    }

    /// Full-width `u128^2 -> u256`.
    ///
    /// This avoids one 64x64->128 multiply compared to `mul_u128(a, a)`.
    #[inline]
    pub fn sqr_u128(a: u128) -> Self {
        const MASK64: u128 = (1u128 << 64) - 1;

        let a0 = a & MASK64;
        let a1 = a >> 64;

        let p00 = a0 * a0;
        let p01 = a0 * a1;
        let p11 = a1 * a1;

        // mid = 2 * p01 (may overflow, which contributes into the high limb at bit 192).
        let (mid, mid_overflow) = p01.overflowing_add(p01);
        let mid_lo_shift = mid << 64;
        let mid_hi = mid >> 64;

        let (lo, carry_lo) = p00.overflowing_add(mid_lo_shift);

        let mut hi = p11
            .wrapping_add(mid_hi)
            .wrapping_add((mid_overflow as u128) << 64);
        if carry_lo {
            hi = hi.wrapping_add(1);
        }

        Self { hi, lo }
    }

    /// Return the bit at index `i` (`0` is least-significant bit of `lo`).
    #[inline]
    pub fn bit(self, i: usize) -> bool {
        if i < 128 {
            ((self.lo >> i) & 1) == 1
        } else {
            ((self.hi >> (i - 128)) & 1) == 1
        }
    }
}
