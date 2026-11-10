//! Branch-free signed integer lifts.

use crate::{CanonicalEncoding, Field};

/// The offset of [`lift_biased`].
pub const LIFT_BIAS: u64 = 1 << 63;

/// `x + 2^63` as a field element, by flipping `x`'s top bit: an unsigned
/// value that is nonzero for every `x > i64::MIN` (in a field with
/// `p > 2^64`). Its top bit is still `x`'s inverted sign, but the multiplier
/// never sees a zero or a 0/1 sign selector.
///
/// For a product with a secret integer, multiply this and subtract
/// `F(LIFT_BIAS)` times the factor: the multiplier then never sees a zero or
/// sign-selected operand in fields with `p > 2^64`, except at `i64::MIN`.
///
/// Control flow and addresses are independent of `x`, provided field
/// arithmetic and `F::from_u64` are data independent for u32 inputs in a
/// 32-bit field, or u64 inputs otherwise. Splitting 32-bit fields avoids the
/// pinned Fp32 implementation's variable-time reduction of wider inputs.
#[inline]
pub fn lift_biased<F: Field + CanonicalEncoding>(x: i64) -> F {
    let biased = (x as u64) ^ LIFT_BIAS;
    if F::MODULUS_BITS == 32 {
        // Both secret conversions are below 2^32: Fp32's reduction loop
        // executes zero iterations. The radix conversion is public.
        F::from_u64(biased & u64::from(u32::MAX)) + F::from_u64(biased >> 32) * F::from_u64(1 << 32)
    } else {
        F::from_u64(biased)
    }
}

/// `x` as a field element with no branch on `x`, unlike
/// [`Field::from_i64`]: [`lift_biased`] minus the
/// constant bias. Wider fields need no multiplication; 32-bit fields use
/// the fixed radix assembly in [`lift_biased`]. Use this for masks and response
/// digits.
///
/// Control flow and addresses are independent of `x` under the arithmetic
/// assumptions of [`lift_biased`], including data-independent field subtraction.
#[inline]
pub fn lift_ct<F: Field + CanonicalEncoding>(x: i64) -> F {
    lift_biased::<F>(x) - F::from_u64(LIFT_BIAS)
}
