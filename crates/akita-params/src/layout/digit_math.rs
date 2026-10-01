//! Small digit-layout arithmetic helpers.

use jolt_field::{CanonicalEncoding, Field};

/// Smallest integer `s` with `s^2 >= v`.
#[inline]
#[must_use]
pub fn isqrt_ceil(v: u128) -> u128 {
    let s = v.isqrt();
    s + u128::from(s.saturating_mul(s) < v)
}

/// Return the row gadget scalars `1, b, b^2, ...` for `b = 2^log_basis`.
pub fn gadget_row_scalars<F: Field + CanonicalEncoding>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for i in 0..levels {
        if i > 0 {
            power *= base;
        }
        out.push(power);
    }
    out
}
