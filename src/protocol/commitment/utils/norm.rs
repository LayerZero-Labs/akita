//! Infinity norm utilities for ring elements over Z_q.

use crate::algebra::ring::CyclotomicRing;
use crate::CanonicalField;

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `−1` in `Z_q` is `q − 1`.
pub(crate) fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Centered absolute value of a field element.
///
/// Maps canonical representation `v ∈ [0, q)` to `min(v, q − v)`.
#[inline]
pub(crate) fn centered_abs<F: CanonicalField>(x: F, modulus: u128) -> u128 {
    let v = x.to_canonical_u128();
    let half = modulus / 2;
    if v <= half {
        v
    } else {
        modulus - v
    }
}

/// L∞ norm of a single ring element (maximum centered coefficient magnitude).
pub(crate) fn ring_inf_norm<F: CanonicalField, const D: usize>(
    r: &CyclotomicRing<F, D>,
    modulus: u128,
) -> u128 {
    r.coefficients()
        .iter()
        .map(|c| centered_abs(*c, modulus))
        .max()
        .unwrap_or(0)
}

/// L∞ norm of a vector of ring elements.
pub(crate) fn vec_inf_norm<F: CanonicalField, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    modulus: u128,
) -> u128 {
    v.iter()
        .map(|r| ring_inf_norm(r, modulus))
        .max()
        .unwrap_or(0)
}
