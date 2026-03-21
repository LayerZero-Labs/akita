//! Infinity norm utilities for ring elements over Z_q.

use crate::CanonicalField;

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `−1` in `Z_q` is `q − 1`.
pub(crate) fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}
