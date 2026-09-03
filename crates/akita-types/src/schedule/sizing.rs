//! Recursive witness sizing shared by planning and runtime validation.

use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: Field + CanonicalEncoding>(log_basis: u32) -> usize {
    crate::sis::compute_num_digits_field_width(F::MODULUS_BITS, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
#[inline]
pub fn detect_field_modulus<F: Field + CanonicalEncoding>() -> Result<u128, AkitaError> {
    crate::dispatch::field_modulus::<F>()
}
