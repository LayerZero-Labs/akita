//! Small math helpers for commitment internals.

use crate::error::HachiError;

/// Compute `2^exp` with overflow checks.
///
/// # Errors
///
/// Returns `InvalidSetup` if `2^exp` does not fit in `usize`.
pub(in crate::protocol::commitment) fn checked_pow2(exp: usize) -> Result<usize, HachiError> {
    1usize
        .checked_shl(exp as u32)
        .ok_or_else(|| HachiError::InvalidSetup(format!("2^{exp} does not fit usize")))
}
