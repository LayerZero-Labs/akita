//! Guardrails for Labrador/Greyhound protocol plumbing.

use crate::error::HachiError;

/// Maximum recursion levels accepted by the protocol.
///
/// Mirrors the fixed upper bound used by the C reference (`proof *pi[16]`).
pub const LABRADOR_MAX_LEVELS: usize = 4;
/// Upper bound for JL nonce search attempts.
pub const LABRADOR_MAX_JL_NONCE_RETRIES: u64 = 1 << 20;
/// Upper bound on challenge polynomials sampled per call.
pub const LABRADOR_MAX_CHALLENGE_POLYS: usize = 1 << 12;
/// Upper bound for temporary byte allocations in Labrador helpers.
pub const LABRADOR_MAX_TEMP_BYTES: usize = 1 << 27; // 128 MiB

/// Checked conversion from `usize` to `u64`.
///
/// # Errors
///
/// Returns an error when `value` does not fit into `u64`.
pub fn checked_usize_to_u64(value: usize, what: &'static str) -> Result<u64, HachiError> {
    u64::try_from(value)
        .map_err(|_| HachiError::InvalidInput(format!("{what} does not fit in u64: {value}")))
}

/// Ensure a value is a power of two.
///
/// # Errors
///
/// Returns an error if `value` is not a power of two.
pub fn ensure_power_of_two(value: usize, what: &'static str) -> Result<(), HachiError> {
    if !value.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "{what} must be a power of two, got {value}"
        )));
    }
    Ok(())
}

/// Checked `a * b` for allocation sizing.
///
/// # Errors
///
/// Returns an error if multiplication overflows `usize`.
pub fn checked_mul(a: usize, b: usize, what: &'static str) -> Result<usize, HachiError> {
    a.checked_mul(b)
        .ok_or_else(|| HachiError::InvalidInput(format!("overflow while computing {what}")))
}

/// Checked `a + b` for allocation sizing.
///
/// # Errors
///
/// Returns an error if addition overflows `usize`.
pub fn checked_add(a: usize, b: usize, what: &'static str) -> Result<usize, HachiError> {
    a.checked_add(b)
        .ok_or_else(|| HachiError::InvalidInput(format!("overflow while computing {what}")))
}

/// Validate temporary allocation size against guardrail cap.
///
/// # Errors
///
/// Returns an error if `bytes > LABRADOR_MAX_TEMP_BYTES`.
pub fn ensure_temp_allocation_limit(bytes: usize, what: &'static str) -> Result<(), HachiError> {
    if bytes > LABRADOR_MAX_TEMP_BYTES {
        return Err(HachiError::InvalidInput(format!(
            "{what} temporary allocation too large: {bytes} bytes (max {LABRADOR_MAX_TEMP_BYTES})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_mul_detects_overflow() {
        let err = checked_mul(usize::MAX, 2, "overflow-test").unwrap_err();
        assert!(matches!(err, HachiError::InvalidInput(_)));
    }

    #[test]
    fn temp_limit_enforced() {
        let err = ensure_temp_allocation_limit(LABRADOR_MAX_TEMP_BYTES + 1, "tmp").unwrap_err();
        assert!(matches!(err, HachiError::InvalidInput(_)));
    }
}
