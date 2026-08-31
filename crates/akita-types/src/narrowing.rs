//! Checked narrowing conversions from `usize` into the fixed-width integers
//! used by wire formats and plan encodings.
//!
//! These live here so that every module reporting "value does not fit" does so
//! with one message shape and one error variant.

use akita_error::AkitaError;

/// Narrow to `u32`, naming the quantity in the error.
pub(crate) fn usize_to_u32(value: usize, name: &str) -> Result<u32, AkitaError> {
    u32::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u32")))
}

/// Narrow to `u64`, naming the quantity in the error.
pub(crate) fn usize_to_u64(value: usize, name: &str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u64")))
}

/// Narrow to `u8`, naming the quantity in the error.
pub(crate) fn usize_to_u8(value: usize, name: &str) -> Result<u8, AkitaError> {
    u8::try_from(value).map_err(|_| AkitaError::InvalidInput(format!("{name} does not fit u8")))
}
