//! Allocation and length limits for shape-backed wire decoding.

use akita_serialization::{SerializationError, DEFAULT_MAX_SEQUENCE_LEN};

pub(crate) const MAX_PROOF_SHAPE_SEQUENCE_LEN: usize = 1 << 12;

pub(crate) fn checked_shape_len(len: usize) -> Result<(), SerializationError> {
    if len > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max: DEFAULT_MAX_SEQUENCE_LEN,
        });
    }
    Ok(())
}

pub(crate) fn checked_shape_sequence_len(len: usize) -> Result<(), SerializationError> {
    if len > MAX_PROOF_SHAPE_SEQUENCE_LEN {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max: MAX_PROOF_SHAPE_SEQUENCE_LEN,
        });
    }
    Ok(())
}

pub(crate) fn reserve_shape_len<T>(vec: &mut Vec<T>, len: usize) -> Result<(), SerializationError> {
    checked_shape_len(len)?;
    vec.try_reserve_exact(len)
        .map_err(|_| SerializationError::InvalidData("shape-backed allocation failed".to_string()))
}
