//! Temporary alignment adapter for Jolt's current borrowed byte framing.
//!
//! The pinned Jolt SDK encodes every `&[u8]` argument through Postcard. Its
//! variable-width slice length changes the guest address of the borrowed blob,
//! so Akita must include that prefix when aligning the setup matrix. Remove
//! this module and the blob padding record after Jolt provides a first-class
//! aligned borrowed byte argument. The trusted decoder remains correct without
//! this adapter because its misaligned path does not allocate or issue aligned
//! loads.

use akita_error::checked;
use akita_serialization::SerializationError;

const SETUP_MATRIX_WIRE_ALIGNMENT: usize = 8;
const SETUP_MATRIX_LENGTH_BYTES: usize = core::mem::size_of::<u64>();
pub(super) const SETUP_MATRIX_MAX_PADDING_BYTES: usize = 15;

fn postcard_length_prefix_bytes(mut value: usize) -> usize {
    let mut bytes = 1;
    while value >= 0x80 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

pub(super) fn setup_matrix_padding(
    unpadded_blob_len: usize,
    padding_record_offset: usize,
) -> Result<usize, SerializationError> {
    for padding in 0..=SETUP_MATRIX_MAX_PADDING_BYTES {
        let record_len = padding.checked_add(1).ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt setup matrix padding length overflow".to_string(),
            )
        })?;
        let encoded_blob_len = unpadded_blob_len.checked_add(record_len).ok_or_else(|| {
            SerializationError::InvalidData("akita-jolt blob length overflow".to_string())
        })?;
        let matrix_payload_offset =
            checked::sum([padding_record_offset, record_len, SETUP_MATRIX_LENGTH_BYTES])
                .ok_or_else(|| {
                    SerializationError::InvalidData(
                        "akita-jolt setup matrix payload offset overflow".to_string(),
                    )
                })?;
        let framed_matrix_offset = postcard_length_prefix_bytes(encoded_blob_len)
            .checked_add(matrix_payload_offset)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt framed setup matrix offset overflow".to_string(),
                )
            })?;
        if framed_matrix_offset.is_multiple_of(SETUP_MATRIX_WIRE_ALIGNMENT) {
            return Ok(padding);
        }
    }
    Err(SerializationError::InvalidData(
        "akita-jolt could not align the setup matrix in the Postcard input frame".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_serialization::{AkitaSerialize, Compress};

    #[test]
    fn aligns_across_postcard_length_boundaries() {
        assert_eq!(
            SETUP_MATRIX_LENGTH_BYTES,
            0usize.serialized_size(Compress::No)
        );
        for boundary in [128usize, 16_384, 2_097_152, 268_435_456] {
            for unpadded_blob_len in (boundary - 16)..(boundary + 16) {
                for padding_record_offset in 0..SETUP_MATRIX_WIRE_ALIGNMENT {
                    let padding = setup_matrix_padding(unpadded_blob_len, padding_record_offset)
                        .expect("bounded alignment padding");
                    assert!(padding <= SETUP_MATRIX_MAX_PADDING_BYTES);
                    let encoded_blob_len = unpadded_blob_len + 1 + padding;
                    let matrix_payload_offset =
                        padding_record_offset + 1 + padding + SETUP_MATRIX_LENGTH_BYTES;
                    assert!((postcard_length_prefix_bytes(encoded_blob_len)
                        + matrix_payload_offset)
                        .is_multiple_of(SETUP_MATRIX_WIRE_ALIGNMENT));
                }
            }
        }
    }
}
