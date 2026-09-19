use spongefish::{Encoding, NargDeserialize, VerificationError};

/// Canonical unsigned LEB128 nonce used by native grinding messages.
///
/// The encoding is self-delimiting and rejects overflow and redundant terminal
/// zero groups. Schedule-dependent range checks remain at the grinding-plan
/// replay boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeNonce(u32);

struct NativeNonceEncoding {
    bytes: [u8; native_nonce_max_bytes(u32::BITS as u8)],
    len: usize,
}

impl AsRef<[u8]> for NativeNonceEncoding {
    fn as_ref(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl NativeNonce {
    /// Wrap one nonce for canonical native proof transport.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the decoded nonce.
    #[must_use]
    pub const fn into_inner(self) -> u32 {
        self.0
    }
}

impl Encoding<[u8]> for NativeNonce {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut value = self.0;
        let mut encoded = NativeNonceEncoding {
            bytes: [0; native_nonce_max_bytes(u32::BITS as u8)],
            len: 0,
        };
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            encoded.bytes[encoded.len] = byte;
            encoded.len += 1;
            if value == 0 {
                return encoded;
            }
        }
    }
}

impl NargDeserialize for NativeNonce {
    fn deserialize_from_narg(buf: &mut &[u8]) -> Result<Self, VerificationError> {
        let mut remaining = *buf;
        let mut value = 0u32;
        for index in 0..native_nonce_max_bytes(u32::BITS as u8) {
            let (&byte, tail) = remaining.split_first().ok_or(VerificationError)?;
            let payload = byte & 0x7f;
            if index == 4 && payload > 0x0f {
                return Err(VerificationError);
            }
            value |= u32::from(payload) << (index * 7);
            remaining = tail;
            if byte & 0x80 == 0 {
                if index != 0 && payload == 0 {
                    return Err(VerificationError);
                }
                *buf = remaining;
                return Ok(Self(value));
            }
        }
        Err(VerificationError)
    }
}

/// Maximum canonical unsigned LEB128 bytes for a nonce of `nonce_bits` bits.
#[must_use]
pub const fn native_nonce_max_bytes(nonce_bits: u8) -> usize {
    if nonce_bits == 0 {
        0
    } else {
        (nonce_bits as usize).div_ceil(7)
    }
}

/// Number of bytes in one canonical unsigned LEB128 nonce.
#[must_use]
pub const fn native_nonce_encoded_len(value: u32) -> usize {
    if value < (1 << 7) {
        1
    } else if value < (1 << 14) {
        2
    } else if value < (1 << 21) {
        3
    } else if value < (1 << 28) {
        4
    } else {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_canonical_and_self_delimiting() {
        for (value, expected_len) in [
            (0, 1),
            (1, 1),
            (127, 1),
            (128, 2),
            (16_383, 2),
            (16_384, 3),
            (u32::MAX, 5),
        ] {
            let encoded = NativeNonce::new(value).encode().as_ref().to_vec();
            assert_eq!(encoded.len(), expected_len);
            let suffix = [0xa5, 0x5a];
            let mut argument = encoded.clone();
            argument.extend_from_slice(&suffix);
            let mut cursor = argument.as_slice();
            let decoded = NativeNonce::deserialize_from_narg(&mut cursor).unwrap();
            assert_eq!(decoded.into_inner(), value);
            assert_eq!(decoded.encode().as_ref(), encoded);
            assert_eq!(cursor, suffix);
        }
    }

    #[test]
    fn rejects_noncanonical_or_overflowing_encodings_transactionally() {
        for malformed in [
            &[0x80, 0x00][..],
            &[0xff, 0xff, 0xff, 0xff, 0x10][..],
            &[0xff, 0xff, 0xff, 0xff, 0x80][..],
            &[0x80][..],
        ] {
            let mut cursor = malformed;
            let original = cursor;
            assert!(NativeNonce::deserialize_from_narg(&mut cursor).is_err());
            assert_eq!(cursor, original);
        }
    }
}
