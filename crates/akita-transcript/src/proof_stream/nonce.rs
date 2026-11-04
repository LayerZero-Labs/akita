use akita_error::AkitaError;
use spongefish::{FromNarg, NargReader, VerificationError};

use super::invalid_proof;
use crate::{ProverChannel, VerifierChannel};

/// Fold-response nonce: canonical unsigned LEB128 on the wire, absorbed as
/// one `u64` little-endian word.
///
/// The wire encoding is self-delimiting and rejects overflow and redundant
/// terminal zero groups. Schedule-dependent range checks remain at the
/// grinding-plan replay boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonceAtom(u32);

/// Canonical LEB128 wire bytes of one nonce.
pub struct NonceWire {
    bytes: [u8; nonce_max_bytes(u32::BITS as u8)],
    len: usize,
}

impl AsRef<[u8]> for NonceWire {
    fn as_ref(&self) -> &[u8] {
        self.bytes.get(..self.len).unwrap_or_default()
    }
}

impl NonceAtom {
    /// Wrap one nonce for canonical proof transport.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the decoded nonce.
    #[must_use]
    pub const fn into_inner(self) -> u32 {
        self.0
    }

    /// The absorbed form: the nonce as one `u64` little-endian word.
    #[must_use]
    pub fn word(self) -> [u8; 8] {
        u64::from(self.0).to_le_bytes()
    }

    /// The wire form: canonical unsigned LEB128.
    #[must_use]
    pub fn wire(self) -> NonceWire {
        let mut value = self.0;
        let mut encoded = NonceWire {
            bytes: [0; nonce_max_bytes(u32::BITS as u8)],
            len: 0,
        };
        for slot in &mut encoded.bytes {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            *slot = byte;
            encoded.len += 1;
            if value == 0 {
                break;
            }
        }
        encoded
    }
}

/// Send one fold-response nonce: its LEB128 wire bytes, absorbed as its word.
pub fn send_nonce(state: &mut ProverChannel, value: u32) {
    state.prover_message_with(
        &NonceAtom::new(value),
        |nonce| nonce.word(),
        |nonce, out| out.extend_from_slice(nonce.wire().as_ref()),
    );
}

/// Receive one fold-response nonce sent by [`send_nonce`], rejecting a
/// noncanonical encoding.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] and poisons the transcript when the
/// next bytes are not a canonical `u32` LEB128 nonce.
pub fn receive_nonce(state: &mut VerifierChannel<'_>) -> Result<u32, AkitaError> {
    state
        .prover_message_with(NonceAtom::from_narg, |nonce| nonce.word())
        .map(NonceAtom::into_inner)
        .map_err(invalid_proof)
}

impl FromNarg for NonceAtom {
    fn from_narg(reader: &mut NargReader<'_>) -> Result<Self, VerificationError> {
        let mut value = 0u32;
        for index in 0..nonce_max_bytes(u32::BITS as u8) {
            let [byte] = reader.take_array::<1>()?;
            let payload = byte & 0x7f;
            if index == 4 && payload > 0x0f {
                return Err(VerificationError);
            }
            value |= u32::from(payload) << (index * 7);
            if byte & 0x80 == 0 {
                if index != 0 && payload == 0 {
                    return Err(VerificationError);
                }
                return Ok(Self(value));
            }
        }
        Err(VerificationError)
    }
}

/// Maximum canonical unsigned LEB128 bytes for a nonce of `nonce_bits` bits.
#[must_use]
pub const fn nonce_max_bytes(nonce_bits: u8) -> usize {
    if nonce_bits == 0 {
        0
    } else {
        (nonce_bits as usize).div_ceil(7)
    }
}

/// Number of bytes in one canonical unsigned LEB128 nonce.
#[must_use]
pub const fn nonce_encoded_len(value: u32) -> usize {
    let significant_bits = u32::BITS - value.leading_zeros();
    let encoded_bits = if significant_bits == 0 {
        1
    } else {
        significant_bits
    };
    (encoded_bits as usize).div_ceil(7)
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
            let encoded = NonceAtom::new(value).wire().as_ref().to_vec();
            assert_eq!(encoded.len(), expected_len);
            assert_eq!(encoded.len(), nonce_encoded_len(value));
            let suffix = [0xa5, 0x5a];
            let mut argument = encoded.clone();
            argument.extend_from_slice(&suffix);
            let mut reader = NargReader::new(&argument);
            let decoded = reader.read::<NonceAtom>().unwrap();
            assert_eq!(decoded.into_inner(), value);
            assert_eq!(decoded.wire().as_ref(), encoded);
            assert_eq!(decoded.word(), u64::from(value).to_le_bytes());
            assert_eq!(reader.take_array::<2>().unwrap(), suffix);
            assert!(reader.is_empty());
        }
    }

    #[test]
    fn rejects_noncanonical_or_overflowing_encodings_and_poisons() {
        for malformed in [
            &[0x80, 0x00][..],
            &[0xff, 0xff, 0xff, 0xff, 0x10][..],
            &[0xff, 0xff, 0xff, 0xff, 0x80][..],
            &[0x80][..],
        ] {
            let mut reader = NargReader::new(malformed);
            assert!(reader.read::<NonceAtom>().is_err());
            assert!(reader.is_poisoned());
        }
    }
}
