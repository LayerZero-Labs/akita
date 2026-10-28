//! Centered q24 image coefficients stored as three-byte offset words.
//!
//! The accepted integer interval is `[-4*S8, 3*S8]`, where
//! `S8 = (8^8 - 1)/7 = 2_396_745`. The stored word is
//! `y = x + 0x924924`, in **little-endian byte order**. Its eight base-8
//! fields encode balanced digits `((y >> (3*i)) & 7) - 4`, least significant
//! digit first. Every three-byte word is valid. Canonical residues modulo
//! [`Q24`] use centered representatives in `[-7_126_704, 7_126_704]`, a
//! strict subset of the accepted interval. Decoding does not impose this
//! narrower honest-image interval.

use akita_error::{checked, AkitaError};

/// Prime modulus for the q24 trinomial coefficient domain.
pub const Q24: u32 = 14_253_409;
/// Sum of the eight base-8 place weights.
pub const IMAGE_DIGIT_WEIGHT_SUM: i32 = 2_396_745;
/// Smallest integer representable by eight balanced base-8 digits.
pub const IMAGE_MIN: i32 = -4 * IMAGE_DIGIT_WEIGHT_SUM;
/// Largest integer representable by eight balanced base-8 digits.
pub const IMAGE_MAX: i32 = 3 * IMAGE_DIGIT_WEIGHT_SUM;
/// Offset added to an image coefficient before storage.
pub const IMAGE_OFFSET: i32 = -IMAGE_MIN;
/// Largest magnitude of a canonical centered residue modulo [`Q24`].
pub const Q24_CENTERED_BOUND: i32 = (Q24 as i32 - 1) / 2;

/// Eight balanced base-8 image digits, stored without per-digit expansion.
///
/// The private three-byte representation guarantees that every value decodes
/// within [`IMAGE_MIN`] through [`IMAGE_MAX`]. It occupies exactly three bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Q24Image([u8; 3]);

impl Q24Image {
    /// Encode an integer in the full enforced image interval.
    pub fn from_centered(value: i32) -> Result<Self, AkitaError> {
        if !(IMAGE_MIN..=IMAGE_MAX).contains(&value) {
            return Err(AkitaError::InvalidInput(
                "image coefficient is outside the eight-digit centered interval".into(),
            ));
        }
        let [a, b, c, _] = (value + IMAGE_OFFSET).to_le_bytes();
        Ok(Self([a, b, c]))
    }

    /// Encode a canonical residue `0 <= residue < q` using its representative
    /// in `[-(q-1)/2, (q-1)/2]`.
    pub fn from_residue(residue: u32) -> Result<Self, AkitaError> {
        if residue >= Q24 {
            return Err(AkitaError::InvalidInput(
                "image residue is not canonical modulo q24".into(),
            ));
        }
        Self::from_centered(center_residue(residue))
    }

    /// Encode an offset word, rejecting values that need more than 24 bits.
    pub fn from_offset_word(word: u32) -> Result<Self, AkitaError> {
        if word >= 1 << 24 {
            return Err(AkitaError::InvalidInput(
                "image offset word exceeds 24 bits".into(),
            ));
        }
        let [a, b, c, _] = word.to_le_bytes();
        Ok(Self([a, b, c]))
    }

    /// Encode eight digits in `-4..=3`, least significant digit first.
    pub fn from_digits(digits: [i8; 8]) -> Result<Self, AkitaError> {
        let mut word = 0u32;
        for (i, digit) in digits.into_iter().enumerate() {
            if !(-4..=3).contains(&digit) {
                return Err(AkitaError::InvalidInput(
                    "image digit is outside the balanced base-8 interval".into(),
                ));
            }
            word |= ((i32::from(digit) + 4) as u32) << (3 * i);
        }
        Self::from_offset_word(word)
    }

    /// Read three little-endian bytes. All possible byte arrays are valid.
    pub const fn from_bytes(bytes: [u8; 3]) -> Self {
        Self(bytes)
    }

    /// Return the accepted integer representative, including noncanonical
    /// representatives outside the narrower honest-image interval.
    pub fn centered(self) -> i32 {
        self.offset_word() as i32 - IMAGE_OFFSET
    }

    /// Return the canonical residue modulo [`Q24`].
    pub fn residue(self) -> u32 {
        self.centered().rem_euclid(Q24 as i32) as u32
    }

    /// Return the canonical centered representative of the decoded residue.
    pub fn canonical_centered(self) -> i32 {
        center_residue(self.residue())
    }

    /// Return the 24-bit offset word `x + 0x924924`.
    pub const fn offset_word(self) -> u32 {
        let [a, b, c] = self.0;
        u32::from_le_bytes([a, b, c, 0])
    }

    /// Expand the eight balanced base-8 digits, least significant first.
    pub fn digits(self) -> [i8; 8] {
        let word = self.offset_word();
        core::array::from_fn(|i| ((word >> (3 * i)) & 7) as i8 - 4)
    }

    /// Return three bytes in little-endian order.
    pub const fn bytes(self) -> [u8; 3] {
        self.0
    }

    /// Return integer parity as zero or one, directly from the offset word.
    /// The offset is even, so this equals parity of the decoded integer.
    pub const fn parity(self) -> u8 {
        let [low, _, _] = self.0;
        low & 1
    }
}

// Both callers supply canonical residues in 0..Q24.
fn center_residue(residue: u32) -> i32 {
    if residue > Q24_CENTERED_BOUND as u32 {
        residue as i32 - Q24 as i32
    } else {
        residue as i32
    }
}

fn validate_packed_length(coefficients: usize, bytes: usize) -> Result<(), AkitaError> {
    let expected = checked::product([coefficients, 3])
        .ok_or_else(|| AkitaError::InvalidInput("image byte length overflows usize".into()))?;
    if bytes != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: bytes,
        });
    }
    Ok(())
}

/// Encode centered coefficients into consecutive three-byte offset words.
/// The destination must have exactly three bytes per input coefficient.
/// An invalid coefficient can leave an already encoded prefix in `bytes`.
pub fn encode_centered_slice(values: &[i32], bytes: &mut [u8]) -> Result<(), AkitaError> {
    validate_packed_length(values.len(), bytes.len())?;
    for (&value, chunk) in values.iter().zip(bytes.chunks_exact_mut(3)) {
        chunk.copy_from_slice(&Q24Image::from_centered(value)?.bytes());
    }
    Ok(())
}

/// Encode canonical residues into consecutive three-byte centered offset words.
/// The destination must have exactly three bytes per input coefficient.
/// An invalid residue can leave an already encoded prefix in `bytes`.
pub fn encode_residue_slice(values: &[u32], bytes: &mut [u8]) -> Result<(), AkitaError> {
    validate_packed_length(values.len(), bytes.len())?;
    for (&value, chunk) in values.iter().zip(bytes.chunks_exact_mut(3)) {
        chunk.copy_from_slice(&Q24Image::from_residue(value)?.bytes());
    }
    Ok(())
}

/// Decode consecutive three-byte offset words into accepted representatives.
/// The input must have exactly three bytes per output coefficient.
pub fn decode_centered_slice(bytes: &[u8], values: &mut [i32]) -> Result<(), AkitaError> {
    validate_packed_length(values.len(), bytes.len())?;
    for (chunk, value) in bytes.chunks_exact(3).zip(values) {
        let bytes = chunk.try_into().map_err(|_| AkitaError::InvalidProof)?;
        *value = Q24Image::from_bytes(bytes).centered();
    }
    Ok(())
}

/// Decode consecutive three-byte offset words into canonical residues modulo q.
/// The input must have exactly three bytes per output coefficient.
pub fn decode_residue_slice(bytes: &[u8], values: &mut [u32]) -> Result<(), AkitaError> {
    validate_packed_length(values.len(), bytes.len())?;
    for (chunk, value) in bytes.chunks_exact(3).zip(values) {
        let bytes = chunk.try_into().map_err(|_| AkitaError::InvalidProof)?;
        *value = Q24Image::from_bytes(bytes).residue();
    }
    Ok(())
}

#[cfg(test)]
mod tests;
