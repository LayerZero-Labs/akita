use super::BinaryIndexMap;
use akita_error::AkitaError;
use akita_serialization::SerializationError;
use std::io::{Cursor, Read};

/// Selected small commitment modulus. This constant is not SIS admission.
pub const BINARY_SOURCE_Q: u32 = 14_253_409;
/// Honest centered image representatives, inclusive.
pub const CANONICAL_IMAGE_WINDOW: (i32, i32) = (-7_126_704, 7_126_704);
/// Full malicious eight-balanced-digit interval, inclusive.
pub const ENFORCED_IMAGE_WINDOW: (i32, i32) = (-9_586_980, 7_190_235);
/// Four balanced base-eight response digits, inclusive.
pub const BINARY_RESPONSE_INTERVAL: (i32, i32) = (-2340, 1755);
/// Maximum distinct points and maximum polynomial/tensor planes per group.
pub const PRIME_POINT_CAP: usize = 5;
/// Bounded wire manifest size; schedule admission may impose a smaller limit.
pub const MAX_BINARY_POLYNOMIALS: usize = 4096;

/// Commitment profile identities are canonical schedule/profile digests, not certificates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryCommitmentProfiles {
    pub a: [u8; 32],
    pub b: [u8; 32],
}

/// Immutable v1 source descriptor. Its fixed tags identify binary polynomial-basis
/// words, Phi243/Psi1944, signed interleaving, the explicit column transpose,
/// indexed bounded-weight-46 challenges with `Shake256V1` deterministic signs, and the conditional
/// signed-bit/zero-padding agreement contract. None of these tags admits a schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinarySourceDescriptor {
    index: BinaryIndexMap,
    polynomials: Vec<[u8; 32]>,
    profiles: BinaryCommitmentProfiles,
}
impl BinarySourceDescriptor {
    /// Polynomial identities are ordered, unique, and nonempty.
    pub fn new(
        index: BinaryIndexMap,
        polynomials: Vec<[u8; 32]>,
        profiles: BinaryCommitmentProfiles,
    ) -> Result<Self, AkitaError> {
        if polynomials.is_empty() || polynomials.len() > MAX_BINARY_POLYNOMIALS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_BINARY_POLYNOMIALS,
                actual: polynomials.len(),
            });
        }
        akita_error::checked::product([polynomials.len(), index.padded_carriers(), 256])
            .ok_or_else(|| AkitaError::InvalidInput("binary group MLE size overflow".into()))?;
        for (i, id) in polynomials.iter().enumerate() {
            if polynomials.iter().take(i).any(|previous| previous == id) {
                return Err(AkitaError::InvalidInput(
                    "duplicate binary polynomial identity".into(),
                ));
            }
        }
        Ok(Self {
            index,
            polynomials,
            profiles,
        })
    }
    pub fn index_map(&self) -> &BinaryIndexMap {
        &self.index
    }
    pub fn polynomials(&self) -> &[[u8; 32]] {
        &self.polynomials
    }
    pub fn profiles(&self) -> &BinaryCommitmentProfiles {
        &self.profiles
    }

    /// Sole transcript encoding: fixed domain/version/tags, LE integers, then
    /// ordered polynomial identities. Consumers absorb these bytes directly.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"AkitaBinarySource\x01".to_vec();
        // source, basis, Psi, packing/permutation, challenge, agreement identities.
        out.extend_from_slice(&[1, 1, 1, 1, 1, 1]);
        for value in [
            self.index.word_width(),
            self.index.logical_len(),
            self.index.padded_carriers(),
            self.index.columns(),
            self.polynomials.len(),
        ] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        out.extend_from_slice(&BINARY_SOURCE_Q.to_le_bytes());
        for value in [
            CANONICAL_IMAGE_WINDOW.0,
            CANONICAL_IMAGE_WINDOW.1,
            ENFORCED_IMAGE_WINDOW.0,
            ENFORCED_IMAGE_WINDOW.1,
            BINARY_RESPONSE_INTERVAL.0,
            BINARY_RESPONSE_INTERVAL.1,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.push(PRIME_POINT_CAP as u8);
        out.extend_from_slice(&self.profiles.a);
        out.extend_from_slice(&self.profiles.b);
        for id in &self.polynomials {
            out.extend_from_slice(id);
        }
        out
    }

    /// Decode a whole descriptor, rejecting unknown tags, counts, noncanonical
    /// constants and trailing bytes before returning any protocol value.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        let bad =
            || SerializationError::InvalidData("noncanonical binary source descriptor".into());
        let mut reader = Cursor::new(bytes);
        let mut header = [0u8; 24];
        reader.read_exact(&mut header)?;
        if header != *b"AkitaBinarySource\x01\x01\x01\x01\x01\x01\x01" {
            return Err(bad());
        }
        let mut dimensions = [0usize; 5];
        for value in &mut dimensions {
            let mut buf = [0; 8];
            reader.read_exact(&mut buf)?;
            *value = usize::try_from(u64::from_le_bytes(buf)).map_err(|_| bad())?;
        }
        let [ell, logical, padded, columns, count] = dimensions;
        if count == 0 || count > MAX_BINARY_POLYNOMIALS {
            return Err(bad());
        }
        let mut q = [0; 4];
        reader.read_exact(&mut q)?;
        if u32::from_le_bytes(q) != BINARY_SOURCE_Q {
            return Err(bad());
        }
        for expected in [
            CANONICAL_IMAGE_WINDOW.0,
            CANONICAL_IMAGE_WINDOW.1,
            ENFORCED_IMAGE_WINDOW.0,
            ENFORCED_IMAGE_WINDOW.1,
            BINARY_RESPONSE_INTERVAL.0,
            BINARY_RESPONSE_INTERVAL.1,
        ] {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;
            if i32::from_le_bytes(buf) != expected {
                return Err(bad());
            }
        }
        let mut cap = [0];
        reader.read_exact(&mut cap)?;
        if cap != [PRIME_POINT_CAP as u8] {
            return Err(bad());
        }
        let mut profiles = BinaryCommitmentProfiles {
            a: [0; 32],
            b: [0; 32],
        };
        reader.read_exact(&mut profiles.a)?;
        reader.read_exact(&mut profiles.b)?;
        let remaining = bytes
            .len()
            .checked_sub(reader.position() as usize)
            .ok_or_else(bad)?;
        if akita_error::checked::product([count, 32]) != Some(remaining) {
            return Err(bad());
        }
        let index = BinaryIndexMap::new(ell, logical, padded, columns).map_err(|_| bad())?;
        let mut polynomials = Vec::with_capacity(count);
        for _ in 0..count {
            let mut id = [0; 32];
            reader.read_exact(&mut id)?;
            polynomials.push(id);
        }
        Self::new(index, polynomials, profiles).map_err(|_| bad())
    }
}
