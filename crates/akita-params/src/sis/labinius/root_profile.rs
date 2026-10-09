use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::AkitaError;

use super::{LabiniusCoefficientPrime, LabiniusRingDegree, LABINIUS_WIDTH_TABLE_DIGEST};

/// Closed set of binary-root parameter identities. Adding one requires a new variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabiniusRootProfile {
    /// Degree-648/P128, bounded weight 46, signed 16-bit response coefficients.
    D648P128BoundedW46Delta16,
}

impl LabiniusRootProfile {
    /// Every supported root profile.
    pub const ALL: [Self; 1] = [Self::D648P128BoundedW46Delta16];

    /// Stable wire tag; tags are never reassigned.
    pub const fn tag(self) -> u8 {
        match self {
            Self::D648P128BoundedW46Delta16 => 0,
        }
    }

    /// Decode a wire tag, rejecting unknown identities.
    pub fn from_tag(tag: u8) -> Result<Self, AkitaError> {
        match tag {
            0 => Ok(Self::D648P128BoundedW46Delta16),
            _ => Err(AkitaError::InvalidSetup(format!(
                "unknown LaBinius root profile tag {tag}"
            ))),
        }
    }

    /// Binary scalar ring.
    pub const fn scalar_ring(self) -> BinaryScalarRing {
        match self {
            Self::D648P128BoundedW46Delta16 => BinaryScalarRing::Cyclotomic243,
        }
    }

    /// Commitment-ring identity.
    pub const fn ring_degree(self) -> LabiniusRingDegree {
        match self {
            Self::D648P128BoundedW46Delta16 => LabiniusRingDegree::D648,
        }
    }

    /// Exact coefficient-prime identity.
    pub const fn coefficient_prime(self) -> LabiniusCoefficientPrime {
        match self {
            Self::D648P128BoundedW46Delta16 => LabiniusCoefficientPrime::P128OffsetA7F7,
        }
    }

    /// Exact challenge family. Fallible construction preserves the no-panic boundary.
    pub fn challenge_profile(self) -> Result<BinaryChallengeProfile, AkitaError> {
        match self {
            Self::D648P128BoundedW46Delta16 => {
                BinaryChallengeProfile::bounded_weight(self.scalar_ring(), 46)
            }
        }
    }

    /// Full accepted response interval `[L, U]`, including both endpoints.
    pub const fn response_interval(self) -> (i128, i128) {
        match self {
            Self::D648P128BoundedW46Delta16 => (-32_768, 32_767),
        }
    }

    /// Domain-separated, unambiguous identity binding every parameter and table bytes.
    pub fn identity_bytes(self) -> Result<Vec<u8>, AkitaError> {
        let challenge = self.challenge_profile()?;
        Ok(encode_identity(
            self.tag(),
            self.scalar_ring(),
            self.coefficient_prime(),
            self.ring_degree(),
            challenge.identity_bytes(),
            self.response_interval(),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ))
    }
}

fn encode_identity(
    tag: u8,
    scalar_ring: BinaryScalarRing,
    prime: LabiniusCoefficientPrime,
    degree: LabiniusRingDegree,
    challenge: &[u8],
    interval: (i128, i128),
    table_digest: [u8; 32],
) -> Vec<u8> {
    let mut bytes = b"akita/labinius/root-profile/v1\0".to_vec();
    bytes.push(tag);
    bytes.push(match scalar_ring {
        BinaryScalarRing::Cyclotomic243 => 0,
        BinaryScalarRing::Cyclotomic729 => 1,
    });
    bytes.extend_from_slice(&prime.modulus().to_le_bytes());
    bytes.extend_from_slice(&degree.degree().to_le_bytes());
    // Challenge identity is length framed without platform-dependent integers.
    for byte in challenge {
        bytes.extend_from_slice(&[1, *byte]);
    }
    bytes.push(0);
    bytes.extend_from_slice(&interval.0.to_le_bytes());
    bytes.extend_from_slice(&interval.1.to_le_bytes());
    bytes.extend_from_slice(&table_digest);
    bytes
}

#[cfg(test)]
#[path = "root_profile_tests.rs"]
mod tests;
