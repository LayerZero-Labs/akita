use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::AkitaError;

use super::{
    LabiniusCommitmentModulus, LabiniusRingDegree, LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST,
};

/// Closed set of binary-root parameter identities. Adding one requires a new variant.
///
/// A profile fixes the commitment ring, the commitment prime and the fold
/// challenge family. The response interval follows from the geometry
/// ([`super::LabiniusFoldResponse`]) and the proof prime is admitted separately
/// ([`super::LabiniusRootShape::derive_encoding`]); neither is part of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabiniusRootProfile {
    /// Degree-648 ring over the 25-bit commitment prime, bounded weight 46.
    D648Q25BoundedW46,
}

impl LabiniusRootProfile {
    /// Binary scalar ring.
    pub const fn scalar_ring(self) -> BinaryScalarRing {
        match self {
            Self::D648Q25BoundedW46 => BinaryScalarRing::Cyclotomic243,
        }
    }

    /// Commitment-ring identity.
    pub const fn ring_degree(self) -> LabiniusRingDegree {
        match self {
            Self::D648Q25BoundedW46 => LabiniusRingDegree::D648,
        }
    }

    /// Commitment modulus.
    pub const fn commitment_modulus(self) -> LabiniusCommitmentModulus {
        match self {
            Self::D648Q25BoundedW46 => LabiniusCommitmentModulus::Q25Plus14561,
        }
    }

    /// Exact challenge family. Fallible construction preserves the no-panic boundary.
    pub fn challenge_profile(self) -> Result<BinaryChallengeProfile, AkitaError> {
        match self {
            Self::D648Q25BoundedW46 => {
                BinaryChallengeProfile::bounded_weight(self.scalar_ring(), 46)
            }
        }
    }

    /// Domain-separated identity binding every parameter and the bytes of the
    /// certified table: the domain, the scalar ring, `q` and the ring degree as
    /// little-endian `u32`, the length-prefixed challenge identity, and the
    /// table digest.
    pub fn identity_bytes(self) -> Result<Vec<u8>, AkitaError> {
        let challenge = self.challenge_profile()?;
        let challenge = challenge.identity_bytes();
        let challenge_len = u64::try_from(challenge.len()).map_err(|_| {
            AkitaError::InvalidSetup("LaBinius challenge identity exceeds u64".into())
        })?;
        let mut bytes = b"akita/labinius/root-profile/v1\0".to_vec();
        bytes.push(match self.scalar_ring() {
            BinaryScalarRing::Cyclotomic243 => 0,
            BinaryScalarRing::Cyclotomic729 => 1,
        });
        bytes.extend_from_slice(&self.commitment_modulus().modulus().to_le_bytes());
        bytes.extend_from_slice(&self.ring_degree().degree().to_le_bytes());
        bytes.extend_from_slice(&challenge_len.to_le_bytes());
        bytes.extend_from_slice(challenge);
        bytes.extend_from_slice(&LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST);
        Ok(bytes)
    }
}
