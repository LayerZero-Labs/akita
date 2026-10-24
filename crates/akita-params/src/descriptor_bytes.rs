//! Shared canonical byte helpers for Fiat-Shamir descriptor digests.

use crate::layout::SisModulusProfileId;
use sha3::{Digest, Keccak256};

/// Descriptor schema version for the in-development transcript preamble.
pub const AKITA_INSTANCE_DESCRIPTOR_VERSION: u32 = 6;

/// Fixed-size Keccak-256 digest used inside the descriptor.
pub type DescriptorDigest = [u8; 32];

pub(crate) fn push_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

pub(crate) fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_u128(bytes: &mut Vec<u8>, value: u128) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub fn sis_modulus_profile_tag(family: SisModulusProfileId) -> u8 {
    match family {
        SisModulusProfileId::Q32Offset99 => 0,
        SisModulusProfileId::Q64Offset59 => 1,
        SisModulusProfileId::Q128OffsetA7F7 => 2,
    }
}

/// Hash canonical descriptor bytes with Akita's Keccak-256 primitive.
///
/// This is the original Keccak-256 (pad byte `0x01`), the hash Jolt's
/// `jolt-inlines-keccak256` guest inline computes, so a recursion guest checks
/// every bound digest with one hash family. The transcript sponge feature does
/// not change it. Domain separation and version bytes are owned by the
/// caller's canonical descriptor. This shared primitive prevents catalog and transcript identity
/// code from implementing divergent hash truncation rules.
pub fn digest_descriptor_bytes(bytes: &[u8]) -> DescriptorDigest {
    let digest = Keccak256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

#[cfg(test)]
mod tests {
    use super::digest_descriptor_bytes;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn descriptor_digest_is_keccak256() {
        // Original Keccak-256 (pad byte 0x01), not SHA3-256, whose empty-input
        // digest is a7ffc6f8...
        assert_eq!(
            hex(&digest_descriptor_bytes(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
        assert_eq!(
            hex(&digest_descriptor_bytes(b"abc")),
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        );
    }
}
