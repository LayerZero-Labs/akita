//! Shared canonical byte helpers for Fiat-Shamir descriptor digests.

use crate::layout::SisModulusProfileId;
use jolt_transcript::DigestHasher;
use sha3::Digest;

/// Descriptor schema version for the in-development transcript preamble.
pub const AKITA_INSTANCE_DESCRIPTOR_VERSION: u32 = 6;

/// Fixed-size digest used inside the descriptor.
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

/// Hash canonical descriptor bytes with Jolt's digest hash
/// ([`jolt_transcript::DIGEST_HASH`]).
///
/// SHA3-256 by default. An application whose recursion guest checks digests
/// with Jolt's `jolt-inlines-keccak256` inline selects Keccak-256 (pad byte
/// `0x01`) with the `keccak256-digests` feature; the session tag names the
/// selection. The transcript sponge feature does not change it. Domain
/// separation and version bytes are owned by the caller's canonical
/// descriptor. This shared primitive prevents catalog and transcript identity
/// code from implementing divergent hash truncation rules.
pub fn digest_descriptor_bytes(bytes: &[u8]) -> DescriptorDigest {
    let digest = DigestHasher::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

#[cfg(test)]
mod tests {
    use jolt_transcript::{DigestHash, DIGEST_HASH};

    use super::digest_descriptor_bytes;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn descriptor_digest_is_the_selected_hash() {
        // FIPS 202 SHA3-256 (pad byte 0x06) by default; original Keccak-256
        // (pad byte 0x01) under `keccak256-digests`.
        let (empty, abc) = match DIGEST_HASH {
            DigestHash::Sha3_256 => (
                "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
                "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532",
            ),
            DigestHash::Keccak256 => (
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
                "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
            ),
        };
        assert_eq!(hex(&digest_descriptor_bytes(b"")), empty);
        assert_eq!(hex(&digest_descriptor_bytes(b"abc")), abc);
    }
}
