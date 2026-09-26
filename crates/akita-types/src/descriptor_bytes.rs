//! Shared canonical byte helpers for Fiat-Shamir descriptor digests.

use crate::layout::SisModulusProfileId;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

/// Descriptor schema version for the in-development transcript preamble.
pub const AKITA_INSTANCE_DESCRIPTOR_VERSION: u32 = 5;

/// Fixed-size Blake2b digest used inside the descriptor.
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

pub(crate) fn sis_modulus_profile_tag(family: SisModulusProfileId) -> u8 {
    match family {
        SisModulusProfileId::Q32Offset99 => 0,
        SisModulusProfileId::Q64Offset59 => 1,
        SisModulusProfileId::Q128OffsetA7F7 => 2,
    }
}

pub(crate) fn blake2b_256(bytes: &[u8]) -> DescriptorDigest {
    type Blake2b256 = Blake2b<U32>;
    let digest = Blake2b256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Hash canonical descriptor bytes with Akita's Blake2b-256 primitive.
///
/// Domain separation and version bytes are owned by the caller's canonical
/// descriptor. This shared primitive prevents catalog and transcript identity
/// code from implementing divergent hash truncation rules.
pub fn digest_descriptor_bytes(bytes: &[u8]) -> DescriptorDigest {
    blake2b_256(bytes)
}
