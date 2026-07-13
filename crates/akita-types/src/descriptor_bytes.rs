//! Shared canonical byte helpers for Fiat-Shamir descriptor digests.

use crate::layout::SisModulusProfileId;

pub(crate) fn push_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

pub(crate) fn push_usize_vec(bytes: &mut Vec<u8>, values: &[usize]) {
    push_usize(bytes, values.len());
    for &value in values {
        push_usize(bytes, value);
    }
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
