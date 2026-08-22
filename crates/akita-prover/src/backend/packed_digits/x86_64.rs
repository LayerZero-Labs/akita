//! BMI2 extraction with AVX2 signed-byte extension.

use core::arch::x86_64::*;

use super::DIGITS_PER_BLOCK;

#[inline]
pub(super) fn try_decode_full_block(
    encoded: &[u8],
    bit_width: u8,
    output: &mut [i8; DIGITS_PER_BLOCK],
) -> bool {
    if !std::arch::is_x86_feature_detected!("bmi2") || !std::arch::is_x86_feature_detected!("avx2")
    {
        return false;
    }
    unsafe { decode_full_block_bmi2_avx2(encoded, bit_width, output) };
    true
}

#[target_feature(enable = "bmi2,avx2")]
unsafe fn decode_full_block_bmi2_avx2(
    encoded: &[u8],
    bit_width: u8,
    output: &mut [i8; DIGITS_PER_BLOCK],
) {
    debug_assert!(encoded.len() >= usize::from(bit_width) * 8 + 16);
    if bit_width == 8 {
        let first = unsafe { _mm256_loadu_si256(encoded.as_ptr().cast()) };
        let second = unsafe { _mm256_loadu_si256(encoded.as_ptr().add(32).cast()) };
        unsafe {
            _mm256_storeu_si256(output.as_mut_ptr().cast(), first);
            _mm256_storeu_si256(output.as_mut_ptr().add(32).cast(), second);
        }
        return;
    }

    let width = usize::from(bit_width);
    let deposit_mask = u64::from((1u8 << bit_width) - 1) * 0x0101_0101_0101_0101;
    let sign = _mm256_set1_epi8((1u8 << (bit_width - 1)) as i8);
    for half in 0..2 {
        let mut raw_groups = [0u64; 4];
        for (lane, raw) in raw_groups.iter_mut().enumerate() {
            let group = half * 4 + lane;
            let source = unsafe {
                core::ptr::read_unaligned(encoded.as_ptr().add(group * width).cast::<u64>())
            };
            *raw = _pdep_u64(source, deposit_mask);
        }
        let raw = unsafe { _mm256_loadu_si256(raw_groups.as_ptr().cast()) };
        let signed = _mm256_sub_epi8(_mm256_xor_si256(raw, sign), sign);
        unsafe {
            _mm256_storeu_si256(output.as_mut_ptr().add(half * 32).cast(), signed);
        }
    }
}
