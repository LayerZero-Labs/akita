//! Fast scalar projection kernel: branchless binary-sign LUT + unchecked `i32`
//! accumulation over `i8` digits.
//!
//! # Safety contract
//!
//! Callers must ensure:
//! - `digits.len() == cols`
//! - every `|digits[i]| <= MAX_JL_DIGIT` (so each digit fits `i8`)
//! - `cols * MAX_JL_DIGIT <= i32::MAX` so each row sum fits `i32`

use crate::jl::MAX_JL_DIGIT;

/// Binary sign for a packed bit: `0 -> -1`, `1 -> +1`.
/// `i32` form, used by the scalar remainder tail.
pub(crate) const SIGN_LUT: [i32; 2] = [-1, 1];

/// `i8` form of [`SIGN_LUT`], consumed by the SIMD kernels.
const SIGN_LUT_I8: [i8; 2] = [-1, 1];

const fn signs_for_byte(byte: u8) -> [i8; 8] {
    [
        SIGN_LUT_I8[(byte & 1) as usize],
        SIGN_LUT_I8[((byte >> 1) & 1) as usize],
        SIGN_LUT_I8[((byte >> 2) & 1) as usize],
        SIGN_LUT_I8[((byte >> 3) & 1) as usize],
        SIGN_LUT_I8[((byte >> 4) & 1) as usize],
        SIGN_LUT_I8[((byte >> 5) & 1) as usize],
        SIGN_LUT_I8[((byte >> 6) & 1) as usize],
        SIGN_LUT_I8[((byte >> 7) & 1) as usize],
    ]
}

const fn build_signs_for_byte_lut() -> [[i8; 8]; 256] {
    let mut lut = [[0i8; 8]; 256];
    let mut byte = 0u8;
    loop {
        lut[byte as usize] = signs_for_byte(byte);
        if byte == 255 {
            break;
        }
        byte += 1;
    }
    lut
}

/// Pre-decoded binary signs for every packed row byte (`256 × 8` `i8`s, 2 KiB).
/// One L1-resident table shared by the scalar, NEON, and x86 SIMD kernels.
pub(crate) static BINARY_SIGNS_FOR_BYTE: [[i8; 8]; 256] = build_signs_for_byte_lut();

/// Accumulate one projection coordinate with unchecked `i32` arithmetic over
/// `i8` digits.
///
/// On aarch64 the NEON kernel covers every path; this function is retained for
/// differential tests and for non-NEON targets.
#[inline]
pub(super) fn project_row(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);
    debug_assert!(cols <= i32::MAX as usize / MAX_JL_DIGIT as usize);

    let full_bytes = cols >> 3;
    let remainder = cols & 0b111;
    let mut coeff_idx = 0usize;
    let mut acc = 0i32;

    for &byte in row.iter().take(full_bytes) {
        let signs = &BINARY_SIGNS_FOR_BYTE[byte as usize];
        // SAFETY: digit-bound + column-count contract guarantees no `i32` overflow.
        acc += i32::from(signs[0]) * i32::from(digits[coeff_idx])
            + i32::from(signs[1]) * i32::from(digits[coeff_idx + 1])
            + i32::from(signs[2]) * i32::from(digits[coeff_idx + 2])
            + i32::from(signs[3]) * i32::from(digits[coeff_idx + 3])
            + i32::from(signs[4]) * i32::from(digits[coeff_idx + 4])
            + i32::from(signs[5]) * i32::from(digits[coeff_idx + 5])
            + i32::from(signs[6]) * i32::from(digits[coeff_idx + 6])
            + i32::from(signs[7]) * i32::from(digits[coeff_idx + 7]);
        coeff_idx += 8;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let bit = (byte >> lane) & 1;
            acc += SIGN_LUT[bit as usize] * i32::from(digits[coeff_idx]);
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    acc
}
