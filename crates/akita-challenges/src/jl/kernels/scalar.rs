//! Fast scalar projection kernel: branchless ternary LUT + unchecked `i32`
//! accumulation over `i8` digits.
//!
//! # Safety contract
//!
//! Callers must ensure:
//! - `digits.len() == cols`
//! - every `|digits[i]| <= MAX_JL_DIGIT` (so each digit fits `i8`)
//! - `cols * MAX_JL_DIGIT <= i32::MAX` so each row sum fits `i32`

#[cfg(any(not(target_arch = "aarch64"), test))]
use crate::jl::MAX_JL_DIGIT;

/// Ternary sign for a packed 2-bit pair: `00 -> -1`, `11 -> +1`, `01`/`10 -> 0`.
/// `i32` form, used by the scalar remainder tail.
pub(crate) const SIGN_LUT: [i32; 4] = [-1, 0, 0, 1];

/// `i8` form of [`SIGN_LUT`], consumed by the SIMD kernels.
const SIGN_LUT_I8: [i8; 4] = [-1, 0, 0, 1];

const fn signs_for_byte(byte: u8) -> [i8; 4] {
    [
        SIGN_LUT_I8[(byte & 3) as usize],
        SIGN_LUT_I8[((byte >> 2) & 3) as usize],
        SIGN_LUT_I8[((byte >> 4) & 3) as usize],
        SIGN_LUT_I8[((byte >> 6) & 3) as usize],
    ]
}

const fn build_signs_for_byte_lut() -> [[i8; 4]; 256] {
    let mut lut = [[0i8; 4]; 256];
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

/// Pre-decoded ternary signs for every packed row byte (`256 × 4` `i8`s, 1 KiB).
/// One L1-resident table shared by the scalar, NEON, and x86 SIMD kernels.
pub(crate) static SIGNS_FOR_BYTE: [[i8; 4]; 256] = build_signs_for_byte_lut();

/// Accumulate one projection coordinate with unchecked `i32` arithmetic over
/// `i8` digits.
///
/// On aarch64 the NEON kernel covers every path; this function is retained for
/// differential tests and for non-NEON targets.
#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
pub(super) fn project_row(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    debug_assert_eq!(digits.len(), cols);
    debug_assert!(cols <= i32::MAX as usize / MAX_JL_DIGIT as usize);

    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let mut coeff_idx = 0usize;
    let mut acc = 0i32;

    for &byte in row.iter().take(full_bytes) {
        let signs = &SIGNS_FOR_BYTE[byte as usize];
        // SAFETY: digit-bound + column-count contract guarantees no `i32` overflow.
        acc += i32::from(signs[0]) * i32::from(digits[coeff_idx])
            + i32::from(signs[1]) * i32::from(digits[coeff_idx + 1])
            + i32::from(signs[2]) * i32::from(digits[coeff_idx + 2])
            + i32::from(signs[3]) * i32::from(digits[coeff_idx + 3]);
        coeff_idx += 4;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            acc += SIGN_LUT[pair as usize] * i32::from(digits[coeff_idx]);
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    acc
}
