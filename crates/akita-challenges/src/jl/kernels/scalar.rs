//! Fast scalar projection kernel: branchless binary-sign LUT + unchecked `i32`
//! accumulation over `i8` digits.

use crate::jl::packed_byte::{BINARY_SIGNS_FOR_BYTE, SIGN_LUT_I32};
use crate::jl::MAX_JL_DIGIT;

/// Accumulate one projection coordinate with unchecked `i32` arithmetic.
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
            acc += SIGN_LUT_I32[bit as usize] * i32::from(digits[coeff_idx]);
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    acc
}
