//! Checked `i64` projection kernel used as the correctness oracle.

use akita_field::AkitaError;

use super::bit_to_sign;

/// Add `sign(bit) * value` into `acc` with checked `i64` arithmetic.
#[inline]
fn accum_bit(acc: i64, bit: u8, value: i64) -> Result<i64, AkitaError> {
    let sign = bit_to_sign(bit);
    let term = if sign < 0 {
        value
            .checked_neg()
            .ok_or_else(|| AkitaError::InvalidInput(jl_overflow_msg()))?
    } else {
        value
    };
    acc.checked_add(term)
        .ok_or_else(|| AkitaError::InvalidInput(jl_overflow_msg()))
}

fn jl_overflow_msg() -> String {
    "JL projection coordinate exceeds i64 range".to_string()
}

/// Accumulate one projection coordinate with checked `i64` arithmetic.
pub(crate) fn project_row(row: &[u8], centered: &[i64], cols: usize) -> Result<i64, AkitaError> {
    let full_bytes = cols >> 3;
    let remainder = cols & 0b111;
    let mut coeff_idx = 0usize;
    let mut acc: i64 = 0;

    for &byte in row.iter().take(full_bytes) {
        acc = accum_bit(acc, byte & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 1) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 2) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 3) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 4) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 5) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 6) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_bit(acc, (byte >> 7) & 1, centered[coeff_idx])?;
        coeff_idx += 1;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let bit = (byte >> lane) & 1;
            acc = accum_bit(acc, bit, centered[coeff_idx])?;
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    Ok(acc)
}
