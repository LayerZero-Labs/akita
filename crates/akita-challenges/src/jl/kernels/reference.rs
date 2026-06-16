//! Checked `i64` projection kernel used as the correctness oracle.

use akita_field::AkitaError;

use super::pair_to_sign;

/// Add `sign(pair) * value` into `acc` with checked `i64` arithmetic.
#[inline]
fn accum_pair(acc: i64, pair: u8, value: i64) -> Result<i64, AkitaError> {
    let sign = pair_to_sign(pair);
    if sign == 0 {
        return Ok(acc);
    }
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
    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let mut coeff_idx = 0usize;
    let mut acc: i64 = 0;

    for &byte in row.iter().take(full_bytes) {
        acc = accum_pair(acc, byte & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_pair(acc, (byte >> 2) & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_pair(acc, (byte >> 4) & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_pair(acc, (byte >> 6) & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            acc = accum_pair(acc, pair, centered[coeff_idx])?;
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    Ok(acc)
}
