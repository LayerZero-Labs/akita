#![allow(dead_code)]

use super::*;

#[cfg(test)]
#[must_use]
pub(crate) fn recompose_balanced_i8_digits(digits: &[i8], log_basis: u32) -> i64 {
    let b = 1i128 << log_basis;
    let half_b = 1i128 << (log_basis - 1);
    let mut acc = 0i128;
    let mut pow = 1i128;
    for &digit in digits {
        let mut balanced = i128::from(digit);
        if balanced >= half_b {
            balanced -= b;
        }
        acc += balanced * pow;
        pow *= b;
    }
    acc as i64
}

#[cfg(test)]
#[must_use]
pub(crate) fn balanced_digits_from_i64(value: i64, num_digits: usize, log_basis: u32) -> Vec<i8> {
    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;
    let mut digits = Vec::with_capacity(num_digits);
    let mut c = i128::from(value);
    for _ in 0..num_digits {
        let d = c & mask;
        let balanced = if d >= half_b { d - b } else { d };
        c = (c - balanced) >> log_basis;
        digits.push(balanced as i8);
    }
    digits
}

/// Build Golomb-Rice `z` payload from centered fold-response ring coefficients.
#[cfg(test)]
pub(crate) fn encode_z_segment_from_centered<const D: usize>(
    centered: &[[i32; D]],
    positions_per_block: usize,
    depth_commit: usize,
    rice_low_bits: u32,
    zigzag_w_z: u32,
) -> Result<Vec<u8>, AkitaError> {
    let inner_width = positions_per_block * depth_commit;
    if !centered.len().is_multiple_of(inner_width) {
        return Err(AkitaError::InvalidInput(
            "z_folded length does not match layout".to_string(),
        ));
    }
    let values = centered_rows_to_i64(centered);
    encode_z_segment_from_centered_flat(&values, rice_low_bits, zigzag_w_z)
}

#[cfg(test)]
fn centered_rows_to_i64<const D: usize>(rows: &[[i32; D]]) -> Vec<i64> {
    rows.iter()
        .flat_map(|row| row.iter().map(|&n| i64::from(n)))
        .collect()
}
