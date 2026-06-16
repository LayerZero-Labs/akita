//! Per-4-column sign-weight LUT helpers for fused JL MLE evaluation.
//!
//! [`BYTE_TO_TERNARY4`] indexes the same four-lane sign alphabet as the projection
//! kernels' per-byte decode table, collapsed so `01`/`10` both map to
//! the zero digit before the 81-pattern DP builds field-weight sums.

use akita_field::FieldCore;

use crate::jl::JlProjectionMatrix;

use super::common::accum_sign_weight;

/// Map each packed matrix byte to its canonical 81-pattern index.
pub(super) static BYTE_TO_TERNARY4: [u8; 256] = build_byte_to_ternary4_lut();

const fn pair_to_ternary_digit(pair: u8) -> usize {
    match pair & 0b11 {
        0 => 0,     // 00 -> -1
        1 | 2 => 1, // 01 / 10 -> 0
        3 => 2,     // 11 -> +1
        _ => 1,
    }
}

const fn sign_from_digit(digit: usize) -> i8 {
    match digit {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => 0,
    }
}

const fn ternary4_index_from_byte(byte: u8) -> u8 {
    let d0 = pair_to_ternary_digit(byte & 0b11);
    let d1 = pair_to_ternary_digit((byte >> 2) & 0b11);
    let d2 = pair_to_ternary_digit((byte >> 4) & 0b11);
    let d3 = pair_to_ternary_digit((byte >> 6) & 0b11);
    (d0 * 27 + d1 * 9 + d2 * 3 + d3) as u8
}

const fn build_byte_to_ternary4_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut byte = 0u8;
    loop {
        lut[byte as usize] = ternary4_index_from_byte(byte);
        if byte == 255 {
            break;
        }
        byte += 1;
    }
    lut
}

/// Build the 81 canonical sign-weight sums via axis extension (no per-entry muls).
#[inline]
fn build_sign_weight_lut_81<L: FieldCore>(weights: &[L; 4], lut81: &mut [L; 81]) {
    debug_assert_eq!(lut81.len(), 81);

    let mut layer = [L::zero(); 81];
    for (axis, &weight) in weights.iter().enumerate() {
        let prev_len = 3usize.pow(axis as u32);
        let next_len = 3usize.pow((axis + 1) as u32);
        let mut next = [L::zero(); 81];
        for (prev_idx, &prev) in layer.iter().enumerate().take(prev_len) {
            for digit in 0..3 {
                let out_idx = prev_idx * 3 + digit;
                next[out_idx] = accum_sign_weight(prev, sign_from_digit(digit), weight);
            }
        }
        layer[..next_len].copy_from_slice(&next[..next_len]);
    }
    lut81.copy_from_slice(&layer);
}

/// Build a direct 256-entry byte LUT from four `eq_w` weights in a column window.
#[inline]
pub(super) fn build_sign_weight_lut_256<L: FieldCore>(weights: &[L; 4], lut256: &mut [L; 256]) {
    let mut lut81 = [L::zero(); 81];
    build_sign_weight_lut_81(weights, &mut lut81);
    expand_lut81_to256(&lut81, lut256);
}

/// Reference builder: expand every packed-byte pattern via [`SIGNS_FOR_BYTE`].
#[cfg(test)]
pub(super) fn build_sign_weight_lut_256_reference<L: FieldCore>(
    weights: &[L; 4],
    lut256: &mut [L; 256],
) {
    use crate::jl::kernels::SIGNS_FOR_BYTE;

    for (b, signs) in SIGNS_FOR_BYTE.iter().enumerate() {
        let mut acc = L::zero();
        for (&sign, &weight) in signs.iter().zip(weights.iter()) {
            acc = accum_sign_weight(acc, sign, weight);
        }
        lut256[b] = acc;
    }
}

#[inline]
fn expand_lut81_to256<L: FieldCore>(lut81: &[L; 81], lut256: &mut [L; 256]) {
    for (b, &idx) in BYTE_TO_TERNARY4.iter().enumerate() {
        lut256[b] = lut81[idx as usize];
    }
}

/// Add one LUT-selected partial row sum per matrix row for `row[..][byte_idx]`.
#[inline]
pub(super) fn accumulate_rows_from_byte_lut<L: FieldCore>(
    row_acc: &mut [L],
    matrix: &JlProjectionMatrix,
    byte_idx: usize,
    lut256: &[L; 256],
) {
    let n_rows = row_acc.len();
    let mut j = 0usize;
    while j + 8 <= n_rows {
        let b0 = matrix.row_slice(j)[byte_idx];
        let b1 = matrix.row_slice(j + 1)[byte_idx];
        let b2 = matrix.row_slice(j + 2)[byte_idx];
        let b3 = matrix.row_slice(j + 3)[byte_idx];
        let b4 = matrix.row_slice(j + 4)[byte_idx];
        let b5 = matrix.row_slice(j + 5)[byte_idx];
        let b6 = matrix.row_slice(j + 6)[byte_idx];
        let b7 = matrix.row_slice(j + 7)[byte_idx];
        row_acc[j] += lut256[b0 as usize];
        row_acc[j + 1] += lut256[b1 as usize];
        row_acc[j + 2] += lut256[b2 as usize];
        row_acc[j + 3] += lut256[b3 as usize];
        row_acc[j + 4] += lut256[b4 as usize];
        row_acc[j + 5] += lut256[b5 as usize];
        row_acc[j + 6] += lut256[b6 as usize];
        row_acc[j + 7] += lut256[b7 as usize];
        j += 8;
    }
    while j < n_rows {
        let b = matrix.row_slice(j)[byte_idx];
        row_acc[j] += lut256[b as usize];
        j += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;

    type F = Fp64<4294967197>;

    #[test]
    fn byte_to_ternary4_collapses_zero_pairs() {
        // lanes 01 and 10 both decode to sign 0.
        let byte_01 = 0b00_01_00_01u8;
        let byte_10 = 0b00_10_00_10u8;
        assert_eq!(
            BYTE_TO_TERNARY4[byte_01 as usize],
            BYTE_TO_TERNARY4[byte_10 as usize]
        );
        assert_eq!(
            BYTE_TO_TERNARY4[byte_01 as usize],
            ternary4_index_from_byte(byte_01)
        );
    }

    #[test]
    fn dp_lut81_matches_reference_patterns() {
        let weights = [
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let mut fast = [F::zero(); 81];
        let mut slow = [F::zero(); 81];
        build_sign_weight_lut_81(&weights, &mut fast);

        for (idx, slow_entry) in slow.iter_mut().enumerate().take(81) {
            let d3 = idx % 3;
            let d2 = (idx / 3) % 3;
            let d1 = (idx / 9) % 3;
            let d0 = (idx / 27) % 3;
            let digits = [d0, d1, d2, d3];
            let mut acc = F::zero();
            for (digit, &weight) in digits.into_iter().zip(weights.iter()) {
                acc = accum_sign_weight(acc, sign_from_digit(digit), weight);
            }
            *slow_entry = acc;
        }
        assert_eq!(fast, slow);
    }

    #[test]
    fn lut256_dp_matches_reference() {
        let weights = [
            F::from_u64(5),
            F::from_u64(9),
            F::from_u64(17),
            F::from_u64(23),
        ];
        let mut fast = [F::zero(); 256];
        let mut reference = [F::zero(); 256];
        build_sign_weight_lut_256(&weights, &mut fast);
        build_sign_weight_lut_256_reference(&weights, &mut reference);
        assert_eq!(fast, reference);
    }
}
