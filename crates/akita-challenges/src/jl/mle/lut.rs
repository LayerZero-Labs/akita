//! Per-byte sign-weight LUT helpers for fused JL MLE evaluation.
//!
//! A packed binary JL byte covers eight matrix columns, so the production LUT
//! has exactly 256 entries: one signed sum for each byte pattern.

use akita_field::FieldCore;

use crate::jl::JlProjectionMatrix;

use super::common::accum_sign_weight;

/// Build a direct 256-entry byte LUT from eight `eq_w` weights in a column window.
#[inline]
pub(super) fn build_sign_weight_lut_256<L: FieldCore>(weights: &[L; 8], lut256: &mut [L; 256]) {
    for (byte, out) in lut256.iter_mut().enumerate() {
        let mut acc = L::zero();
        for (lane, &weight) in weights.iter().enumerate() {
            let sign = if ((byte >> lane) & 1) == 0 { -1 } else { 1 };
            acc = accum_sign_weight(acc, sign, weight);
        }
        *out = acc;
    }
}

/// Reference builder: expand every packed-byte pattern via [`BINARY_SIGNS_FOR_BYTE`].
#[cfg(test)]
pub(super) fn build_sign_weight_lut_256_reference<L: FieldCore>(
    weights: &[L; 8],
    lut256: &mut [L; 256],
) {
    use crate::jl::kernels::BINARY_SIGNS_FOR_BYTE;

    for (b, signs) in BINARY_SIGNS_FOR_BYTE.iter().enumerate() {
        let mut acc = L::zero();
        for (&sign, &weight) in signs.iter().zip(weights.iter()) {
            acc = accum_sign_weight(acc, sign, weight);
        }
        lut256[b] = acc;
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
    fn lut256_binary_matches_reference() {
        let weights = [
            F::from_u64(5),
            F::from_u64(9),
            F::from_u64(17),
            F::from_u64(23),
            F::from_u64(31),
            F::from_u64(37),
            F::from_u64(41),
            F::from_u64(43),
        ];
        let mut fast = [F::zero(); 256];
        let mut reference = [F::zero(); 256];
        build_sign_weight_lut_256(&weights, &mut fast);
        build_sign_weight_lut_256_reference(&weights, &mut reference);
        assert_eq!(fast, reference);
    }
}
