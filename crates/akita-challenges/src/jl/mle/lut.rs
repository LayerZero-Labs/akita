//! Per-byte sign-weight LUT helpers for fused JL MLE evaluation.

use akita_field::FieldCore;

use crate::jl::JlProjectionMatrix;

use super::common::accum_sign_weight;

/// Build a direct 16-entry nibble LUT from four `eq_w` weights.
#[inline]
pub(super) fn build_sign_weight_lut_16<L: FieldCore>(weights: &[L; 4], lut16: &mut [L; 16]) {
    for (bits, out) in lut16.iter_mut().enumerate() {
        let mut acc = L::zero();
        for (lane, &weight) in weights.iter().enumerate() {
            let sign = if ((bits >> lane) & 1) == 0 { -1 } else { 1 };
            acc = accum_sign_weight(acc, sign, weight);
        }
        *out = acc;
    }
}

/// Add two LUT-selected partial row sums per matrix row for `row[..][byte_idx]`.
#[inline]
pub(super) fn accumulate_rows_from_nibble_luts<L: FieldCore>(
    row_acc: &mut [L],
    matrix: &JlProjectionMatrix,
    byte_idx: usize,
    lo_lut: &[L; 16],
    hi_lut: &[L; 16],
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
        row_acc[j] += lo_lut[(b0 & 0x0f) as usize] + hi_lut[(b0 >> 4) as usize];
        row_acc[j + 1] += lo_lut[(b1 & 0x0f) as usize] + hi_lut[(b1 >> 4) as usize];
        row_acc[j + 2] += lo_lut[(b2 & 0x0f) as usize] + hi_lut[(b2 >> 4) as usize];
        row_acc[j + 3] += lo_lut[(b3 & 0x0f) as usize] + hi_lut[(b3 >> 4) as usize];
        row_acc[j + 4] += lo_lut[(b4 & 0x0f) as usize] + hi_lut[(b4 >> 4) as usize];
        row_acc[j + 5] += lo_lut[(b5 & 0x0f) as usize] + hi_lut[(b5 >> 4) as usize];
        row_acc[j + 6] += lo_lut[(b6 & 0x0f) as usize] + hi_lut[(b6 >> 4) as usize];
        row_acc[j + 7] += lo_lut[(b7 & 0x0f) as usize] + hi_lut[(b7 >> 4) as usize];
        j += 8;
    }
    while j < n_rows {
        let b = matrix.row_slice(j)[byte_idx];
        row_acc[j] += lo_lut[(b & 0x0f) as usize] + hi_lut[(b >> 4) as usize];
        j += 1;
    }
}
