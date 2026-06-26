//! Shared JL MLE layout helpers and binary-sign accumulation primitives.

use akita_algebra::PaddedHypercube;
use akita_field::{AkitaError, FieldCore};

use crate::jl::packed_byte::{bit_to_sign, BINARY_SIGNS_FOR_BYTE};
use crate::jl::JlProjectionMatrix;

/// Hypercube geometry for JL row/column batching (`eq` tables pad to powers of two).
pub(super) struct JlMleLayout {
    pub row_bits: usize,
    pub col_bits: usize,
    pub row_hyper: usize,
    pub col_hyper: usize,
}

impl JlMleLayout {
    pub(super) fn new(matrix: &JlProjectionMatrix) -> Result<Self, AkitaError> {
        if matrix.n_rows() == 0 || matrix.cols() == 0 {
            return Err(AkitaError::InvalidInput(
                "JL MLE requires a non-empty matrix".to_string(),
            ));
        }
        let row = PaddedHypercube::from_live_len(matrix.n_rows())?;
        let col = PaddedHypercube::from_live_len(matrix.cols())?;
        Ok(Self {
            row_bits: row.log_len,
            col_bits: col.log_len,
            row_hyper: row.padded_len,
            col_hyper: col.padded_len,
        })
    }
}

pub(super) fn validate_eq_tables<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
) -> Result<(), AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    if e_j.len() != layout.row_hyper {
        return Err(AkitaError::InvalidSize {
            expected: layout.row_hyper,
            actual: e_j.len(),
        });
    }
    if e_w.len() < layout.col_hyper {
        return Err(AkitaError::InvalidSize {
            expected: layout.col_hyper,
            actual: e_w.len(),
        });
    }
    Ok(())
}

pub(super) fn validate_mle_points<L: FieldCore>(
    layout: &JlMleLayout,
    r_J: &[L],
    r_w: Option<&[L]>,
) -> Result<(), AkitaError> {
    if r_J.len() != layout.row_bits {
        return Err(AkitaError::InvalidSize {
            expected: layout.row_bits,
            actual: r_J.len(),
        });
    }
    if let Some(r_w) = r_w {
        if r_w.len() != layout.col_bits {
            return Err(AkitaError::InvalidSize {
                expected: layout.col_bits,
                actual: r_w.len(),
            });
        }
    }
    Ok(())
}

#[inline]
pub(super) fn matrix_sign_at(matrix: &JlProjectionMatrix, row: usize, col: usize) -> i8 {
    debug_assert!(row < matrix.n_rows() && col < matrix.cols());
    let bit = (matrix.row_slice(row)[col >> 3] >> (col & 0b111)) & 1;
    bit_to_sign(bit)
}

#[inline]
pub(super) fn accum_sign_weight<L: FieldCore>(acc: L, sign: i8, weight: L) -> L {
    debug_assert!(sign == 1 || sign == -1);
    if sign < 0 {
        acc - weight
    } else {
        acc + weight
    }
}

/// Visit each `(lane_idx, sign)` pair for `n_cols` columns starting at `col0`.
pub(super) fn walk_signs_in_range(
    row: &[u8],
    col0: usize,
    n_cols: usize,
    mut visit: impl FnMut(usize, i8),
) {
    let col_end = col0 + n_cols;
    let mut col = col0;
    let mut idx = 0usize;

    let misalign = col & 0b111;
    if misalign > 0 {
        let lane_limit = (8 - misalign).min(n_cols);
        let byte = row[col >> 3];
        for lane in misalign..misalign + lane_limit {
            visit(idx, bit_to_sign((byte >> lane) & 1));
            idx += 1;
            col += 1;
        }
    }

    let remaining = col_end - col;
    let full_bytes = remaining >> 3;
    let byte_base = col >> 3;
    for b in 0..full_bytes {
        let signs = &BINARY_SIGNS_FOR_BYTE[row[byte_base + b] as usize];
        for &sign in signs {
            visit(idx, sign);
            idx += 1;
        }
        col += 8;
    }

    let tail = col_end - col;
    if tail > 0 {
        let byte = row[col >> 3];
        for lane in 0..tail {
            visit(idx, bit_to_sign((byte >> lane) & 1));
            idx += 1;
        }
    }
    debug_assert_eq!(idx, n_cols);
}

/// Accumulate `Σ_{k=0}^{n_cols-1} sign(row, col0+k) · weights[k]`.
pub(super) fn accumulate_row_weight_range<L: FieldCore>(
    row: &[u8],
    col0: usize,
    n_cols: usize,
    weights: &[L],
) -> L {
    debug_assert_eq!(weights.len(), n_cols);
    let mut acc = L::zero();
    walk_signs_in_range(row, col0, n_cols, |k, sign| {
        acc = accum_sign_weight(acc, sign, weights[k]);
    });
    acc
}

/// Scatter `weight` into `g[k] += sign(row, col0+k) · weight`.
pub(super) fn scatter_row_weight_range<L: FieldCore>(
    g: &mut [L],
    row: &[u8],
    col0: usize,
    weight: L,
) {
    let n_cols = g.len();
    walk_signs_in_range(row, col0, n_cols, |k, sign| {
        g[k] = accum_sign_weight(g[k], sign, weight);
    });
}
