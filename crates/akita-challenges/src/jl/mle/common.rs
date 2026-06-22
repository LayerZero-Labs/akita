//! Shared JL MLE layout helpers and ternary accumulation primitives.

use akita_field::{AkitaError, FieldCore};

use crate::jl::kernels::{pair_to_sign, SIGNS_FOR_BYTE};
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
        let row_bits = hypercube_bits(matrix.n_rows());
        let col_bits = hypercube_bits(matrix.cols());
        Ok(Self {
            row_bits,
            col_bits,
            row_hyper: 1usize << row_bits,
            col_hyper: 1usize << col_bits,
        })
    }
}

pub(super) fn hypercube_bits(n: usize) -> usize {
    debug_assert!(n > 0);
    n.next_power_of_two().trailing_zeros() as usize
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
pub(super) fn entry_sign(matrix: &JlProjectionMatrix, row: usize, col: usize) -> i8 {
    if row >= matrix.n_rows() || col >= matrix.cols() {
        return 0;
    }
    let shift = (col & 0b11) << 1;
    let pair = (matrix.row_slice(row)[col >> 2] >> shift) & 0b11;
    pair_to_sign(pair)
}

#[inline]
pub(super) fn accum_sign_weight<L: FieldCore>(acc: L, sign: i8, weight: L) -> L {
    match sign {
        0 => acc,
        1 => acc + weight,
        -1 => acc - weight,
        _ => acc,
    }
}

/// Accumulate `Σ_{k=0}^{n_cols-1} sign(row, col0+k) · weights[k]` with byte-wide LUT decode.
pub(super) fn accumulate_row_weight_range<L: FieldCore>(
    row: &[u8],
    col0: usize,
    n_cols: usize,
    weights: &[L],
) -> L {
    debug_assert_eq!(weights.len(), n_cols);

    let mut acc = L::zero();
    let mut col = col0;
    let mut w_idx = 0usize;
    let col_end = col0 + n_cols;

    let misalign = col & 0b11;
    if misalign > 0 {
        let lane_limit = (4 - misalign).min(n_cols);
        let byte = row[col >> 2];
        for lane in misalign..misalign + lane_limit {
            let pair = (byte >> (lane << 1)) & 0b11;
            acc = accum_sign_weight(acc, pair_to_sign(pair), weights[w_idx]);
            w_idx += 1;
            col += 1;
        }
    }

    let remaining = col_end - col;
    let full_bytes = remaining >> 2;
    let byte_base = col >> 2;
    for b in 0..full_bytes {
        let signs = &SIGNS_FOR_BYTE[row[byte_base + b] as usize];
        for &sign in signs {
            acc = accum_sign_weight(acc, sign, weights[w_idx]);
            w_idx += 1;
        }
        col += 4;
    }

    let tail = col_end - col;
    if tail > 0 {
        let byte = row[col >> 2];
        for lane in 0..tail {
            let pair = (byte >> (lane << 1)) & 0b11;
            acc = accum_sign_weight(acc, pair_to_sign(pair), weights[w_idx]);
            w_idx += 1;
        }
    }

    debug_assert_eq!(w_idx, n_cols);
    acc
}

/// Scatter a single per-row `weight` into `g[k] += sign(row, col0+k) · weight` for
/// `k in 0..g.len()`, decoding signs four-at-a-time with the byte-wide LUT.
///
/// This is the transpose of [`accumulate_row_weight_range`]: the prover's
/// `g[i] = Σ_j eq(r_J, j) · J[j, i]` sweeps rows and scatters each row's scalar
/// weight across its columns, reusing the same packed-byte decode.
pub(super) fn scatter_row_weight_range<L: FieldCore>(
    g: &mut [L],
    row: &[u8],
    col0: usize,
    weight: L,
) {
    let n_cols = g.len();
    let mut col = col0;
    let mut g_idx = 0usize;
    let col_end = col0 + n_cols;

    let misalign = col & 0b11;
    if misalign > 0 {
        let lane_limit = (4 - misalign).min(n_cols);
        let byte = row[col >> 2];
        for lane in misalign..misalign + lane_limit {
            let pair = (byte >> (lane << 1)) & 0b11;
            g[g_idx] = accum_sign_weight(g[g_idx], pair_to_sign(pair), weight);
            g_idx += 1;
            col += 1;
        }
    }

    let remaining = col_end - col;
    let full_bytes = remaining >> 2;
    let byte_base = col >> 2;
    for b in 0..full_bytes {
        let signs = &SIGNS_FOR_BYTE[row[byte_base + b] as usize];
        for &sign in signs {
            g[g_idx] = accum_sign_weight(g[g_idx], sign, weight);
            g_idx += 1;
        }
        col += 4;
    }

    let tail = col_end - col;
    if tail > 0 {
        let byte = row[col >> 2];
        for lane in 0..tail {
            let pair = (byte >> (lane << 1)) & 0b11;
            g[g_idx] = accum_sign_weight(g[g_idx], pair_to_sign(pair), weight);
            g_idx += 1;
        }
    }

    debug_assert_eq!(g_idx, n_cols);
}
