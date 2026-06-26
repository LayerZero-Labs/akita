//! Test and fixture constructors for JL matrices.

use akita_field::AkitaError;

use super::packed_byte::bit_to_sign;
use super::{row_bytes_for, JlProjectionMatrix};

/// Reconstruct a matrix from explicit binary-sign rows.
pub(crate) fn matrix_from_sign_rows(signs: &[Vec<i8>]) -> Result<JlProjectionMatrix, AkitaError> {
    let n_rows = signs.len();
    if n_rows == 0 {
        return Err(AkitaError::InvalidInput(
            "JL matrix requires a non-zero row count".to_string(),
        ));
    }
    let cols = signs[0].len();
    let row_bytes = row_bytes_for(cols)?;
    if signs.iter().any(|row| row.len() != cols) {
        return Err(AkitaError::InvalidInput(
            "JL matrix row length mismatch".to_string(),
        ));
    }

    let packed_len = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(super::jl_geometry_overflow)?;
    let mut packed_rows = vec![0u8; packed_len];
    for (row_idx, row) in signs.iter().enumerate() {
        let row_start = row_idx
            .checked_mul(row_bytes)
            .ok_or_else(super::jl_geometry_overflow)?;
        for (col_idx, &sign) in row.iter().enumerate() {
            let bit: u8 = match sign {
                -1 => 0,
                1 => 1,
                _ => {
                    return Err(AkitaError::InvalidInput(
                        "JL matrix entries must be in {-1, +1}".to_string(),
                    ))
                }
            };
            packed_rows[row_start + (col_idx >> 3)] |= bit << (col_idx & 0b111);
        }
    }

    Ok(JlProjectionMatrix {
        n_rows,
        cols,
        row_bytes,
        packed_rows,
    })
}

/// Binary sign at `(row_idx, col_idx)` for differential tests.
pub(crate) fn matrix_sign_at(
    matrix: &JlProjectionMatrix,
    row_idx: usize,
    col_idx: usize,
) -> Option<i8> {
    if row_idx >= matrix.n_rows() || col_idx >= matrix.cols() {
        return None;
    }
    let bit = (matrix.row_slice(row_idx)[col_idx >> 3] >> (col_idx & 0b111)) & 1;
    Some(bit_to_sign(bit))
}
