//! Multilinear evaluations of packed ternary projection matrices.

use super::{try_zeroed_vec, TernaryProjectionMatrix};
use crate::EqPolynomial;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

/// Evaluate the zero-padded multilinear extension of a ternary matrix.
pub fn eval_ternary_matrix_mle<F: Field>(
    matrix: &TernaryProjectionMatrix,
    row_point: &[F],
    col_point: &[F],
) -> Result<F, AkitaError> {
    let shape = matrix.shape();
    if row_point.len() != shape.row_num_vars()? {
        return Err(AkitaError::InvalidSize {
            expected: shape.row_num_vars()?,
            actual: row_point.len(),
        });
    }
    if col_point.len() != shape.col_num_vars()? {
        return Err(AkitaError::InvalidSize {
            expected: shape.col_num_vars()?,
            actual: col_point.len(),
        });
    }
    let row_eq = EqPolynomial::evals(row_point)?;
    let col_eq = EqPolynomial::evals(col_point)?;
    eval_ternary_matrix_mle_from_eq_tables(matrix, &row_eq, &col_eq)
}

/// Evaluate a ternary matrix MLE from complete padded equality tables.
///
/// The exact table lengths are validated before any indexing. Entries outside
/// the live matrix rectangle are interpreted as zero.
pub fn eval_ternary_matrix_mle_from_eq_tables<F: Field>(
    matrix: &TernaryProjectionMatrix,
    row_eq: &[F],
    col_eq: &[F],
) -> Result<F, AkitaError> {
    let shape = matrix.shape();
    for (actual, expected) in [
        (row_eq.len(), shape.row_domain_len()?),
        (col_eq.len(), shape.col_domain_len()?),
    ] {
        if actual != expected {
            return Err(AkitaError::InvalidSize { expected, actual });
        }
    }
    let mut total = F::zero();
    for (row, &row_weight) in row_eq.iter().take(shape.rows()).enumerate() {
        let (nonzero, positive) = matrix.row_bitplanes_unchecked(row);
        let mut row_sum = F::zero();
        for (byte_index, (&nonzero_byte, &positive_byte)) in
            nonzero.iter().zip(positive).enumerate()
        {
            let mut live = nonzero_byte;
            while live != 0 {
                let bit = live.trailing_zeros() as usize;
                let col = byte_index * 8 + bit;
                if col < shape.cols() {
                    if positive_byte & (1u8 << bit) != 0 {
                        row_sum += col_eq[col];
                    } else {
                        row_sum -= col_eq[col];
                    }
                }
                live &= live - 1;
            }
        }
        total += row_weight * row_sum;
    }
    Ok(total)
}

/// Partially evaluate the row variables of a ternary matrix.
///
/// The result has the padded column-domain length. Padded columns are zero,
/// making the returned table directly usable as the public linear factor in a
/// projection consistency sum-check.
pub fn build_ternary_column_weights<F: Field>(
    matrix: &TernaryProjectionMatrix,
    row_point: &[F],
) -> Result<Vec<F>, AkitaError> {
    let shape = matrix.shape();
    if row_point.len() != shape.row_num_vars()? {
        return Err(AkitaError::InvalidSize {
            expected: shape.row_num_vars()?,
            actual: row_point.len(),
        });
    }
    let row_eq = EqPolynomial::evals(row_point)?;
    let mut weights = try_zeroed_vec(shape.col_domain_len()?, F::zero())?;
    for (row, &row_weight) in row_eq.iter().take(shape.rows()).enumerate() {
        let (nonzero, positive) = matrix.row_bitplanes_unchecked(row);
        for (byte_index, (&nonzero_byte, &positive_byte)) in
            nonzero.iter().zip(positive).enumerate()
        {
            let mut live = nonzero_byte;
            while live != 0 {
                let bit = live.trailing_zeros() as usize;
                let col = byte_index * 8 + bit;
                if col < shape.cols() {
                    if positive_byte & (1u8 << bit) != 0 {
                        weights[col] += row_weight;
                    } else {
                        weights[col] -= row_weight;
                    }
                }
                live &= live - 1;
            }
        }
    }
    Ok(weights)
}

/// Evaluate the MLE of `I_blocks tensor matrix` without materializing repeated
/// matrix blocks.
///
/// `blocks` must be a power of two, so the block-identity MLE is exactly one
/// equality polynomial. Non-power-of-two block counts need a prefix-aware
/// selector and are deliberately rejected by this foundation primitive.
pub fn eval_power_of_two_block_diagonal_mle<F: Field>(
    matrix: &TernaryProjectionMatrix,
    blocks: usize,
    output_block_point: &[F],
    output_row_point: &[F],
    input_block_point: &[F],
    input_col_point: &[F],
) -> Result<F, AkitaError> {
    if !blocks.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "ternary block-diagonal MLE requires a power-of-two block count".into(),
        ));
    }
    let block_num_vars = checked::ceil_log2(blocks)
        .ok_or_else(|| AkitaError::InvalidInput("ternary block-diagonal domain overflow".into()))?;
    for point in [output_block_point, input_block_point] {
        if point.len() != block_num_vars {
            return Err(AkitaError::InvalidSize {
                expected: block_num_vars,
                actual: point.len(),
            });
        }
    }
    Ok(EqPolynomial::mle(output_block_point, input_block_point)?
        * eval_ternary_matrix_mle(matrix, output_row_point, input_col_point)?)
}
