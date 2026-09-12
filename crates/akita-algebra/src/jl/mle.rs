//! Multilinear evaluations of packed ternary projection matrices.

use super::{try_zeroed_vec, TernaryProjectionMatrix, TERNARY_NIBBLE_DECODE};
use crate::EqPolynomial;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

const TERNARY4_PATTERN_COUNT: usize = 81;
const SELECTORS_TO_TERNARY4: [u8; 256] = selectors_to_ternary4();

const fn selectors_to_ternary4() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut selectors = 0usize;
    while selectors < table.len() {
        let mut index = 0usize;
        let mut lane = 0usize;
        while lane < 4 {
            let first = (selectors >> lane) & 1;
            let second = (selectors >> (lane + 4)) & 1;
            index = index * 3 + first + second;
            lane += 1;
        }
        table[selectors] = index as u8;
        selectors += 1;
    }
    table
}

/// Contract live matrix columns into every live row.
///
/// Four column weights define only 81 distinct balanced-ternary sums. The
/// canonical two-plane selectors choose one of those sums for each row, so the
/// matrix scan performs one field lookup and addition per four entries. Column
/// padding is assigned zero weight because its canonical `00` bits would
/// otherwise decode as `-1`.
pub(super) fn contract_columns_to_rows<F: Field>(
    matrix: &TernaryProjectionMatrix,
    col_weights: &[F],
    row_acc: &mut [F],
) -> Result<(), AkitaError> {
    let shape = matrix.shape();
    if col_weights.len() < shape.cols() {
        return Err(AkitaError::InvalidSize {
            expected: shape.cols(),
            actual: col_weights.len(),
        });
    }
    if row_acc.len() != shape.rows() {
        return Err(AkitaError::InvalidSize {
            expected: shape.rows(),
            actual: row_acc.len(),
        });
    }

    row_acc.fill(F::zero());
    for group in 0..shape.col_groups() {
        let col_start = group * 4;
        let live = (shape.cols() - col_start).min(4);
        let mut weights = [F::zero(); 4];
        weights[..live].copy_from_slice(&col_weights[col_start..col_start + live]);
        let lut = build_ternary4_weight_lut(&weights);
        let (first_signs, second_signs) = matrix.sign_groups_unchecked(group);

        for ((rows, &first), &second) in row_acc.chunks_mut(2).zip(first_signs).zip(second_signs) {
            let even_selectors = usize::from((first & 0x0f) | ((second & 0x0f) << 4));
            rows[0] += lut[usize::from(SELECTORS_TO_TERNARY4[even_selectors])];
            if let Some(odd) = rows.get_mut(1) {
                let odd_selectors = usize::from((first >> 4) | (second & 0xf0));
                *odd += lut[usize::from(SELECTORS_TO_TERNARY4[odd_selectors])];
            }
        }
    }
    Ok(())
}

#[inline]
fn build_ternary4_weight_lut<F: Field>(weights: &[F; 4]) -> [F; TERNARY4_PATTERN_COUNT] {
    let mut current = [F::zero(); TERNARY4_PATTERN_COUNT];
    let mut next = [F::zero(); TERNARY4_PATTERN_COUNT];
    let mut active = 1usize;
    for &weight in weights {
        for (previous, &value) in current[..active].iter().enumerate() {
            let output = previous * 3;
            next[output] = value - weight;
            next[output + 1] = value;
            next[output + 2] = value + weight;
        }
        active *= 3;
        std::mem::swap(&mut current, &mut next);
    }
    current
}

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
    let mut row_acc = try_zeroed_vec(shape.rows(), F::zero())?;
    contract_columns_to_rows(matrix, col_eq, &mut row_acc)?;
    Ok(row_eq
        .iter()
        .zip(row_acc)
        .take(shape.rows())
        .fold(F::zero(), |total, (&row_weight, row_sum)| {
            total + row_weight * row_sum
        }))
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
    for group in 0..shape.col_groups() {
        let col_start = group * 4;
        let live = (shape.cols() - col_start).min(4);
        let group_weights = &mut weights[col_start..col_start + live];
        let (first_signs, second_signs) = matrix.sign_groups_unchecked(group);
        for ((row_weights, &first), &second) in row_eq[..shape.rows()]
            .chunks(2)
            .zip(first_signs)
            .zip(second_signs)
        {
            let even_selectors = usize::from((first & 0x0f) | ((second & 0x0f) << 4));
            let even_signs = TERNARY_NIBBLE_DECODE[even_selectors].to_ne_bytes();
            accumulate_row_weight(group_weights, &even_signs, row_weights[0]);

            if let Some(&odd_weight) = row_weights.get(1) {
                let odd_selectors = usize::from((first >> 4) | (second & 0xf0));
                let odd_signs = TERNARY_NIBBLE_DECODE[odd_selectors].to_ne_bytes();
                accumulate_row_weight(group_weights, &odd_signs, odd_weight);
            }
        }
    }
    Ok(weights)
}

fn validate_block_tensor_shape(
    live_len: usize,
    blocks: usize,
    inner_len: usize,
) -> Result<(), AkitaError> {
    if !blocks.is_power_of_two() || !inner_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "JL tensor table requires power-of-two block and inner dimensions".into(),
        ));
    }
    let expected = checked::product([blocks, inner_len])
        .ok_or_else(|| AkitaError::InvalidInput("JL tensor live length overflow".into()))?;
    if live_len != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: live_len,
        });
    }
    Ok(())
}

/// Evaluate a compact block-major table in canonical JL tensor-axis order.
pub fn eval_block_tensor_mle<F: Field>(
    live: &[F],
    blocks: usize,
    inner_len: usize,
    point: &[F],
) -> Result<F, AkitaError> {
    let block_vars = checked::ceil_log2(blocks)
        .ok_or_else(|| AkitaError::InvalidInput("JL tensor block domain overflow".into()))?;
    let inner_vars = checked::ceil_log2(inner_len)
        .ok_or_else(|| AkitaError::InvalidInput("JL tensor inner domain overflow".into()))?;
    let expected = checked::sum([block_vars, inner_vars])
        .ok_or_else(|| AkitaError::InvalidInput("JL tensor point dimension overflow".into()))?;
    if point.len() != expected {
        return Err(AkitaError::InvalidPointDimension {
            expected,
            actual: point.len(),
        });
    }
    validate_block_tensor_shape(live.len(), blocks, inner_len)?;
    crate::poly::multilinear_eval(live, point)
}

/// Build `eq(r_b, b) * J~(r_u, v)` with `v` as the low-order axis.
pub fn build_block_projection_weight_table<F: Field>(
    matrix: &TernaryProjectionMatrix,
    blocks: usize,
    output_point: &[F],
) -> Result<Vec<F>, AkitaError> {
    if !blocks.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "JL repeated-block weights require power-of-two blocks".into(),
        ));
    }
    let block_vars = checked::ceil_log2(blocks)
        .ok_or_else(|| AkitaError::InvalidInput("JL block domain overflow".into()))?;
    let row_vars = matrix.shape().row_num_vars()?;
    let expected = checked::sum([block_vars, row_vars])
        .ok_or_else(|| AkitaError::InvalidInput("JL output point dimension overflow".into()))?;
    if output_point.len() != expected {
        return Err(AkitaError::InvalidPointDimension {
            expected,
            actual: output_point.len(),
        });
    }
    let (row_point, block_point) = output_point.split_at(row_vars);
    let block_eq = EqPolynomial::evals(block_point)?;
    let column_weights = build_ternary_column_weights(matrix, row_point)?;
    let len = checked::product([blocks, column_weights.len()])
        .ok_or_else(|| AkitaError::InvalidInput("JL weight table length overflow".into()))?;
    let mut weights = super::try_zeroed_vec(len, F::zero())?;
    for (block_slots, &block_weight) in weights
        .chunks_exact_mut(column_weights.len())
        .zip(block_eq.iter())
    {
        for (slot, &column_weight) in block_slots.iter_mut().zip(column_weights.iter()) {
            *slot = block_weight * column_weight;
        }
    }
    Ok(weights)
}

#[inline]
fn accumulate_row_weight<F: Field>(output: &mut [F], signs: &[u8; 4], row_weight: F) {
    for (column, &sign) in output.iter_mut().zip(signs) {
        match sign as i8 {
            -1 => *column -= row_weight,
            1 => *column += row_weight,
            _ => {}
        }
    }
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

/// Evaluate the terminal factor of one block projection reduction.
///
/// Complete points use the canonical compact-vector bit order: local row or
/// column variables first, followed by block variables. This is the single
/// checked splitter shared by prover and verifier.
pub fn eval_power_of_two_block_projection_reduction_factor<F: Field>(
    matrix: &TernaryProjectionMatrix,
    blocks: usize,
    output_point: &[F],
    input_point: &[F],
) -> Result<F, AkitaError> {
    let row_vars = matrix.shape().row_num_vars()?;
    let col_vars = matrix.shape().col_num_vars()?;
    let block_vars = checked::ceil_log2(blocks)
        .ok_or_else(|| AkitaError::InvalidInput("ternary block domain overflow".into()))?;
    for (point, local_vars) in [(output_point, row_vars), (input_point, col_vars)] {
        let expected = checked::sum([local_vars, block_vars])
            .ok_or_else(|| AkitaError::InvalidInput("ternary point dimension overflow".into()))?;
        if point.len() != expected {
            return Err(AkitaError::InvalidPointDimension {
                expected,
                actual: point.len(),
            });
        }
    }
    let (output_row, output_block) = output_point.split_at(row_vars);
    let (input_col, input_block) = input_point.split_at(col_vars);
    eval_power_of_two_block_diagonal_mle(
        matrix,
        blocks,
        output_block,
        output_row,
        input_block,
        input_col,
    )
}
