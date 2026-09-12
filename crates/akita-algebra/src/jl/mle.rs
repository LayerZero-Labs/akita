//! Multilinear evaluations of packed ternary projection matrices.

use super::{
    try_zeroed_vec, TernaryProjectionMatrix, TernaryProjectionShape, TERNARY_NIBBLE_DECODE,
};
use crate::EqPolynomial;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

#[cfg(target_arch = "x86_64")]
mod x86_64;

const TERNARY4_PATTERN_COUNT: usize = 81;
const SELECTORS_TO_TERNARY4: [u8; 256] = selectors_to_ternary4();

// M4 Max measurements show that Rayon starts winning at 4K columns. At that
// boundary smaller panels expose enough work to all host threads; larger
// inputs use coarser panels to amortize scheduling, scratch, and reduction.
#[cfg(feature = "parallel")]
const PARALLEL_COLUMN_WEIGHT_MIN_COLS: usize = 1 << 12;
#[cfg(feature = "parallel")]
const SMALL_COLUMN_WEIGHT_PANEL_COLS: usize = 1 << 8;
#[cfg(feature = "parallel")]
const LARGE_COLUMN_WEIGHT_PANEL_COLS: usize = 1 << 10;
#[cfg(feature = "parallel")]
const PARALLEL_CONTRACTION_MIN_COLS: usize = 1 << 12;
#[cfg(feature = "parallel")]
const SMALL_CONTRACTION_PANEL_GROUPS: usize = 64;
#[cfg(feature = "parallel")]
const LARGE_CONTRACTION_PANEL_GROUPS: usize = 256;
#[cfg(feature = "parallel")]
const LARGE_PANEL_MIN_COLS: usize = 1 << 14;

impl TernaryProjectionShape {
    /// Upper bound on temporary field elements for a column-to-row contraction.
    ///
    /// Excludes the caller's input and output. Covers the parallel path even
    /// when admission and execution use different Rayon pools.
    pub fn field_contraction_scratch_len(self) -> Result<usize, AkitaError> {
        #[cfg(feature = "parallel")]
        if self.cols() >= PARALLEL_CONTRACTION_MIN_COLS {
            let panels =
                checked::div_ceil(self.col_groups(), contraction_panel_groups(self.cols()))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("ternary contraction panel overflow".into())
                    })?;
            return checked::product([panels, self.rows()]).ok_or_else(|| {
                AkitaError::InvalidInput("ternary contraction scratch overflow".into())
            });
        }
        Ok(0)
    }

    /// Upper bound on temporary field elements for row-variable contraction.
    ///
    /// Excludes the returned column-weight table. Counts the row equality
    /// table, its construction frontier conservatively, and all row lookup
    /// tables. Small bounded stack temporaries are not materialized tables.
    pub fn column_weight_scratch_len(self) -> Result<usize, AkitaError> {
        let row_tables = checked::product([2, self.row_domain_len()?]).ok_or_else(|| {
            AkitaError::InvalidInput("ternary row-table workspace overflow".into())
        })?;
        let lut_elements =
            checked::product([column_weight_lut_count(self)?, TERNARY4_PATTERN_COUNT]).ok_or_else(
                || AkitaError::InvalidInput("ternary row-lookup workspace overflow".into()),
            )?;
        checked::sum([row_tables, lut_elements]).ok_or_else(|| {
            AkitaError::InvalidInput("ternary column-weight workspace overflow".into())
        })
    }

    /// Upper bound on materialized field elements for a complete matrix MLE.
    ///
    /// Includes equality-table construction frontiers, row accumulation, and
    /// the largest supported parallel contraction buffer; excludes input points.
    pub fn matrix_mle_scratch_len(self) -> Result<usize, AkitaError> {
        let rows = self.row_domain_len()?;
        let cols = self.col_domain_len()?;
        let row_build = checked::product([2, rows]);
        let col_build =
            checked::product([2, cols]).and_then(|columns| checked::sum([rows, columns]));
        let contraction = checked::sum([
            rows,
            cols,
            self.rows(),
            self.field_contraction_scratch_len()?,
        ]);
        match (row_build, col_build, contraction) {
            (Some(row), Some(col), Some(contract)) => Ok(row.max(col).max(contract)),
            _ => Err(AkitaError::InvalidInput(
                "ternary matrix-MLE workspace overflow".into(),
            )),
        }
    }
}

#[cfg(feature = "parallel")]
fn contraction_panel_groups(cols: usize) -> usize {
    if cols < LARGE_PANEL_MIN_COLS {
        SMALL_CONTRACTION_PANEL_GROUPS
    } else {
        LARGE_CONTRACTION_PANEL_GROUPS
    }
}

fn column_weight_lut_count(shape: TernaryProjectionShape) -> Result<usize, AkitaError> {
    if shape.rows() >= 4 && shape.cols() >= 128 {
        checked::div_ceil(shape.rows(), 4)
            .ok_or_else(|| AkitaError::InvalidInput("ternary row-table count overflow".into()))
    } else {
        Ok(0)
    }
}

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

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && shape.cols() >= PARALLEL_CONTRACTION_MIN_COLS {
        use rayon::prelude::*;

        let panel_groups = contraction_panel_groups(shape.cols());
        let scratch_len = shape.field_contraction_scratch_len()?;
        if let Ok(mut scratch) = try_zeroed_vec(scratch_len, F::zero()) {
            scratch
                .par_chunks_mut(shape.rows())
                .enumerate()
                .for_each(|(panel, panel_acc)| {
                    let group_start = panel * panel_groups;
                    let group_end = (group_start + panel_groups).min(shape.col_groups());
                    contract_column_group_range(
                        matrix,
                        col_weights,
                        panel_acc,
                        group_start..group_end,
                    );
                });
            row_acc.fill(F::zero());
            for panel_acc in scratch.chunks_exact(shape.rows()) {
                for (output, &partial) in row_acc.iter_mut().zip(panel_acc) {
                    *output += partial;
                }
            }
            return Ok(());
        }
    }

    row_acc.fill(F::zero());
    contract_column_group_range(matrix, col_weights, row_acc, 0..shape.col_groups());
    Ok(())
}

fn contract_column_group_range<F: Field>(
    matrix: &TernaryProjectionMatrix,
    col_weights: &[F],
    row_acc: &mut [F],
    groups: std::ops::Range<usize>,
) {
    let shape = matrix.shape();
    #[cfg(target_arch = "x86_64")]
    let row_kernel = x86_64::selected_row_kernel::<F>(shape.rows() / 2);
    for group in groups {
        let col_start = group * 4;
        let live = (shape.cols() - col_start).min(4);
        let mut weights = [F::zero(); 4];
        weights[..live].copy_from_slice(&col_weights[col_start..col_start + live]);
        let lut = build_ternary4_weight_lut(&weights);
        let (first_signs, second_signs) = matrix.sign_groups_unchecked(group);
        #[cfg(target_arch = "x86_64")]
        let first_scalar_pair = row_kernel.map_or(0, |kernel| {
            // SAFETY: selection checked every feature required by this exact
            // target-feature function. Matrix construction guarantees equal,
            // complete sign planes and the caller validated the row output.
            unsafe { kernel(first_signs, second_signs, row_acc, &lut) }
        });
        #[cfg(not(target_arch = "x86_64"))]
        let first_scalar_pair = 0;
        accumulate_selector_pairs(first_signs, second_signs, row_acc, &lut, first_scalar_pair);
    }
}

fn accumulate_selector_pairs<F: Field>(
    first_signs: &[u8],
    second_signs: &[u8],
    row_acc: &mut [F],
    lut: &[F; TERNARY4_PATTERN_COUNT],
    first_pair: usize,
) {
    for ((rows, &first), &second) in row_acc
        .chunks_mut(2)
        .zip(first_signs)
        .zip(second_signs)
        .skip(first_pair)
    {
        let even_selectors = usize::from((first & 0x0f) | ((second & 0x0f) << 4));
        rows[0] += lut[usize::from(SELECTORS_TO_TERNARY4[even_selectors])];
        if let Some(odd) = rows.get_mut(1) {
            let odd_selectors = usize::from((first >> 4) | (second & 0xf0));
            *odd += lut[usize::from(SELECTORS_TO_TERNARY4[odd_selectors])];
        }
    }
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
    // Transpose four rows of selectors at a time. Their 81 possible weighted
    // sums are shared by every column, including all parallel panels.
    let count = column_weight_lut_count(shape)?;
    let row_luts = if count != 0 {
        try_zeroed_vec(count, [F::zero(); TERNARY4_PATTERN_COUNT])
            .ok()
            .map(|mut tables| {
                for (table, rows) in tables.iter_mut().zip(row_eq[..shape.rows()].chunks(4)) {
                    let mut values = [F::zero(); 4];
                    values[..rows.len()].copy_from_slice(rows);
                    *table = build_ternary4_weight_lut(&values);
                }
                tables
            })
    } else {
        None
    };

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && shape.cols() >= PARALLEL_COLUMN_WEIGHT_MIN_COLS {
        use rayon::prelude::*;

        let panel_cols = if shape.cols() < LARGE_PANEL_MIN_COLS {
            SMALL_COLUMN_WEIGHT_PANEL_COLS
        } else {
            LARGE_COLUMN_WEIGHT_PANEL_COLS
        };
        weights[..shape.cols()]
            .par_chunks_mut(panel_cols)
            .enumerate()
            .for_each(|(panel, panel_weights)| {
                accumulate_column_weight_groups(
                    matrix,
                    &row_eq,
                    row_luts.as_deref(),
                    panel * (panel_cols / 4),
                    panel_weights,
                );
            });
        return Ok(weights);
    }

    accumulate_column_weight_groups(
        matrix,
        &row_eq,
        row_luts.as_deref(),
        0,
        &mut weights[..shape.cols()],
    );
    Ok(weights)
}

fn accumulate_column_weight_groups<F: Field>(
    matrix: &TernaryProjectionMatrix,
    row_eq: &[F],
    row_luts: Option<&[[F; TERNARY4_PATTERN_COUNT]]>,
    first_group: usize,
    output: &mut [F],
) {
    let shape = matrix.shape();
    for (local_group, group_weights) in output.chunks_mut(4).enumerate() {
        let group = first_group + local_group;
        let (first_signs, second_signs) = matrix.sign_groups_unchecked(group);
        if let Some(tables) = row_luts {
            for ((first, second), table) in first_signs
                .chunks(2)
                .zip(second_signs.chunks(2))
                .zip(tables)
            {
                let first = transpose_four_rows(u16::from_le_bytes([
                    first[0],
                    first.get(1).copied().unwrap_or_default(),
                ]));
                let second = transpose_four_rows(u16::from_le_bytes([
                    second[0],
                    second.get(1).copied().unwrap_or_default(),
                ]));
                for (col, value) in group_weights.iter_mut().enumerate() {
                    let selectors =
                        ((first >> (4 * col)) & 15) | (((second >> (4 * col)) & 15) << 4);
                    *value += table[usize::from(SELECTORS_TO_TERNARY4[usize::from(selectors)])];
                }
            }
            continue;
        }
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
}

#[inline]
fn transpose_four_rows(mut bits: u16) -> u16 {
    let swap = (bits ^ (bits >> 3)) & 0x0a0a;
    bits ^= swap ^ (swap << 3);
    let swap = (bits ^ (bits >> 6)) & 0x00cc;
    bits ^ swap ^ (swap << 6)
}

#[cfg(test)]
mod tests {
    #[test]
    fn transpose_four_rows_matches_each_entry_exhaustively() {
        for bits in 0..=u16::MAX {
            let transposed = super::transpose_four_rows(bits);
            for row in 0..4 {
                for col in 0..4 {
                    assert_eq!(
                        (bits >> (row * 4 + col)) & 1,
                        (transposed >> (col * 4 + row)) & 1
                    );
                }
            }
        }
    }
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

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn scratch_bounds_follow_kernel_thresholds_and_padded_domains() {
        for rows in [1, 3, 4, 5, 256] {
            for cols in [1, 127, 128, 4095, 4096, 16383, 16384, 65536] {
                let shape = TernaryProjectionShape::new(rows, cols).unwrap();
                let lut_count = if rows >= 4 && cols >= 128 {
                    rows.div_ceil(4)
                } else {
                    0
                };
                assert_eq!(
                    shape.column_weight_scratch_len().unwrap(),
                    2 * rows.next_power_of_two() + 81 * lut_count,
                );
                #[cfg(feature = "parallel")]
                let panels = if cols < 4096 {
                    0
                } else if cols < 16384 {
                    cols.div_ceil(256)
                } else {
                    cols.div_ceil(1024)
                };
                #[cfg(not(feature = "parallel"))]
                let panels = 0;
                assert_eq!(
                    shape.field_contraction_scratch_len().unwrap(),
                    panels * rows
                );
                let mle_bound = shape.matrix_mle_scratch_len().unwrap();
                assert!(mle_bound >= 2 * rows.next_power_of_two());
                assert!(mle_bound >= rows.next_power_of_two() + 2 * cols.next_power_of_two());
                assert!(
                    mle_bound
                        >= rows.next_power_of_two()
                            + cols.next_power_of_two()
                            + rows
                            + panels * rows
                );
            }
        }
    }
}
