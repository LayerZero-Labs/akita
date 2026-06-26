//! Joint multilinear extension evaluation for dense binary-sign JL matrices.
//!
//! Implements `J̃(r_J, r_w) = Σ_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` and the partial
//! row-weight table `g[i] = Σ_j eq(r_J,j) J[j,i]` used by the consistency sumcheck.
//!
//! # Production eval (`eval_jl_mle_at`)
//!
//! The fused path materializes the two equality tables, then amortizes a
//! per-byte sign-weight LUT across all matrix rows:
//!
//! 1. Outer loop: each byte-aligned 8-column window of `eq(r_w, ·)`.
//! 2. Build two 16-entry LUTs once from the eight weights (one per nibble).
//! 3. Inner loop: every row does two LUT lookups and one field add into `row_acc[j]`.
//! 4. Scalar tail for `cols % 8`.
//! 5. Finish with `Σ_j eq(r_J,j) · row_acc[j]`.
//!
//! On aarch64 fp128 (256 rows, 64K–256K cols), this LUT path beats the row-major
//! scalar baseline (~3×) and deferred-reduction wide variants we tried (~10–30%
//! slower than LUT). See `benches/jl_mle.rs` (`scalar` vs `lut`).
//!
//! Parallelism partitions byte windows into panels when the `parallel` feature is
//! enabled and `n_rows · cols` is large enough.
//!
//! Naive reference helpers in the `reference` submodule back differential tests.

#![allow(non_snake_case)] // `r_J` matches the spec / paper notation.

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

mod common;
mod lut;
mod reference;
#[cfg(test)]
mod tests;

use crate::jl::panel::byte_aligned_panel_cols;
#[cfg(feature = "parallel")]
use crate::jl::panel::{panel_span, parallel_jl_enabled};
use crate::jl::JlProjectionMatrix;
use common::{
    accumulate_row_weight_range, scatter_row_weight_range, validate_eq_tables, validate_mle_points,
    JlMleLayout,
};
use lut::{accumulate_rows_from_nibble_luts, build_sign_weight_lut_16};

pub use reference::{
    build_jl_row_weights_reference, eval_jl_mle_at_reference, eval_mle_from_weights,
};

/// Prover row-weight table `g` after batching JL rows with `eq(r_J, ·)`.
///
/// The returned vector has length `2^{col_bits}` (padded hypercube); entries
/// beyond `matrix.cols()` are zero.
///
/// # Errors
///
/// Returns an error if `r_J` has the wrong length or table allocation fails.
pub fn build_jl_row_weights<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, None)?;
    let e_j = EqPolynomial::evals(r_J)?;
    build_jl_row_weights_from_row_eq(matrix, &e_j)
}

/// Fused `J̃(r_J, r_w)` given precomputed `eq(r_J, ·)` and `eq(r_w, ·)` tables.
///
/// Bench / differential hook: production callers use [`eval_jl_mle_at`], which
/// builds these tables once per call.
#[doc(hidden)]
pub fn eval_jl_mle_at_from_eq_tables<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
) -> Result<L, AkitaError> {
    validate_eq_tables(matrix, e_j, e_w)?;
    Ok(eval_jl_mle_at_from_eq_tables_impl(matrix, e_j, e_w))
}

/// Row-weight table given a precomputed `eq(r_J, ·)` vector.
#[doc(hidden)]
pub fn build_jl_row_weights_from_row_eq<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    if e_j.len() != layout.row_hyper {
        return Err(AkitaError::InvalidSize {
            expected: layout.row_hyper,
            actual: e_j.len(),
        });
    }

    let mut g = vec![L::zero(); layout.col_hyper];
    let cols = matrix.cols();
    let n_rows = matrix.n_rows();
    let panel_cols = byte_aligned_panel_cols(cols);
    let row_eq_total = sum_row_eq(e_j, n_rows);

    fill_row_weights(&mut g, panel_cols, cols, n_rows, matrix, e_j, row_eq_total);
    Ok(g)
}

/// Fused verifier evaluation `J̃(r_J, r_w)` without materializing the weight table.
///
/// Uses the per-4-column sign-weight LUT described in the module docs.
///
/// # Errors
///
/// Returns an error if challenge lengths mismatch the padded row/column hypercube
/// or equality-table allocation would exceed budget.
pub fn eval_jl_mle_at<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let (e_j, e_w) = prepare_eq_tables(matrix, r_J, r_w)?;
    Ok(eval_jl_mle_at_from_eq_tables_impl(matrix, &e_j, &e_w))
}

/// Row-major scalar fused eval with precomputed equality tables.
#[doc(hidden)]
pub fn eval_jl_mle_at_scalar_from_eq_tables<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
) -> Result<L, AkitaError> {
    validate_eq_tables(matrix, e_j, e_w)?;
    let cols = matrix.cols();
    let e_w = &e_w[..cols];
    let n_rows = matrix.n_rows();

    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        return Ok((0..n_rows)
            .into_par_iter()
            .map(|j| e_j[j] * accumulate_row_weight_range(matrix.row_slice(j), 0, cols, e_w))
            .reduce(L::zero, |a, b| a + b));
    }

    let mut total = L::zero();
    for (j, &ej) in e_j.iter().take(n_rows).enumerate() {
        let row_sum = accumulate_row_weight_range(matrix.row_slice(j), 0, cols, e_w);
        total += ej * row_sum;
    }
    Ok(total)
}

/// Row-major scalar fused eval (`benches/jl_mle` baseline).
#[doc(hidden)]
pub fn eval_jl_mle_at_scalar<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let (e_j, e_w) = prepare_eq_tables(matrix, r_J, r_w)?;
    eval_jl_mle_at_scalar_from_eq_tables(matrix, &e_j, &e_w)
}

fn prepare_eq_tables<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<(Vec<L>, Vec<L>), AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, Some(r_w))?;
    Ok((EqPolynomial::evals(r_J)?, EqPolynomial::evals(r_w)?))
}

fn eval_jl_mle_at_from_eq_tables_impl<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
) -> L {
    let n_rows = matrix.n_rows();
    let cols = matrix.cols();
    let e_w = &e_w[..cols];
    let full_cols = cols - (cols & 0b111);
    let tail_cols = cols & 0b111;

    let mut row_acc = vec![L::zero(); n_rows];
    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        row_acc = eval_jl_mle_lut_row_acc_parallel(matrix, e_w, full_cols);
    } else {
        let mut lo_lut = [L::zero(); 16];
        let mut hi_lut = [L::zero(); 16];
        accumulate_lut_byte_range(
            &mut row_acc,
            matrix,
            e_w,
            0,
            full_cols,
            &mut lo_lut,
            &mut hi_lut,
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut lo_lut = [L::zero(); 16];
        let mut hi_lut = [L::zero(); 16];
        accumulate_lut_byte_range(
            &mut row_acc,
            matrix,
            e_w,
            0,
            full_cols,
            &mut lo_lut,
            &mut hi_lut,
        );
    }

    if tail_cols > 0 {
        for (j, acc) in row_acc.iter_mut().enumerate() {
            *acc += accumulate_row_weight_range(
                matrix.row_slice(j),
                full_cols,
                tail_cols,
                &e_w[full_cols..],
            );
        }
    }
    dot_row_eq_weights(e_j, n_rows, &row_acc)
}

#[cfg(feature = "parallel")]
fn eval_jl_mle_lut_row_acc_parallel<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_w: &[L],
    full_cols: usize,
) -> Vec<L> {
    let n_rows = matrix.n_rows();
    let num_bytes = full_cols / 8;
    let bytes_per_panel = panel_span(num_bytes, usize::MAX);
    let num_panels = num_bytes.div_ceil(bytes_per_panel);

    (0..num_panels)
        .into_par_iter()
        .map(|p| {
            let b0 = p * bytes_per_panel;
            let b1 = (b0 + bytes_per_panel).min(num_bytes);
            let mut panel_acc = vec![L::zero(); n_rows];
            let mut lo_lut = [L::zero(); 16];
            let mut hi_lut = [L::zero(); 16];
            accumulate_lut_byte_range(
                &mut panel_acc,
                matrix,
                e_w,
                b0 * 8,
                b1 * 8,
                &mut lo_lut,
                &mut hi_lut,
            );
            panel_acc
        })
        .reduce(
            || vec![L::zero(); n_rows],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

#[inline]
fn accumulate_lut_byte_range<L: FieldCore>(
    row_acc: &mut [L],
    matrix: &JlProjectionMatrix,
    e_w: &[L],
    col0_start: usize,
    col0_end: usize,
    lo_lut: &mut [L; 16],
    hi_lut: &mut [L; 16],
) {
    let mut col0 = col0_start;
    while col0 < col0_end {
        let lo_weights: [L; 4] = [e_w[col0], e_w[col0 + 1], e_w[col0 + 2], e_w[col0 + 3]];
        let hi_weights: [L; 4] = [e_w[col0 + 4], e_w[col0 + 5], e_w[col0 + 6], e_w[col0 + 7]];
        build_sign_weight_lut_16(&lo_weights, lo_lut);
        build_sign_weight_lut_16(&hi_weights, hi_lut);
        accumulate_rows_from_nibble_luts(row_acc, matrix, col0 >> 3, lo_lut, hi_lut);
        col0 += 8;
    }
}

#[inline]
fn dot_row_eq_weights<L: FieldCore>(e_j: &[L], n_rows: usize, row_acc: &[L]) -> L {
    let mut total = L::zero();
    for (j, &ej) in e_j.iter().take(n_rows).enumerate() {
        total += ej * row_acc[j];
    }
    total
}

#[inline]
fn sum_row_eq<L: FieldCore>(e_j: &[L], n_rows: usize) -> L {
    e_j.iter().take(n_rows).copied().sum()
}

fn fill_row_weights<L: FieldCore>(
    g: &mut [L],
    panel_cols: usize,
    cols: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    row_eq_total: L,
) {
    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        g.par_chunks_mut(panel_cols)
            .enumerate()
            .for_each(|(p, g_panel)| {
                scatter_panel(
                    g_panel,
                    p * panel_cols,
                    cols,
                    n_rows,
                    matrix,
                    e_j,
                    row_eq_total,
                );
            });
        return;
    }
    for (p, g_panel) in g.chunks_mut(panel_cols).enumerate() {
        scatter_panel(
            g_panel,
            p * panel_cols,
            cols,
            n_rows,
            matrix,
            e_j,
            row_eq_total,
        );
    }
}

fn scatter_panel<L: FieldCore>(
    g_panel: &mut [L],
    col0: usize,
    cols: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    row_eq_total: L,
) {
    if col0 >= cols {
        return;
    }
    let n = g_panel.len().min(cols - col0);
    let g_active = &mut g_panel[..n];

    if (col0 & 0b111) == 0 {
        scatter_panel_byte_sums(g_active, col0, n_rows, matrix, e_j, row_eq_total);
        return;
    }

    for (j, &w) in e_j.iter().take(n_rows).enumerate() {
        scatter_row_weight_range(g_active, matrix.row_slice(j), col0, w);
    }
}

fn scatter_panel_byte_sums<L: FieldCore>(
    g_active: &mut [L],
    col0: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    row_eq_total: L,
) {
    debug_assert_eq!(col0 & 0b111, 0);

    for byte_col in (0..g_active.len()).step_by(8) {
        let lanes = (g_active.len() - byte_col).min(8);
        let matrix_byte = (col0 + byte_col) >> 3;
        let mut ones = [L::zero(); 8];

        for (j, &w) in e_j.iter().take(n_rows).enumerate() {
            let byte = matrix.row_slice(j)[matrix_byte];
            for (lane, acc) in ones.iter_mut().take(lanes).enumerate() {
                if ((byte >> lane) & 1) != 0 {
                    *acc += w;
                }
            }
        }

        for (lane, &sum_ones) in ones.iter().take(lanes).enumerate() {
            g_active[byte_col + lane] = sum_ones + sum_ones - row_eq_total;
        }
    }
}
