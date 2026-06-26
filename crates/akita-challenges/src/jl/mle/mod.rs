//! Joint multilinear extension evaluation for dense binary-sign JL matrices.
//!
//! Implements `J̃(r_J, r_w) = Σ_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` and the partial
//! row-weight table `g[i] = Σ_j eq(r_J,j) J[j,i]` used by the consistency sumcheck.

#![allow(non_snake_case)] // `r_J` matches the spec / paper notation.

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

mod common;
mod lut;
mod reference;
mod row_weights;
#[cfg(test)]
mod tests;

#[cfg(feature = "parallel")]
use crate::jl::panel::panel_span;
#[cfg(feature = "parallel")]
use crate::jl::panel::parallel_jl_enabled;
use crate::jl::JlProjectionMatrix;
use common::{accumulate_row_weight_range, validate_eq_tables, validate_mle_points, JlMleLayout};
use lut::{accumulate_rows_from_nibble_luts, build_sign_weight_lut_16};
use row_weights::{fill_row_weights, row_weight_panel_cols, sum_row_eq};

pub use reference::{
    build_jl_row_weights_reference, eval_jl_mle_at_reference, eval_mle_from_weights,
};

/// Prover row-weight table `g` after batching JL rows with `eq(r_J, ·)`.
pub fn build_jl_row_weights<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, None)?;
    let e_j = EqPolynomial::evals(r_J)?;
    build_jl_row_weights_from_row_eq(matrix, &e_j)
}

/// Row-weight table given a precomputed `eq(r_J, ·)` vector.
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
    let panel_cols = row_weight_panel_cols(cols);
    let row_eq_total = sum_row_eq(e_j, n_rows);
    fill_row_weights(&mut g, panel_cols, cols, n_rows, matrix, e_j, row_eq_total);
    Ok(g)
}

/// Fused verifier evaluation `J̃(r_J, r_w)` without materializing the weight table.
pub fn eval_jl_mle_at<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let (e_j, e_w) = prepare_eq_tables(matrix, r_J, r_w)?;
    Ok(eval_jl_mle_at_from_eq_tables_impl(matrix, &e_j, &e_w))
}

/// Fused `J̃(r_J, r_w)` given precomputed equality tables.
pub fn eval_jl_mle_at_from_eq_tables<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
) -> Result<L, AkitaError> {
    validate_eq_tables(matrix, e_j, e_w)?;
    Ok(eval_jl_mle_at_from_eq_tables_impl(matrix, e_j, e_w))
}

/// Row-major scalar fused eval with precomputed equality tables.
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

    let mut row_acc = {
        #[cfg(feature = "parallel")]
        if parallel_jl_enabled(n_rows, cols) {
            eval_jl_mle_lut_row_acc_parallel(matrix, e_w, full_cols)
        } else {
            accumulate_lut_row_acc_serial(matrix, e_w, full_cols)
        }
        #[cfg(not(feature = "parallel"))]
        {
            accumulate_lut_row_acc_serial(matrix, e_w, full_cols)
        }
    };

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

fn accumulate_lut_row_acc_serial<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_w: &[L],
    full_cols: usize,
) -> Vec<L> {
    let n_rows = matrix.n_rows();
    let mut row_acc = vec![L::zero(); n_rows];
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
    row_acc
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
