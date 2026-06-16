//! Joint multilinear extension evaluation for dense ternary JL matrices.
//!
//! Implements `J̃(r_J, r_w) = Σ_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` and the partial
//! row-weight table `g[i] = Σ_j eq(r_J,j) J[j,i]` used by the consistency sumcheck.
//!
//! # Production eval (`eval_jl_mle_at`)
//!
//! The fused path materializes the two equality tables, then amortizes a
//! per-4-column sign-weight LUT across all matrix rows:
//!
//! 1. Outer loop: each byte-aligned 4-column window of `eq(r_w, ·)`.
//! 2. Build a 256-entry LUT once from the four weights (81 canonical ternary
//!    patterns via axis extension, remapped through `BYTE_TO_TERNARY4` so
//!    `01`/`10` zero pairs collapse correctly; same four-lane sign alphabet as the
//!    projection kernels' byte decode table).
//! 3. Inner loop: every row does one LUT lookup and one field add into `row_acc[j]`.
//! 4. Scalar tail for `cols % 4`.
//! 5. Finish with `Σ_j eq(r_J,j) · row_acc[j]`.
//!
//! On aarch64 fp128 (256 rows, 64K–256K cols), this LUT path beats the row-major
//! scalar baseline (~3×) and deferred-reduction wide variants we tried (~10–30%
//! slower than LUT). See `benches/jl_mle.rs` (`scalar` vs `lut`).
//!
//! Parallelism partitions quad windows into panels when the `parallel` feature is
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
    accumulate_row_weight_range, scatter_row_weight_range, validate_mle_points, JlMleLayout,
};
use lut::{accumulate_rows_from_byte_lut, build_sign_weight_lut_256};

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
) -> L {
    eval_jl_mle_at_from_eq_tables_impl(matrix, e_j, e_w)
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

    fill_row_weights(&mut g, panel_cols, cols, n_rows, matrix, e_j);
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
) -> L {
    let cols = matrix.cols();
    let e_w = &e_w[..cols];
    let n_rows = matrix.n_rows();

    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        return (0..n_rows)
            .into_par_iter()
            .map(|j| e_j[j] * accumulate_row_weight_range(matrix.row_slice(j), 0, cols, e_w))
            .reduce(L::zero, |a, b| a + b);
    }

    let mut total = L::zero();
    for (j, &ej) in e_j.iter().take(n_rows).enumerate() {
        let row_sum = accumulate_row_weight_range(matrix.row_slice(j), 0, cols, e_w);
        total += ej * row_sum;
    }
    total
}

/// Row-major scalar fused eval (`benches/jl_mle` baseline).
#[doc(hidden)]
pub fn eval_jl_mle_at_scalar<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let (e_j, e_w) = prepare_eq_tables(matrix, r_J, r_w)?;
    Ok(eval_jl_mle_at_scalar_from_eq_tables(matrix, &e_j, &e_w))
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
    let full_cols = cols - (cols & 0b11);
    let tail_cols = cols & 0b11;

    let mut row_acc = vec![L::zero(); n_rows];
    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        row_acc = eval_jl_mle_lut_row_acc_parallel(matrix, e_w, full_cols);
    } else {
        let mut lut = [L::zero(); 256];
        accumulate_lut_quad_range(&mut row_acc, matrix, e_w, 0, full_cols, &mut lut);
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut lut = [L::zero(); 256];
        accumulate_lut_quad_range(&mut row_acc, matrix, e_w, 0, full_cols, &mut lut);
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
    let num_quads = full_cols / 4;
    let quads_per_panel = panel_span(num_quads, usize::MAX);
    let num_panels = num_quads.div_ceil(quads_per_panel);

    (0..num_panels)
        .into_par_iter()
        .map(|p| {
            let q0 = p * quads_per_panel;
            let q1 = (q0 + quads_per_panel).min(num_quads);
            let mut panel_acc = vec![L::zero(); n_rows];
            let mut lut = [L::zero(); 256];
            accumulate_lut_quad_range(&mut panel_acc, matrix, e_w, q0 * 4, q1 * 4, &mut lut);
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
fn accumulate_lut_quad_range<L: FieldCore>(
    row_acc: &mut [L],
    matrix: &JlProjectionMatrix,
    e_w: &[L],
    col0_start: usize,
    col0_end: usize,
    lut: &mut [L; 256],
) {
    let mut col0 = col0_start;
    while col0 < col0_end {
        let weights: [L; 4] = [e_w[col0], e_w[col0 + 1], e_w[col0 + 2], e_w[col0 + 3]];
        build_sign_weight_lut_256(&weights, lut);
        accumulate_rows_from_byte_lut(row_acc, matrix, col0 >> 2, lut);
        col0 += 4;
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

fn fill_row_weights<L: FieldCore>(
    g: &mut [L],
    panel_cols: usize,
    cols: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
) {
    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        g.par_chunks_mut(panel_cols)
            .enumerate()
            .for_each(|(p, g_panel)| {
                scatter_panel(g_panel, p * panel_cols, cols, n_rows, matrix, e_j);
            });
        return;
    }
    for (p, g_panel) in g.chunks_mut(panel_cols).enumerate() {
        scatter_panel(g_panel, p * panel_cols, cols, n_rows, matrix, e_j);
    }
}

fn scatter_panel<L: FieldCore>(
    g_panel: &mut [L],
    col0: usize,
    cols: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
) {
    if col0 >= cols {
        return;
    }
    let n = g_panel.len().min(cols - col0);
    let g_active = &mut g_panel[..n];
    for (j, &w) in e_j.iter().take(n_rows).enumerate() {
        scatter_row_weight_range(g_active, matrix.row_slice(j), col0, w);
    }
}
