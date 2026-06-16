//! Fused JL matrix MLE evaluation: row-weight table and LUT-amortized `J̃(r_J, r_w)`.

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use super::common::{
    accumulate_row_weight_range, scatter_row_weight_range, validate_mle_points, JlMleLayout,
};
use super::lut::{accumulate_rows_from_byte_lut, build_sign_weight_lut_256};
use crate::jl::panel::{byte_aligned_panel_cols, panel_span};
use crate::jl::JlProjectionMatrix;

const PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

/// `g[i] = Σ_j eq(r_J, j) · J[j, i]` via row-major byte-wide scatter.
pub(super) fn build_row_weights<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, None)?;

    let e_j = EqPolynomial::evals(r_J)?;
    let mut g = vec![L::zero(); layout.col_hyper];
    let cols = matrix.cols();
    let n_rows = matrix.n_rows();
    let panel_cols = byte_aligned_panel_cols(cols);

    fill_row_weights(&mut g, panel_cols, cols, n_rows, matrix, &e_j);
    Ok(g)
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
    if use_parallel_mle(n_rows, cols) {
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
        scatter_row_weight_range(g_active, matrix.row_bytes_slice(j), col0, w);
    }
}

/// Production `J̃(r_J, r_w)` with per-4-column sign-weight LUT (see `mle` module docs).
pub(super) fn eval_lut<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let (e_j, e_w) = prepare_eq_tables(matrix, r_J, r_w)?;
    Ok(eval_lut_from_eq_tables(matrix, &e_j, &e_w))
}

/// Row-major scalar fused eval (`benches/jl_mle` baseline and differential tests).
pub(super) fn eval_scalar_rowmajor<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let (e_j, e_w) = prepare_eq_tables(matrix, r_J, r_w)?;
    let cols = matrix.cols();
    let e_w = &e_w[..cols];
    let n_rows = matrix.n_rows();

    #[cfg(feature = "parallel")]
    if use_parallel_mle(n_rows, cols) {
        return Ok((0..n_rows)
            .into_par_iter()
            .map(|j| e_j[j] * accumulate_row_weight_range(matrix.row_bytes_slice(j), 0, cols, e_w))
            .reduce(L::zero, |a, b| a + b));
    }

    let mut total = L::zero();
    for (j, &ej) in e_j.iter().take(n_rows).enumerate() {
        let row_sum = accumulate_row_weight_range(matrix.row_bytes_slice(j), 0, cols, e_w);
        total += ej * row_sum;
    }
    Ok(total)
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

fn eval_lut_from_eq_tables<L: FieldCore>(matrix: &JlProjectionMatrix, e_j: &[L], e_w: &[L]) -> L {
    let n_rows = matrix.n_rows();
    let cols = matrix.cols();
    let e_w = &e_w[..cols];
    let full_cols = cols - (cols & 0b11);
    let tail_cols = cols & 0b11;

    let mut row_acc = vec![L::zero(); n_rows];
    #[cfg(feature = "parallel")]
    if use_parallel_mle(n_rows, cols) {
        row_acc = eval_lut_row_acc_parallel(matrix, e_w, full_cols);
    } else {
        accumulate_lut_quads_sequential(&mut row_acc, matrix, e_w, full_cols);
    }
    #[cfg(not(feature = "parallel"))]
    accumulate_lut_quads_sequential(&mut row_acc, matrix, e_w, full_cols);

    accumulate_lut_tail(matrix, &mut row_acc, e_w, full_cols, tail_cols);
    dot_row_eq_weights(e_j, n_rows, &row_acc)
}

fn accumulate_lut_quads_sequential<L: FieldCore>(
    row_acc: &mut [L],
    matrix: &JlProjectionMatrix,
    e_w: &[L],
    full_cols: usize,
) {
    let mut lut = [L::zero(); 256];
    accumulate_quad_range(row_acc, matrix, e_w, 0, full_cols, &mut lut);
}

#[cfg(feature = "parallel")]
fn eval_lut_row_acc_parallel<L: FieldCore>(
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
            accumulate_quad_range(&mut panel_acc, matrix, e_w, q0 * 4, q1 * 4, &mut lut);
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
fn accumulate_quad_range<L: FieldCore>(
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
fn accumulate_lut_tail<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    row_acc: &mut [L],
    e_w: &[L],
    full_cols: usize,
    tail_cols: usize,
) {
    if tail_cols == 0 {
        return;
    }
    for (j, acc) in row_acc.iter_mut().enumerate() {
        *acc += accumulate_row_weight_range(
            matrix.row_bytes_slice(j),
            full_cols,
            tail_cols,
            &e_w[full_cols..],
        );
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

fn use_parallel_mle(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= PARALLEL_ELEMS_THRESHOLD
}
