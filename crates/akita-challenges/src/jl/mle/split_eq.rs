//! Row-major contractions for JL matrix MLE evaluation.

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use super::common::{
    accumulate_row_weight_range, scatter_row_weight_range, validate_mle_points, JlMleLayout,
};
use super::lut::{accumulate_rows_from_byte_lut, build_sign_weight_lut_256};
use crate::jl::JlProjectionMatrix;

/// `g[i] = Σ_j eq(r_J, j) · J[j, i]` via a row-major byte-wide scatter.
///
/// The full row-eq table `eq(r_J, ·)` is small (`2^{row_bits}`, the image
/// dimension), so unlike the fused eval there is no benefit to splitting the row
/// axis; the output `g` is the only large allocation. Columns are partitioned
/// into byte-aligned panels whose `g` slice stays L1-resident and is reused
/// across every row, so each packed matrix byte is read exactly once.
pub(super) fn build_jl_row_weights_split_eq<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, None)?;

    let e_j = EqPolynomial::evals(r_J)?;
    let mut g = vec![L::zero(); layout.col_hyper];
    let cols = matrix.cols();
    let n_rows = matrix.n_rows();
    let panel_cols = row_weight_panel_cols(cols);

    fill_row_weights(&mut g, panel_cols, cols, n_rows, matrix, &e_j);
    Ok(g)
}

/// Columns per row-weight panel: byte-aligned so the per-row packed-byte decode
/// stays aligned, sized so `g`'s panel slice is reused across all rows from L1.
fn row_weight_panel_cols(cols: usize) -> usize {
    const MIN_PANELS: usize = 64;
    const PANEL_COLS_MAX: usize = 4096;
    let by_balance = cols.div_ceil(MIN_PANELS).max(1);
    let aligned = by_balance.div_ceil(4) * 4;
    aligned.min(PANEL_COLS_MAX)
}

/// Scatter every row's eq weight into one column panel of `g`.
///
/// Panels own disjoint column ranges of `g`, so the parallel sweep needs no
/// reduction. Padding-only panels (`col0 >= cols`) and the padded tail of the
/// straddling panel stay zero.
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

#[inline]
fn direct_eq_tables<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<(Vec<L>, Vec<L>), AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, Some(r_w))?;
    Ok((EqPolynomial::evals(r_J)?, EqPolynomial::evals(r_w)?))
}

/// Scalar row-per-row fused `J̃(r_J, r_w)` (differential tests / benches only).
pub(super) fn eval_jl_mle_at_split_eq<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
    parallel: bool,
) -> Result<L, AkitaError> {
    let (e_j, e_w) = direct_eq_tables(matrix, r_J, r_w)?;
    let cols = matrix.cols();
    let e_w = &e_w[..cols];

    #[cfg(feature = "parallel")]
    if parallel {
        let total = (0..matrix.n_rows())
            .into_par_iter()
            .map(|j| e_j[j] * accumulate_row_weight_range(matrix.row_bytes_slice(j), 0, cols, e_w))
            .reduce(L::zero, |a, b| a + b);
        return Ok(total);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    Ok(eval_jl_mle_at_rows_scalar(matrix, &e_j, e_w))
}

fn eval_jl_mle_at_rows_scalar<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
) -> L {
    let cols = matrix.cols();
    let mut acc = L::zero();
    for (j, &ej) in e_j.iter().take(matrix.n_rows()).enumerate() {
        let row_sum = accumulate_row_weight_range(matrix.row_bytes_slice(j), 0, cols, e_w);
        acc += ej * row_sum;
    }
    acc
}

/// Production fused `J̃(r_J, r_w)` with a per-4-column sign-weight LUT.
///
/// Iteration order (LUT amortized over rows):
/// 1. outer: each byte-aligned 4-column window of `e_w`
/// 2. build `lut256` once from `eq_w[col0..col0+4]` (81-pattern DP + byte remap)
/// 3. inner: every row does one LUT lookup + one add into `row_acc[j]`
/// 4. scalar tail for `cols % 4`
/// 5. finish with `Σ_j e_j[j] · row_acc[j]`
pub(super) fn eval_jl_mle_at_split_eq_lut<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
    parallel: bool,
) -> Result<L, AkitaError> {
    let (e_j, e_w) = direct_eq_tables(matrix, r_J, r_w)?;
    Ok(eval_jl_mle_at_rows_lut(
        matrix,
        &e_j,
        &e_w,
        parallel && use_parallel_mle(matrix.n_rows(), matrix.cols()),
    ))
}

fn eval_jl_mle_at_rows_lut<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
    parallel: bool,
) -> L {
    let n_rows = matrix.n_rows();
    let cols = matrix.cols();
    let e_w = &e_w[..cols];
    let full_cols = cols - (cols & 0b11);
    let tail_cols = cols & 0b11;

    #[cfg(feature = "parallel")]
    if parallel {
        return eval_jl_mle_at_rows_lut_par(matrix, e_j, e_w, full_cols, tail_cols);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    let mut row_acc = vec![L::zero(); n_rows];
    let mut lut = [L::zero(); 256];
    let mut col0 = 0usize;
    while col0 < full_cols {
        let weights: [L; 4] = [e_w[col0], e_w[col0 + 1], e_w[col0 + 2], e_w[col0 + 3]];
        build_sign_weight_lut_256(&weights, &mut lut);
        accumulate_rows_from_byte_lut(&mut row_acc, matrix, col0 >> 2, &lut);
        col0 += 4;
    }
    accumulate_lut_tail(matrix, &mut row_acc, e_w, full_cols, tail_cols);
    dot_row_eq_weights(e_j, n_rows, &row_acc)
}

#[cfg(feature = "parallel")]
fn eval_jl_mle_at_rows_lut_par<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    e_w: &[L],
    full_cols: usize,
    tail_cols: usize,
) -> L {
    let n_rows = matrix.n_rows();
    let num_quads = full_cols / 4;
    let quads_per_panel = lut_quad_panel_count(num_quads);
    let num_panels = num_quads.div_ceil(quads_per_panel);

    let mut row_acc = (0..num_panels)
        .into_par_iter()
        .map(|p| {
            let q0 = p * quads_per_panel;
            let q1 = (q0 + quads_per_panel).min(num_quads);
            let mut panel_acc = vec![L::zero(); n_rows];
            let mut lut = [L::zero(); 256];
            for q in q0..q1 {
                let col0 = q * 4;
                let weights: [L; 4] = [e_w[col0], e_w[col0 + 1], e_w[col0 + 2], e_w[col0 + 3]];
                build_sign_weight_lut_256(&weights, &mut lut);
                accumulate_rows_from_byte_lut(&mut panel_acc, matrix, col0 >> 2, &lut);
            }
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
        );

    accumulate_lut_tail(matrix, &mut row_acc, e_w, full_cols, tail_cols);
    dot_row_eq_weights(e_j, n_rows, &row_acc)
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

/// Quad windows per parallel panel: enough independent work per task without one
/// task per 4-column window at tail scale.
pub(super) fn lut_quad_panel_count(num_quads: usize) -> usize {
    const MIN_PANELS: usize = 64;
    num_quads.div_ceil(MIN_PANELS).max(1)
}

#[inline]
fn dot_row_eq_weights<L: FieldCore>(e_j: &[L], n_rows: usize, row_acc: &[L]) -> L {
    let mut total = L::zero();
    for (j, &ej) in e_j.iter().take(n_rows).enumerate() {
        total += ej * row_acc[j];
    }
    total
}

const JL_MLE_PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

pub(super) fn use_parallel_mle(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= JL_MLE_PARALLEL_ELEMS_THRESHOLD
}
