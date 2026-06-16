//! Dao–Thaler tensor split-eq contraction for JL matrix MLE evaluation.

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use super::common::{
    accumulate_row_weight_range, scatter_row_weight_range, validate_mle_points, JlMleLayout,
};
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

/// Shared split-eq tile geometry and precomputed eq tables for the fused eval.
struct SplitEqSetup<L> {
    j_in_len: usize,
    j_out_len: usize,
    w_in_len: usize,
    w_out_len: usize,
    e_j_out: Vec<L>,
    e_w_out: Vec<L>,
    /// Inner tensor weights `W[j_i, i_i] = e_J_in[j_i] · e_w_in[i_i]`.
    w_inner: Vec<L>,
}

fn split_eq_setup<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<SplitEqSetup<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, Some(r_w))?;

    let (m_ji, _) = layout.row_split();
    let (m_wi, _) = layout.col_split();
    let j_in_len = 1usize << m_ji;
    let j_out_len = layout.row_hyper >> m_ji;
    let w_in_len = 1usize << m_wi;
    let w_out_len = layout.col_hyper >> m_wi;

    let e_j_in = EqPolynomial::evals(&r_J[..m_ji])?;
    let e_j_out = EqPolynomial::evals(&r_J[m_ji..])?;
    let e_w_in = EqPolynomial::evals(&r_w[..m_wi])?;
    let e_w_out = EqPolynomial::evals(&r_w[m_wi..])?;

    let w_inner: Vec<L> = e_j_in
        .iter()
        .flat_map(|&ej| e_w_in.iter().map(move |&ew| ej * ew))
        .collect();
    debug_assert_eq!(w_inner.len(), j_in_len * w_in_len);

    Ok(SplitEqSetup {
        j_in_len,
        j_out_len,
        w_in_len,
        w_out_len,
        e_j_out,
        e_w_out,
        w_inner,
    })
}

/// Fused `J̃(r_J, r_w)` via tensor split-eq on row and column hypercubes.
pub(super) fn eval_jl_mle_at_split_eq<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let s = split_eq_setup(matrix, r_J, r_w)?;

    let mut total = L::zero();
    for j_o in 0..s.j_out_len {
        for i_o in 0..s.w_out_len {
            let outer = s.e_j_out[j_o] * s.e_w_out[i_o];
            let inner =
                inner_tile_contribution(matrix, j_o, i_o, s.j_in_len, s.w_in_len, &s.w_inner);
            total += outer * inner;
        }
    }
    Ok(total)
}

#[inline]
fn inner_tile_contribution<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    j_o: usize,
    i_o: usize,
    j_in_len: usize,
    w_in_len: usize,
    w_inner: &[L],
) -> L {
    let col0 = i_o * w_in_len;
    let col1 = (col0 + w_in_len).min(matrix.cols());
    if col0 >= col1 {
        return L::zero();
    }
    let n_cols = col1 - col0;

    let mut acc = L::zero();
    for j_i in 0..j_in_len {
        let j = j_o * j_in_len + j_i;
        if j >= matrix.n_rows() {
            continue;
        }
        let row = matrix.row_bytes_slice(j);
        let w_base = j_i * w_in_len;
        let weights = &w_inner[w_base..w_base + n_cols];
        acc += accumulate_row_weight_range(row, col0, n_cols, weights);
    }
    acc
}

/// Parallel split-eq eval when `parallel` is enabled and the matrix is large enough.
pub(super) fn eval_jl_mle_at_split_eq_parallel<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
    parallel: bool,
) -> Result<L, AkitaError> {
    #[cfg(feature = "parallel")]
    if parallel {
        return eval_jl_mle_at_split_eq_par(matrix, r_J, r_w);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    eval_jl_mle_at_split_eq(matrix, r_J, r_w)
}

#[cfg(feature = "parallel")]
fn eval_jl_mle_at_split_eq_par<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let s = split_eq_setup(matrix, r_J, r_w)?;

    let tile_count = s.j_out_len * s.w_out_len;
    let total = (0..tile_count)
        .into_par_iter()
        .map(|t| {
            let j_o = t / s.w_out_len;
            let i_o = t % s.w_out_len;
            let outer = s.e_j_out[j_o] * s.e_w_out[i_o];
            outer * inner_tile_contribution(matrix, j_o, i_o, s.j_in_len, s.w_in_len, &s.w_inner)
        })
        .reduce(L::zero, |a, b| a + b);

    Ok(total)
}

const JL_MLE_PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

pub(super) fn use_parallel_mle(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= JL_MLE_PARALLEL_ELEMS_THRESHOLD
}
