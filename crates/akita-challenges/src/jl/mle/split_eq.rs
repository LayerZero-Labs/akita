//! Dao–Thaler tensor split-eq contraction for JL matrix MLE evaluation.

#![allow(clippy::needless_range_loop)]

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use super::common::{
    accum_sign_weight, accumulate_row_weight_range, entry_sign, validate_mle_points, JlMleLayout,
};
use crate::jl::JlProjectionMatrix;

/// `g[i] = Σ_j eq(r_J, j) · J[j, i]` with split-eq on the row hypercube.
pub(super) fn build_jl_row_weights_split_eq<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, None)?;

    let (m_ji, _m_jo) = layout.row_split();
    let j_in_len = 1usize << m_ji;
    let j_out_len = layout.row_hyper >> m_ji;

    let e_j_in = EqPolynomial::evals(&r_J[..m_ji])?;
    let e_j_out = EqPolynomial::evals(&r_J[m_ji..])?;

    let mut g = vec![L::zero(); layout.col_hyper];
    for i in 0..matrix.cols() {
        let mut gi = L::zero();
        for j_o in 0..j_out_len {
            let mut inner = L::zero();
            for j_i in 0..j_in_len {
                let j = j_o * j_in_len + j_i;
                if j >= matrix.n_rows() {
                    continue;
                }
                let w = e_j_in[j_i];
                let sign = entry_sign(matrix, j, i);
                inner = accum_sign_weight(inner, sign, w);
            }
            gi += e_j_out[j_o] * inner;
        }
        g[i] = gi;
    }
    Ok(g)
}

/// Fused `J̃(r_J, r_w)` via tensor split-eq on row and column hypercubes.
pub(super) fn eval_jl_mle_at_split_eq<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, Some(r_w))?;

    let (m_ji, _m_jo) = layout.row_split();
    let (m_wi, _m_wo) = layout.col_split();
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

    let mut total = L::zero();
    for j_o in 0..j_out_len {
        for i_o in 0..w_out_len {
            let outer = e_j_out[j_o] * e_w_out[i_o];
            let inner = inner_tile_contribution(matrix, j_o, i_o, j_in_len, w_in_len, &w_inner)?;
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
) -> Result<L, AkitaError> {
    let col0 = i_o * w_in_len;
    let col1 = (col0 + w_in_len).min(matrix.cols());
    if col0 >= col1 {
        return Ok(L::zero());
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
        let weights = &w_inner[w_base..w_base + w_in_len];
        let local_weights = &weights[..n_cols];
        acc += accumulate_row_weight_range(row, col0, n_cols, local_weights);
    }
    Ok(acc)
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

    let tile_count = j_out_len * w_out_len;
    let partials: Vec<L> = (0..tile_count)
        .into_par_iter()
        .map(|t| {
            let j_o = t / w_out_len;
            let i_o = t % w_out_len;
            let outer = e_j_out[j_o] * e_w_out[i_o];
            let inner = inner_tile_contribution(matrix, j_o, i_o, j_in_len, w_in_len, &w_inner)
                .expect("inner tile");
            outer * inner
        })
        .collect();

    Ok(partials.into_iter().fold(L::zero(), |acc, v| acc + v))
}

const JL_MLE_PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

pub(super) fn use_parallel_mle(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= JL_MLE_PARALLEL_ELEMS_THRESHOLD
}
