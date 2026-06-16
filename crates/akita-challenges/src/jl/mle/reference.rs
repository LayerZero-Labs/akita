//! Naive Θ(`n_rows · cols`) JL matrix MLE evaluators for differential tests.

#![allow(clippy::needless_range_loop)]

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use super::common::{accum_sign_weight, entry_sign, validate_mle_points, JlMleLayout};
use crate::jl::JlProjectionMatrix;

/// `g[i] = Σ_j eq(r_J, j) · J[j, i]` by direct summation.
pub(super) fn build_jl_row_weights_reference<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, None)?;

    let e_j = EqPolynomial::evals(r_J)?;
    let mut g = vec![L::zero(); layout.col_hyper];
    for i in 0..matrix.cols() {
        let mut acc = L::zero();
        for j in 0..layout.row_hyper {
            if j >= matrix.n_rows() {
                continue;
            }
            let sign = entry_sign(matrix, j, i);
            if sign == 0 {
                continue;
            }
            acc = accum_sign_weight(acc, sign, e_j[j]);
        }
        g[i] = acc;
    }
    Ok(g)
}

/// `Σ_{j,i} eq(r_J, j) eq(r_w, i) J[j,i]` by direct summation.
pub(super) fn eval_jl_mle_at_reference<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    let layout = JlMleLayout::new(matrix)?;
    validate_mle_points(&layout, r_J, Some(r_w))?;

    let e_j = EqPolynomial::evals(r_J)?;
    let e_w = EqPolynomial::evals(r_w)?;
    let mut acc = L::zero();
    for j in 0..layout.row_hyper {
        if j >= matrix.n_rows() {
            continue;
        }
        for i in 0..layout.col_hyper {
            if i >= matrix.cols() {
                continue;
            }
            let sign = entry_sign(matrix, j, i);
            if sign == 0 {
                continue;
            }
            let weight = e_j[j] * e_w[i];
            acc = accum_sign_weight(acc, sign, weight);
        }
    }
    Ok(acc)
}

/// `Σ_i eq(r_w, i) g[i]` for a precomputed weight table.
pub fn eval_mle_from_weights<L: FieldCore>(g: &[L], r_w: &[L]) -> Result<L, AkitaError> {
    if g.len() != (1usize << r_w.len()) {
        return Err(AkitaError::InvalidSize {
            expected: 1usize << r_w.len(),
            actual: g.len(),
        });
    }
    let e_w = EqPolynomial::evals(r_w)?;
    Ok(g.iter()
        .zip(e_w.iter())
        .fold(L::zero(), |acc, (&gi, &ew)| acc + gi * ew))
}
