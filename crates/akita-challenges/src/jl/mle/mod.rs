//! Joint multilinear extension evaluation for dense ternary JL matrices.
//!
//! Implements `J̃(r_J, r_w) = Σ_{j,i} eq(r_J,j) eq(r_w,i) J[j,i]` and the partial
//! row-weight table `g[i] = Σ_j eq(r_J,j) J[j,i]` used by the consistency sumcheck.
//! The production path uses Dao–Thaler tensor split-eq; naive reference helpers
//! back differential tests. SIMD kernels are deferred (see spec PR1b).

#![allow(non_snake_case)] // `r_J` matches the spec / paper notation.

mod common;
mod reference;
mod split_eq;

use akita_field::{AkitaError, FieldCore};

use crate::jl::JlProjectionMatrix;

pub use reference::eval_mle_from_weights;

/// Fused verifier evaluation `J̃(r_J, r_w)` without materializing the weight table.
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
    split_eq::eval_jl_mle_at_split_eq_parallel(
        matrix,
        r_J,
        r_w,
        split_eq::use_parallel_mle(matrix.n_rows(), matrix.cols()),
    )
}

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
    split_eq::build_jl_row_weights_split_eq(matrix, r_J)
}

/// Naive reference `J̃(r_J, r_w)` for differential tests.
#[doc(hidden)]
pub fn eval_jl_mle_at_reference<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    reference::eval_jl_mle_at_reference(matrix, r_J, r_w)
}

/// Naive reference row-weight builder for differential tests.
#[doc(hidden)]
pub fn build_jl_row_weights_reference<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
) -> Result<Vec<L>, AkitaError> {
    reference::build_jl_row_weights_reference(matrix, r_J)
}

#[cfg(test)]
mod tests;
