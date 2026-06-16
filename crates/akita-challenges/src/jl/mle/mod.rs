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
//!    patterns via axis extension, remapped through [`lut::BYTE_TO_TERNARY4`] so
//!    `01`/`10` zero pairs collapse correctly; same sign alphabet as
//!    [`crate::jl::kernels::SIGNS_FOR_BYTE`]).
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
//! Naive reference helpers in [`reference`] back differential tests.

#![allow(non_snake_case)] // `r_J` matches the spec / paper notation.

mod common;
mod fused;
mod lut;
mod reference;
#[cfg(test)]
mod tests;

use akita_field::{AkitaError, FieldCore};

use crate::jl::JlProjectionMatrix;

pub use reference::eval_mle_from_weights;
#[doc(hidden)]
pub use reference::{build_jl_row_weights_reference, eval_jl_mle_at_reference};

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
    fused::eval_lut(matrix, r_J, r_w)
}

/// Row-major scalar fused eval (`benches/jl_mle` baseline).
#[doc(hidden)]
pub fn eval_jl_mle_at_scalar<L: FieldCore>(
    matrix: &JlProjectionMatrix,
    r_J: &[L],
    r_w: &[L],
) -> Result<L, AkitaError> {
    fused::eval_scalar_rowmajor(matrix, r_J, r_w)
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
    fused::build_row_weights(matrix, r_J)
}
