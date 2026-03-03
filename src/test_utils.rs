//! Shared test configuration and helpers.
//!
//! This module is only compiled under `#[cfg(test)]` and provides common
//! building blocks for both unit tests (inside `src/`) and integration
//! tests (inside `tests/`).

use std::array::from_fn;

use crate::algebra::{CyclotomicRing, Fp64};
use crate::error::HachiError;
use crate::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, CommitmentConfig, DecompositionParams,
    HachiCommitmentLayout,
};
use crate::{FieldCore, FromSmallInt};

/// Default test field: a 32-bit prime `p = 4294967197`.
pub type F = Fp64<4294967197>;
/// Ring degree used in tests.
pub const D: usize = 64;

/// Minimal commitment config for fast unit tests.
#[derive(Clone)]
pub struct TinyConfig;

impl CommitmentConfig for TinyConfig {
    const D: usize = 64;
    const N_A: usize = 2;
    const N_B: usize = 2;
    const N_D: usize = 2;
    const CHALLENGE_WEIGHT: usize = 3;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(1, 1, &Self::decomposition())
    }
}

/// Number of ring elements per block (`2^m_vars`).
pub const BLOCK_LEN: usize = 2;
/// Number of blocks (`2^r_vars`).
pub const NUM_BLOCKS: usize = 2;
/// Gadget base exponent (`b = 2^log_basis()`), derived from `TinyConfig`.
pub fn log_basis() -> u32 {
    TinyConfig::decomposition().log_basis
}
/// Inner Ajtai row count from `TinyConfig`.
pub const N_A: usize = TinyConfig::N_A;

/// Decomposition depth for original coefficients under `TinyConfig`.
pub fn num_digits_commit() -> usize {
    let d = TinyConfig::decomposition();
    compute_num_digits(d.log_commit_bound, d.log_basis)
}

/// Decomposition depth for the folded witness `z_pre` under `TinyConfig`.
pub fn num_digits_fold() -> usize {
    let d = TinyConfig::decomposition();
    compute_num_digits_fold(1, TinyConfig::CHALLENGE_WEIGHT, d.log_basis)
}

/// Dense matrix-vector multiply over cyclotomic rings.
///
/// Matrix rows may be wider than `vec` (e.g. when matrices are widened for
/// multi-level folding); extra columns are treated as multiplying zero.
pub fn mat_vec_mul(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    mat.iter()
        .map(|row| {
            assert!(row.len() >= vec.len());
            row.iter()
                .zip(vec.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, x)| {
                    acc + (*a * *x)
                })
        })
        .collect()
}

/// Generate deterministic test blocks of ring elements.
pub fn sample_blocks() -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..NUM_BLOCKS)
        .map(|bi| {
            (0..BLOCK_LEN)
                .map(|bj| {
                    let coeffs = from_fn(|k| F::from_u64((bi * 1_000 + bj * 100 + k) as u64));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect()
}

/// Generate deterministic inner opening-point scalars.
pub fn sample_a() -> Vec<F> {
    (0..BLOCK_LEN)
        .map(|j| F::from_u64((j * 10 + 1) as u64))
        .collect()
}

/// Generate deterministic outer opening-point scalars.
pub fn sample_b() -> Vec<F> {
    (0..NUM_BLOCKS)
        .map(|i| F::from_u64((i * 7 + 3) as u64))
        .collect()
}

/// Recompose a gadget-decomposed ring element: `sum_i parts[i] * b^i`.
pub fn field_gadget_recompose(
    parts: &[CyclotomicRing<F, D>],
    log_basis: u32,
) -> CyclotomicRing<F, D> {
    let b = F::from_u64(1u64 << log_basis);
    let mut result = CyclotomicRing::<F, D>::zero();
    let mut b_power = F::one();
    for part in parts {
        result += part.scale(&b_power);
        b_power = b_power * b;
    }
    result
}

/// Recompose `z_hat` chunks (num_digits_fold-width) back to `z_pre` elements.
pub fn recompose_z_hat(z_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    z_hat
        .chunks(num_digits_fold())
        .map(|chunk| field_gadget_recompose(chunk, log_basis()))
        .collect()
}

/// Recompose a vector of gadget-decomposed elements (num_digits_commit-width chunks).
pub fn gadget_recompose_vec(x_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    x_hat
        .chunks(num_digits_commit())
        .map(|chunk| field_gadget_recompose(chunk, log_basis()))
        .collect()
}

/// Recompose a vector of i8 gadget-decomposed digit planes (num_digits_commit-width chunks).
pub fn gadget_recompose_vec_i8(x_hat: &[[i8; D]]) -> Vec<CyclotomicRing<F, D>> {
    x_hat
        .chunks(num_digits_commit())
        .map(|chunk| CyclotomicRing::gadget_recompose_pow2_i8(chunk, log_basis()))
        .collect()
}

/// Alias for [`gadget_recompose_vec`] (same num_digits_commit-width recomposition).
pub fn field_gadget_recompose_vec(v: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    v.chunks(num_digits_commit())
        .map(|chunk| field_gadget_recompose(chunk, log_basis()))
        .collect()
}

/// Compute `a^T * G^{-1}(z)`: recompose `z` then inner-product with `a`.
pub fn a_transpose_gadget_times_vec(a: &[F], z: &[CyclotomicRing<F, D>]) -> CyclotomicRing<F, D> {
    let recomposed = field_gadget_recompose_vec(z);
    assert_eq!(recomposed.len(), a.len());
    recomposed
        .iter()
        .zip(a.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (z_j, a_j)| {
            acc + z_j.scale(a_j)
        })
}
