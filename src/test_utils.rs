//! Shared test configuration and helpers.
//!
//! This module is only compiled under `#[cfg(test)]` and provides common
//! building blocks for both unit tests (inside `src/`) and integration
//! tests (inside `tests/`).

use crate::algebra::{CyclotomicRing, Fp64};
use crate::protocol::commitment::CommitmentConfig;
use crate::{CanonicalField, FieldCore};

pub type F = Fp64<4294967197>;
pub const D: usize = 64;

#[derive(Clone)]
pub struct TinyConfig;

impl CommitmentConfig for TinyConfig {
    const D: usize = 64;
    const M: usize = 1;
    const R: usize = 1;
    const N_A: usize = 2;
    const N_B: usize = 2;
    const N_D: usize = 2;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 9;
    const TAU: usize = 4;
    const BETA: u128 = 1_000_000;
    const CHALLENGE_WEIGHT: usize = 3;
}

pub const BLOCK_LEN: usize = 1 << TinyConfig::M;
pub const NUM_BLOCKS: usize = 1 << TinyConfig::R;
pub const DELTA: usize = TinyConfig::DELTA;
pub const LOG_BASIS: u32 = TinyConfig::LOG_BASIS;
pub const N_A: usize = TinyConfig::N_A;
pub const TAU: usize = TinyConfig::TAU;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn mat_vec_mul(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    mat.iter()
        .map(|row| {
            assert_eq!(row.len(), vec.len());
            row.iter()
                .zip(vec.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, x)| {
                    acc + (*a * *x)
                })
        })
        .collect()
}

pub fn sample_blocks() -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..NUM_BLOCKS)
        .map(|bi| {
            (0..BLOCK_LEN)
                .map(|bj| {
                    let coeffs =
                        std::array::from_fn(|k| F::from_u64((bi * 1_000 + bj * 100 + k) as u64));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect()
}

pub fn sample_a() -> Vec<F> {
    (0..BLOCK_LEN)
        .map(|j| F::from_u64((j * 10 + 1) as u64))
        .collect()
}

pub fn sample_b() -> Vec<F> {
    (0..NUM_BLOCKS)
        .map(|i| F::from_u64((i * 7 + 3) as u64))
        .collect()
}

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

pub fn recompose_z_hat(z_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    z_hat
        .chunks(TAU)
        .map(|chunk| field_gadget_recompose(chunk, LOG_BASIS))
        .collect()
}

pub fn gadget_recompose_vec(x_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    x_hat
        .chunks(DELTA)
        .map(|chunk| field_gadget_recompose(chunk, LOG_BASIS))
        .collect()
}

pub fn field_gadget_recompose_vec(v: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    v.chunks(DELTA)
        .map(|chunk| field_gadget_recompose(chunk, LOG_BASIS))
        .collect()
}

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
