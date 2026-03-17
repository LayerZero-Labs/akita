#![allow(dead_code)]

//! Shared test configuration and helpers.

use std::array::from_fn;

use crate::algebra::{CyclotomicRing, Fp64};
use crate::error::HachiError;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::{
    compute_num_digits, compute_num_digits_fold, CommitmentConfig, DecompositionParams,
    HachiCommitmentLayout,
};
use crate::{FieldCore, FromSmallInt};

pub(crate) type F = Fp64<4294967197>;
pub(crate) const D: usize = 64;

#[derive(Clone)]
pub(crate) struct TinyConfig;

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

pub(crate) const BLOCK_LEN: usize = 2;
pub(crate) const NUM_BLOCKS: usize = 2;

pub(crate) fn log_basis() -> u32 {
    TinyConfig::decomposition().log_basis
}

pub(crate) const N_A: usize = TinyConfig::N_A;

pub(crate) fn num_digits_commit() -> usize {
    let d = TinyConfig::decomposition();
    compute_num_digits(d.log_commit_bound, d.log_basis)
}

pub(crate) fn num_digits_open() -> usize {
    let d = TinyConfig::decomposition();
    let log_open = d.log_open_bound.unwrap_or(d.log_commit_bound);
    compute_num_digits(log_open, d.log_basis)
}

pub(crate) fn num_digits_fold() -> usize {
    let d = TinyConfig::decomposition();
    compute_num_digits_fold(1, TinyConfig::CHALLENGE_WEIGHT, d.log_basis)
}

pub(crate) fn mat_vec_mul(
    mat: &FlatMatrix<F>,
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let view = mat.view::<D>();
    (0..view.num_rows())
        .map(|i| {
            let row = view.row(i);
            assert!(row.len() >= vec.len());
            row.iter()
                .zip(vec.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, x)| {
                    acc + (*a * *x)
                })
        })
        .collect()
}

pub(crate) fn sample_blocks() -> Vec<Vec<CyclotomicRing<F, D>>> {
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

pub(crate) fn sample_a() -> Vec<F> {
    (0..BLOCK_LEN)
        .map(|j| F::from_u64((j * 10 + 1) as u64))
        .collect()
}

pub(crate) fn sample_b() -> Vec<F> {
    (0..NUM_BLOCKS)
        .map(|i| F::from_u64((i * 7 + 3) as u64))
        .collect()
}

pub(crate) fn field_gadget_recompose(
    parts: &[CyclotomicRing<F, D>],
    log_basis: u32,
) -> CyclotomicRing<F, D> {
    let b = F::from_u64(1u64 << log_basis);
    let mut result = CyclotomicRing::<F, D>::zero();
    let mut b_power = F::one();
    for part in parts {
        result += part.scale(&b_power);
        b_power *= b;
    }
    result
}

pub(crate) fn recompose_z_hat(z_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    z_hat
        .chunks(num_digits_fold())
        .map(|chunk| field_gadget_recompose(chunk, log_basis()))
        .collect()
}

pub(crate) fn gadget_recompose_vec(x_hat: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    x_hat
        .chunks(num_digits_commit())
        .map(|chunk| field_gadget_recompose(chunk, log_basis()))
        .collect()
}

pub(crate) fn gadget_recompose_vec_i8(x_hat: &[[i8; D]]) -> Vec<CyclotomicRing<F, D>> {
    x_hat
        .chunks(num_digits_commit())
        .map(|chunk| CyclotomicRing::gadget_recompose_pow2_i8(chunk, log_basis()))
        .collect()
}

pub(crate) fn field_gadget_recompose_vec(v: &[CyclotomicRing<F, D>]) -> Vec<CyclotomicRing<F, D>> {
    v.chunks(num_digits_commit())
        .map(|chunk| field_gadget_recompose(chunk, log_basis()))
        .collect()
}

pub(crate) fn a_transpose_gadget_times_vec(
    a: &[F],
    z: &[CyclotomicRing<F, D>],
) -> CyclotomicRing<F, D> {
    let recomposed = field_gadget_recompose_vec(z);
    assert_eq!(recomposed.len(), a.len());
    recomposed
        .iter()
        .zip(a.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (z_j, a_j)| {
            acc + z_j.scale(a_j)
        })
}
