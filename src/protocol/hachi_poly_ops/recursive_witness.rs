//! Recursive witness helpers for later Hachi prove levels.
//!
//! Recursive levels do not operate on a caller-provided polynomial anymore.
//! Instead they carry a flat digit witness `w` that is re-chunked under the
//! current ring dimension `D` on demand. [`RecursiveWitnessFlat`] owns the
//! D-agnostic digit buffer, while [`RecursiveWitnessView`] provides the
//! zero-copy D-specific operations used by recursive folding and handoff paths.

use crate::algebra::ring::sparse_challenge::SparseChallenge;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_i8, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8,
};
use crate::protocol::hachi_poly_ops::helpers::{
    balanced_digit_decompose_fold_partitioned, build_decompose_fold_witness,
};
use crate::protocol::hachi_poly_ops::{CommitInnerWitness, DecomposeFoldWitness};
use crate::{cfg_fold_reduce, cfg_into_iter, cfg_iter, CanonicalField, FieldCore};
use std::array::from_fn;
use std::marker::PhantomData;

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RecursiveWitnessFlat {
    digits: Vec<i8>,
}

impl RecursiveWitnessFlat {
    pub(crate) fn from_i8_digits(digits: Vec<i8>) -> Self {
        Self { digits }
    }

    pub(crate) fn as_i8_digits(&self) -> &[i8] {
        &self.digits
    }

    pub(crate) fn len(&self) -> usize {
        self.digits.len()
    }

    pub(crate) fn view<F: FieldCore, const D: usize>(
        &self,
    ) -> Result<RecursiveWitnessView<'_, F, D>, HachiError> {
        RecursiveWitnessView::from_i8_digits(&self.digits)
    }
}

impl AsRef<[i8]> for RecursiveWitnessFlat {
    fn as_ref(&self) -> &[i8] {
        self.as_i8_digits()
    }
}

/// D-specific zero-copy view over a flat recursive witness.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecursiveWitnessView<'a, F: FieldCore, const D: usize> {
    coeffs: &'a [[i8; D]],
    padded_ring_elems: usize,
    _marker: PhantomData<F>,
}

impl<'a, F: FieldCore, const D: usize> RecursiveWitnessView<'a, F, D> {
    pub(crate) fn from_i8_digits(digits: &'a [i8]) -> Result<Self, HachiError> {
        let (coeffs, remainder) = digits.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(HachiError::InvalidSize {
                expected: D,
                actual: digits.len(),
            });
        }

        Ok(Self {
            coeffs,
            padded_ring_elems: coeffs.len().next_power_of_two().max(1),
            _marker: PhantomData,
        })
    }

    #[inline]
    fn block_slice(&self, block_idx: usize, block_len: usize) -> &'a [[i8; D]] {
        let start = block_idx * block_len;
        if start >= self.coeffs.len() {
            &[]
        } else {
            &self.coeffs[start..(start + block_len).min(self.coeffs.len())]
        }
    }

    pub(crate) fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }
}

impl<'a, F, const D: usize> RecursiveWitnessView<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
        let total = cfg_fold_reduce!(
            0..self.coeffs.len().min(scalars.len()),
            || [F::zero(); D],
            |mut acc: [F; D], idx| {
                let scalar = scalars[idx];
                let digit = &self.coeffs[idx];
                for (coeff, &d) in acc.iter_mut().zip(digit.iter()) {
                    if d != 0 {
                        *coeff += scalar * F::from_i8(d);
                    }
                }
                acc
            },
            |mut a: [F; D], b: [F; D]| {
                for (a_coeff, b_coeff) in a.iter_mut().zip(b.iter()) {
                    *a_coeff += *b_coeff;
                }
                a
            }
        );
        CyclotomicRing::from_coefficients(total)
    }

    pub(crate) fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (ring, &scalar) in self
                    .block_slice(block_idx, block_len)
                    .iter()
                    .zip(scalars.iter())
                {
                    for (coeff, &d) in acc.iter_mut().zip(ring.iter()) {
                        if d != 0 {
                            *coeff += scalar * F::from_i8(d);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks(fold_scalars, block_len);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "RecursiveWitnessView::decompose_fold")]
    pub(crate) fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let inner_width = block_len * num_digits;
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        let active_blocks = challenges.len().min(num_blocks);

        let q = (-F::one()).to_canonical_u128() + 1;
        let coeffs = self.coeffs;
        let coeff_accum = balanced_digit_decompose_fold_partitioned::<D>(
            coeffs,
            challenges,
            active_blocks,
            block_len,
            num_digits,
            inner_width,
        );
        build_decompose_fold_witness::<F, D>(coeff_accum, q)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn commit_inner(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<[i8; D]>>, HachiError> {
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        let coeff_len = self.coeffs.len();

        let t_all = if num_digits_commit == 1 {
            let block_slices: Vec<&[[i8; D]]> = (0..num_blocks)
                .map(|block_idx| self.block_slice(block_idx, block_len))
                .collect();
            mat_vec_mul_ntt_digits_i8(ntt_a, &block_slices)
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = self
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
                .map(|block_idx| {
                    let start = block_idx * block_len;
                    if start >= coeff_len {
                        &[] as &[CyclotomicRing<F, D>]
                    } else {
                        &ring_elems[start..(start + block_len).min(coeff_len)]
                    }
                })
                .collect();
            mat_vec_mul_ntt_i8(ntt_a, &block_slices, num_digits_commit, log_basis)
        };

        let results = cfg_into_iter!(t_all)
            .map(|t_i| decompose_rows_i8(&t_i, num_digits_open, log_basis))
            .collect();
        Ok(results)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn commit_inner_witness(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let num_blocks = self.num_ring_elems().div_ceil(block_len);
        let coeff_len = self.coeffs.len();

        let t = if num_digits_commit == 1 {
            let block_slices: Vec<&[[i8; D]]> = (0..num_blocks)
                .map(|block_idx| self.block_slice(block_idx, block_len))
                .collect();
            mat_vec_mul_ntt_digits_i8(ntt_a, &block_slices)
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = self
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
                .map(|block_idx| {
                    let start = block_idx * block_len;
                    if start >= coeff_len {
                        &[] as &[CyclotomicRing<F, D>]
                    } else {
                        &ring_elems[start..(start + block_len).min(coeff_len)]
                    }
                })
                .collect();
            mat_vec_mul_ntt_i8(ntt_a, &block_slices, num_digits_commit, log_basis)
        };

        let t_hat = cfg_iter!(t)
            .map(|t_i| decompose_rows_i8(t_i, num_digits_open, log_basis))
            .collect();
        Ok(CommitInnerWitness { t, t_hat })
    }
}
