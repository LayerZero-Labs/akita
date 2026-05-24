//! Recursive witness helpers for later Akita prove levels.
//!
//! Recursive levels do not operate on a caller-provided polynomial anymore.
//! Instead they carry a flat digit witness `w` that is re-chunked under the
//! current ring dimension `D` on demand. [`RecursiveWitnessFlat`] owns the
//! D-agnostic digit buffer, while [`RecursiveWitnessView`] provides the
//! zero-copy D-specific operations used by recursive folding and handoff paths.

#![allow(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};

use crate::backend::poly_helpers::{
    balanced_digit_decompose_fold_partitioned, build_decompose_fold_witness,
};
use crate::compute::{CommitComputeBackend, RecursiveWitnessCommitRowsPlan};
use crate::kernels::linear::decompose_rows_i8_into;
use akita_types::FlatDigitBlocks;
use std::marker::PhantomData;

use crate::{CommitInnerWitness, DecomposeFoldWitness};

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecursiveWitnessFlat {
    digits: Vec<i8>,
}

impl RecursiveWitnessFlat {
    pub fn from_i8_digits(digits: Vec<i8>) -> Self {
        Self { digits }
    }

    pub fn as_i8_digits(&self) -> &[i8] {
        &self.digits
    }

    pub fn len(&self) -> usize {
        self.digits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.digits.is_empty()
    }

    pub fn view<F: FieldCore, const D: usize>(
        &self,
    ) -> Result<RecursiveWitnessView<'_, F, D>, AkitaError> {
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
pub struct RecursiveWitnessView<'a, F: FieldCore, const D: usize> {
    coeffs: &'a [[i8; D]],
    padded_ring_elems: usize,
    _marker: PhantomData<F>,
}

impl<'a, F: FieldCore, const D: usize> RecursiveWitnessView<'a, F, D> {
    pub fn from_i8_digits(digits: &'a [i8]) -> Result<Self, AkitaError> {
        let (coeffs, remainder) = digits.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
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
    fn block_elem(
        &self,
        block_idx: usize,
        col_idx: usize,
        num_blocks: usize,
    ) -> Option<&'a [i8; D]> {
        self.coeffs.get(block_idx + col_idx * num_blocks)
    }

    pub fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }
}

impl<'a, F, const D: usize> RecursiveWitnessView<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn evaluate_ring(&self, scalars: &[F]) -> CyclotomicRing<F, D> {
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

    pub fn fold_blocks(
        &self,
        scalars: &[F],
        block_len: usize,
        num_blocks: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (col_idx, &scalar) in scalars.iter().take(block_len).enumerate() {
                    let Some(ring) = self.block_elem(block_idx, col_idx, num_blocks) else {
                        break;
                    };
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

    pub fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
        num_blocks: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        cfg_into_iter!(0..num_blocks)
            .map(|block_idx| {
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (col_idx, scalar) in scalars.iter().take(block_len).enumerate() {
                    let Some(digits) = self.block_elem(block_idx, col_idx, num_blocks) else {
                        break;
                    };
                    let ring = CyclotomicRing::<F, D>::from_coefficients(
                        digits.map(|digit| F::from_i8(digit)),
                    );
                    ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
                }
                acc
            })
            .collect()
    }

    pub fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
        num_blocks: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks(fold_scalars, block_len, num_blocks);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        (eval, folded)
    }

    pub fn evaluate_and_fold_ring(
        &self,
        eval_outer_scalars: &[CyclotomicRing<F, D>],
        fold_scalars: &[CyclotomicRing<F, D>],
        block_len: usize,
        num_blocks: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        let folded = self.fold_blocks_ring(fold_scalars, block_len, num_blocks);
        let eval = folded
            .iter()
            .zip(eval_outer_scalars.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + (*f_i * *s_i)
            });
        (eval, folded)
    }

    #[tracing::instrument(skip_all, name = "RecursiveWitnessView::decompose_fold")]
    pub fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_blocks: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let inner_width = block_len * num_digits;
        let active_blocks = challenges.len().min(num_blocks);

        let q = (-F::one()).to_canonical_u128() + 1;
        let coeffs = self.coeffs;
        let coeff_accum = balanced_digit_decompose_fold_partitioned::<D>(
            coeffs,
            challenges,
            active_blocks,
            block_len,
            num_blocks,
            num_digits,
            inner_width,
        );
        build_decompose_fold_witness::<F, D>(coeff_accum, q)
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn commit_inner<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_rows: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<FlatDigitBlocks<D>, AkitaError>
    where
        B: CommitComputeBackend<F>,
    {
        let t_all = backend.recursive_witness_commit_rows(
            prepared,
            RecursiveWitnessCommitRowsPlan {
                coeffs: self.coeffs,
                n_rows,
                block_len,
                num_blocks,
                num_digits_commit,
                log_basis,
            },
        )?;

        let block_sizes: Vec<usize> = t_all
            .iter()
            .map(|t_i| t_i.len() * num_digits_open)
            .collect();
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t_all))
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t_all.iter())
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        Ok(t_hat)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub fn commit_inner_witness<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        n_rows: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<CommitInnerWitness<F, D>, AkitaError>
    where
        B: CommitComputeBackend<F>,
    {
        let t = backend.recursive_witness_commit_rows(
            prepared,
            RecursiveWitnessCommitRowsPlan {
                coeffs: self.coeffs,
                n_rows,
                block_len,
                num_blocks,
                num_digits_commit,
                log_basis,
            },
        )?;

        let block_sizes: Vec<usize> = t.iter().map(|t_i| t_i.len() * num_digits_open).collect();
        let mut t_hat = FlatDigitBlocks::zeroed(block_sizes)?;
        let dst_blocks = t_hat.split_blocks_mut();
        #[cfg(feature = "parallel")]
        cfg_into_iter!(dst_blocks)
            .zip(cfg_iter!(t))
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        #[cfg(not(feature = "parallel"))]
        dst_blocks
            .into_iter()
            .zip(t.iter())
            .for_each(|(dst, t_i)| decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis));
        Ok(CommitInnerWitness {
            recomposed_inner_rows: t,
            decomposed_inner_rows: t_hat,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn logical_rows_use_strided_column_major_indices() {
        let digits: Vec<i8> = (0..20).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w
            .view::<akita_field::Prime128OffsetA7F7, 2>()
            .expect("view");
        let num_blocks = 4;
        let block_len = (w.len() / 2).div_ceil(num_blocks);

        let row = |block_idx: usize| -> Vec<[i8; 2]> {
            (0..block_len)
                .filter_map(|col_idx| view.block_elem(block_idx, col_idx, num_blocks).copied())
                .collect()
        };

        assert_eq!(row(0), vec![[0, 1], [8, 9], [16, 17]]);
        assert_eq!(row(1), vec![[2, 3], [10, 11], [18, 19]]);
        assert_eq!(row(2), vec![[4, 5], [12, 13]]);
        assert_eq!(row(3), vec![[6, 7], [14, 15]]);
    }

    fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_u64(offset + idx as u64 + 1)
        }))
    }

    #[test]
    fn ring_fold_matches_dense_multiplication_reference() {
        const D: usize = 4;
        let digits = vec![1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12];
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let scalars = vec![ring::<D>(10), ring::<D>(20)];
        let got = view.fold_blocks_ring(&scalars, 2, 2);

        let expected = (0..2)
            .map(|block_idx| {
                (0..2).fold(CyclotomicRing::<F, D>::zero(), |acc, col_idx| {
                    let Some(digits) = view.block_elem(block_idx, col_idx, 2) else {
                        return acc;
                    };
                    let coeff = CyclotomicRing::from_coefficients(digits.map(F::from_i8));
                    acc + coeff * scalars[col_idx]
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(got, expected);
    }
}
