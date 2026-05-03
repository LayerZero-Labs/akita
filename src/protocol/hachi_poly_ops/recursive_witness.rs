//! Recursive witness helpers for later Hachi prove levels.
//!
//! Recursive levels do not operate on a caller-provided polynomial anymore.
//! Instead they carry a flat digit witness `w` that is re-chunked under the
//! current ring dimension `D` on demand. [`RecursiveWitnessFlat`] owns the
//! D-agnostic digit buffer, while [`RecursiveWitnessView`] provides the
//! zero-copy D-specific operations used by recursive folding and handoff paths.

use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_digits_i8_strided, mat_vec_mul_ntt_i8_strided,
};
use crate::protocol::hachi_poly_ops::helpers::{
    balanced_digit_decompose_fold_partitioned, build_decompose_fold_witness,
};
use crate::{CanonicalField, FieldCore};
use akita_algebra::ring::sparse_challenge::SparseChallenge;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::HachiError;
use akita_prover::{CommitInnerWitness, DecomposeFoldWitness};
use akita_types::FlatDigitBlocks;
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
    fn block_elem(
        &self,
        block_idx: usize,
        col_idx: usize,
        num_blocks: usize,
    ) -> Option<&'a [i8; D]> {
        self.coeffs.get(block_idx + col_idx * num_blocks)
    }

    pub(crate) fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }
}

impl<'a, F, const D: usize> RecursiveWitnessView<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    #[cfg(test)]
    #[allow(dead_code)]
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

    pub(crate) fn fold_blocks(
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

    pub(crate) fn evaluate_and_fold(
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

    #[tracing::instrument(skip_all, name = "RecursiveWitnessView::decompose_fold")]
    pub(crate) fn decompose_fold(
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
    pub(crate) fn commit_inner(
        &self,
        ntt_a: &NttSlotCache<D>,
        n_rows: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, HachiError> {
        let t_all = if num_digits_commit == 1 {
            mat_vec_mul_ntt_digits_i8_strided(
                ntt_a,
                n_rows,
                matrix_stride,
                self.coeffs,
                num_blocks,
                block_len,
            )
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = self
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            mat_vec_mul_ntt_i8_strided(
                ntt_a,
                n_rows,
                matrix_stride,
                &ring_elems,
                num_blocks,
                block_len,
                num_digits_commit,
                log_basis,
            )
        };

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
    pub(crate) fn commit_inner_witness(
        &self,
        ntt_a: &NttSlotCache<D>,
        n_rows: usize,
        block_len: usize,
        num_blocks: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let t = if num_digits_commit == 1 {
            mat_vec_mul_ntt_digits_i8_strided(
                ntt_a,
                n_rows,
                matrix_stride,
                self.coeffs,
                num_blocks,
                block_len,
            )
        } else {
            let ring_elems: Vec<CyclotomicRing<F, D>> = self
                .coeffs
                .iter()
                .map(|digit| {
                    let coeffs = from_fn(|k| F::from_i8(digit[k]));
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect();
            mat_vec_mul_ntt_i8_strided(
                ntt_a,
                n_rows,
                matrix_stride,
                &ring_elems,
                num_blocks,
                block_len,
                num_digits_commit,
                log_basis,
            )
        };

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
        Ok(CommitInnerWitness { t, t_hat })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_rows_use_strided_column_major_indices() {
        let digits: Vec<i8> = (0..20).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w
            .view::<crate::protocol::config::proof_optimized::fp128::Field, 2>()
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
}
