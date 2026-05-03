//! Dense polynomial: all ring coefficients materialized in memory.
//!
//! [`DensePoly`] implements [`HachiPolyOps`](super::HachiPolyOps) via standard
//! dense algorithms — balanced-digit decomposition, NTT-based matrix-vector
//! multiply, and parallel block folds.

use crate::protocol::commitment::utils::crt_ntt::NttSlotCache;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_i8_into, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_i8_dense_single_row,
    try_centered_i8,
};
use crate::protocol::hachi_poly_ops::helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness,
    decompose_ring_single_digit, sparse_mul_acc, try_small_i8_cache_from_ring_coeffs,
    DecomposeParams,
};
use crate::protocol::hachi_poly_ops::{CommitInnerWitness, DecomposeFoldWitness, HachiPolyOps};
use crate::{CanonicalField, FieldCore};
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::ring::sparse_challenge::SparseChallenge;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::HachiError;
use akita_types::{DirectWitnessProof, FlatDigitBlocks, FlatRingVec};

/// Dense polynomial: all ring coefficients materialized in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePoly<F: FieldCore, const D: usize> {
    /// Actual multilinear variable count of the source witness.
    num_vars: usize,
    /// Ring coefficients in sequential block order.
    pub coeffs: Vec<CyclotomicRing<F, D>>,
    small_i8_coeffs: Option<Vec<[i8; D]>>,
}

impl<F: FieldCore + CanonicalField, const D: usize> DensePoly<F, D> {
    /// Pack field-element evaluations into ring elements.
    ///
    /// The first `α = log₂(D)` variables become coefficient slots within each
    /// ring element; the remaining variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two or if
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, HachiError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(HachiError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let expected_len = 1usize
            .checked_shl(num_vars as u32)
            .ok_or_else(|| HachiError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(HachiError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let outer_len = expected_len.div_ceil(D);
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut coeffs = Vec::with_capacity(outer_len);
        let mut small_i8_coeffs = Vec::with_capacity(outer_len);
        let mut all_small_i8 = true;

        for i in 0..outer_len {
            let start = i * D;
            let end = ((i + 1) * D).min(expected_len);
            let slice = &evals[start..end];
            let mut ring = CyclotomicRing::<F, D>::zero();
            for (coeff_idx, coeff) in slice.iter().enumerate() {
                ring.coeffs[coeff_idx] = *coeff;
            }
            coeffs.push(ring);

            if all_small_i8 {
                let mut digits = [0i8; D];
                for (coeff_idx, coeff) in slice.iter().enumerate() {
                    if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                        digits[coeff_idx] = centered;
                    } else {
                        all_small_i8 = false;
                        break;
                    }
                }
                if all_small_i8 {
                    small_i8_coeffs.push(digits);
                }
            }
        }

        Ok(Self {
            num_vars,
            coeffs,
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
        })
    }

    /// Wrap an existing vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() * D` overflows `usize`.
    pub fn from_ring_coeffs(coeffs: Vec<CyclotomicRing<F, D>>) -> Self {
        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs);
        let total = coeffs
            .len()
            .checked_mul(D)
            .expect("ring elems * D overflow");
        Self {
            num_vars: total.trailing_zeros() as usize,
            coeffs,
            small_i8_coeffs,
        }
    }
}

impl<F, const D: usize> HachiPolyOps<F, D> for DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    type CommitCache = NttSlotCache<D>;

    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fold_blocks(&self, scalars: &[F], block_len: usize) -> Vec<CyclotomicRing<F, D>> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);
        cfg_into_iter!(0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                let end = (start + block_len).min(n);
                let block = &self.coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    acc += b_j.scale(&a_j);
                }
                acc
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F, D> {
        let n = self.coeffs.len();
        let coeffs = &self.coeffs;

        let q = (-F::one()).to_canonical_u128() + 1;
        let threshold = decompose_centering_threshold(num_digits, log_basis, q);
        let params = DecomposeParams {
            threshold,
            q,
            mask: (1i128 << log_basis) - 1,
            half_b: 1i128 << (log_basis - 1),
            b_val: 1i128 << log_basis,
            log_basis,
            overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
        };

        if num_digits == 1 {
            if let Some(small_coeffs) = &self.small_i8_coeffs {
                let coeff_accum: Vec<[i32; D]> = {
                    let _span =
                        tracing::info_span!("dense_single_digit_cached_accumulate").entered();
                    cfg_into_iter!(0..block_len)
                        .map(|elem_idx| {
                            let mut z_local = [0i32; D];

                            for (block_idx, c_i) in challenges.iter().enumerate() {
                                let global_idx = block_idx * block_len + elem_idx;
                                if global_idx >= small_coeffs.len() {
                                    continue;
                                }
                                sparse_mul_acc::<D>(&small_coeffs[global_idx], c_i, &mut z_local);
                            }

                            z_local
                        })
                        .collect()
                };

                let _span = tracing::info_span!("dense_single_digit_convert").entered();
                return build_decompose_fold_witness::<F, D>(coeff_accum, params.q);
            }

            let coeff_accum: Vec<[i32; D]> = {
                let _span = tracing::info_span!("dense_single_digit_accumulate").entered();
                cfg_into_iter!(0..block_len)
                    .map(|elem_idx| {
                        let mut z_local = [0i32; D];
                        let mut digit_plane = [0i8; D];

                        for (block_idx, c_i) in challenges.iter().enumerate() {
                            let global_idx = block_idx * block_len + elem_idx;
                            if global_idx >= n {
                                continue;
                            }
                            let ring = &coeffs[global_idx];
                            decompose_ring_single_digit::<F, D>(ring, &mut digit_plane, &params);
                            sparse_mul_acc::<D>(&digit_plane, c_i, &mut z_local);
                        }

                        z_local
                    })
                    .collect()
            };

            let _span = tracing::info_span!("dense_single_digit_convert").entered();
            return build_decompose_fold_witness::<F, D>(coeff_accum, params.q);
        }

        let centered_coeffs = {
            let _span = tracing::info_span!("dense_multi_digit_accumulate").entered();
            balanced_ring_decompose_fold_partitioned::<F, D>(
                coeffs, challenges, block_len, num_digits, &params,
            )
        };

        let _span = tracing::info_span!("dense_multi_digit_convert").entered();
        build_decompose_fold_witness::<F, D>(centered_coeffs, params.q)
    }

    #[tracing::instrument(skip_all, name = "DensePoly::commit_inner")]
    fn commit_inner(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<FlatDigitBlocks<D>, HachiError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
                }
            })
            .collect();

        if n_a == 1 {
            let t = mat_vec_mul_ntt_i8_dense_single_row(
                ntt_a,
                matrix_stride,
                &block_slices,
                num_digits_commit,
                log_basis,
            );
            let mut t_hat = FlatDigitBlocks::zeroed(vec![num_digits_open; t.len()])?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(
                        std::slice::from_ref(t_i),
                        dst,
                        num_digits_open,
                        log_basis,
                    )
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks.into_iter().zip(t.iter()).for_each(|(dst, t_i)| {
                decompose_rows_i8_into(std::slice::from_ref(t_i), dst, num_digits_open, log_basis)
            });
            return Ok(t_hat);
        }

        let t_all = mat_vec_mul_ntt_i8_dense(
            ntt_a,
            n_a,
            matrix_stride,
            &block_slices,
            num_digits_commit,
            log_basis,
        );

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

    fn commit_inner_witness(
        &self,
        _a_matrix: &FlatMatrix<F>,
        ntt_a: &NttSlotCache<D>,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        log_basis: u32,
        matrix_stride: usize,
    ) -> Result<CommitInnerWitness<F, D>, HachiError> {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
                }
            })
            .collect();

        if n_a == 1 {
            let t_single = mat_vec_mul_ntt_i8_dense_single_row(
                ntt_a,
                matrix_stride,
                &block_slices,
                num_digits_commit,
                log_basis,
            );
            let mut t_hat = FlatDigitBlocks::zeroed(vec![num_digits_open; t_single.len()])?;
            let dst_blocks = t_hat.split_blocks_mut();
            #[cfg(feature = "parallel")]
            cfg_into_iter!(dst_blocks)
                .zip(cfg_iter!(t_single))
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(
                        std::slice::from_ref(t_i),
                        dst,
                        num_digits_open,
                        log_basis,
                    )
                });
            #[cfg(not(feature = "parallel"))]
            dst_blocks
                .into_iter()
                .zip(t_single.iter())
                .for_each(|(dst, t_i)| {
                    decompose_rows_i8_into(
                        std::slice::from_ref(t_i),
                        dst,
                        num_digits_open,
                        log_basis,
                    )
                });
            let t = t_single.into_iter().map(|ring| vec![ring]).collect();
            return Ok(CommitInnerWitness { t, t_hat });
        }

        let t = mat_vec_mul_ntt_i8_dense(
            ntt_a,
            n_a,
            matrix_stride,
            &block_slices,
            num_digits_commit,
            log_basis,
        );
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

    fn direct_root_witness(&self) -> Result<DirectWitnessProof<F>, HachiError> {
        let live_len = 1usize.checked_shl(self.num_vars as u32).ok_or_else(|| {
            HachiError::InvalidInput(format!("2^{} does not fit usize", self.num_vars))
        })?;
        let mut coeffs = Vec::with_capacity(live_len);
        let mut remaining = live_len;
        for ring in &self.coeffs {
            let take = remaining.min(D);
            coeffs.extend_from_slice(&ring.coefficients()[..take]);
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        Ok(DirectWitnessProof::FieldElements(FlatRingVec::from_coeffs(
            coeffs,
        )))
    }
}

/// Test-only helpers for [`DensePoly`].
///
/// These live outside the production `HachiPolyOps` trait because they are
/// only used by cross-check tests (e.g. verifying that fused prover paths
/// match a straight-line reference implementation).
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::DensePoly;
    use crate::FieldCore;
    use akita_algebra::CyclotomicRing;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// Reference ring-space evaluation for [`DensePoly`].
    ///
    /// Computes the global weighted sum `y = Σᵢ scalars[i] · self.coeffs[i]`.
    #[allow(dead_code)]
    pub(crate) fn evaluate_ring_dense<F, const D: usize>(
        poly: &DensePoly<F, D>,
        scalars: &[F],
    ) -> CyclotomicRing<F, D>
    where
        F: FieldCore,
    {
        #[cfg(feature = "parallel")]
        {
            poly.coeffs
                .par_iter()
                .zip(scalars.par_iter())
                .fold(
                    || CyclotomicRing::<F, D>::zero(),
                    |acc, (f_i, w_i)| acc + f_i.scale(w_i),
                )
                .reduce(|| CyclotomicRing::<F, D>::zero(), |a, b| a + b)
        }
        #[cfg(not(feature = "parallel"))]
        {
            poly.coeffs
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, w_i)| {
                    acc + f_i.scale(w_i)
                })
        }
    }
}
