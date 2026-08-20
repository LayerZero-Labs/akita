//! Dense polynomial opening and fold operations.
//!
//! Storage is D-free; every ring-shaped operation takes the ring dimension as
//! a method const generic and views the flat coefficients at kernel entry.

use super::poly::DensePoly;
use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness,
    cached_digit_decompose_fold_partitioned, decompose_ring_single_digit, sparse_mul_acc,
    DecomposeParams,
};
use crate::DecomposeFoldWitness;
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
use akita_types::SubfieldMultiplierOpeningPoint;

impl<F> DensePoly<F>
where
    F: FieldCore + CanonicalField,
{
    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let coeffs = self
            .ring_coeffs::<D>()
            .expect("DensePoly::fold_blocks: invalid ring view");
        let n = coeffs.len();
        let num_live_blocks = n.div_ceil(num_positions_per_block);
        cfg_into_iter!(0..num_live_blocks)
            .map(|i| {
                let start = i * num_positions_per_block;
                let end = (start + num_positions_per_block).min(n);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.scale_accumulate_into(&mut acc, a_j);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let coeffs = self
            .ring_coeffs::<D>()
            .expect("DensePoly::fold_blocks_ring: invalid ring view");
        let n = coeffs.len();
        let num_live_blocks = n.div_ceil(num_positions_per_block);
        cfg_into_iter!(0..num_live_blocks)
            .map(|i| {
                let start = i * num_positions_per_block;
                let end = (start + num_positions_per_block).min(n);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.mul_accumulate_sparse_rhs_into(&a_j, &mut acc);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks::<D>(position_weights, num_positions_per_block),
            live_block_weights,
        )
    }

    pub(crate) fn evaluate_and_fold_subfield<const D: usize>(
        &self,
        multipliers: &SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let position_weights = multipliers.materialize_position_rings::<D>()?;
        let live_block_weights = multipliers.materialize_fold_rings::<D>()?;
        Ok(
            crate::backend::poly_helpers::fused_evaluate_and_fold_materialized(
                self.fold_blocks_ring(&position_weights, num_positions_per_block),
                &live_block_weights,
            ),
        )
    }

    #[tracing::instrument(skip_all, name = "DensePoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> DecomposeFoldWitness<F> {
        let coeffs = self
            .ring_coeffs::<D>()
            .expect("DensePoly::decompose_fold: invalid ring view");
        let n = coeffs.len();

        if let Some(digit_planes) = self.digit_planes_for::<D>(num_digits, log_basis) {
            let coeff_accum = {
                let _span = tracing::info_span!("dense_cached_digit_accumulate").entered();
                cached_digit_decompose_fold_partitioned::<F, D>(
                    digit_planes,
                    challenges,
                    num_positions_per_block,
                    num_digits,
                    log_basis,
                )
            };
            let modulus = (-F::one()).to_canonical_u128() + 1;
            return build_decompose_fold_witness::<F, D>(coeff_accum, modulus);
        }

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
            if let Some(small_coeffs) = self.small_i8_ring_coeffs::<D>() {
                let coeff_accum: Vec<[i32; D]> = {
                    let _span =
                        tracing::info_span!("dense_single_digit_cached_accumulate").entered();
                    cfg_into_iter!(0..num_positions_per_block)
                        .map(|elem_idx| {
                            let mut z_local = [0i32; D];

                            for (block_idx, c_i) in challenges.iter().enumerate() {
                                let global_idx = block_idx * num_positions_per_block + elem_idx;
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
                cfg_into_iter!(0..num_positions_per_block)
                    .map(|elem_idx| {
                        let mut z_local = [0i32; D];
                        let mut digit_plane = [0i8; D];

                        for (block_idx, c_i) in challenges.iter().enumerate() {
                            let global_idx = block_idx * num_positions_per_block + elem_idx;
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
                coeffs,
                challenges,
                num_positions_per_block,
                num_digits,
                &params,
            )
        };

        let _span = tracing::info_span!("dense_multi_digit_convert").entered();
        build_decompose_fold_witness::<F, D>(centered_coeffs, params.q)
    }
}
