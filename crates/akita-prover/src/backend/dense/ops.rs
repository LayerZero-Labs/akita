//! Dense polynomial opening and fold operations.
//!
//! Storage is D-free; every ring-shaped operation takes the ring dimension as
//! a method const generic and views the flat coefficients at kernel entry.

use super::poly::DensePoly;
use crate::backend::packed_digits::PackedSignedDigitView;
use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness,
    decompose_ring_single_digit, packed_digit_decompose_fold_partitioned, sparse_mul_acc,
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

const PACKED_RECONSTRUCTION_CHUNK: usize = 64;

/// Sequential field-coefficient reader for one balanced ring stored in packed
/// digit planes.
///
/// The reader decodes and reconstructs one SIMD-sized coefficient chunk at a
/// time. Operation-specific kernels can consume each chunk immediately instead
/// of materializing a temporary `CyclotomicRing`.
pub(super) struct PackedBalancedRingReader<'a, F: FieldCore, const D: usize> {
    digits: PackedSignedDigitView<'a>,
    num_digits: usize,
    log_basis: u32,
    decoded_start: usize,
    decoded_len: usize,
    decoded: [F; PACKED_RECONSTRUCTION_CHUNK],
}

impl<'a, F, const D: usize> PackedBalancedRingReader<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    pub(super) fn new(
        digit_planes: PackedSignedDigitView<'a>,
        ring_index: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Self, AkitaError> {
        let ring_width = num_digits.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("packed balanced ring width overflow".into())
        })?;
        let start = ring_index.checked_mul(ring_width).ok_or_else(|| {
            AkitaError::InvalidInput("packed balanced ring offset overflow".into())
        })?;
        let end = start.checked_add(ring_width).ok_or_else(|| {
            AkitaError::InvalidInput("packed balanced ring extent overflow".into())
        })?;
        let digit_span = num_digits.checked_mul(log_basis as usize).ok_or_else(|| {
            AkitaError::InvalidInput("packed balanced digit span overflow".into())
        })?;
        if digit_span > 126 {
            return Err(AkitaError::InvalidInput(
                "packed balanced ring exceeds signed-128 reconstruction".into(),
            ));
        }
        Ok(Self {
            digits: digit_planes.slice(start..end)?,
            num_digits,
            log_basis,
            decoded_start: D,
            decoded_len: 0,
            decoded: [F::zero(); PACKED_RECONSTRUCTION_CHUNK],
        })
    }

    fn fill_chunk(&mut self, coefficient: usize) -> Result<(), AkitaError> {
        if coefficient >= D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: coefficient,
            });
        }
        let chunk_start = coefficient / PACKED_RECONSTRUCTION_CHUNK * PACKED_RECONSTRUCTION_CHUNK;
        let chunk_len = (D - chunk_start).min(PACKED_RECONSTRUCTION_CHUNK);
        let mut signed = [0i128; PACKED_RECONSTRUCTION_CHUNK];
        let mut decoded = [0i8; PACKED_RECONSTRUCTION_CHUNK];
        let mut shift = 0usize;
        for digit_index in 0..self.num_digits {
            let digit_start = digit_index
                .checked_mul(D)
                .and_then(|offset| offset.checked_add(chunk_start))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("packed balanced digit offset overflow".into())
                })?;
            self.digits
                .decode_range(digit_start, &mut decoded[..chunk_len])?;
            for (accumulator, &digit) in signed[..chunk_len].iter_mut().zip(&decoded) {
                *accumulator += i128::from(digit) << shift;
            }
            shift += self.log_basis as usize;
        }
        for (output, &value) in self.decoded[..chunk_len].iter_mut().zip(&signed) {
            *output = F::from_i128(value);
        }
        self.decoded_start = chunk_start;
        self.decoded_len = chunk_len;
        Ok(())
    }

    pub(super) fn coefficient(&mut self, index: usize) -> Result<F, AkitaError> {
        if index < self.decoded_start || index >= self.decoded_start + self.decoded_len {
            self.fill_chunk(index)?;
        }
        Ok(self.decoded[index - self.decoded_start])
    }

    fn for_each_chunk(
        &mut self,
        mut consume: impl FnMut(usize, &[F]) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        for start in (0..D).step_by(PACKED_RECONSTRUCTION_CHUNK) {
            self.fill_chunk(start)?;
            consume(start, &self.decoded[..self.decoded_len])?;
        }
        Ok(())
    }

    fn materialize(mut self) -> Result<CyclotomicRing<F, D>, AkitaError> {
        let mut output = CyclotomicRing::zero();
        self.for_each_chunk(|start, coefficients| {
            output.coefficients_mut()[start..start + coefficients.len()]
                .copy_from_slice(coefficients);
            Ok(())
        })?;
        Ok(output)
    }
}

pub(super) fn packed_balanced_fold_blocks<F, const D: usize>(
    digit_planes: PackedSignedDigitView<'_>,
    num_rings: usize,
    num_digits: usize,
    log_basis: u32,
    scalars: &[F],
    num_positions_per_block: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    cfg_into_iter!(0..num_rings.div_ceil(num_positions_per_block))
        .map(|block_index| {
            let start = block_index * num_positions_per_block;
            let end = (start + num_positions_per_block).min(num_rings);
            let mut accumulator = CyclotomicRing::<F, D>::zero();
            for (ring_index, &scalar) in (start..end).zip(scalars) {
                let mut ring = PackedBalancedRingReader::<F, D>::new(
                    digit_planes,
                    ring_index,
                    num_digits,
                    log_basis,
                )?;
                ring.for_each_chunk(|coefficient_start, coefficients| {
                    for (output, &source) in accumulator.coefficients_mut()
                        [coefficient_start..coefficient_start + coefficients.len()]
                        .iter_mut()
                        .zip(coefficients)
                    {
                        *output = source.mul_add(scalar, *output);
                    }
                    Ok(())
                })?;
            }
            Ok(accumulator)
        })
        .collect()
}

pub(super) fn packed_balanced_fold_blocks_ring<F, const D: usize>(
    digit_planes: PackedSignedDigitView<'_>,
    num_rings: usize,
    num_digits: usize,
    log_basis: u32,
    scalars: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    cfg_into_iter!(0..num_rings.div_ceil(num_positions_per_block))
        .map(|block_index| {
            let start = block_index * num_positions_per_block;
            let end = (start + num_positions_per_block).min(num_rings);
            let mut accumulator = CyclotomicRing::<F, D>::zero();
            for (ring_index, scalar) in (start..end).zip(scalars) {
                let ring = PackedBalancedRingReader::<F, D>::new(
                    digit_planes,
                    ring_index,
                    num_digits,
                    log_basis,
                )?
                .materialize()?;
                ring.mul_accumulate_sparse_rhs_into(scalar, &mut accumulator);
            }
            Ok(accumulator)
        })
        .collect()
}

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
                packed_digit_decompose_fold_partitioned::<F, D>(
                    digit_planes,
                    n,
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
