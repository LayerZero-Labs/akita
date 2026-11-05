//! Dense polynomial opening and fold operations.
//!
//! Storage is D-free; every ring-shaped operation takes the ring dimension as
//! a method const generic and views the flat coefficients at kernel entry.

use super::poly::DensePoly;
use crate::opaque::DecomposeFoldWitness;
use crate::sources::block_digits::{CachedDigits, DigitShape, LiveRings};
use crate::sources::poly_helpers::{element_partitioned_decompose_fold, SignedDigitBasis};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_types::SubfieldMultiplierOpeningPoint;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};

/// Reject a digit shape the dense fold kernels cannot run, before any cache
/// build or decomposition setup can panic on it.
fn validate_digit_shape(num_digits: usize, log_basis: u32) -> Result<(), AkitaError> {
    SignedDigitBasis::new(log_basis)?;
    if num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense fold digit count must be nonzero".into(),
        ));
    }
    Ok(())
}

impl<F> DensePoly<F>
where
    F: Field + CanonicalEncoding,
{
    pub(crate) fn decompose_fold_chunked<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        chunk_ranges: &[std::ops::Range<usize>],
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Vec<DecomposeFoldWitness>, AkitaError> {
        validate_digit_shape(num_digits, log_basis)?;
        let coeffs = self.nonzero_ring_coeffs::<D>()?;
        let shape = DigitShape {
            num_digits,
            log_basis,
        };
        let rows = if let Some((planes, basis)) = self.digit_planes_for::<D>(num_digits, log_basis)
        {
            element_partitioned_decompose_fold::<_, D>(
                &CachedDigits::new(planes, num_digits, basis)?,
                challenges,
                chunk_ranges,
                num_positions_per_block,
                shape,
            )?
        } else {
            let params = BalancedDecomposePow2Params::new(num_digits, log_basis);
            element_partitioned_decompose_fold::<_, D>(
                &LiveRings {
                    coeffs,
                    params: &params,
                    basis: SignedDigitBasis::new(log_basis)?,
                },
                challenges,
                chunk_ranges,
                num_positions_per_block,
                shape,
            )?
        };
        Ok(rows
            .into_iter()
            .map(DecomposeFoldWitness::from_centered_rows)
            .collect())
    }

    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let coeffs = self.nonzero_ring_coeffs::<D>()?;
        let n = coeffs.len();
        let num_live_blocks = self.num_ring_elems_at(D).div_ceil(num_positions_per_block);
        Ok(cfg_into_iter!(0..num_live_blocks)
            .map(|i| {
                let start = (i * num_positions_per_block).min(n);
                let end = (start + num_positions_per_block).min(n);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.scale_accumulate_into(&mut acc, a_j);
                }
                acc
            })
            .collect())
    }

    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let coeffs = self.nonzero_ring_coeffs::<D>()?;
        let n = coeffs.len();
        let num_live_blocks = self.num_ring_elems_at(D).div_ceil(num_positions_per_block);
        Ok(cfg_into_iter!(0..num_live_blocks)
            .map(|i| {
                let start = (i * num_positions_per_block).min(n);
                let end = (start + num_positions_per_block).min(n);
                let block = &coeffs[start..end];
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (b_j, &a_j) in block.iter().zip(scalars.iter()) {
                    b_j.mul_accumulate_sparse_rhs_into(&a_j, &mut acc);
                }
                acc
            })
            .collect())
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        Ok(crate::sources::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks::<D>(position_weights, num_positions_per_block)?,
            live_block_weights,
        ))
    }

    pub(crate) fn evaluate_and_fold_subfield<const D: usize>(
        &self,
        multipliers: &SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let position_weights = multipliers.materialize_position_rings::<D>()?;
        let live_block_weights = multipliers.materialize_fold_rings::<D>()?;
        Ok(
            crate::sources::poly_helpers::fused_evaluate_and_fold_materialized(
                self.fold_blocks_ring(&position_weights, num_positions_per_block)?,
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
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        self.decompose_fold_chunked::<D>(
            challenges,
            std::slice::from_ref(&(0..challenges.len())),
            num_positions_per_block,
            num_digits,
            log_basis,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| AkitaError::InvalidInput("dense fold produced no chunk".into()))
    }
}
