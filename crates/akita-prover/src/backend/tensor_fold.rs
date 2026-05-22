use super::poly_helpers::{decompose_ring_full_challenge_accumulate, DecomposeParams};
use crate::CenteredCoeff;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField};

pub(super) fn balanced_ring_decompose_fold_tensor_partitioned<F: CanonicalField, const D: usize>(
    poly_coeffs: &[&[CyclotomicRing<F, D>]],
    tensor: &TensorChallengeSet,
    block_len: usize,
    num_digits: usize,
    p: &DecomposeParams,
) -> Result<Vec<[CenteredCoeff; D]>, AkitaError> {
    if block_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense tensor decompose-fold requires non-zero block_len and num_digits".to_string(),
        ));
    }
    tensor.validate::<D>()?;
    if poly_coeffs.len() != tensor.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: tensor.num_claims(),
            actual: poly_coeffs.len(),
        });
    }

    let blocks_per_claim = tensor.blocks_per_claim()?;
    let total_blocks = tensor.total_blocks()?;

    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len.max(1)).max(1);
    let elem_chunk = block_len.div_ceil(actual_threads);
    let chunks = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let elem_start = tid * elem_chunk;
            if elem_start >= block_len {
                return Ok(Vec::new());
            }
            let elem_end = (elem_start + elem_chunk).min(block_len);
            let mut acc = vec![[0 as CenteredCoeff; D]; (elem_end - elem_start) * num_digits];
            let mut rotated = [[0 as CenteredCoeff; D]; D];

            for block_idx in 0..total_blocks {
                let claim_idx = block_idx / blocks_per_claim;
                let local_block_idx = block_idx % blocks_per_claim;
                let coeff_start = local_block_idx * block_len + elem_start;
                let coeffs = poly_coeffs[claim_idx];
                if coeff_start >= coeffs.len() {
                    continue;
                }
                let coeff_end = (local_block_idx * block_len + elem_end).min(coeffs.len());
                if coeff_start >= coeff_end {
                    continue;
                }

                let (_, _, left, right) = tensor.factors_for_logical_block(block_idx)?;
                fill_rotated_tensor_challenge::<D>(&mut rotated, left, right)?;
                for (local_elem_idx, ring) in coeffs[coeff_start..coeff_end].iter().enumerate() {
                    let base = local_elem_idx * num_digits;
                    decompose_ring_full_challenge_accumulate::<F, D>(
                        ring,
                        &rotated,
                        &mut acc[base..base + num_digits],
                        p,
                    );
                }
            }

            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}

pub(super) fn fill_rotated_tensor_challenge<const D: usize>(
    table: &mut [[CenteredCoeff; D]],
    left: &SparseChallenge,
    right: &SparseChallenge,
) -> Result<(), AkitaError> {
    debug_assert!(D.is_power_of_two());
    debug_assert!(table.len() >= D);
    let mut dense = [0 as CenteredCoeff; D];
    for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
        for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
            let degree = left_pos as usize + right_pos as usize;
            let (base_pos, base_sign) = if degree < D {
                (degree, 1 as CenteredCoeff)
            } else {
                (degree - D, -1 as CenteredCoeff)
            };
            let coeff = CenteredCoeff::from(left_coeff)
                .checked_mul(CenteredCoeff::from(right_coeff))
                .and_then(|coeff| coeff.checked_mul(base_sign))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("tensor challenge coefficient overflow".to_string())
                })?;
            dense[base_pos] = dense[base_pos].checked_add(coeff).ok_or_else(|| {
                AkitaError::InvalidInput("tensor challenge coefficient overflow".to_string())
            })?;
        }
    }

    for (shift, row) in table.iter_mut().enumerate().take(D) {
        row[shift..D].copy_from_slice(&dense[..D - shift]);
        for (dst, src) in row[..shift].iter_mut().zip(dense[D - shift..].iter()) {
            *dst = -*src;
        }
    }
    Ok(())
}
