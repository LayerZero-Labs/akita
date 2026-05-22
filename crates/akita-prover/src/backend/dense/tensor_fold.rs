use super::DensePoly;
use crate::backend::poly_helpers::{
    build_decompose_fold_witness, integer_mul_acc, DecomposeParams,
};
use crate::backend::tensor_fold::balanced_ring_decompose_fold_tensor_partitioned;
use crate::{CenteredCoeff, DecomposeFoldWitness};
use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_challenges::{IntegerChallenge, TensorChallengeSet};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};

impl<F: FieldCore + CanonicalField, const D: usize> DensePoly<F, D> {
    pub(super) fn decompose_fold_batched_tensor_dense(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError> {
        if polys.is_empty() {
            return Ok(None);
        }
        let q = (-F::one()).to_canonical_u128() + 1;
        if let Some(digit_planes) = polys
            .iter()
            .map(|poly| poly.digit_planes_for(num_digits, log_basis))
            .collect::<Option<Vec<_>>>()
        {
            let centered_coeffs = {
                let _span = tracing::info_span!("dense_tensor_cached_digit_accumulate").entered();
                accumulate_cached_digit_planes_tensor::<D>(
                    &digit_planes,
                    tensor,
                    block_len,
                    num_digits,
                )?
            };
            let _span = tracing::info_span!("dense_tensor_cached_digit_convert").entered();
            return Ok(Some(build_decompose_fold_witness::<F, D>(
                centered_coeffs,
                q,
            )));
        }

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
        let coeff_slices = polys
            .iter()
            .map(|poly| poly.coeffs.as_slice())
            .collect::<Vec<_>>();
        let centered_coeffs = {
            let _span = tracing::info_span!("dense_tensor_accumulate").entered();
            balanced_ring_decompose_fold_tensor_partitioned::<F, D>(
                &coeff_slices,
                tensor,
                block_len,
                num_digits,
                &params,
            )?
        };
        let _span = tracing::info_span!("dense_tensor_convert").entered();
        Ok(Some(build_decompose_fold_witness::<F, D>(
            centered_coeffs,
            params.q,
        )))
    }
}

fn accumulate_cached_digit_planes_tensor<const D: usize>(
    digit_planes_by_poly: &[&[[i8; D]]],
    tensor: &TensorChallengeSet,
    block_len: usize,
    num_digits: usize,
) -> Result<Vec<[CenteredCoeff; D]>, AkitaError> {
    if block_len == 0 || num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "dense cached tensor decompose-fold requires non-zero block_len and num_digits"
                .to_string(),
        ));
    }
    if digit_planes_by_poly.len() != tensor.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: tensor.num_claims(),
            actual: digit_planes_by_poly.len(),
        });
    }
    tensor.validate::<D>()?;
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

            for block_idx in 0..total_blocks {
                let (claim_idx, local_block_idx, left, right) =
                    tensor.factors_for_logical_block(block_idx)?;
                let challenge = IntegerChallenge::tensor_product::<D>(left, right)?;
                let digit_planes = digit_planes_by_poly[claim_idx];

                for elem_idx in elem_start..elem_end {
                    let ring_idx = local_block_idx * block_len + elem_idx;
                    let plane_base = ring_idx * num_digits;
                    if plane_base >= digit_planes.len() {
                        continue;
                    }
                    let out_base = (elem_idx - elem_start) * num_digits;
                    for digit_idx in 0..num_digits {
                        let Some(digit_plane) = digit_planes.get(plane_base + digit_idx) else {
                            continue;
                        };
                        integer_mul_acc::<D>(
                            digit_plane,
                            &challenge,
                            &mut acc[out_base + digit_idx],
                        );
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}
