use super::{SparseRingBlockEntry, SparseRingPoly};
use crate::backend::poly_helpers::build_decompose_fold_witness;
use crate::backend::tensor_fold::{
    fill_rotated_sparse_challenge_i64, narrow_tensor_accum_to_i32, sparse_i64_mul_acc_i64,
    validate_tensor_blocks,
};
use crate::DecomposeFoldWitness;
use akita_challenges::TensorChallenges as TensorChallengeSet;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};

pub(super) fn decompose_fold_batched_tensor_sparse<F, const D: usize>(
    polys: &[&SparseRingPoly<F, D>],
    tensor: &TensorChallengeSet,
    block_len: usize,
    num_digits: usize,
) -> Result<DecomposeFoldWitness<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let mut flat_blocks = Vec::new();
    for poly in polys {
        let blocks = poly.blocks_for(block_len)?;
        flat_blocks.extend((0..blocks.num_blocks()).map(|idx| blocks.block(idx)));
    }
    let expected_blocks = tensor.total_blocks()?;
    if flat_blocks.len() != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: flat_blocks.len(),
        });
    }
    validate_tensor_blocks::<D>(tensor, expected_blocks)?;
    let inner_width = block_len.checked_mul(num_digits).ok_or_else(|| {
        AkitaError::InvalidSetup("sparse tensor fold inner width overflow".to_string())
    })?;
    let accum_i64 = sparse_accumulate_tensor::<D>(&flat_blocks, tensor, inner_width, num_digits)?;
    let coeff_accum = narrow_tensor_accum_to_i32::<D>(accum_i64)?;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    Ok(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
}

fn sparse_accumulate_tensor<const D: usize>(
    blocks: &[&[SparseRingBlockEntry]],
    tensor: &TensorChallengeSet,
    inner_width: usize,
    num_digits: usize,
) -> Result<Vec<[i64; D]>, AkitaError> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);
    let chunks: Vec<Vec<[i64; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let mut acc = vec![[0i64; D]; pos_end - pos_start];
            let mut tmp = vec![[0i64; D]; pos_end - pos_start];
            let mut rotated = vec![[0i64; D]; D];

            for claim_idx in 0..tensor.num_claims {
                for left_idx in 0..tensor.left_len {
                    tmp.fill([0i64; D]);
                    for right_idx in 0..tensor.right_len {
                        let block_idx = claim_idx * tensor.left_len * tensor.right_len
                            + left_idx * tensor.right_len
                            + right_idx;
                        let entries = blocks[block_idx];
                        let lo =
                            entries.partition_point(|e| e.pos_in_block() * num_digits < pos_start);
                        let hi =
                            entries.partition_point(|e| e.pos_in_block() * num_digits < pos_end);
                        if lo >= hi {
                            continue;
                        }
                        let right = &tensor.right[claim_idx * tensor.right_len + right_idx];
                        fill_rotated_sparse_challenge_i64::<D>(&mut rotated, right);
                        for entry in &entries[lo..hi] {
                            let local_pos = entry.pos_in_block() * num_digits - pos_start;
                            let rot = &rotated[entry.coeff_idx()];
                            let dst = &mut tmp[local_pos];
                            let weight = i64::from(entry.value);
                            for k in 0..D {
                                dst[k] += weight * rot[k];
                            }
                        }
                    }
                    let left = &tensor.left[claim_idx * tensor.left_len + left_idx];
                    for (src, dst) in tmp.iter().zip(acc.iter_mut()) {
                        sparse_i64_mul_acc_i64::<D>(src, left, dst);
                    }
                }
            }
            acc
        })
        .collect();
    Ok(chunks.into_iter().flatten().collect())
}
