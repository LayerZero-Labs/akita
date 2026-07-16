use super::*;

/// Accumulates one-hot decompose-fold rows in compressed position order.
///
/// The returned vector has `positions_per_block` rows. Callers expand each row across
/// `num_digits` later, inserting zero rows for higher digit planes.
///
/// `blocks` is a slice-of-slices view over per-block entries. Both
/// single-polynomial callers (which collect once via `FlatBlocks::block`)
/// and batched callers (which concatenate slices across polynomials) feed
/// through the same signature.
pub(super) fn onehot_accumulate<E, const D: usize>(
    blocks: &[&[E]],
    challenges: &[SparseChallenge],
    live_block_count: usize,
    positions_per_block: usize,
) -> Vec<[i32; D]>
where
    E: OneHotEntry,
{
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(positions_per_block).max(1);
    let pos_chunk = positions_per_block.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= positions_per_block {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(positions_per_block);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(live_block_count) {
                let entries = blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);

                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    for &ci in entry.coeffs() {
                        let rot = &rotated[ci as usize];
                        for k in 0..D {
                            dst[k] += rot[k] as i32;
                        }
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}

// Tensor accumulators use `[i64; D]` because each per-block challenge is a
// product of two sparse samples. The witness boundary narrows back to
// `[i32; D]` after checking the selected schedule's coefficient envelope.

pub(super) fn onehot_accumulate_tensor<E, const D: usize>(
    blocks: &[&[E]],
    tensor: &TensorChallengeSet,
    live_block_count: usize,
    positions_per_block: usize,
) -> Result<Vec<[i64; D]>, AkitaError>
where
    E: OneHotEntry,
{
    let tensor_blocks = tensor.total_blocks()?;
    if tensor_blocks != live_block_count {
        return Err(AkitaError::InvalidSize {
            expected: live_block_count,
            actual: tensor_blocks,
        });
    }
    if blocks.len() != live_block_count {
        return Err(AkitaError::InvalidSize {
            expected: live_block_count,
            actual: blocks.len(),
        });
    }
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(positions_per_block).max(1);
    let pos_chunk = positions_per_block.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i64; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= positions_per_block {
                return Ok(Vec::new());
            }
            let pos_end = (pos_start + pos_chunk).min(positions_per_block);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i64; D]; len];
            let mut tmp = vec![[0i64; D]; len];
            let mut rotated = vec![[0i64; D]; D];

            for claim_idx in 0..tensor.num_claims {
                for high_idx in 0..tensor.fold_high_len() {
                    tmp.fill([0i64; D]);
                    for low_idx in 0..tensor.fold_low_len {
                        let local_block = high_idx * tensor.fold_low_len + low_idx;
                        if local_block >= tensor.live_blocks_per_claim {
                            break;
                        }
                        let block_idx = claim_idx * tensor.live_blocks_per_claim + local_block;
                        let entries = blocks[block_idx];
                        let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                        let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                        if lo >= hi {
                            continue;
                        }

                        let fold_low = &tensor.fold_low[claim_idx * tensor.fold_low_len + low_idx];
                        fill_rotated_sparse_challenge_i64::<D>(&mut rotated, fold_low);

                        for entry in &entries[lo..hi] {
                            let dst = &mut tmp[entry.pos_in_block() - pos_start];
                            for &ci in entry.coeffs() {
                                let rot = &rotated[ci as usize];
                                for k in 0..D {
                                    dst[k] += rot[k];
                                }
                            }
                        }
                    }
                    let fold_high =
                        &tensor.fold_high[claim_idx * tensor.fold_high_len() + high_idx];
                    for (src, dst) in tmp.iter().zip(acc.iter_mut()) {
                        sparse_i64_mul_acc_i64::<D>(src, fold_high, dst);
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<_, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}
