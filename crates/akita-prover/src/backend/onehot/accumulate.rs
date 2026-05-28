use super::*;

/// Position-parallel accumulation for multi-chunk one-hot witnesses.
///
/// `multi_chunk_blocks` is a slice-of-slices view over per-block entries.
/// Both single-polynomial callers (which collect once via
/// `FlatBlocks::block`) and batched callers (which concatenate slices
/// across polynomials) feed through the same signature.
pub(super) fn multi_chunk_onehot_accumulate<const D: usize>(
    multi_chunk_blocks: &[&[MultiChunkEntry]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = multi_chunk_blocks[block_idx];
                let lo = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_start);
                let hi = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);

                for entry in &entries[lo..hi] {
                    let local_pos = entry.pos_in_block() * num_digits - pos_start;
                    for &ci in entry.nonzero_coeffs() {
                        let rot = &rotated[ci as usize];
                        let dst = &mut acc[local_pos];
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

/// Position-partitioned accumulation for single-chunk one-hot witnesses,
/// where each nonzero ring element carries exactly one hot coefficient.
///
/// See [`multi_chunk_onehot_accumulate`] for the block-view convention.
pub(super) fn single_chunk_onehot_accumulate<const D: usize>(
    single_chunk_blocks: &[&[SingleChunkEntry]],
    challenges: &[SparseChallenge],
    num_blocks: usize,
    block_len: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_blocks) {
                let entries = single_chunk_blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);
                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    let rot = &rotated[entry.coeff_idx()];
                    for k in 0..D {
                        dst[k] += rot[k] as i32;
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

pub(super) fn multi_chunk_onehot_accumulate_tensor<const D: usize>(
    multi_chunk_blocks: &[&[MultiChunkEntry]],
    tensor: &TensorChallengeSet,
    num_blocks: usize,
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
                return Ok(Vec::new());
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i64; D]; len];
            let mut rotated = vec![[0i64; D]; D];

            for (block_idx, entries) in multi_chunk_blocks.iter().enumerate().take(num_blocks) {
                let lo = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_start);
                let hi = entries.partition_point(|e| e.pos_in_block() * num_digits < pos_end);
                if lo >= hi {
                    continue;
                }

                let (_, _, left, right) = tensor.factors_for_logical_block(block_idx)?;
                fill_rotated_tensor_challenge::<D>(&mut rotated, left, right)?;

                for entry in &entries[lo..hi] {
                    let local_pos = entry.pos_in_block() * num_digits - pos_start;
                    for &ci in entry.nonzero_coeffs() {
                        let rot = &rotated[ci as usize];
                        let dst = &mut acc[local_pos];
                        for k in 0..D {
                            dst[k] += rot[k];
                        }
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<_, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}

pub(super) fn single_chunk_onehot_accumulate_tensor<const D: usize>(
    single_chunk_blocks: &[&[SingleChunkEntry]],
    tensor: &TensorChallengeSet,
    num_blocks: usize,
    block_len: usize,
) -> Result<Vec<[i64; D]>, AkitaError> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i64; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= block_len {
                return Ok(Vec::new());
            }
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i64; D]; len];
            let mut rotated = vec![[0i64; D]; D];

            for (block_idx, entries) in single_chunk_blocks.iter().enumerate().take(num_blocks) {
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                let (_, _, left, right) = tensor.factors_for_logical_block(block_idx)?;
                fill_rotated_tensor_challenge::<D>(&mut rotated, left, right)?;

                for entry in &entries[lo..hi] {
                    let dst = &mut acc[entry.pos_in_block() - pos_start];
                    let rot = &rotated[entry.coeff_idx()];
                    for k in 0..D {
                        dst[k] += rot[k];
                    }
                }
            }

            Ok(acc)
        })
        .collect::<Result<_, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}
