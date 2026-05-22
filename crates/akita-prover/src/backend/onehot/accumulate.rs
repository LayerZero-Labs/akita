use super::SingleChunkEntry;
use crate::backend::poly_helpers::fill_rotated_challenge;
use crate::CenteredCoeff;
use akita_challenges::IntegerChallenge;
use akita_field::parallel::*;

/// Position-partitioned accumulation for single-chunk one-hot witnesses,
/// where each nonzero ring element carries exactly one hot coefficient.
///
/// See [`multi_chunk_onehot_accumulate`] for the block-view convention.
pub(super) fn single_chunk_onehot_accumulate<const D: usize>(
    single_chunk_blocks: &[&[SingleChunkEntry]],
    challenges: &[IntegerChallenge],
    num_blocks: usize,
    block_len: usize,
) -> Vec<[CenteredCoeff; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(block_len).max(1);
    let pos_chunk = block_len.div_ceil(actual_threads);

    let chunks: Vec<Vec<[CenteredCoeff; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            let pos_end = (pos_start + pos_chunk).min(block_len);
            let len = pos_end - pos_start;
            let mut acc = vec![[0 as CenteredCoeff; D]; len];
            let mut rotated = vec![[0 as CenteredCoeff; D]; D];

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
                        dst[k] += rot[k];
                    }
                }
            }

            acc
        })
        .collect();

    chunks.into_iter().flatten().collect()
}
