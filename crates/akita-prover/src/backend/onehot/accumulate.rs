use super::*;

/// Accumulates one-hot decompose-fold rows in compressed position order.
///
/// The returned vector has `num_positions_per_block` rows. Callers expand each row across
/// `num_digits` later, inserting zero rows for higher digit planes.
///
/// `blocks` is a slice-of-slices view over per-block entries. Both
/// single-polynomial callers (which collect once via `FlatBlocks::block`)
/// and batched callers (which concatenate slices across polynomials) feed
/// through the same signature.
pub(super) fn onehot_accumulate<const D: usize>(
    blocks: &[&[SparseRingBlockEntry]],
    challenges: &[SparseChallenge],
    num_live_blocks: usize,
    num_positions_per_block: usize,
) -> Vec<[i32; D]> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(num_positions_per_block).max(1);
    let pos_chunk = num_positions_per_block.div_ceil(actual_threads);

    let chunks: Vec<Vec<[i32; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= num_positions_per_block {
                return Vec::new();
            }
            let pos_end = (pos_start + pos_chunk).min(num_positions_per_block);
            let len = pos_end - pos_start;
            let mut acc = vec![[0i32; D]; len];
            let mut rotated = vec![[0i16; D]; D];

            for (block_idx, challenge) in challenges.iter().enumerate().take(num_live_blocks) {
                let entries = blocks[block_idx];
                let lo = entries.partition_point(|entry| entry.pos_in_block() < pos_start);
                let hi = entries.partition_point(|entry| entry.pos_in_block() < pos_end);
                if lo >= hi {
                    continue;
                }

                fill_rotated_challenge::<D>(&mut rotated, challenge);

                for entry in &entries[lo..hi] {
                    let pos_in_block = entry.pos_in_block();
                    let coeff_idx = entry.coeff_idx();
                    let dst = &mut acc[pos_in_block - pos_start];
                    let rot = &rotated[coeff_idx];
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
