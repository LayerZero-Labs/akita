use super::accumulate::onehot_accumulate;
use super::*;

const BATCH_BLOCK_TILE: usize = 64;

fn expand_onehot_accum<const D: usize>(
    compressed: Vec<[i32; D]>,
    num_digits: usize,
) -> Vec<[i32; D]> {
    if num_digits == 1 {
        return compressed;
    }

    let mut expanded = Vec::with_capacity(compressed.len().saturating_mul(num_digits));
    for coeffs in compressed {
        expanded.push(coeffs);
        for _ in 1..num_digits {
            expanded.push([0i32; D]);
        }
    }
    expanded
}

fn finish_decompose_fold<F: CanonicalField, const D: usize>(
    compressed_accum: Vec<[i32; D]>,
    num_digits: usize,
) -> DecomposeFoldWitness<F> {
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let coeff_accum = {
        let _span = tracing::info_span!("onehot_expand_accum").entered();
        expand_onehot_accum(compressed_accum, num_digits)
    };
    let _span = tracing::info_span!("onehot_convert").entered();
    build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
}

fn decompose_fold_from_views<F, const D: usize>(
    block_views: &[&[SparseRingBlockEntry]],
    challenges: &[SparseChallenge],
    num_live_blocks: usize,
    num_positions_per_block: usize,
    num_digits: usize,
) -> DecomposeFoldWitness<F>
where
    F: CanonicalField,
{
    let compressed_accum = {
        let _span = tracing::info_span!("onehot_accumulate").entered();
        onehot_accumulate::<D>(
            block_views,
            challenges,
            num_live_blocks,
            num_positions_per_block,
        )
    };
    finish_decompose_fold(compressed_accum, num_digits)
}

impl<F: FieldCore, I: OneHotIndex> OneHotPoly<F, I> {
    pub(super) fn decompose_fold_onehot<const D: usize>(
        &self,
        blocks: &FlatBlocks<SparseRingBlockEntry>,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F>
    where
        F: CanonicalField,
    {
        let num_live_blocks = challenges.len().min(blocks.num_live_blocks());
        let block_views: Vec<&[SparseRingBlockEntry]> =
            (0..num_live_blocks).map(|i| blocks.block(i)).collect();
        decompose_fold_from_views::<F, D>(
            &block_views,
            challenges,
            num_live_blocks,
            num_positions_per_block,
            num_digits,
        )
    }

    pub(super) fn decompose_fold_batched_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: CanonicalField,
    {
        let mut challenge_start = 0usize;
        let mut compressed = vec![[0i32; D]; num_positions_per_block];
        for poly in polys {
            if challenge_start == challenges.len() {
                break;
            }
            let (_, num_blocks) = poly.view_layout(D, num_positions_per_block).ok()?;
            let active_blocks = num_blocks.min(challenges.len() - challenge_start);
            if active_blocks == 0 {
                continue;
            }
            for block_start in (0..active_blocks).step_by(BATCH_BLOCK_TILE) {
                let block_end = (block_start + BATCH_BLOCK_TILE).min(active_blocks);
                let blocks = poly
                    .materialize_block_range(D, num_positions_per_block, block_start..block_end)
                    .ok()?;
                let tile_len = block_end - block_start;
                let views = (0..tile_len)
                    .map(|block| blocks.block(block))
                    .collect::<Vec<_>>();
                let challenge_tile_start = challenge_start + block_start;
                let part = onehot_accumulate::<D>(
                    &views,
                    &challenges[challenge_tile_start..challenge_tile_start + tile_len],
                    tile_len,
                    num_positions_per_block,
                );
                for (dst, src) in compressed.iter_mut().zip(part) {
                    for (dst_coeff, src_coeff) in dst.iter_mut().zip(src) {
                        *dst_coeff += src_coeff;
                    }
                }
            }
            challenge_start += active_blocks;
        }
        if challenge_start == 0 {
            return None;
        }
        Some(finish_decompose_fold(compressed, num_digits))
    }
}
