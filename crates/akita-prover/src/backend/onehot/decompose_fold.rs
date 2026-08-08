use super::accumulate::onehot_accumulate;
use super::*;

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

fn decompose_fold_from_views<E, F, const D: usize>(
    block_views: &[&[E]],
    challenges: &[SparseChallenge],
    num_live_blocks: usize,
    num_positions_per_block: usize,
    num_digits: usize,
) -> DecomposeFoldWitness<F>
where
    E: OneHotEntry,
    F: CanonicalField,
{
    let compressed_accum = {
        let _span = tracing::info_span!("onehot_accumulate").entered();
        onehot_accumulate::<E, D>(
            block_views,
            challenges,
            num_live_blocks,
            num_positions_per_block,
        )
    };
    finish_decompose_fold(compressed_accum, num_digits)
}

impl<F: FieldCore, I: OneHotIndex> OneHotPoly<F, I> {
    pub(super) fn decompose_fold_onehot<E, const D: usize>(
        &self,
        blocks: &FlatBlocks<E>,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F>
    where
        E: OneHotEntry,
        F: CanonicalField,
    {
        let num_live_blocks = challenges.len().min(blocks.num_live_blocks());
        let block_views: Vec<&[E]> = (0..blocks.num_live_blocks())
            .map(|i| blocks.block(i))
            .collect();
        decompose_fold_from_views::<E, F, D>(
            &block_views,
            challenges,
            num_live_blocks,
            num_positions_per_block,
            num_digits,
        )
    }

    pub(super) fn decompose_fold_batched_single_chunk_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let operation_blocks = polys
            .iter()
            .map(|poly| poly.blocks_for_operation(D, num_positions_per_block).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(total_blocks);
        for operation in &operation_blocks {
            let OneHotBlocks::SingleChunk(blocks) = operation.as_ref() else {
                return None;
            };
            for i in 0..blocks.num_live_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        Some(decompose_fold_from_views::<SingleChunkEntry, F, D>(
            &flat_blocks,
            challenges,
            active_blocks,
            num_positions_per_block,
            num_digits,
        ))
    }

    pub(super) fn decompose_fold_batched_multi_chunk_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let operation_blocks = polys
            .iter()
            .map(|poly| poly.blocks_for_operation(D, num_positions_per_block).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(total_blocks);
        for operation in &operation_blocks {
            let OneHotBlocks::MultiChunk(blocks) = operation.as_ref() else {
                return None;
            };
            for i in 0..blocks.num_live_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        Some(decompose_fold_from_views::<MultiChunkEntry, F, D>(
            &flat_blocks,
            challenges,
            active_blocks,
            num_positions_per_block,
            num_digits,
        ))
    }
}
