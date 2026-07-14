use super::accumulate::{onehot_accumulate, onehot_accumulate_tensor};
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
    live_fold_count: usize,
    fold_position_count: usize,
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
            live_fold_count,
            fold_position_count,
        )
    };
    finish_decompose_fold(compressed_accum, num_digits)
}

impl<F: FieldCore, I: OneHotIndex> OneHotPoly<F, I> {
    pub(super) fn decompose_fold_onehot<E, const D: usize>(
        &self,
        blocks: &FlatBlocks<E>,
        challenges: &[SparseChallenge],
        fold_position_count: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F>
    where
        E: OneHotEntry,
        F: CanonicalField,
    {
        let live_fold_count = challenges.len().min(blocks.live_fold_count());
        let block_views: Vec<&[E]> = (0..blocks.live_fold_count())
            .map(|i| blocks.block(i))
            .collect();
        decompose_fold_from_views::<E, F, D>(
            &block_views,
            challenges,
            live_fold_count,
            fold_position_count,
            num_digits,
        )
    }

    pub(super) fn decompose_fold_batched_single_chunk_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        fold_position_count: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let cached_blocks = polys
            .iter()
            .map(|poly| poly.blocks_for(D, fold_position_count).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(total_blocks);
        for cached in &cached_blocks {
            let OneHotBlocks::SingleChunk(blocks) = cached.as_ref() else {
                return None;
            };
            for i in 0..blocks.live_fold_count() {
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
            fold_position_count,
            num_digits,
        ))
    }

    pub(super) fn decompose_fold_batched_multi_chunk_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        fold_position_count: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let cached_blocks = polys
            .iter()
            .map(|poly| poly.blocks_for(D, fold_position_count).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(total_blocks);
        for cached in &cached_blocks {
            let OneHotBlocks::MultiChunk(blocks) = cached.as_ref() else {
                return None;
            };
            for i in 0..blocks.live_fold_count() {
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
            fold_position_count,
            num_digits,
        ))
    }

    /// Tensor-shaped batched decompose-fold for one-hot polynomials.
    pub(super) fn decompose_fold_batched_tensor_onehot<const D: usize>(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        fold_position_count: usize,
        num_digits: usize,
    ) -> Result<Option<DecomposeFoldWitness<F>>, AkitaError>
    where
        F: CanonicalField,
    {
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        let first_blocks = first
            .blocks_for(D, fold_position_count)
            .expect("OneHotPoly::decompose_fold_batched_tensor_onehot: invalid fold_position_count for first polynomial");
        let expected_blocks = tensor.total_blocks()?;
        validate_tensor_blocks::<D>(tensor, expected_blocks)?;
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let cached_blocks = polys
            .iter()
            .map(|poly| poly.blocks_for(D, fold_position_count))
            .collect::<Result<Vec<_>, _>>()?;
        let witness = match first_blocks.as_ref() {
            OneHotBlocks::SingleChunk(_) => {
                let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(expected_blocks);
                for cached in &cached_blocks {
                    let OneHotBlocks::SingleChunk(blocks) = cached.as_ref() else {
                        return Ok(None);
                    };
                    for i in 0..blocks.live_fold_count() {
                        flat_blocks.push(blocks.block(i));
                    }
                }
                if flat_blocks.len() != expected_blocks {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_blocks,
                        actual: flat_blocks.len(),
                    });
                }
                let coeff_accum_i64 = {
                    let _span = tracing::info_span!("onehot_accumulate_tensor").entered();
                    onehot_accumulate_tensor::<SingleChunkEntry, D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        fold_position_count,
                    )?
                };
                let compressed_accum = narrow_tensor_accum_to_i32::<D>(coeff_accum_i64)?;
                let coeff_accum = expand_onehot_accum(compressed_accum, num_digits);
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
            OneHotBlocks::MultiChunk(_) => {
                let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(expected_blocks);
                for cached in &cached_blocks {
                    let OneHotBlocks::MultiChunk(blocks) = cached.as_ref() else {
                        return Ok(None);
                    };
                    for i in 0..blocks.live_fold_count() {
                        flat_blocks.push(blocks.block(i));
                    }
                }
                if flat_blocks.len() != expected_blocks {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_blocks,
                        actual: flat_blocks.len(),
                    });
                }
                let coeff_accum_i64 = {
                    let _span = tracing::info_span!("onehot_accumulate_tensor").entered();
                    onehot_accumulate_tensor::<MultiChunkEntry, D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        fold_position_count,
                    )?
                };
                let compressed_accum = narrow_tensor_accum_to_i32::<D>(coeff_accum_i64)?;
                let coeff_accum = expand_onehot_accum(compressed_accum, num_digits);
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
        };
        Ok(Some(witness))
    }
}
