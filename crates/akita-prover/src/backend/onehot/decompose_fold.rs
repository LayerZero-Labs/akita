use super::accumulate::{
    multi_chunk_onehot_accumulate, multi_chunk_onehot_accumulate_tensor,
    single_chunk_onehot_accumulate, single_chunk_onehot_accumulate_tensor,
};
use super::*;

impl<F: FieldCore, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I> {
    pub(super) fn decompose_fold_single_chunk_onehot(
        &self,
        single_chunk_blocks: &FlatBlocks<SingleChunkEntry>,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let num_blocks = challenges.len().min(single_chunk_blocks.num_blocks());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let block_views: Vec<&[SingleChunkEntry]> = (0..single_chunk_blocks.num_blocks())
            .map(|i| single_chunk_blocks.block(i))
            .collect();

        let coeff_accum_digit0: Vec<[i32; D]> = {
            let _span = tracing::info_span!("onehot_single_chunk_accumulate").entered();
            single_chunk_onehot_accumulate::<D>(&block_views, challenges, num_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_single_chunk_expand").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_single_chunk_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    pub(super) fn decompose_fold_multi_chunk_onehot(
        &self,
        multi_chunk_blocks: &FlatBlocks<MultiChunkEntry>,
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> DecomposeFoldWitness<F, D>
    where
        F: CanonicalField,
    {
        let inner_width = block_len * num_digits;
        let num_blocks = challenges.len().min(multi_chunk_blocks.num_blocks());
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let block_views: Vec<&[MultiChunkEntry]> = (0..multi_chunk_blocks.num_blocks())
            .map(|i| multi_chunk_blocks.block(i))
            .collect();

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_multi_chunk_accumulate").entered();
            multi_chunk_onehot_accumulate::<D>(
                &block_views,
                challenges,
                num_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_multi_chunk_convert").entered();
        build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
    }

    pub(super) fn decompose_fold_batched_single_chunk_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            // `blocks_for` was already called by the public batched entry
            // point; this just reads the cached layout.
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::SingleChunk(blocks) = cached else {
                return None;
            };
            for i in 0..blocks.num_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let coeff_accum_digit0 = {
            let _span = tracing::info_span!("onehot_single_chunk_accumulate_batched").entered();
            single_chunk_onehot_accumulate::<D>(&flat_blocks, challenges, active_blocks, block_len)
        };

        let coeff_accum = if num_digits == 1 {
            coeff_accum_digit0
        } else {
            let _span = tracing::info_span!("onehot_single_chunk_expand_batched").entered();
            let mut expanded = Vec::with_capacity(block_len * num_digits);
            for coeffs in coeff_accum_digit0 {
                expanded.push(coeffs);
                for _ in 1..num_digits {
                    expanded.push([0i32; D]);
                }
            }
            expanded
        };

        let _span = tracing::info_span!("onehot_single_chunk_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }

    pub(super) fn decompose_fold_batched_multi_chunk_onehot(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        block_len: usize,
        num_digits: usize,
    ) -> Option<DecomposeFoldWitness<F, D>>
    where
        F: CanonicalField,
    {
        let total_blocks = challenges.len();
        let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(total_blocks);
        for poly in polys {
            let (_, cached) = poly.block_cache.get()?;
            let OneHotBlocks::MultiChunk(blocks) = cached else {
                return None;
            };
            for i in 0..blocks.num_blocks() {
                flat_blocks.push(blocks.block(i));
            }
        }
        if flat_blocks.is_empty() {
            return None;
        }
        let active_blocks = flat_blocks.len().min(total_blocks);
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let inner_width = block_len * num_digits;

        let coeff_accum = {
            let _span = tracing::info_span!("onehot_multi_chunk_accumulate_batched").entered();
            multi_chunk_onehot_accumulate::<D>(
                &flat_blocks,
                challenges,
                active_blocks,
                inner_width,
                num_digits,
            )
        };

        let _span = tracing::info_span!("onehot_multi_chunk_convert_batched").entered();
        Some(build_decompose_fold_witness::<F, D>(coeff_accum, modulus))
    }

    /// Tensor-shaped batched decompose-fold for one-hot polynomials.
    pub(super) fn decompose_fold_batched_tensor_onehot(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        block_len: usize,
        num_digits: usize,
    ) -> Result<Option<DecomposeFoldWitness<F, D>>, AkitaError>
    where
        F: CanonicalField,
    {
        for poly in polys {
            poly.blocks_for(block_len).expect(
                "OneHotPoly::decompose_fold_batched_tensor_onehot: invalid block_len for one polynomial",
            );
        }
        let Some(first) = polys.first() else {
            return Ok(None);
        };
        let (_, first_blocks) = first
            .block_cache
            .get()
            .expect("block cache was just built above");
        let expected_blocks = tensor
            .left_len
            .checked_mul(tensor.right_len)
            .and_then(|blocks| blocks.checked_mul(tensor.num_claims))
            .ok_or_else(|| AkitaError::InvalidSetup("tensor challenge count overflow".into()))?;
        let modulus = (-F::one()).to_canonical_u128() + 1;

        let witness = match first_blocks {
            OneHotBlocks::SingleChunk(_) => {
                let mut flat_blocks: Vec<&[SingleChunkEntry]> = Vec::with_capacity(expected_blocks);
                for poly in polys {
                    let (_, cached) = poly.block_cache.get().expect("block cache exists");
                    let OneHotBlocks::SingleChunk(blocks) = cached else {
                        return Ok(None);
                    };
                    for i in 0..blocks.num_blocks() {
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
                    let _span =
                        tracing::info_span!("onehot_single_chunk_accumulate_tensor").entered();
                    single_chunk_onehot_accumulate_tensor::<D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        block_len,
                    )?
                };
                let coeff_accum_digit0 = narrow_tensor_accum_to_i32::<D>(coeff_accum_i64)?;
                let coeff_accum = if num_digits == 1 {
                    coeff_accum_digit0
                } else {
                    let _span = tracing::info_span!("onehot_single_chunk_expand_tensor").entered();
                    let mut expanded = Vec::with_capacity(block_len * num_digits);
                    for coeffs in coeff_accum_digit0 {
                        expanded.push(coeffs);
                        for _ in 1..num_digits {
                            expanded.push([0i32; D]);
                        }
                    }
                    expanded
                };
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
            OneHotBlocks::MultiChunk(_) => {
                let mut flat_blocks: Vec<&[MultiChunkEntry]> = Vec::with_capacity(expected_blocks);
                for poly in polys {
                    let (_, cached) = poly.block_cache.get().expect("block cache exists");
                    let OneHotBlocks::MultiChunk(blocks) = cached else {
                        return Ok(None);
                    };
                    for i in 0..blocks.num_blocks() {
                        flat_blocks.push(blocks.block(i));
                    }
                }
                if flat_blocks.len() != expected_blocks {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_blocks,
                        actual: flat_blocks.len(),
                    });
                }
                let inner_width = block_len * num_digits;
                let coeff_accum_i64 = {
                    let _span =
                        tracing::info_span!("onehot_multi_chunk_accumulate_tensor").entered();
                    multi_chunk_onehot_accumulate_tensor::<D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        inner_width,
                        num_digits,
                    )?
                };
                let coeff_accum = narrow_tensor_accum_to_i32::<D>(coeff_accum_i64)?;
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
        };
        Ok(Some(witness))
    }
}
