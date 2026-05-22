use super::{MultiChunkEntry, OneHotBlocks, OneHotIndex, OneHotPoly, SingleChunkEntry};
use crate::backend::poly_helpers::build_decompose_fold_witness;
use crate::backend::tensor_fold::fill_rotated_tensor_challenge;
use crate::{CenteredCoeff, DecomposeFoldWitness};
use akita_challenges::TensorChallengeSet;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};

impl<F, const D: usize, I: OneHotIndex> OneHotPoly<F, D, I>
where
    F: FieldCore + CanonicalField,
{
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
        tensor.validate::<D>()?;
        let expected_blocks = tensor.total_blocks()?;
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
                let coeff_accum_digit0 = {
                    let _span =
                        tracing::info_span!("onehot_single_chunk_accumulate_tensor").entered();
                    single_chunk_onehot_accumulate_tensor::<D>(
                        &flat_blocks,
                        tensor,
                        expected_blocks,
                        block_len,
                    )?
                };
                let coeff_accum = if num_digits == 1 {
                    coeff_accum_digit0
                } else {
                    let _span = tracing::info_span!("onehot_single_chunk_expand_tensor").entered();
                    let mut expanded = Vec::with_capacity(block_len * num_digits);
                    for coeffs in coeff_accum_digit0 {
                        expanded.push(coeffs);
                        for _ in 1..num_digits {
                            expanded.push([0 as CenteredCoeff; D]);
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
                let coeff_accum = {
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
                build_decompose_fold_witness::<F, D>(coeff_accum, modulus)
            }
        };
        Ok(Some(witness))
    }
}

pub(super) fn multi_chunk_onehot_accumulate_tensor<const D: usize>(
    multi_chunk_blocks: &[&[MultiChunkEntry]],
    tensor: &TensorChallengeSet,
    num_blocks: usize,
    inner_width: usize,
    num_digits: usize,
) -> Result<Vec<[CenteredCoeff; D]>, AkitaError> {
    #[cfg(feature = "parallel")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let num_threads = 1;

    let actual_threads = num_threads.min(inner_width.max(1));
    let pos_chunk = inner_width.div_ceil(actual_threads);

    let chunks: Vec<Vec<[CenteredCoeff; D]>> = cfg_into_iter!(0..actual_threads)
        .map(|tid| {
            let pos_start = tid * pos_chunk;
            if pos_start >= inner_width {
                return Ok(Vec::new());
            }
            let pos_end = (pos_start + pos_chunk).min(inner_width);
            let len = pos_end - pos_start;
            let mut acc = vec![[0 as CenteredCoeff; D]; len];
            let mut rotated = vec![[0 as CenteredCoeff; D]; D];

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
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}

pub(super) fn single_chunk_onehot_accumulate_tensor<const D: usize>(
    single_chunk_blocks: &[&[SingleChunkEntry]],
    tensor: &TensorChallengeSet,
    num_blocks: usize,
    block_len: usize,
) -> Result<Vec<[CenteredCoeff; D]>, AkitaError> {
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
        .collect::<Result<Vec<_>, AkitaError>>()?;

    Ok(chunks.into_iter().flatten().collect())
}
