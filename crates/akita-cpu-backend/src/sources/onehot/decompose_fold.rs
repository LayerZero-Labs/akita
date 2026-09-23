use super::*;

mod rotations;

use rotations::{add_rotated, prepare_rotations, PreparedRotations};

#[cfg(feature = "parallel")]
const TASKS_PER_RAYON_WORKER: usize = 4;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;

struct DecomposeSource<'a, F: Field, I: OneHotIndex> {
    poly: &'a OneHotPoly<F, I>,
    challenge_start: usize,
    active_blocks: usize,
    ring_elems: usize,
}

fn decompose_position_chunk<F, I, const D: usize>(
    sources: &[DecomposeSource<'_, F, I>],
    num_positions_per_block: usize,
    _independent_buffers: usize,
) -> usize
where
    F: Field,
    I: OneHotIndex,
{
    let row_alignment = sources
        .iter()
        .map(|source| (source.poly.onehot_k / D).max(1))
        .max()
        .unwrap_or(1);
    #[cfg(feature = "parallel")]
    let target_tasks = rayon::current_num_threads()
        .saturating_mul(TASKS_PER_RAYON_WORKER)
        .min(num_positions_per_block.saturating_mul(_independent_buffers))
        .max(1)
        .div_ceil(_independent_buffers);
    #[cfg(not(feature = "parallel"))]
    let target_tasks = 1usize;
    let thread_balanced_chunk = num_positions_per_block
        .div_ceil(target_tasks)
        .next_multiple_of(row_alignment);
    let cache_sized_chunk = (DECOMPOSE_POSITION_WORKING_SET_TARGET
        / std::mem::size_of::<[i32; D]>())
    .max(row_alignment)
    .next_multiple_of(row_alignment);
    thread_balanced_chunk
        .min(cache_sized_chunk)
        .min(num_positions_per_block)
}

#[inline]
fn accumulate_ring_range<F, I, const D: usize>(
    source: &DecomposeSource<'_, F, I>,
    ring_start: usize,
    ring_end: usize,
    block_start: usize,
    challenge_idx: usize,
    dst: &mut [[i32; D]],
    rotations: &PreparedRotations<'_, D>,
) where
    F: Field,
    I: OneHotIndex,
{
    let poly = source.poly;
    let onehot_k = poly.onehot_k;
    if onehot_k == D {
        for (ring, hot) in poly.indices[ring_start..ring_end]
            .iter()
            .copied()
            .enumerate()
        {
            if let Some(hot) = hot {
                add_rotated(
                    &mut dst[ring_start + ring - block_start],
                    rotations,
                    challenge_idx,
                    hot.as_usize(),
                );
            }
        }
    } else if onehot_k > D {
        let rings_per_chunk = onehot_k / D;
        let chunk_start = ring_start / rings_per_chunk;
        let chunk_end = ring_end.div_ceil(rings_per_chunk);
        for (chunk, hot) in poly.indices[chunk_start..chunk_end]
            .iter()
            .copied()
            .enumerate()
        {
            let Some(hot) = hot else {
                continue;
            };
            let hot = hot.as_usize();
            let ring = (chunk_start + chunk) * rings_per_chunk + hot / D;
            if ring_start <= ring && ring < ring_end {
                add_rotated(
                    &mut dst[ring - block_start],
                    rotations,
                    challenge_idx,
                    hot % D,
                );
            }
        }
    } else {
        let chunks_per_ring = D / onehot_k;
        let chunk_start = ring_start * chunks_per_ring;
        let chunk_end = ring_end * chunks_per_ring;
        for (chunk, hot) in poly.indices[chunk_start..chunk_end]
            .iter()
            .copied()
            .enumerate()
        {
            if let Some(hot) = hot {
                let local_chunk = chunk_start + chunk;
                let ring = local_chunk / chunks_per_ring;
                let lane = local_chunk % chunks_per_ring;
                add_rotated(
                    &mut dst[ring - block_start],
                    rotations,
                    challenge_idx,
                    lane * onehot_k + hot.as_usize(),
                );
            }
        }
    }
}

fn accumulate_indices<F, I, const D: usize>(
    sources: &[DecomposeSource<'_, F, I>],
    challenges: &[SparseChallenge],
    num_positions_per_block: usize,
) -> Vec<[i32; D]>
where
    F: Field,
    I: OneHotIndex,
{
    let rotations = {
        let _span = tracing::info_span!(
            "onehot_prepare_rotations",
            challenges = challenges.len(),
            ring_dimension = D,
        )
        .entered();
        prepare_rotations::<D>(challenges)
    };
    let position_chunk = decompose_position_chunk::<F, I, D>(sources, num_positions_per_block, 1);
    let position_tasks = num_positions_per_block.div_ceil(position_chunk);
    let _span = tracing::info_span!(
        "onehot_accumulate_indices",
        sources = sources.len(),
        challenges = challenges.len(),
        ring_dimension = D,
        rotation_kind = rotations.kind(),
        position_tasks,
        position_chunk,
    )
    .entered();
    let mut compressed = vec![[0i32; D]; num_positions_per_block];
    cfg_chunks_mut!(&mut compressed, position_chunk)
        .enumerate()
        .for_each(|(position_task, dst)| {
            let position_start = position_task * position_chunk;
            let position_end = position_start + dst.len();
            for source in sources {
                for block in 0..source.active_blocks {
                    let block_base = block * num_positions_per_block;
                    let ring_start = (block_base + position_start).min(source.ring_elems);
                    let ring_end = (block_base + position_end).min(source.ring_elems);
                    if ring_start >= ring_end {
                        continue;
                    }
                    accumulate_ring_range(
                        source,
                        ring_start,
                        ring_end,
                        block_base + position_start,
                        source.challenge_start + block,
                        dst,
                        &rotations,
                    );
                }
            }
        });
    compressed
}

fn accumulate_indices_chunked<F, I, const D: usize>(
    sources: &[DecomposeSource<'_, F, I>],
    challenges: &[SparseChallenge],
    chunk_ranges: &[std::ops::Range<usize>],
    num_positions_per_block: usize,
) -> Vec<Vec<[i32; D]>>
where
    F: Field,
    I: OneHotIndex,
{
    if chunk_ranges.is_empty() {
        return Vec::new();
    }
    let rotations = {
        let _span = tracing::info_span!(
            "onehot_prepare_rotations",
            challenges = challenges.len(),
            ring_dimension = D,
        )
        .entered();
        prepare_rotations::<D>(challenges)
    };
    let num_chunks = chunk_ranges.len();
    let position_chunk =
        decompose_position_chunk::<F, I, D>(sources, num_positions_per_block, num_chunks);
    let _span = tracing::info_span!(
        "onehot_accumulate_indices_chunked",
        sources = sources.len(),
        challenges = challenges.len(),
        ring_dimension = D,
        rotation_kind = rotations.kind(),
        chunks = num_chunks,
        position_tasks = num_positions_per_block
            .div_ceil(position_chunk)
            .saturating_mul(num_chunks),
        position_chunk,
    )
    .entered();
    let mut chunks = vec![vec![[0i32; D]; num_positions_per_block]; num_chunks];
    cfg_iter_mut!(&mut chunks)
        .enumerate()
        .for_each(|(chunk_index, buffer)| {
            cfg_chunks_mut!(buffer, position_chunk)
                .enumerate()
                .for_each(|(position_task, dst)| {
                    let position_start = position_task * position_chunk;
                    let position_end = position_start + dst.len();
                    for source in sources {
                        for block in chunk_ranges[chunk_index].clone() {
                            if block >= source.active_blocks {
                                break;
                            }
                            let block_base = block * num_positions_per_block;
                            let ring_start = (block_base + position_start).min(source.ring_elems);
                            let ring_end = (block_base + position_end).min(source.ring_elems);
                            if ring_start >= ring_end {
                                continue;
                            }
                            accumulate_ring_range(
                                source,
                                ring_start,
                                ring_end,
                                block_base + position_start,
                                source.challenge_start + block,
                                dst,
                                &rotations,
                            );
                        }
                    }
                });
        });
    chunks
}

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

pub(super) fn finish_decompose_fold<const D: usize>(
    compressed_accum: Vec<[i32; D]>,
    num_digits: usize,
) -> DecomposeFoldWitness {
    let coeff_accum = {
        let _span = tracing::info_span!("onehot_expand_accum").entered();
        expand_onehot_accum(compressed_accum, num_digits)
    };
    DecomposeFoldWitness::from_centered_rows(coeff_accum)
}

impl<F: Field, I: OneHotIndex> OneHotPoly<F, I> {
    /// Validate a fused decompose-fold batch and lay out its sources.
    ///
    /// Each polynomial consumes exactly `challenges_per_poly` consecutive
    /// challenges and must have exactly that many live blocks.
    fn decompose_sources<'a, const D: usize>(
        polys: &[&'a Self],
        challenges: &[SparseChallenge],
        challenges_per_poly: usize,
        num_positions_per_block: usize,
    ) -> Result<Vec<DecomposeSource<'a, F, I>>, AkitaError> {
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "one-hot decompose_fold requires at least one polynomial".into(),
            ));
        }
        let expected = akita_error::checked::product([polys.len(), challenges_per_poly])
            .ok_or_else(|| {
                AkitaError::InvalidInput("one-hot decompose_fold challenge count overflow".into())
            })?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        let mut sources = Vec::with_capacity(polys.len());
        for (index, &poly) in polys.iter().enumerate() {
            let (ring_elems, num_blocks) = poly.view_layout(D, num_positions_per_block)?;
            if num_blocks != challenges_per_poly {
                return Err(AkitaError::InvalidSize {
                    expected: num_blocks,
                    actual: challenges_per_poly,
                });
            }
            sources.push(DecomposeSource {
                poly,
                challenge_start: index * challenges_per_poly,
                active_blocks: challenges_per_poly,
                ring_elems,
            });
        }
        Ok(sources)
    }

    /// Fused decompose-fold returning one witness per block range in
    /// `chunk_ranges`.
    pub(super) fn decompose_fold_batched_chunked_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        challenges_per_poly: usize,
        chunk_ranges: &[std::ops::Range<usize>],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Result<Vec<DecomposeFoldWitness>, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        let sources = Self::decompose_sources::<D>(
            polys,
            challenges,
            challenges_per_poly,
            num_positions_per_block,
        )?;
        let accumulators = accumulate_indices_chunked::<F, I, D>(
            &sources,
            challenges,
            chunk_ranges,
            num_positions_per_block,
        );
        Ok(cfg_into_iter!(accumulators)
            .map(|accumulator| finish_decompose_fold(accumulator, num_digits))
            .collect())
    }

    /// Fused decompose-fold of `polys`, each consuming exactly
    /// `challenges_per_poly` consecutive challenges.
    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_batched")]
    pub(super) fn decompose_fold_batched_onehot<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        challenges_per_poly: usize,
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Result<DecomposeFoldWitness, AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        let sources = Self::decompose_sources::<D>(
            polys,
            challenges,
            challenges_per_poly,
            num_positions_per_block,
        )?;
        let compressed =
            accumulate_indices::<F, I, D>(&sources, challenges, num_positions_per_block);
        Ok(finish_decompose_fold(compressed, num_digits))
    }
}
